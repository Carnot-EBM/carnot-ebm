"""Exp6448 V554 terminal handoff and V555 queue integrity.

Spec refs: REQ-REPORT-6448, SCENARIO-REPORT-6448-V554-FREEZE,
SCENARIO-REPORT-6448-V555-QUEUE,
SCENARIO-REPORT-6448-GATES-PRIORS-MODELS,
SCENARIO-REPORT-6448-SCHEMA.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
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
MILESTONE_V554 = "2026.08.554"
MILESTONE_V555 = "2026.08.555"
RUN_DATE = "20260815"
EXPERIMENT_ID = "exp6448-v555-terminal-handoff-and-queue-integrity"
SCHEMA = "carnot.experiment_6448.v555_terminal_handoff_and_queue_integrity.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6448_v555_terminal_handoff_and_queue_integrity.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ROADMAP_SCHEMA_RELATIVE_PATH = Path("scripts/roadmap_schema.py")
PRIOR_FAILURE_LINT_RELATIVE_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_RELATIVE_PATH = Path("scripts/exclusion_manifest_lint.py")
ROADMAP_GATE_AUDIT_RELATIVE_PATH = Path("scripts/audit_roadmap_gates.py")
CONDUCTOR_GATES_RELATIVE_PATH = Path("scripts/conductor_gates.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")

EXPECTED_V554_ACTIVATED_TASK_IDS = (
    "exp6436-v554-terminal-handoff-and-queue-preflight",
    "exp6437-generation-to-verdict-receipt-replay-contract",
    "exp6438-powered-verification-cost-repair-ab",
    "exp6439-factor-clause-influence-ab",
    "exp6440-held-factor-revocation-binding-shift-ab",
    "exp6441-prospective-query-conditioned-factor-reuse",
    "exp6442-skill-misevolution-quarantine-rollback-ab",
    "exp6443-fresh-held-restart-csl-replication",
    "exp6444-csl-lifecycle-recomputation-audit",
)
V554_DELIVERABLES_BY_TASK = {
    "exp6436-v554-terminal-handoff-and-queue-preflight": (
        "results/experiment_6436_v554_terminal_handoff_and_queue_preflight.json"
    ),
    "exp6437-generation-to-verdict-receipt-replay-contract": (
        "results/experiment_6437_generation_to_verdict_receipt_replay_contract.json"
    ),
    "exp6438-powered-verification-cost-repair-ab": (
        "results/experiment_6438_powered_verification_cost_repair_ab.json"
    ),
    "exp6439-factor-clause-influence-ab": (
        "results/experiment_6439_factor_clause_influence_ab.json"
    ),
    "exp6440-held-factor-revocation-binding-shift-ab": (
        "results/experiment_6440_held_factor_revocation_binding_shift_ab.json"
    ),
    "exp6441-prospective-query-conditioned-factor-reuse": (
        "results/experiment_6441_prospective_query_conditioned_factor_reuse.json"
    ),
    "exp6442-skill-misevolution-quarantine-rollback-ab": (
        "results/experiment_6442_skill_misevolution_quarantine_rollback_ab.json"
    ),
    "exp6443-fresh-held-restart-csl-replication": (
        "results/experiment_6443_fresh_held_restart_csl_replication.json"
    ),
    "exp6444-csl-lifecycle-recomputation-audit": (
        "results/experiment_6444_csl_lifecycle_recomputation_audit.json"
    ),
}

EXPECTED_V555_TASK_IDS = (
    "exp6448-v555-terminal-handoff-and-queue-integrity",
    "exp6449-generation-to-verdict-path-receipt-contract",
    "exp6450-sota-fixed-policy-candidate-corpus",
    "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
    "exp6452-representation-objective-causal-ab",
    "exp6453-held-verifier-budget-allocation-ab",
    "exp6454-held-exact-constraint-energy-selection-ab",
    "exp6455-prospective-verifier-bounded-factor-weight-csl",
    "exp6456-corrupt-feedback-held-restart-csl-replication",
    "exp6457-independent-verifier-bounded-csl-audit",
    "exp6458-arc-representation-objective-generalization-ab",
    "exp6459-v555-adversarial-capstone",
)
REQUIRED_PRIOR_FAILURE_TASK_IDS = (
    "exp6449-generation-to-verdict-path-receipt-contract",
    "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
    "exp6453-held-verifier-budget-allocation-ab",
    "exp6454-held-exact-constraint-energy-selection-ab",
    "exp6456-corrupt-feedback-held-restart-csl-replication",
    "exp6457-independent-verifier-bounded-csl-audit",
    "exp6458-arc-representation-objective-generalization-ab",
)
LLM_TASK_IDS = (
    "exp6450-sota-fixed-policy-candidate-corpus",
    "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
    "exp6455-prospective-verifier-bounded-factor-weight-csl",
    "exp6456-corrupt-feedback-held-restart-csl-replication",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6448_v555_terminal_handoff_and_queue_integrity "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6448_v555_terminal_handoff_and_queue_integrity.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6448_v555_terminal_handoff_and_queue_integrity.py "
    "-m pytest "
    "tests/python/test_experiment_6448_v555_terminal_handoff_and_queue_integrity.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6448_v555_terminal_handoff_and_queue_integrity.py "
    "--fail-under=100 --show-missing"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6448_v555_terminal_handoff_and_queue_integrity.py",
    ".venv/bin/python -c 'import yaml; from pathlib import Path; "
    "from scripts.roadmap_schema import Roadmap; "
    'Roadmap.model_validate(yaml.safe_load(Path("research-roadmap.yaml").read_text()))'
    "'",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml --complete research-complete.yaml",
    ".venv/bin/python scripts/artifact_convention_audit.py",
    ".venv/bin/python scripts/determination_preservation_lint.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
    RUN_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6448_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    RESEARCH_HARDWARE_WISHLIST_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    ROADMAP_SCHEMA_RELATIVE_PATH,
    PRIOR_FAILURE_LINT_RELATIVE_PATH,
    EXCLUSION_LINT_RELATIVE_PATH,
    ROADMAP_GATE_AUDIT_RELATIVE_PATH,
    CONDUCTOR_GATES_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *[Path(path) for path in V554_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v554_terminal_rows",
    "v554_activated_task_count",
    "v554_missing_zero_byte_or_blocked_artifacts",
    "v554_terminal_claim_determinations",
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
    "v555_queue_integrity_score",
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
    "status": "The status states whether the V555 queue passed or failed closed.",
    "v554_terminal_rows": "Each activated V554 task row preserves its exact artifact state.",
    "v554_activated_task_count": "The V554 denominator comes from Exp6436, not from reconstruction.",
    "v554_missing_zero_byte_or_blocked_artifacts": "Absent and blocked V554 evidence stays visible.",
    "v554_terminal_claim_determinations": "V554 claim boundaries remain blocked unless terminal evidence allows them.",
    "active_roadmap_hash": "The activated V555 queue is content-addressed.",
    "task_count": "V555 readiness requires exactly twelve active tasks.",
    "task_ids_in_order": "Conductor order must stay deterministic.",
    "unique_id_and_deliverable_check": "IDs and result JSON deliverables define the queue identity.",
    "milestone_consistency_check": "Every task must belong to milestone 2026.08.555.",
    "schema_validation": "The Pydantic roadmap schema must accept the activated queue.",
    "prior_failure_validation": "Rerun scopes must state the prior verdict and retirement rule.",
    "exclusion_manifest_validation": "Retired IDs and hard exclusion risks block the queue.",
    "structured_gate_validation": "Every gate must name a valid upstream task, field, operator, and value.",
    "gate_producer_contract_rows": "Each gate field must be declared by its producer artifact contract.",
    "model_policy_validation": "LLM tasks must use local GGUF policy without AutoTokenizer headline paths.",
    "per_unit_row_contract_validation": "Comparative tasks need row-level evidence, not aggregates only.",
    "prompt_terminal_line_validation": "Rendered prompts must end with the conductor and push prohibition.",
    "protected_files_unchanged": "The audit must not mutate protected inputs.",
    "v555_queue_integrity_score": "The scalar gate is one only when every queue contract passes.",
    "blocked_reason": "A failed queue audit must name every failed readiness condition.",
    "gate_check_summary": "Blocked diagnostics must show expected and observed values.",
    "preconditions_checked": "Input files, state, and substrate are recorded before conclusions.",
    "inference_substrate": "This task reads repository evidence without a model call.",
    "verifier_is_oracle": "The audit validates contracts and proves no scientific claim.",
    "field_principles": "Every required field and acceptance gate has a stated purpose.",
    "field_provenance": "Every required field identifies its source kind.",
    "random_seed": "No random sampling is used by this deterministic audit.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes stay attached to the artifact.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict uses a terminal prefix and states the queue boundary.",
    "acceptance_gate:v555_queue_integrity_score": (
        "The queue-integrity scalar must fail closed unless all validation surfaces pass."
    ),
}
FIELD_PROVENANCE = {
    "status": "derived",
    "v554_terminal_rows": "derived",
    "v554_activated_task_count": "derived",
    "v554_missing_zero_byte_or_blocked_artifacts": "derived",
    "v554_terminal_claim_determinations": "derived",
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
    "v555_queue_integrity_score": "derived",
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


def _is_blocked(payload: JsonMap, meta: JsonMap) -> bool:
    if meta.get("error"):
        return False
    status = str(payload.get("status") or "").lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    return status.startswith("blocked") or verdict.startswith("blocked") or "complete_blocked" in status


def _artifact_state(payload: JsonMap, meta: JsonMap) -> str:
    base = _artifact_status(payload, meta)
    if base in {"missing", "zero_byte", "malformed"}:
        return base
    if _is_blocked(payload, meta):
        return "blocked"
    if base.startswith("complete"):
        return "complete"
    return base


def _artifact_meta(root: Path, rel: Path) -> tuple[JsonDict, JsonDict]:
    payload, meta = read_json_mapping(root / rel)
    meta = dict(meta)
    meta["size_bytes"] = (root / rel).stat().st_size if (root / rel).exists() else None
    return payload, meta


def _readiness_fields(payload: JsonMap) -> JsonDict:
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(key, str) and (key.endswith("_score") or key.endswith("_ready"))
    }


def _claim_eligibility_for_task(task_id: str, state: str, payload: JsonMap) -> JsonDict:
    if state in {"missing", "zero_byte", "malformed"}:
        return {"eligible": False, "blockers": [f"{state}_artifact"], "scope": task_id}
    if state == "blocked":
        blockers = payload.get("csl_ineligibility_reasons")
        if not isinstance(blockers, list) or not blockers:
            blockers = [str(payload.get("blocked_reason") or "blocked_artifact")]
        return {"eligible": False, "blockers": blockers, "scope": task_id}
    if task_id == "exp6436-v554-terminal-handoff-and-queue-preflight":
        ready = payload.get("v554_queue_ready_score") == 1.0
        return {
            "eligible": ready,
            "blockers": [] if ready else ["v554_queue_ready_score=0.0"],
            "scope": "queue integrity",
        }
    return {"eligible": False, "blockers": ["no_claim_promoted_by_exp6448"], "scope": task_id}


def _v554_terminal_rows(root: Path) -> tuple[list[JsonDict], JsonDict]:
    source_payload, _source_meta = _artifact_meta(
        root, Path(V554_DELIVERABLES_BY_TASK[EXPECTED_V554_ACTIVATED_TASK_IDS[0]])
    )
    source_ids = source_payload.get("task_ids_in_order")
    task_ids = (
        [str(task_id) for task_id in source_ids]
        if isinstance(source_ids, list) and source_ids
        else list(EXPECTED_V554_ACTIVATED_TASK_IDS)
    )
    rows: list[JsonDict] = []
    for task_id in task_ids:
        rel = Path(V554_DELIVERABLES_BY_TASK.get(task_id, ""))
        payload, meta = _artifact_meta(root, rel)
        state = _artifact_state(payload, meta)
        live = _live_adversarial(root, rel, meta.get("error") is None)
        rows.append(
            {
                "task_id": task_id,
                "deliverable_path": rel.as_posix(),
                "byte_count": meta.get("size_bytes"),
                "artifact_state": state,
                "status": _artifact_status(payload, meta),
                "honest_verdict": payload.get("honest_verdict"),
                "readiness_fields": _readiness_fields(payload),
                "gate_summary": payload.get("gate_check_summary"),
                "current_adversarial_findings": live,
                "final_claim_eligibility": _claim_eligibility_for_task(task_id, state, payload),
            }
        )
    return rows, _v554_claim_determinations(root, rows)


def _v554_claim_determinations(root: Path, rows: Sequence[JsonMap]) -> JsonDict:
    rows_by_task = {str(row.get("task_id")): row for row in rows}
    exp6444, _meta = _artifact_meta(
        root, Path(V554_DELIVERABLES_BY_TASK["exp6444-csl-lifecycle-recomputation-audit"])
    )
    csl_blockers = exp6444.get("csl_ineligibility_reasons")
    if not isinstance(csl_blockers, list) or not csl_blockers:
        csl_blockers = ["v554_csl_evidence_missing_or_blocked"]
    return {
        "v554_queue_integrity": {
            "eligible": False,
            "blockers": ["v554_queue_ready_score=0.0"],
            "source": rows_by_task.get(EXPECTED_V554_ACTIVATED_TASK_IDS[0], {}).get(
                "deliverable_path"
            ),
        },
        "path_receipt": {
            "eligible": False,
            "blockers": ["exp6437_blocked_gate_check_failed"],
            "source": rows_by_task.get("exp6437-generation-to-verdict-receipt-replay-contract", {}).get(
                "deliverable_path"
            ),
        },
        "factor_influence": {
            "eligible": False,
            "blockers": ["missing_or_blocked_v554_factor_chain"],
            "source": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        },
        "prospective_csl": {
            "eligible": False,
            "blockers": csl_blockers,
            "source": V554_DELIVERABLES_BY_TASK["exp6444-csl-lifecycle-recomputation-audit"],
        },
        "internal_arc_reachability": {
            "eligible": False,
            "blockers": ["exp6445_not_activated"],
            "source": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        },
        "public_arc": {
            "eligible": False,
            "blockers": ["no_v554_public_arc_claim_eligible"],
            "source": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        },
        "hardware": {
            "eligible": False,
            "blockers": ["v554_contains_no_authenticated_hardware_evidence"],
            "source": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        },
    }


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
    declares_mandated = bool(MANDATED_GGUF_IDS & named_models) or (
        "mandated gguf" in prompt_lower or "mandatory gguf" in prompt_lower
    )
    failures: list[JsonDict] = []
    if "MODEL_SPECS" not in rendered_prompt:
        failures.append({"task_id": task_id, "reason": "missing_model_specs"})
    if "cached_sota_pair()" not in rendered_prompt and "same resolver" not in prompt_lower:
        failures.append({"task_id": task_id, "reason": "missing_cache_resolver"})
    if not declares_mandated:
        failures.append({"task_id": task_id, "reason": "missing_mandated_gguf_declaration"})
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


def _has_blocked_verdict_contract(prompt: str, required_fields: set[str]) -> bool:
    return "gate_check_summary" in required_fields or "gate_check_summary" in prompt


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


def _operator_override_rows(tasks_by_id: Mapping[str, JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, task in tasks_by_id.items():
        override = str(task.get("operator_override") or "").strip()
        if not override:
            continue
        rows.append(
            {
                "task_id": task_id,
                "operator_override_present": True,
                "cites_standing_transition_directive": (
                    "2026-05-29 operator directive" in override
                    and "standing" in override.lower()
                ),
                "value": override,
            }
        )
    return rows


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
    operator_rows = _operator_override_rows(tasks_by_id)
    exp6448_override_ok = any(
        row["task_id"] == EXPERIMENT_ID and row["cites_standing_transition_directive"]
        for row in operator_rows
    )
    if not exp6448_override_ok:
        failures.append({"task_id": EXPERIMENT_ID, "reason": "missing_standing_transition_override"})
    return {
        "ok": not missing
        and not failures
        and not linter["schema_errors"]
        and not linter["prior_failure_violations"]
        and gate_audit["roadmap_gate_audit_passed"] is True,
        "required_rerun_task_ids": list(REQUIRED_PRIOR_FAILURE_TASK_IDS),
        "missing_required_rerun_task_ids": missing,
        "required_prior_failure_rows": rows,
        "operator_override_rows": operator_rows,
        "failures": failures,
        "validate_prior_failures": linter,
        "gate_audit_prior_missing": gate_audit["n_prior_failures_missing"],
        "gate_audit_failure_details": gate_audit.get("failure_details", []),
        "gate_audit_passed": gate_audit["roadmap_gate_audit_passed"],
    }


def validate_v555_queue_data(
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
        if task.get("milestone") != MILESTONE_V555
    ]
    llm_present: list[str] = []
    model_failures: list[JsonDict] = []
    for task_id in ids:
        prompt = rendered_prompts.get(task_id, "")
        prompt_lower = prompt.lower()
        declares_model_specs = "MODEL_SPECS" in prompt
        is_expected_llm = task_id in LLM_TASK_IDS
        if is_expected_llm or declares_model_specs:
            if task_id not in llm_present:
                llm_present.append(task_id)
            model_failures.extend(_model_policy_failures(task_id, prompt))
        if task_id not in LLM_TASK_IDS and declares_model_specs:
            model_failures.append({"task_id": task_id, "reason": "unexpected_model_specs_task"})
        if "autotokenizer.from_pretrained" in prompt_lower:
            model_failures.append({"task_id": task_id, "reason": "forbidden_autotokenizer_path"})
    comparative_ids: list[str] = []
    per_unit_failures: list[JsonDict] = []
    blocked_summary_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        rendered = rendered_prompts.get(task_id, "")
        lower = rendered.lower()
        required_fields = required_fields_by_id.get(task_id, set())
        is_comparative = any(
            token in lower
            for token in (" compare ", " compares ", "matched arms", " a/b", "matched arm")
        )
        if is_comparative:
            comparative_ids.append(task_id)
            if task.get("per_unit_rows") is not True:
                per_unit_failures.append({"task_id": task_id, "reason": "per_unit_rows_not_true"})
            if "per_unit_rows" not in required_fields:
                per_unit_failures.append(
                    {"task_id": task_id, "reason": "missing_per_unit_rows_required_field"}
                )
            if "emit per_unit_rows" not in lower and "per_unit_rows must contain" not in lower:
                per_unit_failures.append({"task_id": task_id, "reason": "missing_row_emission_rule"})
        if ("blocked_" in lower or "blocked artifact" in lower) and not _has_blocked_verdict_contract(
            rendered, required_fields
        ):
            blocked_summary_failures.append(
                {"task_id": task_id, "reason": "missing_gate_check_summary"}
            )
    prompt_checks = _queue_prompt_checks(tasks, rendered_prompts, render_receipts, root, date)
    prior_validation = _prior_validation(tasks_by_id, root)
    exclusion = _exclusion_validation(root)
    identity_ok = (
        ids == list(EXPECTED_V555_TASK_IDS)
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
            "expected_task_count": len(EXPECTED_V555_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V555_TASK_IDS),
            "missing_expected_task_ids": [
                task_id for task_id in EXPECTED_V555_TASK_IDS if task_id not in ids
            ],
            "extra_task_ids": [task_id for task_id in ids if task_id not in EXPECTED_V555_TASK_IDS],
            "duplicate_task_ids": duplicate_ids,
            "unique_task_ids": not duplicate_ids,
            "duplicate_deliverables": duplicate_deliverables,
            "unique_deliverables": not duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
            "retired_task_ids": retired_task_ids,
        },
        "milestone_consistency_check": {
            "ok": data.get("milestone") == MILESTONE_V555 and not milestone_failures,
            "roadmap_milestone": data.get("milestone"),
            "expected_milestone": MILESTONE_V555,
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


def _failed_checks(report_fields: JsonMap) -> list[JsonDict]:
    failures: list[JsonDict] = []
    if report_fields.get("task_count") != len(EXPECTED_V555_TASK_IDS):
        failures.append(
            {
                "failed_check": "task_count",
                "expected_condition": "task_count == 12",
                "observed_value": report_fields.get("task_count"),
                "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            }
        )
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
            failures.append(
                {
                    "failed_check": field,
                    "expected_condition": f"{field}.ok is true",
                    "observed_value": value,
                    "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
                }
            )
    test_rows = report_fields.get("tests_run", [])
    if isinstance(test_rows, Sequence) and not isinstance(test_rows, (str, bytes)):
        failed_test_rows = [
            dict(row)
            for row in test_rows
            if isinstance(row, Mapping) and row.get("exit_code") != 0
        ]
        if failed_test_rows:
            failures.append(
                {
                    "failed_check": "tests_run",
                    "expected_condition": "every tests_run.exit_code is 0",
                    "observed_value": failed_test_rows,
                    "evidence_path": EXTERNAL_TEST_RECEIPT_PATH.as_posix(),
                }
            )
    return failures


def _first_failed_check(report_fields: JsonMap) -> JsonDict:
    failures = _failed_checks(report_fields)
    if failures:
        first = dict(failures[0])
        first["failed_checks"] = failures
        return first
    return {
        "failed_check": "unknown",
        "failed_checks": [],
        "expected_condition": "all checks pass",
        "observed_value": None,
        "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
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
    v554_rows, determinations = _v554_terminal_rows(root)
    checks = validate_v555_queue_data(roadmap, root, date)
    after_hashes = protected_hashes(root)
    protected = _protected_files_unchanged(before_hashes, after_hashes)
    tests_run = _test_rows(command_receipts)
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
        checks["task_count"] == len(EXPECTED_V555_TASK_IDS)
        and all(checks[name]["ok"] is True for name in check_names)
        and protected["ok"] is True
        and all(row.get("exit_code") == 0 for row in tests_run)
    )
    integrity_score = 1.0 if all_checks_ok else 0.0
    summary = {
        "failed_check": None,
        "expected_condition": "all checks pass",
        "observed_value": "all checks pass",
        "evidence_path": None,
    }
    blocked_reason = None
    status = "complete_v555_queue_integrity_passed"
    if not all_checks_ok:
        summary = _first_failed_check(
            {
                **checks,
                "protected_files_unchanged": protected,
                "tests_run": tests_run,
            }
        )
        blocked_reason = "; ".join(
            f"{failure['failed_check']}: expected {failure['expected_condition']}"
            for failure in summary.get("failed_checks", [summary])
        )
        status = "complete_blocked_v555_queue_integrity_failed"
    honest_verdict = {
        "complete_v555_queue_integrity_passed": (
            "complete_v555_queue_integrity_passed: V554 terminal facts are "
            "preserved and the twelve-task V555 queue validates"
        ),
        "complete_blocked_v555_queue_integrity_failed": (
            "complete_blocked_v555_queue_integrity_failed: V554 terminal facts "
            "are preserved but a V555 queue contract failed"
        ),
    }[status]
    report: JsonDict = {
        "status": status,
        "v554_terminal_rows": v554_rows,
        "v554_activated_task_count": len(v554_rows),
        "v554_missing_zero_byte_or_blocked_artifacts": [
            row
            for row in v554_rows
            if row["artifact_state"] in {"missing", "zero_byte", "blocked", "malformed"}
        ],
        "v554_terminal_claim_determinations": determinations,
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
        "v555_queue_integrity_score": integrity_score,
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
            "active_roadmap": path_receipt(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "v554_artifacts": [
                path_receipt(root / Path(path)) for path in V554_DELIVERABLES_BY_TASK.values()
            ],
            "v554_missing_or_zero_byte": [
                row
                for row in v554_rows
                if row["artifact_state"] in {"missing", "zero_byte", "malformed"}
            ],
            "exclusion_manifest": path_receipt(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            "roadmap_schema": path_receipt(root / ROADMAP_SCHEMA_RELATIVE_PATH),
            "protected_files": [path.as_posix() for path in PROTECTED_RELATIVE_PATHS],
            "research_compute_started": False,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": duration_s,
        "tests_run": tests_run,
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
    determinations = report.get("v554_terminal_claim_determinations")
    if isinstance(determinations, Mapping):
        for claim in ("v554_queue_integrity", "prospective_csl", "public_arc", "hardware"):
            if determinations.get(claim, {}).get("eligible") is not False:
                errors.append(f"{claim} must remain blocked")
    else:
        errors.append("v554_terminal_claim_determinations must be a mapping")
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
        for field in list(REQUIRED_ARTIFACT_FIELDS) + ["acceptance_gate:v555_queue_integrity_score"]:
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
        report.get("task_count") == len(EXPECTED_V555_TASK_IDS)
        and all(
            isinstance(report.get(field), Mapping) and report[field].get("ok") is True
            for field in check_fields
        )
        and isinstance(protected, Mapping)
        and protected.get("ok") is True
    )
    if report.get("v555_queue_integrity_score") == 1.0 and not checks_pass:
        errors.append("integrity score cannot pass with failed checks")
    if report.get("v555_queue_integrity_score") == 0.0 and not report.get("blocked_reason"):
        errors.append("blocked report must name blocked_reason")
    if (
        str(report.get("status") or "").startswith("complete_blocked")
        and isinstance(summary, Mapping)
        and not summary.get("failed_check")
    ):
        errors.append("blocked report must name failed_check")
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
    print(f"{RESULT_RELATIVE_PATH.name}: {report['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
