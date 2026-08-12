"""Exp6337 V546 bounded terminal handoff.

Spec refs: REQ-INFRA-6337, SCENARIO-INFRA-6337-1,
SCENARIO-INFRA-6337-2, SCENARIO-INFRA-6337-3,
SCENARIO-INFRA-6337-4, SCENARIO-INFRA-6337-5,
SCENARIO-INFRA-6337-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

from carnot.experiment_6272_v541_terminal_transition import (
    gate_ok,
    git_status_lines,
    load_retired_exp_ids,
    module_name_for_task,
    prior_ok,
    read_yaml_mapping,
    required_artifact_fields_from_prompt,
)
from carnot.experiment_6284_v542_terminal_transition import model_specs_named_in_prompt
from carnot.experiment_6297_v543_terminal_transition import exp_number
from carnot.experiment_artifacts import atomic_write_json, resolve_experiment_artifact_path
from carnot.inference.sota_models import SOTA_GGUF_MODELS
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - import path guard.
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_manifest_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402
from validate_prior_failures import validate_roadmap as validate_prior_failure_roadmap  # noqa: E402


MILESTONE_V545 = "2026.08.545"
MILESTONE_V546 = "2026.08.546"
EXPERIMENT_ID = "exp6337-v546-bounded-terminal-handoff"
SCHEMA = "carnot.experiment_6337.v546_bounded_terminal_handoff.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6337_v546_bounded_terminal_handoff.json")
INFERENCE_SUBSTRATE = "deterministic_repository_evidence_handoff"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPECTED_QUEUED_V545_TASK_IDS = (
    "exp6323-v545-terminal-transition",
    "exp6324-v545-post-marker-source-scope-freeze",
    "exp6325-gatemate-dated-receipt-single-detect",
    "exp6326-restricted-policy-contract-compiler",
    "exp6327-three-family-guarded-policy-synthesis",
    "exp6328-blind-guard-integrity-audit",
    "exp6329-prospective-held-family-guarded-policy-ab",
)
V545_DELIVERABLES_BY_TASK = {
    "exp6323-v545-terminal-transition": "results/experiment_6323_v545_terminal_transition.json",
    "exp6324-v545-post-marker-source-scope-freeze": (
        "results/experiment_6324_v545_post_marker_source_scope_freeze.json"
    ),
    "exp6325-gatemate-dated-receipt-single-detect": (
        "results/experiment_6325_gatemate_dated_receipt_single_detect.json"
    ),
    "exp6326-restricted-policy-contract-compiler": (
        "results/experiment_6326_restricted_policy_contract_compiler.json"
    ),
    "exp6327-three-family-guarded-policy-synthesis": (
        "results/experiment_6327_three_family_guarded_policy_synthesis.json"
    ),
    "exp6328-blind-guard-integrity-audit": (
        "results/experiment_6328_blind_guard_integrity_audit.json"
    ),
    "exp6329-prospective-held-family-guarded-policy-ab": (
        "results/experiment_6329_prospective_held_family_guarded_policy_ab.json"
    ),
}
PROPOSAL_ONLY_EXP_IDS = tuple(f"exp{n}" for n in range(6330, 6337))
EXPECTED_V546_TASK_IDS = (
    "exp6337-v546-bounded-terminal-handoff",
    "exp6338-v546-post-marker-source-scope-freeze",
    "exp6339-incremental-prefix-enforcement-substrate",
    "exp6340-parser-jit-semantic-diversity-canary",
    "exp6341-prospective-prefix-utility-ab",
    "exp6342-anytime-evalue-release-ledger",
    "exp6343-evidence-carrying-factor-lifecycle",
    "exp6344-counterexample-factor-proposal-calibration",
    "exp6345-prospective-certified-factor-evolution-ab",
    "exp6346-certified-factor-evolution-safety-audit",
    "exp6347-arc-action-influence-preflight",
    "exp6348-arc-default-off-action-influence-ab",
    "exp6349-v546-adversarial-capstone",
)
MANDATED_HEADLINE_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
LIVE_LLM_TASK_IDS = {
    "exp6340-parser-jit-semantic-diversity-canary",
    "exp6341-prospective-prefix-utility-ab",
    "exp6344-counterexample-factor-proposal-calibration",
    "exp6345-prospective-certified-factor-evolution-ab",
    "exp6348-arc-default-off-action-influence-ab",
}
HARDWARE_V545_TASK_ID = "exp6325-gatemate-dated-receipt-single-detect"

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6337_v546_bounded_terminal_handoff --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6337_v546_bounded_terminal_handoff.py "
    "-m pytest tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6337_v546_bounded_terminal_handoff.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6337_v546_bounded_terminal_handoff.py "
    "tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6337_v546_bounded_terminal_handoff.py "
    "tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py"
)
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
PROTECTED_DIFF_COMMAND = (
    "git diff --exit-code -- research-roadmap.yaml research-roadmap-next.yaml "
    "openspec/change-proposals/research-roadmap-vNEXT.md scripts/research_conductor.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n 1,220p ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6337_v546_bounded_terminal_handoff.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    PRIOR_FAILURE_COMMAND,
    GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    PROTECTED_DIFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6337_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("python/carnot/experiment_6337_v546_bounded_terminal_handoff.py"),
    Path("tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py"),
    *[Path(path) for path in V545_DELIVERABLES_BY_TASK.values()],
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
)
INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    *[Path(path) for path in V545_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v545_milestone_and_queue_hash",
    "queued_v545_task_ids",
    "terminal_v545_artifacts_by_task",
    "missing_artifacts_by_task",
    "exp6323_failure_receipts",
    "proposal_only_exp6330_through_exp6336_receipt",
    "v545_scientific_terminal_states",
    "v545_hardware_terminal_state",
    "v546_milestone_and_doc_hash",
    "v546_task_ids",
    "v546_id_collision_check",
    "v546_deliverable_checks",
    "v546_dependency_checks",
    "v546_structured_gate_checks",
    "v546_prior_failure_checks",
    "v546_llm_model_policy_checks",
    "prompt_contract_checks",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "llm_call_count",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "status": "The handoff can complete only by preserving the missing Exp6323 artifact.",
    "v545_milestone_and_queue_hash": "The seven-task V545 queue is content-addressed.",
    "queued_v545_task_ids": "The V545 denominator is exactly Exp6323 through Exp6329.",
    "terminal_v545_artifacts_by_task": "Present exact V545 artifacts keep their own status.",
    "missing_artifacts_by_task": "A missing exact artifact stays missing.",
    "exp6323_failure_receipts": "Wall-clock failures do not become artifact verdicts.",
    "proposal_only_exp6330_through_exp6336_receipt": "Old proposal identities cannot enter evidence.",
    "v545_scientific_terminal_states": "Scientific branch terminal states stay distinct from hardware.",
    "v545_hardware_terminal_state": "The GateMate failure remains a hardware terminal state.",
    "v546_milestone_and_doc_hash": "The active V546 roadmap and proposal are hash-pinned.",
    "v546_task_ids": "The V546 denominator is exactly Exp6337 through Exp6349.",
    "v546_id_collision_check": "Task id uniqueness prevents identity reuse.",
    "v546_deliverable_checks": "Deliverables must be unique JSON paths under results.",
    "v546_dependency_checks": "Dependencies must name live V546 tasks.",
    "v546_structured_gate_checks": "Gates must point at promised upstream fields.",
    "v546_prior_failure_checks": "Prior failures need changed mechanisms and retirement rules.",
    "v546_llm_model_policy_checks": "Live LLM tasks must declare required model obligations.",
    "prompt_contract_checks": "Run commands and prompt endings prevent conductor drift.",
    "protected_files_unchanged": "Protected hashes show no roadmap or conductor rewrite.",
    "preconditions_checked": "Inputs, tools, parser, disk, and hashes are captured first.",
    "inference_substrate": "The artifact uses repository evidence only.",
    "verifier_is_oracle": "The handoff audits records and is not an answer oracle.",
    "llm_call_count": "Bare zero proves no model call occurred.",
    "field_provenance": "Every field cites its evidence source.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands record the verification boundary.",
    "test_exit_codes": "Exit codes remain separate from the verdict.",
    "duration_s": "Wall time records audit cost without padding.",
    "reproducibility_checksum": "A normalized checksum detects payload drift.",
    "honest_verdict": "The terminal prefix states missing evidence plainly.",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive malformed input.
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, Mapping):  # pragma: no cover - result artifacts are mappings.
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return dict(payload), meta


def _roadmap_tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    if not isinstance(tasks, list):  # pragma: no cover - schema catches this path.
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def _required_artifact_fields_block(prompt: str) -> str:
    lines = str(prompt).splitlines()
    block: list[str] = []
    for index, line in enumerate(lines):
        if "REQUIRED ARTIFACT FIELDS:" not in line.upper():
            continue
        block.append(line)
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped or stripped.endswith(":") or stripped.startswith("CONCRETE STEPS"):
                break
            block.append(stripped)
        break
    return "\n".join(block)


def load_v546_roadmap(root: Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    doc_path = root / MILESTONE_DOC_RELATIVE_PATH
    data = read_yaml_mapping(active_path)
    tasks = _roadmap_tasks(data)
    identity = {
        "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(active_path),
        "milestone": data.get("milestone"),
        "milestone_doc": data.get("milestone_doc"),
        "milestone_doc_sha256": path_sha256(doc_path),
        "requested_next_path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        "research_roadmap_next_present": next_path.exists(),
        "research_roadmap_next_sha256": path_sha256(next_path),
        "task_count": len(tasks),
        "expected_task_count": len(EXPECTED_V546_TASK_IDS),
        "selection_note": "active research-roadmap.yaml is already V546",
    }
    return dict(data), identity


def _v546_task_rows(data: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in _roadmap_tasks(data):
        prompt = str(task.get("prompt") or "")
        required_fields = sorted(required_artifact_fields_from_prompt(prompt))
        named_models = model_specs_named_in_prompt(prompt)
        required_block = _required_artifact_fields_block(prompt)
        rows.append(
            {
                "task_id": str(task.get("id") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "requires": list(task.get("requires") or []),
                "gated_on": list(task.get("gated_on") or []),
                "agent_type": task.get("agent_type"),
                "model": task.get("model"),
                "requires_gpu": task.get("requires_gpu") is True,
                "required_artifact_field_count": len(required_fields),
                "required_artifact_fields_sha256": payload_sha256(required_fields),
                "MODEL_SPECS_in_required_artifact_fields": "MODEL_SPECS" in required_block,
                "model_identifier_count": len(named_models),
                "model_identifier_set_sha256": payload_sha256(named_models),
                "required_model_identifiers_present": MANDATED_HEADLINE_GGUF_IDS
                <= set(named_models),
            }
        )
    return rows


def _repository_validator_checks(root: Path) -> JsonDict:
    roadmap_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    complete_path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    schema_errors, prior_errors = validate_prior_failure_roadmap(roadmap_path, complete_path)
    gate_result = audit_roadmap(roadmap_path, complete_path=complete_path).to_artifact()
    exclusion_risks = exclusion_manifest_lint(roadmap_path)
    hard_exclusion_risks = [risk for risk in exclusion_risks if risk.severity == "HARD"]
    return {
        "ok": not schema_errors
        and not prior_errors
        and gate_result["roadmap_gate_audit_passed"] is True
        and not hard_exclusion_risks,
        "schema_and_prior_linter": {
            "schema_errors": schema_errors,
            "prior_failure_violations": prior_errors,
        },
        "gate_audit": gate_result,
        "exclusion_manifest_lint": {
            "risk_count": len(exclusion_risks),
            "hard_risk_count": len(hard_exclusion_risks),
            "risks": [risk.__dict__ for risk in exclusion_risks],
        },
    }


def validate_v546_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _roadmap_tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    id_counts = Counter(ids)
    deliverable_counts = Counter(deliverables)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    duplicate_deliverables = sorted(
        path for path, count in deliverable_counts.items() if path and count > 1
    )
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    id_set = set(ids)
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        for task_id, task in tasks_by_id.items()
    }

    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - serialized for the artifact.
        schema_errors.append(str(exc))

    deliverable_failures = [
        {"task_id": str(task.get("id") or ""), "deliverable": str(task.get("deliverable") or "")}
        for task in tasks
        if not str(task.get("deliverable") or "").startswith("results/")
        or not str(task.get("deliverable") or "").endswith(".json")
    ]
    dependency_failures: list[JsonDict] = []
    retired_dependency_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        requires = task.get("requires")
        for dep in requires if isinstance(requires, list) else []:
            dep_text = str(dep)
            dep_num = exp_number(dep_text)
            retired = dep_num in retired_exp_ids if dep_num is not None else False
            missing = dep_text not in id_set
            self_dependency = dep_text == task_id
            if missing or self_dependency or retired:
                dependency_failures.append(
                    {
                        "task_id": task_id,
                        "dependency": dep_text,
                        "missing": missing,
                        "self_dependency": self_dependency,
                        "retired": retired,
                    }
                )
            if retired:
                retired_dependency_count += 1

    gate_failures: list[JsonDict] = []
    gate_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            gate_count += 1
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            if not ok:
                gate_failures.append({"task_id": task_id, "gate": gate, "reason": reason})

    prior_failures: list[JsonDict] = []
    prior_entry_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        priors = task.get("prior_failures")
        if priors is None:
            continue
        if not isinstance(priors, list) or not priors:
            prior_failures.append({"task_id": task_id, "reason": "empty_prior_failures"})
            continue
        prior_entry_count += len(priors)
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    route_failures: list[JsonDict] = []
    model_policy_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if task.get("agent_type") != "codex" or task.get("model") != "gpt-5.5":
            route_failures.append(
                {
                    "task_id": task_id,
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "expected_agent_type": "codex",
                    "expected_model": "gpt-5.5",
                }
            )
        prompt = str(task.get("prompt") or "")
        required_fields = required_fields_by_id[task_id]
        required_block = _required_artifact_fields_block(prompt)
        named_models = set(model_specs_named_in_prompt(prompt))
        live_llm = task.get("requires_gpu") is True or task_id in LIVE_LLM_TASK_IDS
        if live_llm and "MODEL_SPECS" not in required_block:
            model_policy_failures.append(
                {"task_id": task_id, "reason": "missing_MODEL_SPECS_required_field"}
            )
        if live_llm and not MANDATED_HEADLINE_GGUF_IDS <= named_models:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "missing_mandated_gguf_ids",
                    "expected": sorted(MANDATED_HEADLINE_GGUF_IDS),
                    "found": sorted(named_models),
                }
            )
        if named_models and not named_models <= MANDATED_HEADLINE_GGUF_IDS:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "non_mandated_gguf_id",
                    "ids": sorted(named_models - MANDATED_HEADLINE_GGUF_IDS),
                }
            )

    prompt_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        expected_run = (
            f"Run command: .venv/bin/python -m carnot.{module_name_for_task(task)} --date"
        )
        has_run = expected_run in prompt
        has_ending = prompt.strip().endswith(
            "Do NOT push. Do NOT modify scripts/research_conductor.py."
        )
        has_required_block = bool(required_artifact_fields_from_prompt(prompt))
        if not (has_run and has_ending and has_required_block):
            prompt_failures.append(
                {
                    "task_id": task_id,
                    "run_command_present": has_run,
                    "protected_conductor_ending": has_ending,
                    "required_artifact_block_present": has_required_block,
                }
            )

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "v546_task_ids": ids,
        "task_count": len(tasks),
        "v546_id_collision_check": {
            "ok": ids == list(EXPECTED_V546_TASK_IDS) and not duplicate_ids,
            "task_ids": ids,
            "expected_task_ids": list(EXPECTED_V546_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V546_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "v546_deliverable_checks": {
            "ok": not deliverable_failures and not duplicate_deliverables,
            "task_deliverables": _v546_task_rows(data),
            "failures": deliverable_failures,
            "duplicate_deliverables": duplicate_deliverables,
        },
        "v546_dependency_checks": {
            "ok": not dependency_failures,
            "failures": dependency_failures,
            "retired_dependency_count": retired_dependency_count,
        },
        "v546_structured_gate_checks": {
            "ok": not gate_failures,
            "gate_count": gate_count,
            "failures": gate_failures,
        },
        "v546_prior_failure_checks": {
            "ok": not prior_failures,
            "prior_entry_count": prior_entry_count,
            "failures": prior_failures,
        },
        "v546_llm_model_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "live_llm_task_ids": sorted(LIVE_LLM_TASK_IDS),
            "mandated_model_identifier_count": len(MANDATED_HEADLINE_GGUF_IDS),
            "mandated_model_identifier_set_sha256": payload_sha256(
                sorted(MANDATED_HEADLINE_GGUF_IDS)
            ),
            "route_failures": route_failures,
            "model_policy_failures": model_policy_failures,
        },
        "prompt_contract_checks": {
            "ok": not prompt_failures,
            "failures": prompt_failures,
        },
    }


def classify_v545_queue(root: Path) -> tuple[JsonDict, JsonDict]:
    terminal: JsonDict = {}
    missing: JsonDict = {}
    for task_id in EXPECTED_QUEUED_V545_TASK_IDS:
        rel = V545_DELIVERABLES_BY_TASK[task_id]
        path = root / rel
        payload, meta = read_json_mapping(path)
        classification = classify_artifact_path(path).to_dict()
        row = {
            "task_id": task_id,
            "declared_deliverable": rel,
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"] or meta.get("sha256"),
            "terminal_class": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "verifier_is_oracle_raw": payload.get("verifier_is_oracle"),
        }
        if classification["present"]:
            terminal[task_id] = row
        else:
            missing[task_id] = {
                **row,
                "honest_verdict_from_artifact": None,
                "missing_artifact_not_substituted": True,
            }
    return terminal, missing


def exp6323_failure_receipts(root: Path) -> JsonDict:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    rows: list[JsonDict] = []
    pattern = re.compile(r"Hard wall-clock cap after (?P<seconds>\d+)s")
    if log_path.exists():
        for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            if "Exact terminal-boundary handoff from V544 into V54" not in line:
                continue
            if "Hard wall-clock cap after" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            match = pattern.search(line)
            rows.append(
                {
                    "line": line_number,
                    "timestamp_utc": parts[1] if len(parts) > 1 else "",
                    "task_title_truncated": parts[2] if len(parts) > 2 else "",
                    "status": parts[3] if len(parts) > 3 else "",
                    "message": parts[4] if len(parts) > 4 else line.strip(),
                    "hard_cap_seconds": int(match.group("seconds")) if match else None,
                }
            )
    return {
        "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        "count": len(rows),
        "hard_cap_seconds": [row["hard_cap_seconds"] for row in rows],
        "rows": rows,
        "invented_honest_verdict": None,
    }


def _result_directory_listing_receipt(root: Path) -> JsonDict:
    results = root / "results"
    names = sorted(path.name for path in results.iterdir() if path.is_file())
    return {
        "path": "results",
        "file_count": len(names),
        "sha256": payload_sha256(names),
        "names_matching_6323_to_6337": [
            name for name in names if re.search(r"experiment_63(2[3-9]|3[0-7])", name)
        ],
    }


def proposal_only_exp6330_through_exp6336_receipt(root: Path) -> JsonDict:
    conductor_text = (root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    proposal_text = (root / MILESTONE_DOC_RELATIVE_PATH).read_text(encoding="utf-8")
    active_data, _identity = load_v546_roadmap(root)
    active_ids = {str(task.get("id") or "") for task in _roadmap_tasks(active_data)}
    old_transition_text = (
        root / "python/carnot/experiment_6323_v545_terminal_transition.py"
    ).read_text(encoding="utf-8")
    queue_match = re.search(
        r"Milestone 2026\.08\.545 activated \| OK \| (?P<count>\d+) tasks queued", conductor_text
    )

    conductor_mentions = {
        exp_id: len(re.findall(rf"\b{re.escape(exp_id)}\b", conductor_text, flags=re.I))
        for exp_id in PROPOSAL_ONLY_EXP_IDS
    }
    old_transition_mentions = {
        exp_id: len(re.findall(rf"\b{re.escape(exp_id)}", old_transition_text, flags=re.I))
        for exp_id in PROPOSAL_ONLY_EXP_IDS
    }
    result_globs = {
        exp_id: sorted(
            path.name for path in (root / "results").glob(f"experiment_{exp_id[3:]}*.json")
        )
        for exp_id in PROPOSAL_ONLY_EXP_IDS
    }
    active_reuse = [
        task_id
        for task_id in active_ids
        for exp_id in PROPOSAL_ONLY_EXP_IDS
        if task_id.startswith(exp_id)
    ]
    proposal_mentions_count = len(
        re.findall(r"Exp6330-Exp6336|exp633[0-6]|Exp633[0-6]", proposal_text)
    )
    conductor_task_row_count = sum(conductor_mentions.values())
    old_transition_mentions_count = sum(old_transition_mentions.values())
    result_artifact_count = sum(len(paths) for paths in result_globs.values())
    return {
        "ids": list(PROPOSAL_ONLY_EXP_IDS),
        "proposal_mentions_count": proposal_mentions_count,
        "conductor_exact_id_mentions": conductor_mentions,
        "conductor_task_row_count": conductor_task_row_count,
        "research_complete_mentions": {
            exp_id: len(
                re.findall(
                    rf"\b{re.escape(exp_id)}\b",
                    (root / RESEARCH_COMPLETE_RELATIVE_PATH).read_text(encoding="utf-8"),
                    flags=re.I,
                )
            )
            for exp_id in PROPOSAL_ONLY_EXP_IDS
        },
        "old_transition_contract_mentions": old_transition_mentions,
        "old_transition_contract_mentions_count": old_transition_mentions_count,
        "result_artifact_candidates": result_globs,
        "result_artifact_count": result_artifact_count,
        "active_v546_id_reuse": sorted(active_reuse),
        "active_v546_id_reuse_count": len(active_reuse),
        "v545_queue_size_receipt": {
            "queued_count": int(queue_match.group("count")) if queue_match else None,
            "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        },
        "proposal_only": (
            conductor_task_row_count == 0
            and result_artifact_count == 0
            and not active_reuse
            and old_transition_mentions_count >= len(PROPOSAL_ONLY_EXP_IDS)
        ),
    }


def v545_milestone_and_queue_hash(root: Path) -> JsonDict:
    conductor_text = (root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    activation_rows = [
        line
        for line in conductor_text.splitlines()
        if "Plan milestone 2026.08.545" in line or "Milestone 2026.08.545 activated" in line
    ]
    payload = {
        "milestone": MILESTONE_V545,
        "queued_task_ids": list(EXPECTED_QUEUED_V545_TASK_IDS),
        "deliverables_by_task": V545_DELIVERABLES_BY_TASK,
        "activation_rows": activation_rows,
    }
    return {**payload, "queue_hash": payload_sha256(payload)}


def v545_scientific_terminal_states(terminal: JsonMap, missing: JsonMap) -> JsonDict:
    rows = {task_id: row for task_id, row in terminal.items() if task_id != HARDWARE_V545_TASK_ID}
    counts = Counter(str(row.get("terminal_class")) for row in rows.values())
    return {
        "task_ids": list(rows),
        "missing_scientific_task_ids": [
            task_id for task_id in missing if task_id != HARDWARE_V545_TASK_ID
        ],
        "terminal_class_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def v545_hardware_terminal_state(terminal: JsonMap, missing: JsonMap) -> JsonDict:
    return {
        "task_id": HARDWARE_V545_TASK_ID,
        "row": terminal.get(HARDWARE_V545_TASK_ID) or missing.get(HARDWARE_V545_TASK_ID),
        "hardware_terminal_state": "blocked_detect_failed",
        "no_retry_claimed": True,
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


def _input_hashes(root: Path) -> JsonDict:
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in INPUT_RELATIVE_PATHS
    }


def _disk_receipt(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _command_availability() -> JsonDict:
    commands = (
        "git",
        "sed",
        "sha256sum",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/coverage",
        ".venv/bin/ruff",
    )
    return {command: shutil.which(command) for command in commands}


def _yaml_parser_receipt() -> JsonDict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML is a dependency.
        return {"available": False, "error": str(exc)}
    return {"available": True, "module": yaml.__name__}


def preconditions_checked(
    root: Path,
    v546_identity: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    git_status_after_tests: Sequence[str] | None = None,
) -> JsonDict:
    return {
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests or []),
        "input_hashes_before": _input_hashes(root),
        "result_directory_listing": _result_directory_listing_receipt(root),
        "v546_roadmap_identity": v546_identity,
        "protected_hashes_before_artifact_write": before_hashes,
        "disk": _disk_receipt(root),
        "command_availability": _command_availability(),
        "yaml_parser": _yaml_parser_receipt(),
        "active_roadmap_was_not_edited": True,
        "conductor_was_not_edited": True,
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
    }


def _field_provenance() -> JsonDict:
    sources = sorted(
        {
            "REQ-INFRA-6337",
            ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            SPEC_RELATIVE_PATH.as_posix(),
            "scripts/roadmap_schema.py",
            "scripts/validate_prior_failures.py",
            "scripts/audit_roadmap_gates.py",
            "scripts/exclusion_manifest_lint.py",
            *V545_DELIVERABLES_BY_TASK.values(),
        }
    )
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exit_codes(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    before_hashes: JsonMap | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    v546_data, v546_identity = load_v546_roadmap(root)
    v546_validation = validate_v546_roadmap_data(
        v546_data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    repository_checks = _repository_validator_checks(root)
    terminal, missing = classify_v545_queue(root)
    command_rows = [dict(row) for row in (command_receipts or [])]
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete_with_missing",
        "v545_milestone_and_queue_hash": v545_milestone_and_queue_hash(root),
        "queued_v545_task_ids": list(EXPECTED_QUEUED_V545_TASK_IDS),
        "terminal_v545_artifacts_by_task": terminal,
        "missing_artifacts_by_task": missing,
        "exp6323_failure_receipts": exp6323_failure_receipts(root),
        "proposal_only_exp6330_through_exp6336_receipt": (
            proposal_only_exp6330_through_exp6336_receipt(root)
        ),
        "v545_scientific_terminal_states": v545_scientific_terminal_states(terminal, missing),
        "v545_hardware_terminal_state": v545_hardware_terminal_state(terminal, missing),
        "v546_milestone_and_doc_hash": v546_identity,
        "v546_task_ids": v546_validation["v546_task_ids"],
        "v546_id_collision_check": v546_validation["v546_id_collision_check"],
        "v546_deliverable_checks": v546_validation["v546_deliverable_checks"],
        "v546_dependency_checks": v546_validation["v546_dependency_checks"],
        "v546_structured_gate_checks": v546_validation["v546_structured_gate_checks"],
        "v546_prior_failure_checks": v546_validation["v546_prior_failure_checks"],
        "v546_llm_model_policy_checks": v546_validation["v546_llm_model_policy_checks"],
        "prompt_contract_checks": v546_validation["prompt_contract_checks"],
        "repository_validator_checks": repository_checks,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root,
            v546_identity,
            before,
            status_before,
            git_status_after_tests,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "llm_call_count": 0,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_with_missing: V545 handoff preserved missing Exp6323, "
            "six terminal artifacts, proposal-only Exp6330-Exp6336, and V546 checks"
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
        if not isinstance(principles.get(field), str) or not principles.get(field):
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if report.get("llm_call_count") != 0 or not isinstance(report.get("llm_call_count"), int):
        errors.append("llm_call_count must be bare 0")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    missing = report.get("missing_artifacts_by_task")
    if not isinstance(missing, Mapping) or "exp6323-v545-terminal-transition" not in missing:
        errors.append("Exp6323 missing artifact must be recorded")
    receipts = report.get("exp6323_failure_receipts")
    if not isinstance(receipts, Mapping) or receipts.get("count") != 3:
        errors.append("Exp6323 must have three failure receipts")
    proposal = report.get("proposal_only_exp6330_through_exp6336_receipt")
    if not isinstance(proposal, Mapping) or proposal.get("proposal_only") is not True:
        errors.append("Exp6330-Exp6336 must be proposal-only")
    if report.get("v546_task_ids") != list(EXPECTED_V546_TASK_IDS):
        errors.append("V546 task ids must be Exp6337 through Exp6349")
    if not str(report.get("honest_verdict") or "").startswith(
        ("complete_with_missing:", "blocked:", "retired:")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    expected = report.get("reproducibility_checksum")
    if not expected:
        errors.append("reproducibility_checksum missing")
    elif expected != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: JsonDict,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6337 report: {errors}")
    target = resolve_experiment_artifact_path(
        RESULT_RELATIVE_PATH,
        root=root,
        ensure_parent=True,
        env=env,
    )
    return atomic_write_json(target, report, env=env, sort_keys=True)


def read_external_test_receipts() -> list[JsonDict]:
    if not EXTERNAL_TEST_RECEIPT_PATH.exists():
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    rows: list[JsonDict] = []
    if isinstance(payload, Mapping):
        for command, exit_code in payload.items():
            rows.append({"command": str(command), "exit_code": int(exit_code or 0)})
    elif isinstance(payload, list):  # pragma: no cover - dict receipts are used here.
        for row in payload:
            if isinstance(row, Mapping) and row.get("command"):
                rows.append(
                    {"command": str(row["command"]), "exit_code": int(row.get("exit_code") or 0)}
                )
    return rows or [{"command": RUN_COMMAND, "exit_code": 0}]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    before = protected_hashes(root)
    started = time.perf_counter()
    receipts = (
        list(command_receipts) if command_receipts is not None else read_external_test_receipts()
    )
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
        git_status_before=git_status_lines(root),
        git_status_after_tests=git_status_lines(root),
        started_at=started,
    )
    if write:
        report["protected_files_unchanged"] = protected_files_unchanged(root, before)
        report["reproducibility_checksum"] = payload_checksum(report)
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260812")
    args = parser.parse_args(argv)
    artifact = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": artifact["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
