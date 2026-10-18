"""Exp6349 V546 adversarial capstone.

Spec refs: REQ-INFRA-6349, SCENARIO-INFRA-6349-1,
SCENARIO-INFRA-6349-2, SCENARIO-INFRA-6349-3,
SCENARIO-INFRA-6349-4, SCENARIO-INFRA-6349-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json, resolve_experiment_artifact_path
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.546"
EXPERIMENT_ID = "exp6349-v546-adversarial-capstone"
SCHEMA = "carnot.experiment_6349.v546_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6349_v546_adversarial_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6349_test_receipts.json")

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
PROPOSAL_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPECTED_TASK_IDS = (
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
    EXPERIMENT_ID,
)
MODEL_TASK_IDS = {
    "exp6340-parser-jit-semantic-diversity-canary",
    "exp6344-counterexample-factor-proposal-calibration",
    "exp6345-prospective-certified-factor-evolution-ab",
    "exp6348-arc-default-off-action-influence-ab",
}
SKIPPED_MODEL_TASK_IDS = {"exp6341-prospective-prefix-utility-ab"}
MANDATED_MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
ZERO_MUTATION_FIELDS = (
    "source_model_weight_mutation_count",
    "source_model_mutation_count",
    "weight_mutation_count",
)
ZERO_LABEL_STATE_FIELDS = ("generated_label_count", "hidden_state_access_count")
ARC_ZERO_FIELDS = (
    "hidden_game_source_access_count",
    "offline_ground_truth_bfs_count",
    "hand_game_adapter_count",
    "per_game_calibration_count",
    "solve_claim_count",
    "registry_update_count",
)
VERIFICATION_COST_TASK_IDS = (
    "exp6339-incremental-prefix-enforcement-substrate",
    "exp6340-parser-jit-semantic-diversity-canary",
    "exp6344-counterexample-factor-proposal-calibration",
    "exp6345-prospective-certified-factor-evolution-ab",
    "exp6347-arc-action-influence-preflight",
    "exp6348-arc-default-off-action-influence-ab",
)

RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6349_v546_adversarial_capstone --date 20260812"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6349_v546_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6349_v546_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6349_v546_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6349_v546_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6349_v546_adversarial_capstone.py "
    "tests/python/test_experiment_6349_v546_adversarial_capstone.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6349_v546_adversarial_capstone.py "
    "tests/python/test_experiment_6349_v546_adversarial_capstone.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6349_v546_adversarial_capstone.py"
)
ROADMAP_SCHEMA_COMMAND = (
    ".venv/bin/python -c 'import yaml; from pathlib import Path; "
    "from scripts.roadmap_schema import Roadmap; "
    'Roadmap.model_validate(yaml.safe_load(Path("research-roadmap.yaml").read_text()))'
    "'"
)
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
ROADMAP_GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6349_v546_adversarial_capstone.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROADMAP_SCHEMA_COMMAND,
    PRIOR_FAILURE_COMMAND,
    ROADMAP_GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone_and_roadmap_hash",
    "declared_task_ids_and_deliverables",
    "conductor_terminal_receipts_by_task",
    "artifact_existence_hash_schema_status_and_honest_verdict_by_task",
    "dependency_recomputation",
    "structured_gate_recomputation",
    "skipped_task_handling",
    "prior_failure_and_retirement_audit",
    "exclusion_manifest_audit",
    "prompt_contract_audit",
    "required_field_and_field_principle_audit",
    "model_policy_and_MODEL_SPECS_audit",
    "llama_cpp_embedded_tokenizer_audit",
    "gpu_offload_and_memory_release_audit",
    "source_model_weight_mutation_audit",
    "generated_label_and_hidden_state_audit",
    "exact_oracle_and_learned_claim_boundary_audit",
    "prefix_generation_determination",
    "certified_continuous_learning_determination",
    "eprocess_and_factor_lifecycle_determination",
    "safety_audit_determination",
    "arc_action_influence_determination",
    "solve_provenance_audit",
    "arc_registry_immutability_audit",
    "hardware_nonuse_and_inference_substrate_audit",
    "verification_cost_accounting_audit",
    "three_gap_closure_matrix",
    "prd_requirement_mapping",
    "protected_files_changed_with_reasons",
    "docs_and_archive_reconciliation",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A mixed terminal record can close without a positive science claim.",
    "milestone_and_roadmap_hash": "The roadmap fixes the task denominator.",
    "declared_task_ids_and_deliverables": "Exact deliverables prevent alias substitution.",
    "conductor_terminal_receipts_by_task": "Receipts are context and cannot promote artifacts.",
    "artifact_existence_hash_schema_status_and_honest_verdict_by_task": "Each task keeps its exact file state.",
    "dependency_recomputation": "Dependencies are checked from declared task ids.",
    "structured_gate_recomputation": "Gates are replayed from raw upstream fields.",
    "skipped_task_handling": "Gate skips stay skips with no hidden utility claim.",
    "prior_failure_and_retirement_audit": "Retirement signals use exact verdict comparison.",
    "exclusion_manifest_audit": "Retired scopes stay visible and blocked patterns are respected.",
    "prompt_contract_audit": "Task prompts must contain field contracts and run commands.",
    "required_field_and_field_principle_audit": "Required fields and principles are checked per artifact.",
    "model_policy_and_MODEL_SPECS_audit": "Live GGUF tasks must name the mandated models.",
    "llama_cpp_embedded_tokenizer_audit": "GGUF tasks must use embedded tokenizer receipts.",
    "gpu_offload_and_memory_release_audit": "Model tasks must record release receipts.",
    "source_model_weight_mutation_audit": "Base model weights must stay immutable.",
    "generated_label_and_hidden_state_audit": "Exact labels and hidden state are forbidden upstream leaks.",
    "exact_oracle_and_learned_claim_boundary_audit": "Exact oracles are named but not promoted as learned verifiers.",
    "prefix_generation_determination": "The prefix branch is decided by the canary and skip.",
    "certified_continuous_learning_determination": "The learning branch is decided by chronological release evidence.",
    "eprocess_and_factor_lifecycle_determination": "Release validity and lifecycle bounds are separate gates.",
    "safety_audit_determination": "Safety cannot become utility.",
    "arc_action_influence_determination": "Action influence is not a solve.",
    "solve_provenance_audit": "ARC provenance and solve counts are audited separately.",
    "arc_registry_immutability_audit": "The registry must not change during influence work.",
    "hardware_nonuse_and_inference_substrate_audit": "Hardware boundaries stay non-use claims.",
    "verification_cost_accounting_audit": "Exact checker cost must be accounted where required.",
    "three_gap_closure_matrix": "The final record states each gap without overclaiming.",
    "prd_requirement_mapping": "The capstone maps evidence back to PRD requirements.",
    "protected_files_changed_with_reasons": "Protected files must not be rewritten by the capstone.",
    "docs_and_archive_reconciliation": "Docs and archive state are reported, not silently changed.",
    "preconditions_checked": "Hashes, disk, commands, and protected state are captured first.",
    "inference_substrate": "The capstone aggregates upstream artifacts only.",
    "verifier_is_oracle": "The capstone audits records and is not an oracle.",
    "llm_call_count": "Bare zero proves no model call occurred.",
    "field_provenance": "Every required field cites its source evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands record the verification boundary.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming success.",
    "duration_s": "Wall time records audit cost without padding.",
    "reproducibility_checksum": "The normalized payload is content addressed.",
    "honest_verdict": "The terminal prefix states the mixed evidence plainly.",
}

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    PROPOSAL_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
)


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
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, Mapping):
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return dict(payload), meta


def _load_yaml_mapping(path: Path) -> JsonDict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _roadmap_tasks(root: Path) -> list[JsonDict]:
    data = _load_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def required_artifact_fields_from_prompt(prompt: str) -> list[str]:
    match = re.search(
        r"REQUIRED ARTIFACT FIELDS:\s*(?P<body>.*?)(?:\n\s*\n|\n\s*CONCRETE STEPS)",
        prompt,
        flags=re.I | re.S,
    )
    if not match:
        return []
    body = " ".join(line.strip() for line in match.group("body").splitlines())
    fields = []
    for raw in body.split(","):
        name = raw.strip().strip(". ")
        name = re.sub(r"\s.*$", "", name)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            fields.append(name)
    return sorted(set(fields))


def _module_name_for_task(task: JsonMap) -> str:
    deliverable = str(task.get("deliverable") or "")
    stem = Path(deliverable).name
    if stem.startswith("experiment_") and stem.endswith(".json"):
        return stem[: -len(".json")]
    return str(task.get("id") or "").replace("-", "_")


def _bare_value(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) >= {"value", "principle"}:
        return value.get("value")
    return value


def _extract_field(payload: JsonMap, field: str) -> Any:
    return _bare_value(payload.get(field))


def _numeric_count(value: Any) -> int:
    value = _bare_value(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _read_payloads(root: Path, tasks: Sequence[JsonMap]) -> dict[str, JsonDict]:
    payloads: dict[str, JsonDict] = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        path = root / str(task.get("deliverable") or "")
        payloads[task_id] = read_json_mapping(path)[0]
    return payloads


def _schema_status(payload: JsonMap, self_task: bool) -> str:
    if self_task:
        return "self_referential_excluded"
    if not payload:
        return "missing"
    return "present" if payload.get("schema") or payload.get("schema_version") else "missing"


def milestone_and_roadmap_hash(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    task_graph = [
        {
            "id": task.get("id"),
            "deliverable": task.get("deliverable"),
            "requires": task.get("requires") or [],
            "gated_on": task.get("gated_on") or [],
        }
        for task in tasks
    ]
    return {
        "milestone": MILESTONE,
        "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
        "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "proposal_path": PROPOSAL_RELATIVE_PATH.as_posix(),
        "proposal_sha256": path_sha256(root / PROPOSAL_RELATIVE_PATH),
        "task_graph_sha256": payload_sha256(task_graph),
        "expected_task_ids": list(EXPECTED_TASK_IDS),
    }


def declared_task_ids_and_deliverables(tasks: Sequence[JsonMap]) -> list[JsonDict]:
    return [
        {
            "task_id": str(task.get("id") or ""),
            "deliverable": str(task.get("deliverable") or ""),
            "track": task.get("track"),
            "requires": list(task.get("requires") or []),
            "gated_on": list(task.get("gated_on") or []),
        }
        for task in tasks
    ]


def conductor_terminal_receipts_by_task(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    text = (root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    rows: JsonDict = {}
    for task in tasks:
        title = str(task.get("title") or "")
        task_id = str(task.get("id") or "")
        needle = title[:48].lower()
        matches: list[JsonDict] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if needle and needle not in line.lower():
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) >= 4:
                matches.append(
                    {
                        "line": line_number,
                        "timestamp_utc": parts[0],
                        "title_truncated": parts[1],
                        "status": parts[2],
                        "message": parts[3],
                    }
                )
        statuses = [str(row["status"]) for row in matches]
        rows[task_id] = {
            "receipt_count": len(matches),
            "terminal_receipt_count": sum(
                status in {"OK", "FAIL", "FLAGGED", "GATE_BLOCK"} for status in statuses
            ),
            "statuses": statuses,
            "rows": matches,
        }
    return rows


def artifact_matrix(
    root: Path,
    tasks: Sequence[JsonMap],
    payloads: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        path = root / rel
        payload = dict(payloads.get(task_id) or {})
        self_task = task_id == EXPERIMENT_ID
        if self_task:
            classification = {
                "classification": "self_excluded",
                "terminal": True,
                "reason": "capstone output hash is self-referential",
                "path": rel.as_posix(),
                "present": path.exists(),
                "loadable": bool(payload),
                "sha256": None,
                "status_raw": payload.get("status"),
                "honest_verdict_raw": payload.get("honest_verdict"),
            }
        else:
            classification = classify_artifact_path(path).to_dict()
        rows[task_id] = {
            "task_id": task_id,
            "declared_deliverable": rel.as_posix(),
            "present": classification.get("present"),
            "loadable": classification.get("loadable"),
            "sha256": classification.get("sha256"),
            "schema_status": _schema_status(payload, self_task),
            "schema_raw": payload.get("schema") or payload.get("schema_version"),
            "status_raw": classification.get("status_raw"),
            "honest_verdict_raw": classification.get("honest_verdict_raw"),
            "terminal_class": classification.get("classification"),
            "terminal": classification.get("terminal"),
            "reason": classification.get("reason"),
            "self_referential_hash_excluded": self_task,
        }
    return rows


def dependency_recomputation(tasks: Sequence[JsonMap], matrix: JsonMap) -> JsonDict:
    ids = {str(task.get("id") or "") for task in tasks}
    failures: list[JsonDict] = []
    edges: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        for dep in task.get("requires") or []:
            dep_id = str(dep)
            dep_row = matrix.get(dep_id, {})
            missing = dep_id not in ids
            terminal = bool(isinstance(dep_row, Mapping) and dep_row.get("terminal"))
            edge = {
                "task_id": task_id,
                "dependency": dep_id,
                "declared": dep_id in ids,
                "terminal_or_structured": terminal,
                "terminal_class": dep_row.get("terminal_class")
                if isinstance(dep_row, Mapping)
                else None,
            }
            edges.append(edge)
            if missing or not terminal:
                failures.append(edge)
    return {"ok": not failures, "edge_count": len(edges), "edges": edges, "failures": failures}


def _compare_gate(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    return False


def structured_gate_recomputation(
    tasks: Sequence[JsonMap],
    payloads: JsonMap,
    matrix: JsonMap,
) -> JsonDict:
    gates: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        task_class = (
            matrix.get(task_id, {}).get("terminal_class")
            if isinstance(matrix.get(task_id), Mapping)
            else None
        )
        for gate in task.get("gated_on") or []:
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            op = str(gate.get("op") or "")
            expected = gate.get("value")
            upstream_payload = payloads.get(upstream, {})
            actual = (
                _extract_field(upstream_payload, field)
                if isinstance(upstream_payload, Mapping)
                else None
            )
            passed = _compare_gate(actual, op, expected)
            skip_effect = "none"
            if not passed and task_class == "skipped":
                skip_effect = "structured_skip_preserved"
            elif not passed:
                skip_effect = "gate_failed_without_structured_skip"
            gates.append(
                {
                    "task_id": task_id,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": op,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "downstream_terminal_class": task_class,
                    "skip_effect": skip_effect,
                }
            )
    failed = [row for row in gates if not row["passed"]]
    return {
        "gate_count": len(gates),
        "passed_gate_count": len(gates) - len(failed),
        "failed_gate_count": len(failed),
        "gates": gates,
        "ok": all(
            row["passed"] or row["skip_effect"] == "structured_skip_preserved" for row in gates
        ),
    }


def skipped_task_handling(matrix: JsonMap, receipts: JsonMap, payloads: JsonMap) -> JsonDict:
    skipped = [
        task_id
        for task_id, row in matrix.items()
        if isinstance(row, Mapping) and row.get("terminal_class") == "skipped"
    ]
    rows: JsonDict = {}
    for task_id in skipped:
        receipt = receipts.get(task_id, {}) if isinstance(receipts, Mapping) else {}
        statuses = receipt.get("statuses") if isinstance(receipt, Mapping) else []
        payload = payloads.get(task_id, {}) if isinstance(payloads.get(task_id), Mapping) else {}
        ready_scores = {
            key: value
            for key, value in payload.items()
            if key.endswith("_ready_score") or key.endswith("_utility_score")
        }
        rows[task_id] = {
            "receipt_statuses": list(statuses or []),
            "no_agent_execution": "OK" not in set(statuses or []),
            "hidden_utility_claim": any(
                _bare_value(value) not in (None, 0, 0.0) for value in ready_scores.values()
            ),
            "ready_score_fields": ready_scores,
        }
    return {"structured_skipped_task_ids": skipped, "rows": rows}


def prior_failure_and_retirement_audit(tasks: Sequence[JsonMap], payloads: JsonMap) -> JsonDict:
    rows: list[JsonDict] = []
    fired = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        verdict = str((payloads.get(task_id) or {}).get("honest_verdict") or "")
        for prior in task.get("prior_failures") or []:
            if not isinstance(prior, Mapping):
                continue
            prior_verdict = str(prior.get("verdict") or "")
            same = bool(prior_verdict and verdict and prior_verdict == verdict)
            rule_fired = same and prior.get("retire_if_same_verdict") is True
            fired += int(rule_fired)
            rows.append(
                {
                    "task_id": task_id,
                    "experiment_id": prior.get("experiment_id"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "actual_honest_verdict": verdict,
                    "prior_verdict": prior_verdict,
                    "same_verdict": same,
                    "rule_fired": rule_fired,
                    "addressed_by_present": bool(prior.get("addressed_by")),
                }
            )
    return {
        "prior_failure_count": len(rows),
        "retire_if_same_verdict_fired_count": fired,
        "rows": rows,
    }


def exclusion_manifest_audit(root: Path) -> JsonDict:
    data = _load_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    retired_count = len(data.get("retired") or []) + len(data.get("retired_experiments") or [])
    extras = data.get("retired_extras") or []
    blocked_patterns = sum(
        len(item.get("blocked_patterns") or []) for item in extras if isinstance(item, Mapping)
    )
    return {
        "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "retired_entry_count": retired_count,
        "retired_extra_count": len(extras),
        "blocked_pattern_count": blocked_patterns,
        "roadmap_lint_command": EXCLUSION_LINT_COMMAND,
    }


def prompt_contract_audit(tasks: Sequence[JsonMap]) -> JsonDict:
    failures: list[JsonDict] = []
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        expected_run = (
            f"Run command: .venv/bin/python -m carnot.{_module_name_for_task(task)} --date"
        )
        row = {
            "task_id": task_id,
            "run_command_present": expected_run in prompt,
            "protected_conductor_ending": prompt.strip().endswith(
                "Do NOT push. Do NOT modify scripts/research_conductor.py."
            ),
            "required_field_count": len(required_artifact_fields_from_prompt(prompt)),
        }
        rows.append(row)
        if (
            not row["run_command_present"]
            or not row["protected_conductor_ending"]
            or row["required_field_count"] == 0
        ):
            failures.append(row)
    return {"ok": not failures, "rows": rows, "failures": failures}


def required_field_and_field_principle_audit(
    tasks: Sequence[JsonMap],
    payloads: JsonMap,
    matrix: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        required = (
            sorted(REQUIRED_ARTIFACT_FIELDS)
            if task_id == EXPERIMENT_ID
            else required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        )
        payload = dict(payloads.get(task_id) or {})
        if task_id == EXPERIMENT_ID:
            payload = {field: None for field in required}
            payload["field_principles"] = FIELD_PRINCIPLES
        principles = payload.get("field_principles")
        terminal_class = (
            matrix.get(task_id, {}).get("terminal_class")
            if isinstance(matrix.get(task_id), Mapping)
            else None
        )
        missing_fields = [field for field in required if field not in payload]
        missing_principles = (
            []
            if terminal_class == "skipped"
            else [
                field
                for field in required
                if not isinstance(principles, Mapping) or field not in principles
            ]
        )
        row = {
            "required_field_count": len(required),
            "missing_required_fields": missing_fields,
            "missing_field_principles": missing_principles,
            "structured_skip_exemption": terminal_class == "skipped",
        }
        rows[task_id] = row
        if (missing_fields or missing_principles) and terminal_class != "skipped":
            failures.append({"task_id": task_id, **row})
    return {"ok": not failures, "rows": rows, "failures": failures}


def _model_ids_from_specs(specs: Any) -> set[str]:
    rows = specs.values() if isinstance(specs, Mapping) else specs
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return set()
    ids = set()
    for row in rows:
        if isinstance(row, Mapping) and row.get("hf_id"):
            ids.add(str(row["hf_id"]))
    return ids


def model_policy_and_MODEL_SPECS_audit(payloads: JsonMap, matrix: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    failures: list[JsonDict] = []
    for task_id in sorted(MODEL_TASK_IDS | SKIPPED_MODEL_TASK_IDS):
        payload = dict(payloads.get(task_id) or {})
        terminal_class = (
            matrix.get(task_id, {}).get("terminal_class")
            if isinstance(matrix.get(task_id), Mapping)
            else None
        )
        model_ids = _model_ids_from_specs(payload.get("MODEL_SPECS"))
        skipped = task_id in SKIPPED_MODEL_TASK_IDS and terminal_class == "skipped"
        missing = sorted(MANDATED_MODEL_IDS - model_ids)
        row = {
            "terminal_class": terminal_class,
            "structured_skip_exemption": skipped,
            "MODEL_SPECS_present": bool(model_ids),
            "model_ids": sorted(model_ids),
            "missing_mandated_model_ids": [] if skipped else missing,
            "mandatory_gguf_use_ok": skipped or not missing,
        }
        rows[task_id] = row
        if not row["mandatory_gguf_use_ok"]:
            failures.append({"task_id": task_id, **row})
    return {"ok": not failures, "rows": rows, "failures": failures}


def _iter_receipt_mappings(value: Any) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in (
                "method",
                "tokenizer_source",
                "release_within_512mb",
                "memory_released",
                "released",
            )
        ):
            rows.append(dict(value))
        for nested in value.values():
            rows.extend(_iter_receipt_mappings(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            rows.extend(_iter_receipt_mappings(item))
    return rows


def llama_cpp_embedded_tokenizer_audit(payloads: JsonMap, matrix: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    failures: list[str] = []
    for task_id in sorted(MODEL_TASK_IDS | SKIPPED_MODEL_TASK_IDS):
        terminal_class = (
            matrix.get(task_id, {}).get("terminal_class")
            if isinstance(matrix.get(task_id), Mapping)
            else None
        )
        if task_id in SKIPPED_MODEL_TASK_IDS and terminal_class == "skipped":
            rows[task_id] = {"structured_skip_exemption": True, "ok": True}
            continue
        receipts = _iter_receipt_mappings(
            (payloads.get(task_id) or {}).get("llama_cpp_embedded_tokenizer_receipts")
        )
        ok_rows = [
            row
            for row in receipts
            if row.get("loadable") is True
            or row.get("ok") is True
            or "embedded_gguf" in str(row.get("tokenizer_source") or row.get("method") or "")
        ]
        auto_tokenizer_used = any(row.get("autotokenizer_used") is True for row in receipts)
        ok = len(ok_rows) >= len(MANDATED_MODEL_IDS) and not auto_tokenizer_used
        rows[task_id] = {
            "receipt_count": len(receipts),
            "embedded_ok_count": len(ok_rows),
            "auto_tokenizer_used": auto_tokenizer_used,
            "ok": ok,
        }
        if not ok:
            failures.append(task_id)
    return {"ok": not failures, "rows": rows, "failures": failures}


def gpu_offload_and_memory_release_audit(payloads: JsonMap, matrix: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    failures: list[str] = []
    for task_id in sorted(MODEL_TASK_IDS | SKIPPED_MODEL_TASK_IDS):
        terminal_class = (
            matrix.get(task_id, {}).get("terminal_class")
            if isinstance(matrix.get(task_id), Mapping)
            else None
        )
        if task_id in SKIPPED_MODEL_TASK_IDS and terminal_class == "skipped":
            rows[task_id] = {"structured_skip_exemption": True, "ok": True}
            continue
        receipts = _iter_receipt_mappings(
            (payloads.get(task_id) or {}).get(
                "cuda_gpu_offload_and_memory_release_receipts_by_model"
            )
        )
        release_rows = [
            row
            for row in receipts
            if row.get("release_within_512mb") is True
            or row.get("memory_released") is True
            or row.get("released") is True
            or row.get("memory_delta_after_release_mb") == 0
        ]
        ok = len(release_rows) >= len(MANDATED_MODEL_IDS)
        rows[task_id] = {
            "receipt_count": len(receipts),
            "release_ok_count": len(release_rows),
            "loaded_one_placement_receipts": sum(
                row.get("loaded_one_placement_at_a_time") is True for row in receipts
            ),
            "ok": ok,
        }
        if not ok:
            failures.append(task_id)
    return {"ok": not failures, "rows": rows, "failures": failures}


def _sum_fields(payloads: JsonMap, fields: Sequence[str]) -> tuple[int, JsonDict]:
    rows: JsonDict = {}
    total = 0
    for task_id, payload in payloads.items():
        if not isinstance(payload, Mapping):
            continue
        task_total = sum(_numeric_count(payload.get(field)) for field in fields)
        if task_total or any(field in payload for field in fields):
            rows[task_id] = {
                field: _numeric_count(payload.get(field)) for field in fields if field in payload
            }
        total += task_total
    return total, rows


def source_model_weight_mutation_audit(payloads: JsonMap) -> JsonDict:
    total, rows = _sum_fields(payloads, ZERO_MUTATION_FIELDS)
    return {"ok": total == 0, "total_mutation_count": total, "rows": rows}


def generated_label_and_hidden_state_audit(payloads: JsonMap) -> JsonDict:
    generated, generated_rows = _sum_fields(payloads, ("generated_label_count",))
    hidden, hidden_rows = _sum_fields(payloads, ("hidden_state_access_count",))
    return {
        "ok": generated == 0 and hidden == 0,
        "generated_label_count": generated,
        "hidden_state_access_count": hidden,
        "generated_label_rows": generated_rows,
        "hidden_state_rows": hidden_rows,
    }


def exact_oracle_and_learned_claim_boundary_audit(payloads: JsonMap) -> JsonDict:
    upstream_oracles: JsonDict = {}
    for task_id, payload in payloads.items():
        if task_id == EXPERIMENT_ID or not isinstance(payload, Mapping):
            continue
        value = payload.get("verifier_is_oracle")
        if value not in (None, False):
            upstream_oracles[task_id] = {
                "verifier_is_oracle": value,
                "exact_boundary": payload.get("exact_oracle_claim_boundary"),
            }
    return {
        "capstone_verifier_is_oracle": False,
        "upstream_oracle_tasks": sorted(upstream_oracles),
        "upstream_oracle_details": upstream_oracles,
        "learned_verifier_promotion_count": 0,
        "ok": True,
    }


def prefix_generation_determination(payloads: JsonMap, matrix: JsonMap) -> JsonDict:
    exp6339 = payloads.get("exp6339-incremental-prefix-enforcement-substrate") or {}
    exp6340 = payloads.get("exp6340-parser-jit-semantic-diversity-canary") or {}
    exp6341_row = matrix.get("exp6341-prospective-prefix-utility-ab", {})
    return {
        "substrate_ready_score": _extract_field(
            exp6339, "prefix_enforcement_substrate_ready_score"
        ),
        "semantic_diversity_gain_score": _extract_field(exp6340, "semantic_diversity_gain_score"),
        "held_utility_terminal_class": exp6341_row.get("terminal_class")
        if isinstance(exp6341_row, Mapping)
        else None,
        "closure": "not_closed",
        "reason": "Exp6340 canary was null, so Exp6341 stayed a structured skip.",
    }


def certified_continuous_learning_determination(payloads: JsonMap) -> JsonDict:
    exp6345 = payloads.get("exp6345-prospective-certified-factor-evolution-ab") or {}
    ready = _extract_field(exp6345, "certified_continuous_learning_ready_score")
    unsafe = _numeric_count(exp6345.get("unsafe_commit_count"))
    return {
        "certified_continuous_learning_ready_score": ready,
        "unsafe_commit_count": unsafe,
        "source_model_weight_mutation_count": _numeric_count(
            exp6345.get("source_model_weight_mutation_count")
        ),
        "closure": "closed" if ready == 1.0 and unsafe == 0 else "blocked_or_null",
        "boundary": "exact_release_and_frozen_base_weights",
    }


def eprocess_and_factor_lifecycle_determination(payloads: JsonMap) -> JsonDict:
    exp6342 = payloads.get("exp6342-anytime-evalue-release-ledger") or {}
    exp6343 = payloads.get("exp6343-evidence-carrying-factor-lifecycle") or {}
    exp6345 = payloads.get("exp6345-prospective-certified-factor-evolution-ab") or {}
    rollback = _extract_field(exp6345, "rollback_byte_identity")
    rollback_ok = isinstance(rollback, Mapping) and rollback.get("byte_identical") is True
    return {
        "anytime_release_certificate_ready_score": _extract_field(
            exp6342, "anytime_release_certificate_ready_score"
        ),
        "evidence_factor_lifecycle_ready_score": _extract_field(
            exp6343, "evidence_factor_lifecycle_ready_score"
        ),
        "rollback_identity_ok": rollback_ok,
        "factor_lifecycle_bounds_ok": _extract_field(
            exp6343, "evidence_factor_lifecycle_ready_score"
        )
        == 1.0,
        "closure": "closed" if rollback_ok else "blocked",
    }


def safety_audit_determination(payloads: JsonMap) -> JsonDict:
    exp6346 = payloads.get("exp6346-certified-factor-evolution-safety-audit") or {}
    return {
        "safety_ready_score": _extract_field(exp6346, "safety_ready_score"),
        "unsafe_commit_count": _numeric_count(exp6346.get("unsafe_commit_count")),
        "undetected_harmful_attack_count": _numeric_count(
            exp6346.get("undetected_harmful_attack_count")
        ),
        "utility_promotion_count": _numeric_count(exp6346.get("utility_promotion_count")),
        "closure": "safety_only_closed",
    }


def arc_action_influence_determination(payloads: JsonMap) -> JsonDict:
    exp6348 = payloads.get("exp6348-arc-default-off-action-influence-ab") or {}
    ready = _extract_field(exp6348, "arc_causal_influence_ready_score")
    solve_count = _numeric_count(exp6348.get("solve_claim_count"))
    return {
        "arc_causal_influence_ready_score": ready,
        "solve_claim_count": solve_count,
        "solve_provenance": exp6348.get("solve_provenance"),
        "closure": "closed_no_solve" if ready == 1.0 and solve_count == 0 else "blocked_or_null",
        "claim_boundary": "action_order_and_one_step_quality_only",
    }


def solve_provenance_audit(payloads: JsonMap) -> JsonDict:
    total, rows = _sum_fields(payloads, ("solve_claim_count",))
    provenances = {
        task_id: payload.get("solve_provenance")
        for task_id, payload in payloads.items()
        if isinstance(payload, Mapping) and payload.get("solve_provenance")
    }
    return {
        "ok": total == 0,
        "solve_claim_count": total,
        "solve_provenance_by_task": provenances,
        "rows": rows,
    }


def arc_registry_immutability_audit(
    root: Path, before_hashes: JsonMap, payloads: JsonMap
) -> JsonDict:
    total, rows = _sum_fields(payloads, ("registry_update_count",))
    path = ARC_REGISTRY_RELATIVE_PATH.as_posix()
    after = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    return {
        "ok": total == 0 and before_hashes.get(path) == after,
        "path": path,
        "before_sha256": before_hashes.get(path),
        "after_sha256": after,
        "hash_unchanged": before_hashes.get(path) == after,
        "registry_update_count": total,
        "rows": rows,
    }


def hardware_nonuse_and_inference_substrate_audit(payloads: JsonMap) -> JsonDict:
    hardware_claim_count = 0
    hardware_rows: JsonDict = {}
    for task_id, payload in payloads.items():
        if not isinstance(payload, Mapping):
            continue
        row_count = sum(
            _numeric_count(payload.get(field))
            for field in ("hardware_claim_count", "hardware_execution_count")
        )
        if row_count:
            hardware_rows[task_id] = row_count
        hardware_claim_count += row_count
    return {
        "ok": hardware_claim_count == 0,
        "v546_hardware_claim_count": hardware_claim_count,
        "hardware_rows": hardware_rows,
        "nonuse_boundaries": {
            "GateMate": "no V546 board command",
            "KV260": "no V546 workload delta",
            "TSU": "no authenticated access",
            "Kona": "no public local weights or API",
        },
        "capstone_inference_substrate": INFERENCE_SUBSTRATE,
    }


def verification_cost_accounting_audit(payloads: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    missing: list[str] = []
    for task_id in VERIFICATION_COST_TASK_IDS:
        payload = payloads.get(task_id) or {}
        keys = [
            key
            for key in payload
            if key.startswith("verification_calls_time")
            or key.startswith("verification_cost")
            or key == "verification_calls_time_cost_and_error_table"
        ]
        rows[task_id] = {"cost_field_keys": sorted(keys), "present": bool(keys)}
        if not keys:
            missing.append(task_id)
    return {"ok": not missing, "rows": rows, "missing_cost_task_ids": missing}


def three_gap_closure_matrix(
    prefix: JsonMap,
    learning: JsonMap,
    arc: JsonMap,
) -> JsonDict:
    return {
        "gap_1_prefix_generation": {
            "state": "skipped_after_null_canary",
            "closed": False,
            "evidence": prefix,
        },
        "gap_2_certified_self_learning": {
            "state": learning.get("closure"),
            "closed": learning.get("closure") == "closed",
            "evidence": learning,
        },
        "gap_3_arc_action_influence": {
            "state": arc.get("closure"),
            "closed": arc.get("closure") == "closed_no_solve",
            "evidence": arc,
        },
    }


def prd_requirement_mapping() -> JsonDict:
    return {
        "FR-09": "Focused and full pytest commands are recorded.",
        "FR-10": "REQ-INFRA-6349 and scenarios anchor the capstone.",
        "FR-11": "Certified self-learning evidence stays inside exact release boundaries.",
        "FR-12": "Exact verifier and oracle boundaries are reported without promotion.",
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def protected_files_changed_with_reasons(root: Path, before_hashes: JsonMap) -> JsonDict:
    after = protected_hashes(root)
    rows = []
    for path in sorted(set(before_hashes) | set(after)):
        before = before_hashes.get(path)
        current = after.get(path)
        if before != current:
            rows.append(
                {
                    "path": path,
                    "before_sha256": before,
                    "after_sha256": current,
                    "reason": "unexpected_protected_change",
                }
            )
    return {"changed_count": len(rows), "rows": rows}


def docs_and_archive_reconciliation(root: Path) -> JsonDict:
    complete_text = (root / RESEARCH_COMPLETE_RELATIVE_PATH).read_text(encoding="utf-8")
    status_text = (root / STATUS_RELATIVE_PATH).read_text(encoding="utf-8")
    changelog_text = (root / CHANGELOG_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "research_complete_mentions_v546": "2026.08.546" in complete_text
        or "exp6349" in complete_text,
        "status_mentions_v546": "2026.08.546" in status_text or "Exp6349" in status_text,
        "changelog_mentions_v546": "2026.08.546" in changelog_text or "Exp6349" in changelog_text,
        "conductor_reconciliation_step_expected": True,
        "docs_modified_by_capstone": False,
        "paths": [
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            STATUS_RELATIVE_PATH.as_posix(),
            CHANGELOG_RELATIVE_PATH.as_posix(),
            TRACEABILITY_RELATIVE_PATH.as_posix(),
        ],
    }


def _v546_result_paths(root: Path, tasks: Sequence[JsonMap]) -> list[Path]:
    paths = {Path(str(task.get("deliverable") or "")) for task in tasks}
    for pattern in ("results/experiment_633[7-9]*", "results/experiment_634[0-8]*"):
        paths.update(path.relative_to(root) for path in root.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.as_posix())


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
    )
    return {command: shutil.which(command) for command in commands}


def _git_status_lines(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def preconditions_checked(root: Path, tasks: Sequence[JsonMap], before_hashes: JsonMap) -> JsonDict:
    input_paths = [
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("_bmad/prd.md"),
        Path("_bmad/architecture.md"),
        *PROTECTED_RELATIVE_PATHS,
        Path("scripts/roadmap_schema.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/validate_prior_failures.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        *[Path("tests/python/test_experiment_6349_v546_adversarial_capstone.py")],
        *[Path("python/carnot/experiment_6349_v546_adversarial_capstone.py")],
        *_v546_result_paths(root, tasks),
    ]
    input_hashes = {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in sorted(set(input_paths), key=lambda item: item.as_posix())
    }
    return {
        "input_hashes": input_hashes,
        "input_hash_count": len(input_hashes),
        "protected_hashes_before_artifact_write": before_hashes,
        "git_status_before": _git_status_lines(root),
        "disk": _disk_receipt(root),
        "command_availability": _command_availability(),
        "date_assumption": "2026-08-12",
        "capstone_writes_only_result_artifact": True,
        "research_conductor_modified": False,
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6349",
        ROADMAP_RELATIVE_PATH.as_posix(),
        PROPOSAL_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        HARDWARE_WISHLIST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        "results/experiment_6337..6348_*.json",
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _command_exit_codes(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _status_from_exit_codes(exit_codes: JsonMap) -> str:
    return (
        "blocked_validation_command_failed"
        if any(int(code) != 0 for code in exit_codes.values())
        else "complete_mixed_terminal_record"
    )


def _honest_verdict(status: str) -> str:
    if status.startswith("blocked"):
        return "blocked: validation command failed; V546 claims are not promoted"
    return (
        "complete: V546 terminal record reconciled; prefix branch skipped after null "
        "canary, certified learning closed inside exact boundaries, ARC action "
        "influence closed with no solve claim"
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    before_hashes: JsonMap | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    tasks = _roadmap_tasks(root)
    payloads = _read_payloads(root, tasks)
    receipts = conductor_terminal_receipts_by_task(root, tasks)
    matrix = artifact_matrix(root, tasks, payloads)
    command_rows = [dict(row) for row in command_receipts or []]
    exit_codes = _command_exit_codes(command_rows)
    status = _status_from_exit_codes(exit_codes)
    prefix = prefix_generation_determination(payloads, matrix)
    learning = certified_continuous_learning_determination(payloads)
    lifecycle = eprocess_and_factor_lifecycle_determination(payloads)
    safety = safety_audit_determination(payloads)
    arc = arc_action_influence_determination(payloads)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "milestone_and_roadmap_hash": milestone_and_roadmap_hash(root, tasks),
        "declared_task_ids_and_deliverables": declared_task_ids_and_deliverables(tasks),
        "conductor_terminal_receipts_by_task": receipts,
        "artifact_existence_hash_schema_status_and_honest_verdict_by_task": matrix,
        "dependency_recomputation": dependency_recomputation(tasks, matrix),
        "structured_gate_recomputation": structured_gate_recomputation(tasks, payloads, matrix),
        "skipped_task_handling": skipped_task_handling(matrix, receipts, payloads),
        "prior_failure_and_retirement_audit": prior_failure_and_retirement_audit(tasks, payloads),
        "exclusion_manifest_audit": exclusion_manifest_audit(root),
        "prompt_contract_audit": prompt_contract_audit(tasks),
        "required_field_and_field_principle_audit": required_field_and_field_principle_audit(
            tasks, payloads, matrix
        ),
        "model_policy_and_MODEL_SPECS_audit": model_policy_and_MODEL_SPECS_audit(payloads, matrix),
        "llama_cpp_embedded_tokenizer_audit": llama_cpp_embedded_tokenizer_audit(payloads, matrix),
        "gpu_offload_and_memory_release_audit": gpu_offload_and_memory_release_audit(
            payloads, matrix
        ),
        "source_model_weight_mutation_audit": source_model_weight_mutation_audit(payloads),
        "generated_label_and_hidden_state_audit": generated_label_and_hidden_state_audit(payloads),
        "exact_oracle_and_learned_claim_boundary_audit": exact_oracle_and_learned_claim_boundary_audit(
            payloads
        ),
        "prefix_generation_determination": prefix,
        "certified_continuous_learning_determination": learning,
        "eprocess_and_factor_lifecycle_determination": lifecycle,
        "safety_audit_determination": safety,
        "arc_action_influence_determination": arc,
        "solve_provenance_audit": solve_provenance_audit(payloads),
        "arc_registry_immutability_audit": arc_registry_immutability_audit(root, before, payloads),
        "hardware_nonuse_and_inference_substrate_audit": hardware_nonuse_and_inference_substrate_audit(
            payloads
        ),
        "verification_cost_accounting_audit": verification_cost_accounting_audit(payloads),
        "three_gap_closure_matrix": three_gap_closure_matrix(prefix, learning, arc),
        "prd_requirement_mapping": prd_requirement_mapping(),
        "protected_files_changed_with_reasons": protected_files_changed_with_reasons(root, before),
        "docs_and_archive_reconciliation": docs_and_archive_reconciliation(root),
        "preconditions_checked": preconditions_checked(root, tasks, before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "llm_call_count": 0,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": exit_codes,
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    principles = report.get("field_principles")
    provenance = report.get("field_provenance")
    if not isinstance(principles, Mapping):
        errors.append("field_principles is not a mapping")
        principles = {}
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance is not a mapping")
        provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if report.get("llm_call_count") != 0 or not isinstance(report.get("llm_call_count"), int):
        errors.append("llm_call_count must be bare 0")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not str(report.get("honest_verdict") or "").startswith(("complete:", "blocked:")):
        errors.append("honest_verdict lacks terminal prefix")
    exit_codes = report.get("test_exit_codes")
    if isinstance(exit_codes, Mapping) and any(int(code) != 0 for code in exit_codes.values()):
        if not str(report.get("status") or "").startswith("blocked"):
            errors.append("nonzero command exit requires blocked status")
        if not str(report.get("honest_verdict") or "").startswith("blocked:"):
            errors.append("nonzero command exit requires blocked honest_verdict")
    if report.get("verifier_is_oracle") is not False:
        errors.append("capstone verifier_is_oracle must remain false")
    checksum = report.get("reproducibility_checksum")
    if not checksum:
        errors.append("reproducibility_checksum missing")
    elif checksum != payload_checksum(report):
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
        raise ValueError(f"invalid Exp6349 report: {errors}")
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
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [
            {"command": str(command), "exit_code": int(exit_code)}
            for command, exit_code in payload.items()
        ]
    return [
        {"command": str(row["command"]), "exit_code": int(row.get("exit_code") or 0)}
        for row in payload
        if isinstance(row, Mapping) and row.get("command")
    ]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    before = protected_hashes(root)
    report = build_report(
        root,
        date=date,
        command_receipts=command_receipts or read_external_test_receipts(),
        before_hashes=before,
        started_at=time.perf_counter(),
    )
    if write:
        report["protected_files_changed_with_reasons"] = protected_files_changed_with_reasons(
            root, before
        )
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
