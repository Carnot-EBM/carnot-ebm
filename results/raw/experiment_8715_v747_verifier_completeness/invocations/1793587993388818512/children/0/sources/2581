"""Build the deterministic V604 manifest and evidence-admission receipt.

The reducer keeps the roadmap contract separate from scientific admission.
A clean manifest cannot approve science. A flagged source cannot enter a
headline through a downstream reducer. See REQ-REPORT-6898.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
Verifier = Callable[[Path], Mapping[str, Any]]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6898_v604_evidence_admissibility_contract.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_PATH = Path("scripts/conductor_gates.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
V603_ARTIFACT_PATHS = {
    6885: Path("results/experiment_6885_v603_executable_manifest_branch_contract.json"),
    6887: Path("results/experiment_6887_three_family_relation_proposal_corpus.json"),
    6888: Path("results/experiment_6888_independent_relation_qualification.json"),
}
V604_MILESTONE = "2026.09.604"
EXPECTED_NUMBERS = tuple(range(6898, 6911))
SCIENCE_SOURCE_IDS = frozenset({6887, 6888})
INFERENCE_SUBSTRATE = "deterministic_manifest_and_admissibility_audit_no_llm"
RANDOM_SEED = 6898
PROMPT_SECTIONS = ("CONTEXT:", "EXISTING CODE TO READ FIRST:", "TASK:", "CONCRETE STEPS:")
FINAL_SENTENCE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
BLOCKED_VERDICT = "complete_blocked_v604_evidence_admissibility_contract"
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ROW_COLLECTION_FIELDS = (
    "rows",
    "document_task_rows",
    "yaml_task_rows",
    "document_yaml_parity_rows",
    "gate_contract_rows",
    "prior_failure_contract_rows",
    "prompt_ending_rows",
    "branch_root_rows",
    "ungated_tail_rows",
    "adversarial_recheck_rows",
    "artifact_admissibility_rows",
    "dependency_taint_rows",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    *ROW_COLLECTION_FIELDS,
    "quarantined_artifact_ids",
    "task_count",
    "experiment_range",
    "random_seed",
    "reproducibility_checksum",
    "v604_manifest_contract_ready_score",
    "v603_admissible_science_source_count",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets later reducers reject incompatible receipts.",
    "experiment_id": "The fixed identity binds this receipt to Exp6898.",
    "run_date": "The date separates this audit from later roadmap states.",
    "status": "A terminal status proves the audit completed its inventory.",
    "field_principles": "A reason per field keeps later audits understandable.",
    "preconditions_checked": "Exact source checks make missing evidence actionable.",
    "inference_substrate": "The no-LLM declaration prevents this audit from posing as science.",
    "duration_s": "Measured wall time shows that the deterministic reducer ran.",
    "source_artifact_hashes": "Hashes bind conclusions to the exact primary bytes.",
    "rows": "Combined task, contract, and admission rows prevent hidden omissions.",
    "document_task_rows": "Document rows preserve the design as an independent source.",
    "yaml_task_rows": "YAML rows preserve the executable queue seen by the conductor.",
    "document_yaml_parity_rows": "One row per task exposes every identity and order mismatch.",
    "gate_contract_rows": "Exact gate rows prevent dead or misspelled branch edges.",
    "prior_failure_contract_rows": "Primary verdict checks stop unchanged failures returning silently.",
    "prompt_ending_rows": "Exact endings prevent truncated or mutable run instructions.",
    "branch_root_rows": "Root rows prove that the advisory audit cannot gate science.",
    "ungated_tail_rows": "The tail row keeps terminal synthesis runnable after branch blocks.",
    "adversarial_recheck_rows": "Fresh verifier output prevents stale flag decisions.",
    "artifact_admissibility_rows": "One row per artifact makes every admission decision explicit.",
    "dependency_taint_rows": "Separate taint rows stop a clean reducer from laundering a bad source.",
    "quarantined_artifact_ids": "The exact quarantine set is machine-readable for later consumers.",
    "task_count": "The fixed count detects partial roadmap activation before science dispatch.",
    "experiment_range": "The inclusive range detects hidden gaps and reused task numbers.",
    "random_seed": "The fixed seed states that this ordered audit has no hidden randomness.",
    "reproducibility_checksum": "A content hash detects silent changes while ignoring elapsed time.",
    "v604_manifest_contract_ready_score": "This advisory score covers document and YAML checks only.",
    "v603_admissible_science_source_count": "This count includes only clean, traceable science sources.",
    "gate_check_summary": "Expected and observed values make each blocked result actionable.",
    "verifier_is_oracle": "False prevents this structural audit from approving a scientific claim.",
    "verdict_class": "The closed class records the evidence boundary without free-form promotion.",
    "honest_verdict": "The terminal prefix lets the conductor classify the completed audit safely.",
}
SOURCE_PATHS = {
    "active_v604_roadmap": ROADMAP_PATH,
    "v604_design": DESIGN_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "conductor_gates": GATE_PATH,
    "prior_failure_validator": PRIOR_VALIDATOR_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "adversarial_verifier": ADVERSARIAL_PATH,
    "module": Path("python/carnot/experiment_6898_v604_evidence_admissibility_contract.py"),
    "wrapper": Path("scripts/experiments/experiment_6898_v604_evidence_admissibility_contract.py"),
    "focused_tests": Path(
        "tests/python/test_experiment_6898_v604_evidence_admissibility_contract.py"
    ),
    "spec": SPEC_PATH,
    **{f"v603_exp{number}": path for number, path in V603_ARTIFACT_PATHS.items()},
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 bytes for a deterministic identity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 identity in the receipt's explicit format."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash one file and preserve absence as null evidence."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def experiment_number(value: Any) -> int | None:
    """Read an experiment number without treating a Boolean as an integer."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"(?:exp|experiment_)?(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one check with exact expected and observed values."""

    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and expose the first failure for blocked artifacts."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _task_id_from_deliverable(number: int, deliverable: str) -> str:
    """Derive the executable task ID encoded by a standard result path."""

    stem = Path(deliverable).stem
    suffix = stem.removeprefix(f"experiment_{number}_").replace("_", "-")
    return f"exp{number}-{suffix}"


def _parse_condition(text: str) -> tuple[str, Any]:
    """Parse the comparison form used by the design gate table."""

    match = re.fullmatch(r"\s*(==|!=|>=|<=|>|<|exists|in)\s*(.*?)\s*", text)
    if not match:
        return text.strip(), None
    raw = match.group(2)
    return match.group(1), yaml.safe_load(raw) if raw else None


def parse_design(text: str) -> JsonDict:
    """Parse V604 task and gate rows independently from executable YAML."""

    milestone_match = re.search(r"^\*\*Milestone:\*\*\s*`([^`]+)`", text, re.MULTILINE)
    if not milestone_match:
        raise ValueError("design milestone is required")
    tasks: list[JsonDict] = []
    sections = re.finditer(
        r"^### Exp(?P<number>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^### Exp\d+:|^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    for match in sections:
        number = int(match.group("number"))
        if number not in EXPECTED_NUMBERS:
            continue
        found = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", match.group("body"))
        if not found:
            raise ValueError(f"Exp{number} design section is missing deliverable")
        deliverable = found.group(1)
        tasks.append(
            {
                "order": len(tasks) + 1,
                "number": number,
                "task_id": _task_id_from_deliverable(number, deliverable),
                "title": match.group("title").strip(),
                "milestone": milestone_match.group(1),
                "deliverable": deliverable,
                "gates": [],
            }
        )
    by_number = {row["number"]: row for row in tasks}
    pattern = re.compile(
        r"^\|\s*Exp(?P<down>\d+)\s*\|\s*`exp(?P<up>\d+)\.(?P<field>[^`]+)`\s*"
        r"\|\s*`(?P<condition>[^`]+)`\s*\|$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        downstream = int(match.group("down"))
        upstream = int(match.group("up"))
        if downstream not in by_number:
            continue
        op, value = _parse_condition(match.group("condition"))
        upstream_id = by_number.get(upstream, {}).get("task_id", f"exp{upstream}-missing")
        by_number[downstream]["gates"].append(
            {
                "upstream": upstream_id,
                "artifact_field": match.group("field"),
                "op": op,
                "value": value,
            }
        )
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def _required_fields(prompt: str) -> set[str]:
    """Read field tokens only from the prompt's required-field paragraph."""

    match = re.search(
        r"REQUIRED ARTIFACT FIELDS:(.*?)(?:\n\s*\n|\n\s*Run command:|\Z)",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    return set(re.findall(r"\b[a-z][a-z0-9_]+\b", match.group(1))) if match else set()


def parse_roadmap(document: Any) -> JsonDict:
    """Parse the active mapping while preserving exact gate and prior values."""

    if not isinstance(document, Mapping):
        raise ValueError("roadmap mapping is required")
    values = document.get("tasks")
    if not isinstance(values, list):
        raise ValueError("roadmap tasks list is required")
    rows: list[JsonDict] = []
    for order, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise ValueError(f"malformed roadmap task at order {order}")
        gates = value.get("gated_on", []) or []
        priors = value.get("prior_failures", []) or []
        if not isinstance(gates, list) or not isinstance(priors, list):
            raise ValueError(f"malformed task lists at order {order}")
        if any(not isinstance(gate, Mapping) for gate in gates):
            raise ValueError(f"malformed gate at order {order}")
        prompt = str(value.get("prompt") or "")
        rows.append(
            {
                "order": order,
                "number": experiment_number(value.get("id")),
                "task_id": str(value.get("id") or ""),
                "title": str(value.get("title") or ""),
                "milestone": str(value.get("milestone") or ""),
                "deliverable": str(value.get("deliverable") or ""),
                "gates": [
                    {
                        "upstream": str(gate.get("upstream") or ""),
                        "artifact_field": str(gate.get("artifact_field") or ""),
                        "op": str(gate.get("op") or ""),
                        "value": deepcopy(gate.get("value")),
                    }
                    for gate in gates
                ],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields": _required_fields(prompt),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def schema_gate_operators(source: str) -> set[str]:
    """Read allowed gate operators from the schema Literal declaration."""

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "GateOp" for target in node.targets
        ):
            values = {
                item.value
                for item in ast.walk(node.value)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if values:
                return values
    raise ValueError("GateOp Literal declaration is missing")


def conductor_gate_operators(source: str) -> set[str]:
    """Read operator branches implemented by the current gate evaluator."""

    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_eval_op"
        ),
        None,
    )
    if function is None:
        raise ValueError("_eval_op is missing")
    values: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != "op":
            continue
        for comparator in node.comparators:
            values.update(
                item.value
                for item in ast.walk(comparator)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    if not values:
        raise ValueError("_eval_op has no operator branches")
    return values


def _shape(row: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only task identity fields in public comparison rows."""

    if row is None:
        return None
    return {
        "order": row.get("order"),
        "number": row.get("number"),
        "task_id": row.get("task_id"),
        "milestone": row.get("milestone"),
        "deliverable": row.get("deliverable"),
    }


def _public_yaml_row(row: Mapping[str, Any]) -> JsonDict:
    """Replace full prompt text with its content identity."""

    return {
        **_shape(row),
        "title": row.get("title"),
        "gates": deepcopy(row.get("gates", [])),
        "prior_failure_count": len(row.get("prior_failures", [])),
        "prompt_sha256": sha256_bytes(str(row.get("prompt") or "").encode()),
    }


def _task_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Emit one exact contract row for each required task position."""

    rows: list[JsonDict] = []
    for index, expected_number in enumerate(EXPECTED_NUMBERS):
        document = document_tasks[index] if index < len(document_tasks) else None
        executable = yaml_tasks[index] if index < len(yaml_tasks) else None
        comparisons = {
            "document_present": document is not None,
            "yaml_present": executable is not None,
            "document_order": bool(document and document.get("number") == expected_number),
            "yaml_order": bool(executable and executable.get("number") == expected_number),
            "task_id": bool(
                document and executable and document.get("task_id") == executable.get("task_id")
            ),
            "deliverable": bool(
                document
                and executable
                and document.get("deliverable") == executable.get("deliverable")
            ),
            "milestone": bool(
                document and executable and document.get("milestone") == executable.get("milestone")
            ),
        }
        rows.append(
            {
                "order": index + 1,
                "expected_number": expected_number,
                "document": _shape(document),
                "yaml": _shape(executable),
                "comparisons": comparisons,
                "passed": all(comparisons.values()),
            }
        )
    return rows


def _prompt_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Check the required sections and exact final two non-empty lines."""

    rows: list[JsonDict] = []
    for index, number in enumerate(EXPECTED_NUMBERS):
        document = document_tasks[index] if index < len(document_tasks) else None
        executable = yaml_tasks[index] if index < len(yaml_tasks) else None
        prompt = str(executable.get("prompt") or "") if executable else ""
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        expected_command = None
        if document:
            stem = Path(str(document.get("deliverable") or "")).stem
            expected_command = (
                "Run command: cd {project_root} && .venv/bin/python "
                f"scripts/experiments/{stem}.py --date {{date}}"
            )
        sections = {section: section in prompt for section in PROMPT_SECTIONS}
        observed_command = lines[-2] if len(lines) >= 2 else None
        observed_sentence = lines[-1] if lines else None
        rows.append(
            {
                "order": index + 1,
                "number": number,
                "task_id": executable.get("task_id") if executable else None,
                "required_sections": sections,
                "expected_run_command": expected_command,
                "observed_run_command": observed_command,
                "expected_final_sentence": FINAL_SENTENCE,
                "observed_final_sentence": observed_sentence,
                "passed": bool(
                    document
                    and executable
                    and all(sections.values())
                    and observed_command == expected_command
                    and observed_sentence == FINAL_SENTENCE
                ),
            }
        )
    return rows


def _gate_rows(
    document_tasks: Sequence[Mapping[str, Any]],
    yaml_tasks: Sequence[Mapping[str, Any]],
    valid_operators: set[str],
) -> list[JsonDict]:
    """Validate every gate against design, order, operator, and producer fields."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    yaml_by_id = {row.get("task_id"): row for row in yaml_tasks}
    rows: list[JsonDict] = []
    for number in EXPECTED_NUMBERS:
        expected_gates = list(document_by_number.get(number, {}).get("gates", []))
        observed_gates = list(yaml_by_number.get(number, {}).get("gates", []))
        for index in range(max(len(expected_gates), len(observed_gates))):
            expected = expected_gates[index] if index < len(expected_gates) else None
            observed = observed_gates[index] if index < len(observed_gates) else None
            upstream = yaml_by_id.get(observed.get("upstream")) if observed else None
            field = str(observed.get("artifact_field") or "") if observed else ""
            downstream = yaml_by_number.get(number)
            checks = {
                "matches_document": expected == observed,
                "operator_valid": bool(observed and observed.get("op") in valid_operators),
                "upstream_present": upstream is not None,
                "upstream_earlier": bool(
                    upstream
                    and downstream
                    and upstream.get("order", 0) < downstream.get("order", 0)
                ),
                "producer_field_declared": bool(
                    upstream and field in upstream.get("required_fields", set())
                ),
                "advisory_manifest_not_consumed": experiment_number(
                    observed.get("upstream") if observed else None
                )
                != 6898,
            }
            rows.append(
                {
                    "downstream_number": number,
                    "gate_index": index,
                    "expected": deepcopy(expected),
                    "observed": deepcopy(observed),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def _prior_rows(
    yaml_tasks: Sequence[Mapping[str, Any]], prior_artifacts: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Check all four prior-failure fields against exact primary artifacts."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    for task in yaml_tasks:
        for index, value in enumerate(task.get("prior_failures", [])):
            prior = dict(value) if isinstance(value, Mapping) else {}
            prior_id = prior.get("experiment_id")
            primary = prior_artifacts.get(str(prior_id))
            primary_verdict = primary.get("honest_verdict") if primary else None
            failed = [field for field in required if field not in prior]
            if not isinstance(prior_id, str) or experiment_number(prior_id) is None:
                failed.append("experiment_id")
            if primary_verdict != prior.get("verdict"):
                failed.append("verdict")
            if (
                not isinstance(prior.get("addressed_by"), str)
                or not prior.get("addressed_by", "").strip()
            ):
                failed.append("addressed_by")
            if prior.get("retire_if_same_verdict") is not True:
                failed.append("retire_if_same_verdict")
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "prior_index": index,
                    "experiment_id": prior_id,
                    "declared_verdict": prior.get("verdict"),
                    "primary_verdict": primary_verdict,
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "failed_subfields": list(dict.fromkeys(failed)),
                    "passed": not failed,
                }
            )
    return rows


def _branch_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Prove three science roots, no advisory fanout, and one ungated tail."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    document_consumers = [
        task.get("task_id")
        for task in document_tasks
        if any(experiment_number(gate.get("upstream")) == 6898 for gate in task.get("gates", []))
    ]
    yaml_consumers = [
        task.get("task_id")
        for task in yaml_tasks
        if any(experiment_number(gate.get("upstream")) == 6898 for gate in task.get("gates", []))
    ]
    roots = [
        {
            "kind": "advisory_manifest_fanout",
            "number": 6898,
            "document_consumers": document_consumers,
            "yaml_consumers": yaml_consumers,
            "passed": not document_consumers and not yaml_consumers,
        }
    ]
    for number in (6899, 6905, 6906):
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        roots.append(
            {
                "kind": "independent_science_root",
                "number": number,
                "document_present_ungated": bool(document and not document.get("gates")),
                "yaml_present_ungated": bool(executable and not executable.get("gates")),
                "observed_gates": deepcopy(executable.get("gates", [])) if executable else None,
                "passed": bool(
                    document
                    and executable
                    and not document.get("gates")
                    and not executable.get("gates")
                ),
            }
        )
    document = document_by_number.get(6910)
    executable = yaml_by_number.get(6910)
    tails = [
        {
            "number": 6910,
            "document_present_ungated": bool(document and not document.get("gates")),
            "yaml_present_ungated": bool(executable and not executable.get("gates")),
            "observed_gates": deepcopy(executable.get("gates", [])) if executable else None,
            "passed": bool(
                document
                and executable
                and not document.get("gates")
                and not executable.get("gates")
            ),
        }
    ]
    return roots, tails


def evaluate_manifest_contract(
    design_text: str,
    roadmap: Mapping[str, Any],
    prior_artifacts: Mapping[str, Mapping[str, Any]],
    schema_source: str,
    gate_source: str,
) -> JsonDict:
    """Evaluate the complete V604 document and executable YAML contract."""

    document = parse_design(design_text)
    executable = parse_roadmap(roadmap)
    document_tasks = document["tasks"]
    yaml_tasks = executable["tasks"]
    valid_operators = schema_gate_operators(schema_source) & conductor_gate_operators(gate_source)
    task_rows = _task_rows(document_tasks, yaml_tasks)
    prompt_rows = _prompt_rows(document_tasks, yaml_tasks)
    gate_rows = _gate_rows(document_tasks, yaml_tasks, valid_operators)
    prior_rows = _prior_rows(yaml_tasks, prior_artifacts)
    root_rows, tail_rows = _branch_rows(document_tasks, yaml_tasks)
    document_numbers = [row["number"] for row in document_tasks]
    yaml_numbers = [row["number"] for row in yaml_tasks]
    document_deliverables = [row["deliverable"] for row in document_tasks]
    yaml_deliverables = [row["deliverable"] for row in yaml_tasks]
    document_ok = (
        document["milestone"] == V604_MILESTONE
        and document_numbers == list(EXPECTED_NUMBERS)
        and len(document_deliverables) == len(set(document_deliverables)) == 13
        and all(
            path.startswith("results/") and path.endswith(".json") for path in document_deliverables
        )
    )
    yaml_ok = (
        executable["milestone"] == V604_MILESTONE
        and yaml_numbers == list(EXPECTED_NUMBERS)
        and len({row["task_id"] for row in yaml_tasks}) == 13
        and len(yaml_deliverables) == len(set(yaml_deliverables)) == 13
        and all(
            path.startswith("results/") and path.endswith(".json") for path in yaml_deliverables
        )
        and all(row["milestone"] == V604_MILESTONE for row in yaml_tasks)
    )
    checks = [
        _check(
            "document_primary_contract",
            {"milestone": V604_MILESTONE, "task_count": 13, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": document["milestone"],
                "task_count": len(document_tasks),
                "numbers": document_numbers,
            },
            document_ok,
        ),
        _check(
            "yaml_executable_contract",
            {"milestone": V604_MILESTONE, "task_count": 13, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": executable["milestone"],
                "task_count": len(yaml_tasks),
                "numbers": yaml_numbers,
            },
            yaml_ok,
        ),
        _check(
            "document_yaml_task_contract",
            True,
            all(row["passed"] for row in task_rows),
            all(row["passed"] for row in task_rows),
        ),
        _check(
            "prompt_contract",
            True,
            all(row["passed"] for row in prompt_rows),
            all(row["passed"] for row in prompt_rows),
        ),
        _check(
            "gate_contract",
            True,
            all(row["passed"] for row in gate_rows),
            all(row["passed"] for row in gate_rows),
        ),
        _check(
            "prior_failure_contract",
            True,
            all(row["passed"] for row in prior_rows),
            all(row["passed"] for row in prior_rows),
        ),
        _check(
            "branch_root_isolation",
            True,
            all(row["passed"] for row in root_rows),
            all(row["passed"] for row in root_rows),
        ),
        _check(
            "ungated_terminal_tail",
            True,
            all(row["passed"] for row in tail_rows),
            all(row["passed"] for row in tail_rows),
        ),
    ]
    hard_failures = [
        {"hard_failure": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    return {
        "passed": not hard_failures,
        "checks": checks,
        "document_task_rows": deepcopy(document_tasks),
        "yaml_task_rows": [_public_yaml_row(row) for row in yaml_tasks],
        "document_yaml_parity_rows": task_rows,
        "gate_contract_rows": gate_rows,
        "prior_failure_contract_rows": prior_rows,
        "prompt_ending_rows": prompt_rows,
        "branch_root_rows": root_rows,
        "ungated_tail_rows": tail_rows,
        "hard_failures": hard_failures,
    }


def _dependency_ids(artifact: Mapping[str, Any]) -> set[int]:
    """Read declared source experiment IDs from hash keys and source paths."""

    dependencies: set[int] = set()
    sources = artifact.get("source_artifact_hashes", {})
    if not isinstance(sources, Mapping):
        return dependencies
    for key, value in sources.items():
        number = experiment_number(key)
        if number is not None:
            dependencies.add(number)
        if isinstance(value, Mapping):
            path_number = experiment_number(value.get("path"))
            if path_number is not None:
                dependencies.add(path_number)
    return dependencies


def evaluate_evidence_admission(
    artifacts: Mapping[int, Mapping[str, Any]], rechecks: Mapping[int, Mapping[str, Any]]
) -> JsonDict:
    """Quarantine current critical sources and every claim that consumes them."""

    critical_ids: set[int] = set()
    hard_failures: list[JsonDict] = []
    warnings: list[JsonDict] = []
    recheck_rows: list[JsonDict] = []
    for number in V603_ARTIFACT_PATHS:
        report = dict(rechecks.get(number, {}))
        flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
        critical = [flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"]
        if critical:
            critical_ids.add(number)
            hard_failures.extend({"experiment_id": number, **flag} for flag in critical)
        warnings.extend(
            {"experiment_id": number, **flag}
            for flag in flags
            if str(flag.get("severity", "")).lower() == "warn"
        )
        recheck_rows.append(
            {
                "experiment_id": number,
                "loaded": report.get("loaded") is True,
                "flag_count": report.get("flag_count", len(flags)),
                "max_severity": report.get("max_severity"),
                "flags": flags,
                "gate_version": report.get("gate_version"),
            }
        )
    quarantined = set(critical_ids)
    taint_rows: list[JsonDict] = []
    changed = True
    while changed:
        changed = False
        for number, artifact in artifacts.items():
            if number in quarantined:
                continue
            tainted_by = sorted(_dependency_ids(artifact) & quarantined)
            if not tainted_by:
                continue
            quarantined.add(number)
            changed = True
            taint_rows.append(
                {
                    "claim_experiment_id": number,
                    "source_experiment_id": tainted_by[0],
                    "all_tainted_sources": tainted_by,
                    "taint_kind": "transitive_source_quarantine",
                    "headline_eligible": False,
                }
            )
    admission_rows: list[JsonDict] = []
    for number in V603_ARTIFACT_PATHS:
        artifact = artifacts.get(number, {})
        report = rechecks.get(number, {})
        critical_kinds = sorted(
            str(flag.get("kind"))
            for flag in report.get("flags", [])
            if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        )
        traceable = bool(artifact.get("source_artifact_hashes"))
        failures: list[str] = []
        if report.get("loaded") is not True:
            failures.append("adversarial_recheck_not_loaded")
        if critical_kinds:
            failures.append("current_critical_flag")
        if not traceable:
            failures.append("independent_traceability_missing")
        if number in quarantined and number not in critical_ids:
            failures.append("transitive_dependency_taint")
        qualified_arms = sorted(
            str(row.get("arm"))
            for row in artifact.get("eligible_arm_rows", [])
            if isinstance(row, Mapping) and row.get("passed") is True
        )
        science_source = number in SCIENCE_SOURCE_IDS
        admissible = not failures
        admission_rows.append(
            {
                "experiment_id": number,
                "science_source": science_source,
                "terminal": bool(artifact.get("honest_verdict")),
                "independently_traceable": traceable,
                "critical_flag_kinds": critical_kinds,
                "dependency_ids": sorted(_dependency_ids(artifact)),
                "locally_qualified_arms": qualified_arms,
                "admission_failures": failures,
                "admissible": admissible,
                "headline_eligible": bool(science_source and admissible),
                "quarantined": number in quarantined,
            }
        )
    count = sum(row["science_source"] and row["admissible"] for row in admission_rows)
    return {
        "adversarial_recheck_rows": recheck_rows,
        "artifact_admissibility_rows": admission_rows,
        "dependency_taint_rows": taint_rows,
        "quarantined_artifact_ids": sorted(quarantined),
        "v603_admissible_science_source_count": count,
        "warnings": warnings,
        "hard_failures": hard_failures,
    }


def read_yaml_mapping(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one YAML mapping and return a stable failure label."""

    if not path.is_file():
        return None, "missing"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, f"invalid_yaml:{type(exc).__name__}"
    if not isinstance(value, Mapping):
        return None, "yaml_mapping_required"
    return dict(value), None


def read_json_mapping(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one JSON object and retain its exact failure class."""

    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(value, Mapping):
        return None, "json_object_required"
    return dict(value), None


def _prior_artifact_path(root: Path, experiment_id: Any) -> Path | None:
    """Map an exact prior task ID to its primary result path."""

    match = re.fullmatch(r"exp(\d+)-(.+)", str(experiment_id or ""), re.IGNORECASE)
    if not match:
        return None
    suffix = match.group(2).replace("-", "_")
    return root / "results" / f"experiment_{match.group(1)}_{suffix}.json"


def _load_prior_artifacts(
    root: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Load the primary artifact declared by every prior-failure row."""

    artifacts: dict[str, JsonDict] = {}
    rows: list[JsonDict] = []
    for task in tasks:
        for prior in task.get("prior_failures", []):
            prior_id = prior.get("experiment_id") if isinstance(prior, Mapping) else None
            path = _prior_artifact_path(root, prior_id)
            payload, error = read_json_mapping(path) if path else (None, "invalid_id")
            if payload is not None:
                artifacts[str(prior_id)] = payload
            rows.append(
                {
                    "experiment_id": prior_id,
                    "path": path.relative_to(root).as_posix() if path else None,
                    "sha256": sha256_path(path) if path else None,
                    "error": error,
                }
            )
    return artifacts, rows


def _empty_evaluation(error: str) -> JsonDict:
    """Return complete empty collections when primary parsing cannot continue."""

    return {
        "passed": False,
        "checks": [_check("contract_evaluation", "valid primary records", error, False)],
        "document_task_rows": [],
        "yaml_task_rows": [],
        "document_yaml_parity_rows": [],
        "gate_contract_rows": [],
        "prior_failure_contract_rows": [],
        "prompt_ending_rows": [],
        "branch_root_rows": [],
        "ungated_tail_rows": [],
        "hard_failures": [{"hard_failure": "contract_evaluation", "observed": error}],
    }


def _empty_admission() -> JsonDict:
    """Return empty evidence rows when required artifacts cannot be read."""

    return {
        "adversarial_recheck_rows": [],
        "artifact_admissibility_rows": [],
        "dependency_taint_rows": [],
        "quarantined_artifact_ids": [],
        "v603_admissible_science_source_count": 0,
        "warnings": [],
        "hard_failures": [],
    }


def _source_hashes(root: Path, prior_rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Bind all required, implementation, and prior sources to exact bytes."""

    rows = {
        name: {"path": path.as_posix(), "sha256": sha256_path(root / path)}
        for name, path in SOURCE_PATHS.items()
    }
    for index, row in enumerate(prior_rows):
        rows[f"prior_{index}_{row.get('experiment_id')}"] = {
            "path": row.get("path"),
            "sha256": row.get("sha256"),
            "error": row.get("error"),
        }
    return rows


def _checksum_value(value: Any) -> Any:
    """Remove only elapsed time and the checksum from reproducible content."""

    if isinstance(value, Mapping):
        return {
            key: _checksum_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_checksum_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic content in the complete receipt."""

    return sha256_bytes(canonical_json(_checksum_value(dict(artifact))))


def _default_verifier(path: Path) -> Mapping[str, Any]:
    """Run the current on-disk adversarial verifier on one artifact."""

    from scripts.adversarial_verify import verify_artifact

    return verify_artifact(path)


def build_artifact(root: Path, run_date: str, verifier: Verifier | None = None) -> JsonDict:
    """Build a complete positive or blocked V604 audit receipt."""

    started = time.perf_counter()
    source_errors: list[JsonDict] = []
    roadmap, roadmap_error = read_yaml_mapping(root / ROADMAP_PATH)
    exclusion, exclusion_error = read_yaml_mapping(root / EXCLUSION_PATH)
    text_sources: dict[str, str] = {}
    for name, path in (
        ("v604_design", DESIGN_PATH),
        ("roadmap_schema", SCHEMA_PATH),
        ("conductor_gates", GATE_PATH),
        ("prior_failure_validator", PRIOR_VALIDATOR_PATH),
        ("roadmap_gate_audit", GATE_AUDIT_PATH),
        ("adversarial_verifier", ADVERSARIAL_PATH),
    ):
        try:
            text_sources[name] = (root / path).read_text(encoding="utf-8")
            if not text_sources[name]:
                source_errors.append({"source": name, "error": "empty"})
        except (OSError, UnicodeError) as exc:
            text_sources[name] = ""
            source_errors.append({"source": name, "error": f"unreadable:{type(exc).__name__}"})
    for name, error in (
        ("active_v604_roadmap", roadmap_error),
        ("exclusion_manifest", exclusion_error),
    ):
        if error:
            source_errors.append({"source": name, "error": error})
    artifacts: dict[int, JsonDict] = {}
    for number, path in V603_ARTIFACT_PATHS.items():
        payload, error = read_json_mapping(root / path)
        terminal = bool(payload and payload.get("honest_verdict") and payload.get("status"))
        if error or not terminal:
            source_errors.append(
                {"source": path.as_posix(), "error": error or "terminal_artifact_required"}
            )
        if payload is not None:
            artifacts[number] = payload
    raw_tasks = roadmap.get("tasks", []) if roadmap else []
    prior_artifacts, prior_rows = _load_prior_artifacts(
        root, [task for task in raw_tasks if isinstance(task, Mapping)]
    )
    source_errors.extend(
        {"source": row.get("path") or row.get("experiment_id"), "error": row["error"]}
        for row in prior_rows
        if row.get("error")
    )
    try:
        design = parse_design(text_sources["v604_design"])
        if design["milestone"] != V604_MILESTONE or [
            row["number"] for row in design["tasks"]
        ] != list(EXPECTED_NUMBERS):
            source_errors.append(
                {
                    "source": "v604_design",
                    "error": "v604_design_contract_required",
                    "observed_milestone": design["milestone"],
                    "observed_numbers": [row["number"] for row in design["tasks"]],
                }
            )
        evaluation = evaluate_manifest_contract(
            text_sources["v604_design"],
            roadmap or {},
            prior_artifacts,
            text_sources["roadmap_schema"],
            text_sources["conductor_gates"],
        )
    except (ValueError, SyntaxError, yaml.YAMLError) as exc:
        source_errors.append({"source": "contract_evaluation", "error": str(exc)})
        evaluation = _empty_evaluation(str(exc))
    admission = _empty_admission()
    if len(artifacts) == len(V603_ARTIFACT_PATHS):
        reports: dict[int, Mapping[str, Any]] = {}
        checker = verifier or _default_verifier
        for number, path in V603_ARTIFACT_PATHS.items():
            try:
                reports[number] = checker(root / path)
            except Exception as exc:  # The artifact must record verifier failure, not disappear.
                source_errors.append(
                    {"source": path.as_posix(), "error": f"verifier_failed:{type(exc).__name__}"}
                )
                reports[number] = {"loaded": False, "flags": []}
        admission = evaluate_evidence_admission(artifacts, reports)
        for row in admission["adversarial_recheck_rows"]:
            if row["loaded"] is not True:
                source_errors.append(
                    {
                        "source": f"exp{row['experiment_id']}",
                        "error": "adversarial_recheck_not_loaded",
                    }
                )
    preconditions = gate_summary(
        [_check("required_source_readability", [], source_errors, not source_errors)]
    )
    checks = [
        _check(
            "preconditions",
            "all required sources readable",
            source_errors or "all required sources readable",
            preconditions["passed"],
        ),
        *evaluation["checks"],
    ]
    ready = all(row["passed"] for row in checks)
    checks.append(_check("v604_manifest_contract_ready_score", 1, int(ready), ready))
    summary = gate_summary(checks)
    contract_failures = [
        {"hard_failure": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    rows = [
        {"row_type": "task_contract", **deepcopy(row)}
        for row in evaluation["document_yaml_parity_rows"]
    ]
    rows.extend({"row_type": "contract_check", **deepcopy(row)} for row in checks)
    rows.extend(
        {"row_type": "artifact_admission", **deepcopy(row)}
        for row in admission["artifact_admissibility_rows"]
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_6898.v604_evidence_admissibility_contract.v1",
        "experiment_id": 6898,
        "run_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}",
        "status": "complete" if ready else "complete_blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root, prior_rows),
        "rows": rows,
        "document_task_rows": evaluation["document_task_rows"],
        "yaml_task_rows": evaluation["yaml_task_rows"],
        "document_yaml_parity_rows": evaluation["document_yaml_parity_rows"],
        "gate_contract_rows": evaluation["gate_contract_rows"],
        "prior_failure_contract_rows": evaluation["prior_failure_contract_rows"],
        "prompt_ending_rows": evaluation["prompt_ending_rows"],
        "branch_root_rows": evaluation["branch_root_rows"],
        "ungated_tail_rows": evaluation["ungated_tail_rows"],
        "adversarial_recheck_rows": admission["adversarial_recheck_rows"],
        "artifact_admissibility_rows": admission["artifact_admissibility_rows"],
        "dependency_taint_rows": admission["dependency_taint_rows"],
        "quarantined_artifact_ids": admission["quarantined_artifact_ids"],
        "task_count": 13,
        "experiment_range": [6898, 6910],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v604_manifest_contract_ready_score": int(ready),
        "v603_admissible_science_source_count": admission["v603_admissible_science_source_count"],
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete_positive_v604_manifest_and_evidence_admissibility_contract"
            if ready
            else BLOCKED_VERDICT
        ),
        "warnings": admission["warnings"],
        "hard_failures": [*contract_failures, *admission["hard_failures"]],
        "transitive_taint": deepcopy(admission["dependency_taint_rows"]),
    }
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute readiness without trusting the advisory score itself."""

    summary = artifact.get("gate_check_summary")
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    base = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("check") != "v604_manifest_contract_ready_score"
    ]
    return bool(base) and all(row.get("passed") is True for row in base)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate fields, principles, scores, terminal shape, and checksum."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not REQUIRED_ARTIFACT_FIELDS.issubset(principles):
        errors.append("field_principles_missing")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    ready = int(_ready_from_artifact(artifact))
    if artifact.get("v604_manifest_contract_ready_score") != ready:
        errors.append("readiness_recomputation_mismatch")
    science_count = sum(
        row.get("science_source") is True and row.get("admissible") is True
        for row in artifact.get("artifact_admissibility_rows", [])
        if isinstance(row, Mapping)
    )
    if artifact.get("v603_admissible_science_source_count") != science_count:
        errors.append("science_source_count_mismatch")
    if ready:
        if artifact.get("status") != "complete" or artifact.get("verdict_class") != "positive":
            errors.append("ready_terminal_shape_mismatch")
    elif (
        artifact.get("status") != "complete_blocked"
        or artifact.get("verdict_class") != "blocked"
        or artifact.get("honest_verdict") != BLOCKED_VERDICT
    ):
        errors.append("blocked_terminal_shape_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write only the requested new receipt after complete serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _date_argument(value: str) -> str:
    """Reject ambiguous execution dates before reading evidence."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Write and validate the receipt at its append-only output path."""

    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_PATH)
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    write_json_atomic(args.output, artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ready": artifact["v604_manifest_contract_ready_score"],
                "admissible_science_sources": artifact["v603_admissible_science_source_count"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path
    raise SystemExit(main())
