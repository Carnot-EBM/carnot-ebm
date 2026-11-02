"""Build the deterministic V603 executable-manifest contract.

The reducer compares the design and active YAML as independent primary files.
It records mismatches instead of repairing either source. This keeps the
advisory audit from becoming a global science gate. See REQ-REPORT-6885.
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
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6885_v603_executable_manifest_branch_contract.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_PATH = Path("scripts/conductor_gates.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")
V602_TERMINAL_PATHS = (
    Path("results/experiment_6874_v602_evidence_substrate_manifest_contract.json"),
    Path("results/experiment_6875_text_anchored_relation_asp_fixture.json"),
)
V603_MILESTONE = "2026.09.603"
EXPECTED_NUMBERS = tuple(range(6885, 6898))
INFERENCE_SUBSTRATE = "deterministic_manifest_contract_no_llm"
RANDOM_SEED = 6885
PROMPT_SECTIONS = ("CONTEXT:", "EXISTING CODE TO READ FIRST:", "TASK:", "CONCRETE STEPS:")
FINAL_SENTENCE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
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
    "rows",
    "document_task_rows",
    "yaml_task_rows",
    "document_yaml_parity_rows",
    "gate_contract_rows",
    "prior_failure_contract_rows",
    "prompt_ending_rows",
    "retired_id_rows",
    "retired_upstream_rows",
    "branch_root_rows",
    "ungated_tail_rows",
    "task_count",
    "experiment_range",
    "random_seed",
    "reproducibility_checksum",
    "v603_manifest_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets future reducers reject incompatible records.",
    "experiment_id": "The fixed identity binds this receipt to Exp6885.",
    "run_date": "The date separates this execution from later roadmap states.",
    "status": "A terminal status proves the reducer finished its evidence inventory.",
    "field_principles": "A reason per field keeps the contract understandable during later audits.",
    "preconditions_checked": "Explicit source checks make missing evidence diagnosable without a rerun.",
    "inference_substrate": "The no-LLM declaration prevents an infrastructure audit from becoming science.",
    "duration_s": "Measured wall time exposes whether the deterministic reducer actually ran.",
    "source_artifact_hashes": "Hashes bind each conclusion to the exact primary bytes read.",
    "rows": "Separate task and check rows prevent omissions from disappearing in totals.",
    "document_task_rows": "Document rows preserve the design independently from executable YAML.",
    "yaml_task_rows": "YAML rows preserve the exact executable queue seen by the conductor.",
    "document_yaml_parity_rows": "Per-position comparisons expose count, order, identity, and milestone drift.",
    "gate_contract_rows": "Exact gate rows prevent a typo or dead edge from blocking a branch.",
    "prior_failure_contract_rows": "Primary verdict checks prevent unchanged failed methods from returning silently.",
    "prompt_ending_rows": "Exact final commands prevent truncated or mutable execution instructions.",
    "retired_id_rows": "Current-ID checks prevent reuse of an excluded experiment identity.",
    "retired_upstream_rows": "Retired-edge checks prevent an executable dependency from becoming permanently dead.",
    "branch_root_rows": "Root rows prove that advisory readiness cannot stop either science branch.",
    "ungated_tail_rows": "Tail rows keep the independent audit and capstone runnable after branch blocks.",
    "task_count": "The fixed count detects the V602 failure mode before science dispatch.",
    "experiment_range": "The inclusive range prevents hidden gaps or reused task numbers.",
    "random_seed": "A fixed identity records that the ordered reduction has no hidden randomness.",
    "reproducibility_checksum": "A content hash detects silent changes while ignoring elapsed time.",
    "v603_manifest_contract_ready_score": "This advisory score is one only for a complete executable contract.",
    "gate_check_summary": "Expected and observed values make every blocked verdict actionable.",
    "verifier_is_oracle": "False prevents this structural reducer from approving a scientific claim.",
    "verdict_class": "The closed class states the evidence boundary without free-form promotion.",
    "honest_verdict": "The terminal prefix lets the conductor classify the completed audit safely.",
}
OWN_SOURCE_PATHS = {
    "module": Path("python/carnot/experiment_6885_v603_executable_manifest_branch_contract.py"),
    "wrapper": Path(
        "scripts/experiments/experiment_6885_v603_executable_manifest_branch_contract.py"
    ),
    "focused_tests": Path(
        "tests/python/test_experiment_6885_v603_executable_manifest_branch_contract.py"
    ),
    "spec": SPEC_PATH,
}
BASE_SOURCE_PATHS = {
    "active_v603_roadmap": ROADMAP_PATH,
    "v603_design": DESIGN_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "conductor_gates": GATE_PATH,
    "prior_failure_validator": PRIOR_VALIDATOR_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "conductor_log": CONDUCTOR_LOG_PATH,
    "known_issues": KNOWN_ISSUES_PATH,
    "v602_exp6874": V602_TERMINAL_PATHS[0],
    "v602_exp6875": V602_TERMINAL_PATHS[1],
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 bytes for a deterministic content identity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 identity in the artifact's explicit format."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash one readable file and preserve absence as null evidence."""

    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one contract check with exact diagnostic values."""

    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every check and expose the first failed expected value."""

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


def _experiment_number(value: Any) -> int | None:
    """Read an experiment number without treating a Boolean as an integer."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"(?:exp|experiment_)?(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _task_id_from_deliverable(number: int, deliverable: str) -> str:
    """Derive the executable task ID encoded by its standard result path."""

    stem = Path(deliverable).stem
    suffix = stem.removeprefix(f"experiment_{number}_").replace("_", "-")
    return f"exp{number}-{suffix}"


def _normalize_gate(value: Mapping[str, Any]) -> JsonDict:
    """Keep the four executable gate fields without changing their spelling."""

    return {
        "upstream": str(value.get("upstream") or ""),
        "artifact_field": str(value.get("artifact_field") or ""),
        "op": str(value.get("op") or ""),
        "value": deepcopy(value.get("value")),
    }


def _parse_condition(text: str) -> tuple[str, Any]:
    """Parse the comparison syntax used by the V603 design gate table."""

    match = re.fullmatch(r"\s*(==|>=)\s*(.+?)\s*", text)
    if not match:
        return text.strip(), None
    return match.group(1), yaml.safe_load(match.group(2))


def parse_design(text: str) -> JsonDict:
    """Parse V603 task and gate rows independently from executable YAML."""

    milestone_match = re.search(r"^\*\*Milestone:\*\*\s*`([^`]+)`", text, re.MULTILINE)
    if not milestone_match:
        raise ValueError("design milestone is required")
    sections = re.finditer(
        r"^### Exp(?P<number>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^### Exp\d+:|^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    tasks: list[JsonDict] = []
    for match in sections:
        number = int(match.group("number"))
        if number not in EXPECTED_NUMBERS:
            continue
        deliverable_match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", match.group("body"))
        if not deliverable_match:
            raise ValueError(f"Exp{number} design section is missing deliverable")
        deliverable = deliverable_match.group(1)
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
        op, expected = _parse_condition(match.group("condition"))
        upstream_id = by_number.get(upstream, {}).get("task_id", f"exp{upstream}-missing")
        by_number[downstream]["gates"].append(
            {
                "upstream": upstream_id,
                "artifact_field": match.group("field"),
                "op": op,
                "value": expected,
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
    """Parse the active mapping while preserving malformed field values."""

    if not isinstance(document, Mapping):
        raise ValueError("roadmap mapping is required")
    values = document.get("tasks")
    if not isinstance(values, list):
        raise ValueError("roadmap tasks list is required")
    rows: list[JsonDict] = []
    for order, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise ValueError(f"malformed roadmap task at order {order}")
        gates_value = value.get("gated_on", []) or []
        priors_value = value.get("prior_failures", []) or []
        if not isinstance(gates_value, list) or not isinstance(priors_value, list):
            raise ValueError(f"malformed task lists at order {order}")
        if any(not isinstance(gate, Mapping) for gate in gates_value):
            raise ValueError(f"malformed gate at order {order}")
        task_id = str(value.get("id") or "")
        prompt = str(value.get("prompt") or "")
        rows.append(
            {
                "order": order,
                "number": _experiment_number(task_id),
                "task_id": task_id,
                "title": str(value.get("title") or ""),
                "milestone": str(value.get("milestone") or ""),
                "deliverable": str(value.get("deliverable") or ""),
                "gates": [_normalize_gate(gate) for gate in gates_value],
                "prior_failures": deepcopy(priors_value),
                "prompt": prompt,
                "required_fields": _required_fields(prompt),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def schema_gate_operators(source: str) -> set[str]:
    """Read allowed gate operators from the schema's Literal declaration."""

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
    """Read branches implemented by the conductor's current operator evaluator."""

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
            for item in ast.walk(comparator):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    values.add(item.value)
    if not values:
        raise ValueError("_eval_op has no operator branches")
    return values


def retired_experiment_ids(manifest: Mapping[str, Any]) -> set[int]:
    """Return active retired IDs after append-only un-retirement corrections."""

    retired: set[int] = set()
    unretired: set[int] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        values = manifest.get(section, []) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            candidates = [value.get("experiment_id"), *value.get("experiment_ids", [])]
            reversals = value.get("un_retired_experiment_ids", [])
            retired.update(
                number for number in map(_experiment_number, candidates) if number is not None
            )
            unretired.update(
                number for number in map(_experiment_number, reversals) if number is not None
            )
    return retired - unretired


def _public_yaml_row(row: Mapping[str, Any]) -> JsonDict:
    """Remove full prompt text while retaining its auditable content identity."""

    return {
        "order": row.get("order"),
        "number": row.get("number"),
        "task_id": row.get("task_id"),
        "title": row.get("title"),
        "milestone": row.get("milestone"),
        "deliverable": row.get("deliverable"),
        "gates": deepcopy(row.get("gates", [])),
        "prior_failure_count": len(row.get("prior_failures", [])),
        "prompt_sha256": sha256_bytes(str(row.get("prompt") or "").encode("utf-8")),
    }


def _task_shape(row: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep the identity fields needed for one parity diagnostic."""

    if row is None:
        return None
    return {
        "order": row.get("order"),
        "number": row.get("number"),
        "task_id": row.get("task_id"),
        "milestone": row.get("milestone"),
        "deliverable": row.get("deliverable"),
    }


def _parity_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare task contracts by conductor position so swaps cannot disappear."""

    rows: list[JsonDict] = []
    for index in range(max(len(document_tasks), len(yaml_tasks))):
        document = document_tasks[index] if index < len(document_tasks) else None
        executable = yaml_tasks[index] if index < len(yaml_tasks) else None
        comparisons = {
            "conductor_order": bool(
                document and executable and document.get("number") == executable.get("number")
            ),
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
                "presence": "both"
                if document and executable
                else "document_only"
                if document
                else "yaml_only",
                "document": _task_shape(document),
                "yaml": _task_shape(executable),
                "comparisons": comparisons,
                "passed": bool(document and executable and all(comparisons.values())),
            }
        )
    return rows


def _prompt_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Check required sections and the exact final two non-empty lines."""

    rows: list[JsonDict] = []
    for index in range(max(len(document_tasks), len(yaml_tasks))):
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
        observed_command = lines[-2] if len(lines) >= 2 else None
        observed_sentence = lines[-1] if lines else None
        sections = {section: section in prompt for section in PROMPT_SECTIONS}
        rows.append(
            {
                "order": index + 1,
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
    retired_ids: set[int],
) -> list[JsonDict]:
    """Validate design parity and every executable gate component."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    yaml_by_id = {row.get("task_id"): row for row in yaml_tasks}
    rows: list[JsonDict] = []
    for number in sorted(
        set(document_by_number) | set(yaml_by_number), key=lambda item: item or -1
    ):
        expected_gates = list(document_by_number.get(number, {}).get("gates", []))
        observed_gates = list(yaml_by_number.get(number, {}).get("gates", []))
        for index in range(max(len(expected_gates), len(observed_gates))):
            expected = expected_gates[index] if index < len(expected_gates) else None
            observed = observed_gates[index] if index < len(observed_gates) else None
            upstream = yaml_by_id.get(observed.get("upstream")) if observed else None
            upstream_number = _experiment_number(observed.get("upstream")) if observed else None
            field = str(observed.get("artifact_field") or "") if observed else ""
            checks = {
                "matches_document": expected == observed,
                "operator_valid": bool(observed and observed.get("op") in valid_operators),
                "upstream_present": upstream is not None,
                "upstream_earlier": bool(
                    upstream
                    and yaml_by_number.get(number)
                    and upstream["order"] < yaml_by_number[number]["order"]
                ),
                "producer_field_declared": bool(
                    upstream and field in upstream.get("required_fields", set())
                ),
                "upstream_not_retired": upstream_number not in retired_ids,
                "manifest_not_consumed": upstream_number != 6885,
            }
            rows.append(
                {
                    "downstream_number": number,
                    "downstream_task_id": yaml_by_number.get(
                        number, document_by_number.get(number, {})
                    ).get("task_id"),
                    "gate_index": index,
                    "expected": deepcopy(expected),
                    "observed": deepcopy(observed),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def _prior_rows(
    yaml_tasks: Sequence[Mapping[str, Any]],
    prior_artifacts: Mapping[str, Mapping[str, Any]],
    retired_ids: set[int],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check all four prior-failure fields against exact primary artifacts."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    warnings: list[JsonDict] = []
    for task in yaml_tasks:
        for index, value in enumerate(task.get("prior_failures", [])):
            prior = dict(value) if isinstance(value, Mapping) else {}
            prior_id = prior.get("experiment_id")
            primary = prior_artifacts.get(str(prior_id))
            primary_verdict = primary.get("honest_verdict") if primary else None
            failed: list[str] = []
            if not isinstance(prior_id, str) or _experiment_number(prior_id) is None:
                failed.append("experiment_id")
            if "verdict" not in prior or primary_verdict != prior.get("verdict"):
                failed.append("verdict")
            if (
                not isinstance(prior.get("addressed_by"), str)
                or not prior.get("addressed_by", "").strip()
            ):
                failed.append("addressed_by")
            if prior.get("retire_if_same_verdict") is not True:
                failed.append("retire_if_same_verdict")
            missing = [field for field in required if field not in prior]
            failed = list(dict.fromkeys([*missing, *failed]))
            number = _experiment_number(prior_id)
            if number in retired_ids:
                warnings.append(
                    {
                        "warning": "retired_prior_failure_reference_allowed",
                        "task_id": task.get("task_id"),
                        "prior_experiment_id": number,
                    }
                )
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "prior_index": index,
                    "experiment_id": prior_id,
                    "declared_verdict": prior.get("verdict"),
                    "primary_verdict": primary_verdict,
                    "primary_artifact_present": primary is not None,
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "failed_subfields": failed,
                    "passed": not failed,
                }
            )
    return rows, warnings


def _retirement_rows(
    document_tasks: Sequence[Mapping[str, Any]],
    yaml_tasks: Sequence[Mapping[str, Any]],
    retired_ids: set[int],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Keep current task IDs separate from structured upstream references."""

    id_rows = [
        {
            "source": source,
            "task_id": row.get("task_id"),
            "number": row.get("number"),
            "retired": row.get("number") in retired_ids,
            "passed": row.get("number") not in retired_ids,
        }
        for source, tasks in (("document", document_tasks), ("yaml", yaml_tasks))
        for row in tasks
    ]
    upstream_rows = [
        {
            "source": source,
            "downstream_task_id": task.get("task_id"),
            "upstream_task_id": gate.get("upstream"),
            "upstream_number": _experiment_number(gate.get("upstream")),
            "retired": _experiment_number(gate.get("upstream")) in retired_ids,
            "passed": _experiment_number(gate.get("upstream")) not in retired_ids,
        }
        for source, tasks in (("document", document_tasks), ("yaml", yaml_tasks))
        for task in tasks
        for gate in task.get("gates", [])
    ]
    return id_rows, upstream_rows


def _branch_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Prove both science roots and both terminal tasks are present and ungated."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    document_consumers = [
        task.get("task_id")
        for task in document_tasks
        if any(_experiment_number(gate.get("upstream")) == 6885 for gate in task.get("gates", []))
    ]
    yaml_consumers = [
        task.get("task_id")
        for task in yaml_tasks
        if any(_experiment_number(gate.get("upstream")) == 6885 for gate in task.get("gates", []))
    ]
    roots = [
        {
            "kind": "advisory_manifest_fanout",
            "number": 6885,
            "expected_consumers": [],
            "document_consumers": document_consumers,
            "yaml_consumers": yaml_consumers,
            "passed": not document_consumers and not yaml_consumers,
        }
    ]
    for number in (6886, 6892):
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        roots.append(
            {
                "kind": "independent_science_root",
                "number": number,
                "task_id": executable.get("task_id") if executable else None,
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
    tails: list[JsonDict] = []
    for number in (6896, 6897):
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        tails.append(
            {
                "number": number,
                "task_id": executable.get("task_id") if executable else None,
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
    return roots, tails


def evaluate_contract(
    design_text: str,
    roadmap: Mapping[str, Any],
    exclusion: Mapping[str, Any],
    prior_artifacts: Mapping[str, Mapping[str, Any]],
    schema_source: str,
    gate_source: str,
) -> JsonDict:
    """Evaluate the full V603 contract without reading or writing files."""

    document = parse_design(design_text)
    executable = parse_roadmap(roadmap)
    document_tasks = document["tasks"]
    yaml_tasks = executable["tasks"]
    retired_ids = retired_experiment_ids(exclusion)
    valid_operators = schema_gate_operators(schema_source) & conductor_gate_operators(gate_source)
    parity_rows = _parity_rows(document_tasks, yaml_tasks)
    prompt_rows = _prompt_rows(document_tasks, yaml_tasks)
    gate_rows = _gate_rows(document_tasks, yaml_tasks, valid_operators, retired_ids)
    prior_rows, warnings = _prior_rows(yaml_tasks, prior_artifacts, retired_ids)
    retired_id_rows, retired_upstream_rows = _retirement_rows(
        document_tasks, yaml_tasks, retired_ids
    )
    root_rows, tail_rows = _branch_rows(document_tasks, yaml_tasks)
    document_numbers = [row["number"] for row in document_tasks]
    yaml_numbers = [row["number"] for row in yaml_tasks]
    document_deliverables = [row["deliverable"] for row in document_tasks]
    yaml_deliverables = [row["deliverable"] for row in yaml_tasks]
    document_ok = (
        document["milestone"] == V603_MILESTONE
        and document_numbers == list(EXPECTED_NUMBERS)
        and len(set(document_deliverables)) == 13
        and all(
            path.startswith("results/") and path.endswith(".json") for path in document_deliverables
        )
    )
    yaml_ok = (
        executable["milestone"] == V603_MILESTONE
        and yaml_numbers == list(EXPECTED_NUMBERS)
        and len({row["task_id"] for row in yaml_tasks}) == 13
        and len(set(yaml_deliverables)) == 13
        and all(
            path.startswith("results/") and path.endswith(".json") for path in yaml_deliverables
        )
        and all(row["milestone"] == V603_MILESTONE for row in yaml_tasks)
    )
    checks = [
        _check(
            "document_primary_contract",
            {"milestone": V603_MILESTONE, "task_count": 13, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": document["milestone"],
                "task_count": len(document_tasks),
                "numbers": document_numbers,
            },
            document_ok,
        ),
        _check(
            "document_yaml_task_contract",
            {"task_count": 13, "all_task_fields_equal": True},
            {
                "document_task_count": len(document_tasks),
                "yaml_task_count": len(yaml_tasks),
                "all_task_fields_equal": all(row["passed"] for row in parity_rows),
            },
            len(parity_rows) == 13 and all(row["passed"] for row in parity_rows),
        ),
        _check(
            "yaml_executable_contract",
            {"milestone": V603_MILESTONE, "task_count": 13, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": executable["milestone"],
                "task_count": len(yaml_tasks),
                "numbers": yaml_numbers,
            },
            yaml_ok,
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
            "current_ids_not_retired",
            True,
            all(row["passed"] for row in retired_id_rows),
            all(row["passed"] for row in retired_id_rows),
        ),
        _check(
            "gate_upstreams_not_retired",
            True,
            all(row["passed"] for row in retired_upstream_rows),
            all(row["passed"] for row in retired_upstream_rows),
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
        "document_yaml_parity_rows": parity_rows,
        "gate_contract_rows": gate_rows,
        "prior_failure_contract_rows": prior_rows,
        "prompt_ending_rows": prompt_rows,
        "retired_id_rows": retired_id_rows,
        "retired_upstream_rows": retired_upstream_rows,
        "branch_root_rows": root_rows,
        "ungated_tail_rows": tail_rows,
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


def _read_json_mapping(path: Path) -> tuple[JsonDict | None, str | None]:
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
    """Map an exact declared prior task ID to its primary result path."""

    match = re.fullmatch(r"exp(\d+)-(.+)", str(experiment_id or ""), re.IGNORECASE)
    if not match:
        return None
    return root / "results" / f"experiment_{match.group(1)}_{match.group(2).replace('-', '_')}.json"


def load_prior_artifacts(
    root: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Load exact primary artifacts for every declared prior-failure row."""

    artifacts: dict[str, JsonDict] = {}
    rows: list[JsonDict] = []
    for task in tasks:
        for prior in task.get("prior_failures", []):
            prior_id = prior.get("experiment_id") if isinstance(prior, Mapping) else None
            path = _prior_artifact_path(root, prior_id)
            payload, error = _read_json_mapping(path) if path else (None, "invalid_id")
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
        "retired_id_rows": [],
        "retired_upstream_rows": [],
        "branch_root_rows": [],
        "ungated_tail_rows": [],
        "warnings": [],
        "hard_failures": [{"hard_failure": "contract_evaluation", "observed": error}],
    }


def _source_hashes(root: Path, prior_rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Bind every required and implementation source to exact bytes."""

    rows = {
        name: {"path": path.as_posix(), "sha256": sha256_path(root / path)}
        for name, path in {**BASE_SOURCE_PATHS, **OWN_SOURCE_PATHS}.items()
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


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Build a complete positive or blocked V603 manifest receipt."""

    started = time.perf_counter()
    source_errors: list[JsonDict] = []
    roadmap, roadmap_error = read_yaml_mapping(root / ROADMAP_PATH)
    exclusion, exclusion_error = read_yaml_mapping(root / EXCLUSION_PATH)
    text_sources: dict[str, str] = {}
    for name, path in (
        ("v603_design", DESIGN_PATH),
        ("roadmap_schema", SCHEMA_PATH),
        ("conductor_gates", GATE_PATH),
        ("prior_failure_validator", PRIOR_VALIDATOR_PATH),
        ("roadmap_gate_audit", GATE_AUDIT_PATH),
    ):
        try:
            text_sources[name] = (root / path).read_text(encoding="utf-8")
            if not text_sources[name]:
                source_errors.append({"source": name, "error": "empty"})
        except (OSError, UnicodeError) as exc:
            source_errors.append({"source": name, "error": f"unreadable:{type(exc).__name__}"})
            text_sources[name] = ""
    for name, error in (
        ("active_v603_roadmap", roadmap_error),
        ("exclusion_manifest", exclusion_error),
    ):
        if error:
            source_errors.append({"source": name, "error": error})
    for path in V602_TERMINAL_PATHS:
        payload, error = _read_json_mapping(root / path)
        terminal = bool(
            payload
            and payload.get("honest_verdict")
            and str(payload.get("status") or "").lower() not in {"", "running", "bootstrap"}
        )
        if error or not terminal:
            source_errors.append(
                {
                    "source": path.as_posix(),
                    "error": error or "terminal_artifact_required",
                }
            )
    raw_tasks = roadmap.get("tasks", []) if roadmap else []
    prior_artifacts, prior_source_rows = load_prior_artifacts(
        root, [task for task in raw_tasks if isinstance(task, Mapping)]
    )
    source_errors.extend(
        {"source": row.get("path") or row.get("experiment_id"), "error": row["error"]}
        for row in prior_source_rows
        if row.get("error")
    )
    try:
        evaluation = evaluate_contract(
            text_sources["v603_design"],
            roadmap or {},
            exclusion or {},
            prior_artifacts,
            text_sources["roadmap_schema"],
            text_sources["conductor_gates"],
        )
    except (ValueError, SyntaxError, yaml.YAMLError) as exc:
        source_errors.append({"source": "contract_evaluation", "error": str(exc)})
        evaluation = _empty_evaluation(str(exc))
    preconditions = gate_summary(
        [
            _check(
                "required_source_readability",
                [],
                source_errors,
                not source_errors,
            )
        ]
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
    checks.append(_check("v603_manifest_contract_ready_score", 1, int(ready), ready))
    summary = gate_summary(checks)
    hard_failures = [
        {"hard_failure": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    rows = [
        {
            "row_type": "task",
            "order": row["order"],
            "task_contract": deepcopy(row),
            "passed": row["passed"],
        }
        for row in evaluation["document_yaml_parity_rows"]
    ]
    rows.extend({"row_type": "contract_check", **deepcopy(row)} for row in checks)
    artifact: JsonDict = {
        "schema": "carnot.experiment_6885.v603_executable_manifest_branch_contract.v1",
        "experiment_id": 6885,
        "run_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}",
        "status": "complete" if ready else "complete_blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root, prior_source_rows),
        "rows": rows,
        "document_task_rows": evaluation["document_task_rows"],
        "yaml_task_rows": evaluation["yaml_task_rows"],
        "document_yaml_parity_rows": evaluation["document_yaml_parity_rows"],
        "gate_contract_rows": evaluation["gate_contract_rows"],
        "prior_failure_contract_rows": evaluation["prior_failure_contract_rows"],
        "prompt_ending_rows": evaluation["prompt_ending_rows"],
        "retired_id_rows": evaluation["retired_id_rows"],
        "retired_upstream_rows": evaluation["retired_upstream_rows"],
        "branch_root_rows": evaluation["branch_root_rows"],
        "ungated_tail_rows": evaluation["ungated_tail_rows"],
        "task_count": 13,
        "experiment_range": [6885, 6897],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v603_manifest_contract_ready_score": int(ready),
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete_positive_v603_executable_manifest_branch_contract_ready"
            if ready
            else "complete_blocked_v603_executable_manifest_branch_contract"
        ),
        "warnings": evaluation["warnings"],
        "hard_failures": hard_failures,
    }
    for row in artifact["gate_contract_rows"]:
        gate = row.get("expected") or row.get("observed") or {}
        field = gate.get("artifact_field") if isinstance(gate, Mapping) else None
        if field:
            artifact["field_principles"][str(field)] = (
                "This exact producer field spelling controls one V603 branch gate."
            )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute readiness without trusting the advisory headline field."""

    summary = artifact.get("gate_check_summary")
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    base = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("check") != "v603_manifest_contract_ready_score"
    ]
    return bool(base) and all(row.get("passed") is True for row in base)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, principles, reduction, and checksum."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    gate_fields = {
        str(gate.get("artifact_field"))
        for row in artifact.get("gate_contract_rows", [])
        if isinstance(row, Mapping)
        for gate in (row.get("expected"), row.get("observed"))
        if isinstance(gate, Mapping) and gate.get("artifact_field")
    }
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not (REQUIRED_ARTIFACT_FIELDS | gate_fields).issubset(
        principles
    ):
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
    if artifact.get("v603_manifest_contract_ready_score") != ready:
        errors.append("readiness_recomputation_mismatch")
    if ready:
        if artifact.get("status") != "complete" or artifact.get("verdict_class") != "positive":
            errors.append("ready_terminal_shape_mismatch")
    elif (
        artifact.get("status") != "complete_blocked"
        or artifact.get("verdict_class") != "blocked"
        or artifact.get("honest_verdict")
        != "complete_blocked_v603_executable_manifest_branch_contract"
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
    """Write and validate the receipt at its requested append-only path."""

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
                "ready": artifact["v603_manifest_contract_ready_score"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path
    raise SystemExit(main())
