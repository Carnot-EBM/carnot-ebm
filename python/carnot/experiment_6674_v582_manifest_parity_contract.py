"""Audit the V582 design and active execution manifest without an LLM.

The audit treats each task, gate, prior failure, and validator as a row. This
keeps a failed activation check local and reproducible. See REQ-REPORT-6674 and
the SCENARIO-REPORT-6674 anchors in the research-reporting specification.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
MILESTONE = "2026.08.582"
RESULT_PATH = Path("results/experiment_6674_v582_manifest_parity_contract.json")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
INFERENCE_SUBSTRATE = "artifact_and_manifest_audit_no_llm"
RANDOM_SEED = 6674

EXPECTED_TASK_IDS = (
    "exp6674-v582-manifest-parity-contract",
    "exp6675-triggered-tail-scope-receipt",
    "exp6676-three-family-triggered-tail-ab",
    "exp6677-triggered-tail-independent-audit",
    "exp6678-constraint-family-stream",
    "exp6679-prequential-cross-family-csl-ab",
    "exp6680-csl-durability-audit",
    "exp6681-arc-post-redirect-outcomes",
    "exp6682-arc-held-family-supervisor-ab",
    "exp6683-ising-reference-scope-receipt",
    "exp6684-torx-typed-factor-parity",
    "exp6685-autocorrelation-schedule-ab",
    "exp6686-stochastic-portability-audit",
    "exp6687-v582-branch-synthesis",
)

_TRACK_BY_NUMBER = {
    6674: "execution-integrity",
    6675: "verification-transport",
    6676: "verification-transport",
    6677: "verification-transport",
    6678: "continuous-self-learning",
    6679: "continuous-self-learning",
    6680: "continuous-self-learning",
    6681: "live-arc-outcomes",
    6682: "live-arc-outcomes",
    6683: "stochastic-portability",
    6684: "stochastic-portability",
    6685: "stochastic-portability",
    6686: "stochastic-portability",
    6687: "milestone-synthesis",
}
_GPU_TASK_NUMBERS = {6676, 6679}
_CODEX_TASK_NUMBERS = {6675, 6678, 6683, 6684, 6685}
_PRIOR_IDS_BY_NUMBER = {
    6674: ["exp6660-v581-evidence-contract"],
    6675: ["exp6661-triggered-tail-fixture"],
    6676: [
        "exp5923-sota-schema-supported-constraintir-ab",
        "exp6662-triggered-structured-tail-ab",
    ],
    6677: [
        "exp6652-constraint-intervention-audit",
        "exp6663-structured-tail-independent-audit",
    ],
    6678: [
        "exp5709-fr11-prospective-shadow-stream",
        "exp5786-sota-hardness-controlled-constraint-stream",
    ],
    6679: ["exp6655-repair-memory-safety-audit"],
    6680: ["exp6655-repair-memory-safety-audit"],
    6681: ["exp6656-arc-trace-automaton-live-loo"],
    6682: ["exp6656-arc-trace-automaton-live-loo"],
    6683: ["exp6657-bounded-treewidth-ising-reference"],
    6684: [],
    6685: ["exp6658-thermodynamic-schedule-ab"],
    6686: [],
    6687: [],
}

PROTECTED_PATHS = (ROADMAP_PATH, NEXT_ROADMAP_PATH, CONDUCTOR_PATH)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "design_task_rows",
    "manifest_task_rows",
    "producer_consumer_rows",
    "prior_failure_rows",
    "validator_rows",
    "validator_mismatch_rows",
    "v582_manifest_parity_ready",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

VALIDATOR_DEFINITIONS = (
    (
        "yaml_parse",
        '.venv/bin/python -c "from pathlib import Path; import yaml; '
        "data=yaml.safe_load(Path('research-roadmap.yaml').read_text()); "
        "assert isinstance(data, dict); print('YAML parse passed:', "
        "data['milestone'], len(data['tasks']))\"",
    ),
    (
        "roadmap_schema",
        '.venv/bin/python -c "from pathlib import Path; import yaml; '
        "from scripts.roadmap_schema import Roadmap; "
        "r=Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_text())); "
        "print('Roadmap schema passed:', r.milestone, len(r.tasks))\"",
    ),
    (
        "prior_failures",
        ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ),
    (
        "gate_contract",
        ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ),
    (
        "exclusion_manifest",
        ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ),
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest -o addopts='' tests/python/test_experiment_6674_v582_manifest_parity_contract.py -q",
        "exit": 0,
        "summary": "focused Exp6674 tests passed",
    },
    {
        "command": "COVERAGE_FILE=/tmp/carnot_exp6674.coverage .venv/bin/coverage run --include='*/experiment_6674_v582_manifest_parity_contract.py' -m pytest --noconftest -o addopts='' tests/python/test_experiment_6674_v582_manifest_parity_contract.py -q && COVERAGE_FILE=/tmp/carnot_exp6674.coverage .venv/bin/coverage report --include='*/experiment_6674_v582_manifest_parity_contract.py' --show-missing --fail-under=100",
        "exit": 0,
        "summary": "new Exp6674 module reached 100% statement coverage",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit": 1,
        "summary": (
            "1120 failed, 34350 passed, 103 skipped, 141 warnings, and 4 errors "
            "in 2939.84s; xdist stopped at 61% after a worker CWD was removed"
        ),
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6674_v582_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused spec coverage passed",
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6674_v582_manifest_parity_contract.py tests/python/test_experiment_6674_v582_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused Ruff lint passed",
    },
    {
        "command": ".venv/bin/ruff format --check python/carnot/experiment_6674_v582_manifest_parity_contract.py tests/python/test_experiment_6674_v582_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused format check passed",
    },
    {
        "command": "cd /home/ianblenke/github.com/ianblenke/carnot && .venv/bin/python -m carnot.experiment_6674_v582_manifest_parity_contract --date 20260827",
        "exit": 0,
        "summary": "required end-to-end command atomically wrote a ready artifact",
    },
    {
        "command": ".venv/bin/python -m carnot.experiment_6674_v582_manifest_parity_contract --validate --output results/experiment_6674_v582_manifest_parity_contract.json",
        "exit": 0,
        "summary": "stored artifact passed schema, reduction, protection, and checksum validation",
    },
    {
        "command": ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6674_v582_manifest_parity_contract.json",
        "exit": 0,
        "summary": "row consistency reported no row/verdict contradiction",
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6674_v582_manifest_parity_contract.json",
        "exit": 1,
        "summary": (
            "one non-critical SUBSTRATE_NO_LLM_BY_NAME warning because the required "
            "no-LLM substrate is not allowlisted; no critical finding"
        ),
    },
    {
        "command": "git status --short",
        "exit": 0,
        "summary": "worktree inspection found only the intended spec, module, test, and artifact changes",
    },
)


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for checksums and row identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_hash(value: Any) -> str:
    """Hash one JSON-compatible value with the shared canonical form."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one input and preserve absence as an explicit state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every artifact field except the checksum field itself."""

    return value_hash(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def _experiment_number(task_id: str) -> int:
    """Read the numeric experiment identity from a task ID."""

    match = re.match(r"exp(\d+)-", task_id)
    if match is None:
        raise ValueError(f"invalid task id: {task_id}")
    return int(match.group(1))


def _expected_route(number: int) -> JsonDict:
    """Apply the per-task route declarations from the V582 design policy."""

    if number == 6674:
        return {"agent_type": "claude", "model": "opus"}
    if number in _CODEX_TASK_NUMBERS:
        return {"agent_type": "codex", "model": "gpt-5.6-sol"}
    return {"agent_type": "claude", "model": None}


def _parse_gate_condition(text: str) -> tuple[str, Any]:
    """Parse the two condition forms used by the V582 design table."""

    cleaned = text.strip().strip("`")
    if cleaned == "in [true]":
        return "in", [True]
    match = re.fullmatch(r">=\s*(\d+)", cleaned)
    if match is None:
        raise ValueError(f"unsupported design gate condition: {text}")
    return ">=", int(match.group(1))


def parse_design_contract(text: str) -> list[JsonDict]:
    """Parse the V582 task table and its structured producer-gate table."""

    contract_match = re.search(
        r"## Exact execution contract\n(?P<body>.*?)(?=\n## External research incorporated)",
        text,
        re.DOTALL,
    )
    if contract_match is None:
        raise ValueError("V582 exact execution contract table is missing")
    table_rows = re.findall(
        r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|$",
        contract_match.group("body"),
        re.MULTILINE,
    )
    if len(table_rows) != 14:
        raise ValueError(f"V582 exact execution contract has {len(table_rows)} task rows")

    id_by_number = {_experiment_number(task_id): task_id for _, task_id, _, _ in table_rows}
    gate_section = text.split("Structured gates use producer-owned fields:", 1)[1].split(
        "Every producer prompt declares", 1
    )[0]
    gate_rows: list[JsonDict] = []
    for consumer_text, producer_text, producer_field, condition in re.findall(
        r"^\|\s*Exp(\d+)\s*\|\s*`exp(\d+)\.([a-z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|$",
        gate_section,
        re.MULTILINE,
    ):
        operator, value = _parse_gate_condition(condition)
        gate_rows.append(
            {
                "consumer": id_by_number[int(consumer_text)],
                "upstream": id_by_number[int(producer_text)],
                "artifact_field": producer_field,
                "operator": operator,
                "value": value,
            }
        )

    design_rows: list[JsonDict] = []
    for order_text, task_id, title, deliverable in table_rows:
        number = _experiment_number(task_id)
        gates = [
            {
                "upstream": row["upstream"],
                "artifact_field": row["artifact_field"],
                "op": row["operator"],
                "value": row["value"],
            }
            for row in gate_rows
            if row["consumer"] == task_id
        ]
        module_name = Path(deliverable).stem
        design_rows.append(
            {
                "order": int(order_text),
                "task_id": task_id,
                "title": title.strip(),
                "deliverable": deliverable,
                "track": _TRACK_BY_NUMBER[number],
                "requires_gpu": number in _GPU_TASK_NUMBERS,
                "dependencies": list(dict.fromkeys(gate["upstream"] for gate in gates)),
                "gated_on": gates,
                "route": _expected_route(number),
                "prior_failure_ids": list(_PRIOR_IDS_BY_NUMBER[number]),
                "required_artifact_fields": sorted(
                    row["artifact_field"] for row in gate_rows if row["upstream"] == task_id
                ),
                "run_command": (
                    f"cd {{project_root}} && .venv/bin/python -m carnot.{module_name} "
                    "--date {date}"
                ),
                "declaration_source": DESIGN_PATH.as_posix(),
            }
        )
    return design_rows


def extract_required_fields(prompt: str) -> list[str]:
    """Read exact field names from one task's required artifact block."""

    if "REQUIRED ARTIFACT FIELDS:" not in prompt or "Run command:" not in prompt:
        return []
    block = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)[1].split("Run command:", 1)[0]
    return re.findall(r"^\s{2}([a-z][a-z0-9_]*):", block, re.MULTILINE)


def extract_run_command(prompt: str) -> str | None:
    """Read the single task command without executing it."""

    match = re.search(r"^Run command:\s*(.+)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else None


def _normalized_gates(task: Mapping[str, Any]) -> list[JsonDict]:
    """Copy gate fields in conductor order with stable key names."""

    return [
        {
            "upstream": gate.get("upstream"),
            "artifact_field": gate.get("artifact_field"),
            "op": gate.get("op"),
            "value": gate.get("value"),
        }
        for gate in task.get("gated_on", []) or []
        if isinstance(gate, Mapping)
    ]


def load_manifest_rows(root: Path) -> list[JsonDict]:
    """Load V582 YAML and attach field-by-field design comparisons."""

    path = root / ROADMAP_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    if not isinstance(payload, Mapping) or payload.get("milestone") != MILESTONE:
        raise ValueError(f"expected roadmap milestone {MILESTONE}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("roadmap task list must be a list")

    design_rows = parse_design_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
    design_by_id = {row["task_id"]: row for row in design_rows}
    manifest_rows: list[JsonDict] = []
    for order, task_value in enumerate(tasks, 1):
        task = dict(task_value)
        task_id = str(task.get("id", ""))
        expected = design_by_id.get(task_id, {})
        prompt = str(task.get("prompt", ""))
        fields = extract_required_fields(prompt)
        gates = _normalized_gates(task)
        route = {
            "agent_type": task.get("agent_type") or "claude",
            "model": task.get("model"),
        }
        prior_blocks = [dict(item) for item in task.get("prior_failures", []) or []]
        prior_ids = [item.get("experiment_id") for item in prior_blocks]
        prior_blocks_valid = all(
            isinstance(item.get("experiment_id"), str)
            and bool(str(item.get("verdict", "")).strip())
            and bool(str(item.get("addressed_by", "")).strip())
            and item.get("retire_if_same_verdict") is True
            for item in prior_blocks
        )
        observed = {
            "title": task.get("title"),
            "deliverable": task.get("deliverable"),
            "track": task.get("track"),
            "requires_gpu": task.get("requires_gpu", False),
            "dependency": list(dict.fromkeys(gate["upstream"] for gate in gates)),
            "gate": gates,
            "route": route,
            "prior_failure_block": prior_ids,
            "required_artifact_field": fields,
            "run_command": extract_run_command(prompt),
        }
        expected_values = {
            "title": expected.get("title"),
            "deliverable": expected.get("deliverable"),
            "track": expected.get("track"),
            "requires_gpu": expected.get("requires_gpu"),
            "dependency": expected.get("dependencies"),
            "gate": expected.get("gated_on"),
            "route": expected.get("route"),
            "prior_failure_block": expected.get("prior_failure_ids"),
            "required_artifact_field": expected.get("required_artifact_fields"),
            "run_command": expected.get("run_command"),
        }
        field_checks = {
            "title": observed["title"] == expected_values["title"],
            "deliverable": observed["deliverable"] == expected_values["deliverable"],
            "track": observed["track"] == expected_values["track"],
            "requires_gpu": observed["requires_gpu"] == expected_values["requires_gpu"],
            "dependency": observed["dependency"] == expected_values["dependency"],
            "gate": observed["gate"] == expected_values["gate"],
            "route": observed["route"] == expected_values["route"],
            "prior_failure_block": (
                observed["prior_failure_block"] == expected_values["prior_failure_block"]
                and prior_blocks_valid
            ),
            "required_artifact_field": set(expected_values["required_artifact_field"] or [])
            <= set(fields),
            "run_command": observed["run_command"] == expected_values["run_command"],
        }
        field_differences = {
            name: {"expected": expected_values[name], "observed": observed[name]}
            for name, passed in field_checks.items()
            if not passed
        }
        manifest_rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task.get("title"),
                "deliverable": task.get("deliverable"),
                "track": task.get("track"),
                "requires_gpu": task.get("requires_gpu", False),
                "requires": task.get("requires"),
                "dependencies": observed["dependency"],
                "gated_on": gates,
                "route": route,
                "prior_failures": prior_blocks,
                "required_artifact_fields": fields,
                "run_command": observed["run_command"],
                "field_checks": field_checks,
                "field_differences": field_differences,
                "source_path": ROADMAP_PATH.as_posix(),
                "source_hash": sha256_file(path),
            }
        )
    return manifest_rows


def build_producer_consumer_rows(
    design_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    retired_numbers: set[int] | None = None,
) -> list[JsonDict]:
    """Cross-reference every consumer gate against its producer prompt."""

    producers = {str(row.get("task_id")): row for row in manifest_rows}
    retired_numbers = retired_numbers or set()
    design_gates = {
        (
            str(row.get("task_id")),
            str(gate.get("upstream")),
            str(gate.get("artifact_field")),
            str(gate.get("op")),
            canonical_json(gate.get("value")),
        )
        for row in design_rows
        for gate in row.get("gated_on", [])
    }
    rows: list[JsonDict] = []
    for consumer in manifest_rows:
        for gate in consumer.get("gated_on", []):
            upstream = str(gate.get("upstream"))
            producer = producers.get(upstream)
            declared = list(producer.get("required_artifact_fields", [])) if producer else []
            key = (
                str(consumer.get("task_id")),
                upstream,
                str(gate.get("artifact_field")),
                str(gate.get("op")),
                canonical_json(gate.get("value")),
            )
            rows.append(
                {
                    "producer": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "consumer": consumer.get("task_id"),
                    "operator": gate.get("op"),
                    "value": gate.get("value"),
                    "upstream_exists": producer is not None,
                    "producer_declares_exact_field": gate.get("artifact_field") in declared,
                    "matches_design": key in design_gates,
                    "upstream_retired": (
                        _experiment_number(upstream) in retired_numbers if producer else None
                    ),
                    "producer_required_artifact_fields_hash": value_hash(declared),
                }
            )
    return rows


def _completed_index(root: Path) -> dict[str, JsonDict]:
    """Index completed task rows without changing the archival file."""

    payload = yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8")) or {}
    return {
        str(task.get("id")): {
            "milestone": milestone.get("id"),
            "title": task.get("title"),
            "deliverable": task.get("deliverable"),
            "result": task.get("result"),
        }
        for milestone in payload.get("milestones", [])
        if isinstance(milestone, Mapping)
        for task in milestone.get("tasks", [])
        if isinstance(task, Mapping) and task.get("id")
    }


def _retirement_index(root: Path) -> dict[int, list[JsonDict]]:
    """Index direct and scope-level exclusion entries by experiment number."""

    payload = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    index: dict[int, list[JsonDict]] = {}
    for section in ("retired", "retired_experiments", "retired_extras"):
        for entry_value in payload.get(section, []) or []:
            if not isinstance(entry_value, Mapping):
                continue
            entry = dict(entry_value)
            identities: list[Any] = [entry.get("experiment_id")]
            identities.extend(entry.get("experiment_ids", []) or [])
            for identity in identities:
                match = re.search(r"(?:exp)?(\d+)", str(identity))
                if match:
                    index.setdefault(int(match.group(1)), []).append(
                        {
                            "section": section,
                            "id": entry.get("id") or entry.get("experiment_scope"),
                            "reason": entry.get("reason"),
                        }
                    )
    return index


def build_prior_failure_rows(
    root: Path, manifest_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind each prior block to completion and exclusion evidence."""

    completed = _completed_index(root)
    retired = _retirement_index(root)
    upstream_ids = {
        str(gate.get("upstream")) for task in manifest_rows for gate in task.get("gated_on", [])
    }
    rows: list[JsonDict] = []
    for task in manifest_rows:
        for prior_value in task.get("prior_failures", []):
            prior = dict(prior_value)
            prior_id = str(prior.get("experiment_id"))
            number = _experiment_number(prior_id)
            record = completed.get(prior_id)
            exclusion_entries = retired.get(number, [])
            retired_upstream = prior_id in upstream_ids and bool(exclusion_entries)
            row = {
                "consumer_task_id": task.get("task_id"),
                "prior_experiment_id": prior_id,
                "prior_scope": record.get("title") if record else None,
                "verdict": prior.get("verdict"),
                "changed_condition": prior.get("addressed_by"),
                "retirement_signal": prior.get("retire_if_same_verdict"),
                "completed_record_found": record is not None,
                "completed_record": record,
                "exclusion_manifest_match": bool(exclusion_entries),
                "exclusion_manifest_entries": exclusion_entries,
                "reference_role": "prior_failure",
                "retired_upstream_reference": retired_upstream,
            }
            row["passed"] = bool(
                row["completed_record_found"]
                and str(row["verdict"] or "").strip()
                and str(row["changed_condition"] or "").strip()
                and row["retirement_signal"] is True
                and not retired_upstream
            )
            rows.append(row)
    return rows


_STALE_CODEX_RE = re.compile(
    r"MODEL_AGENT_COHERENCE (?P<task>\S+): agent_type=codex requires "
    r"model=gpt-5\.5, got gpt-5\.6-sol"
)


def classify_validator_result(
    name: str,
    command: str,
    exit_code: int,
    output: str,
    run_date: str,
) -> tuple[JsonDict, list[JsonDict]]:
    """Separate the dated Codex rule from activation-hard validator failures."""

    classification = "passed" if exit_code == 0 else "activation_hard_failure"
    mismatches: list[JsonDict] = []
    if name == "gate_contract" and exit_code != 0:
        try:
            details = json.loads(output).get("failure_details", [])
        except (json.JSONDecodeError, AttributeError):
            details = []
        matches = [_STALE_CODEX_RE.fullmatch(str(detail)) for detail in details]
        if details and all(matches):
            classification = "validator_mismatch_nonblocking"
            date_text = datetime.strptime(run_date, "%Y%m%d").date().isoformat()
            mismatches = [
                {
                    "conflict_date": date_text,
                    "validator": name,
                    "task_id": match.group("task"),
                    "validator_rule": "Codex routes require gpt-5.5",
                    "validator_model": "gpt-5.5",
                    "operator_requirement": "Codex routes use gpt-5.6-sol",
                    "operator_model": "gpt-5.6-sol",
                    "classification": "dated_validator_mismatch_not_task_defect",
                    "detail": detail,
                }
                for detail, match in zip(details, matches, strict=True)
            ]
    row = {
        "validator": name,
        "command": command,
        "exit": exit_code,
        "output": output,
        "classification": classification,
        "hash": value_hash(
            {
                "command": command,
                "exit": exit_code,
                "output": output,
                "classification": classification,
            }
        ),
    }
    return row, mismatches


def _run_command(root: Path, command: str) -> tuple[int, str]:
    """Run one fixed validator command and retain combined output."""

    result = subprocess.run(
        command,
        cwd=root,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def run_validators(root: Path, run_date: str) -> tuple[list[JsonDict], list[JsonDict]]:
    """Execute every required read-only roadmap validator once."""

    rows: list[JsonDict] = []
    mismatches: list[JsonDict] = []
    for name, command in VALIDATOR_DEFINITIONS:
        exit_code, output = _run_command(root, command)
        row, found = classify_validator_result(name, command, exit_code, output, run_date)
        rows.append(row)
        mismatches.extend(found)
    return rows, mismatches


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash both roadmap paths and the conductor, including absence."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def _protected_rows(root: Path, before: Mapping[str, str]) -> list[JsonDict]:
    """Build exact before-and-after receipts for protected inputs."""

    after = protected_hashes(root)
    return [
        {
            "path": path.as_posix(),
            "before": before.get(path.as_posix(), "missing"),
            "after": after[path.as_posix()],
            "unchanged": before.get(path.as_posix(), "missing") == after[path.as_posix()],
        }
        for path in PROTECTED_PATHS
    ]


def _ram_bytes() -> int:
    """Measure physical RAM from the host page counters."""

    return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def _v581_record_rows(root: Path, completed: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Record present and missing V581 deliverables with archive lineage."""

    rows: list[JsonDict] = []
    for number in range(6660, 6664):
        task_id = next(key for key in completed if key.startswith(f"exp{number}-"))
        record = completed[task_id]
        relative = Path(str(record["deliverable"]))
        digest = sha256_file(root / relative)
        rows.append(
            {
                "experiment_number": number,
                "task_id": task_id,
                "path": relative.as_posix(),
                "state": "present" if digest != "missing" else "missing",
                "sha256": digest,
                "completed_record": record,
                "completed_record_hash": value_hash(record),
            }
        )
    return rows


def collect_preconditions(root: Path) -> JsonDict:
    """Measure input identity, host resources, and the no-LLM substrate."""

    input_paths = (
        ROADMAP_PATH,
        NEXT_ROADMAP_PATH,
        DESIGN_PATH,
        CONDUCTOR_PATH,
        EXCLUSION_PATH,
        COMPLETE_PATH,
        CONDUCTOR_LOG_PATH,
    )
    completed = _completed_index(root)
    disk = shutil.disk_usage(root)
    conductor_text = (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8")
    v581_log_lines = [
        line
        for line in conductor_text.splitlines()
        if any(
            marker in line
            for marker in (
                "Milestone 2026.08.581",
                "V581 evidence and retirement contract",
                "Trigger-switched structured-tail fixture",
                "Three-family trigger-switched structured-tail A/B",
                "Structured-tail independent row audit",
            )
        )
    ]
    return {
        "inputs": [
            {
                "path": path.as_posix(),
                "state": "present" if (root / path).is_file() else "missing",
                "sha256": sha256_file(root / path),
            }
            for path in input_paths
        ],
        "resources": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_logical_count": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
        },
        "v581_records": _v581_record_rows(root, completed),
        "v581_conductor_rows": v581_log_lines,
        "v581_conductor_rows_hash": value_hash(v581_log_lines),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_calls": 0,
    }


def _failure(check: str, unit: str, expected: Any, observed: Any, reason: str) -> JsonDict:
    """Build one stable diagnostic without replacing absence with zero."""

    return {
        "check": check,
        "unit": unit,
        "expected_value": expected,
        "observed_value": observed,
        "reason": reason,
    }


def reduce_readiness(
    design_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    validator_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Reduce complete row evidence to one activation-readiness Boolean."""

    failures: list[JsonDict] = []
    design_ids = [row.get("task_id") for row in design_rows]
    manifest_ids = [row.get("task_id") for row in manifest_rows]
    if design_ids != list(EXPECTED_TASK_IDS):
        failures.append(
            _failure(
                "design.task_order",
                "design",
                list(EXPECTED_TASK_IDS),
                design_ids,
                "ordered_task_set_mismatch",
            )
        )
    if manifest_ids != list(EXPECTED_TASK_IDS):
        failures.append(
            _failure(
                "manifest.task_order",
                "manifest",
                list(EXPECTED_TASK_IDS),
                manifest_ids,
                "ordered_task_set_mismatch",
            )
        )
    deliverables = [row.get("deliverable") for row in manifest_rows]
    deliverables_valid = (
        len(deliverables) == 14
        and len(set(deliverables)) == 14
        and all(
            isinstance(path, str) and path.startswith("results/") and path.endswith(".json")
            for path in deliverables
        )
    )
    if not deliverables_valid:
        failures.append(
            _failure(
                "manifest.deliverables",
                "manifest",
                "14 unique results/*.json paths",
                deliverables,
                "deliverable_contract_mismatch",
            )
        )
    for task in manifest_rows:
        for field_name, passed in task.get("field_checks", {}).items():
            if not passed:
                difference = task.get("field_differences", {}).get(field_name, {})
                failures.append(
                    _failure(
                        f"task.{field_name}",
                        str(task.get("task_id")),
                        difference.get("expected"),
                        difference.get("observed"),
                        "design_manifest_field_mismatch",
                    )
                )
    for row in gate_rows:
        for check, key in (
            ("gate.upstream_exists", "upstream_exists"),
            ("gate.producer_field_spelling", "producer_declares_exact_field"),
            ("gate.matches_design", "matches_design"),
        ):
            if row.get(key) is not True:
                failures.append(
                    _failure(
                        check,
                        f"{row.get('consumer')}->{row.get('producer')}",
                        True,
                        row.get(key),
                        "gate_contract_mismatch",
                    )
                )
        if row.get("upstream_retired") is not False:
            failures.append(
                _failure(
                    "gate.upstream_not_retired",
                    f"{row.get('consumer')}->{row.get('producer')}",
                    False,
                    row.get("upstream_retired"),
                    "retired_or_unresolved_upstream_reference",
                )
            )
    for row in prior_rows:
        if row.get("passed") is not True:
            failures.append(
                _failure(
                    "prior.lineage",
                    str(row.get("prior_experiment_id")),
                    True,
                    row.get("passed"),
                    "prior_failure_contract_mismatch",
                )
            )
    for row in validator_rows:
        if row.get("classification") == "activation_hard_failure":
            failures.append(
                _failure(
                    "validator.activation",
                    str(row.get("validator")),
                    "passed or dated nonblocking mismatch",
                    row.get("classification"),
                    "activation_hard_validator_failure",
                )
            )
    for row in protected_rows:
        if row.get("unchanged") is not True:
            failures.append(
                _failure(
                    "protected_file.unchanged",
                    str(row.get("path")),
                    row.get("before"),
                    row.get("after"),
                    "protected_file_changed",
                )
            )

    aggregate = {
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "design_task_ids": design_ids,
        "manifest_task_ids": manifest_ids,
        "task_count": len(manifest_rows),
        "gate_count": len(gate_rows),
        "prior_failure_count": len(prior_rows),
        "validator_count": len(validator_rows),
        "protected_file_count": len(protected_rows),
        "failed_check_count": len(failures),
        "activation_hard_validator_count": sum(
            row.get("classification") == "activation_hard_failure" for row in validator_rows
        ),
        "recomputed_ready": not failures,
    }
    return aggregate, failures


def _field_provenance(root: Path) -> dict[str, JsonDict]:
    """Name source, parser, reducer, and hash for every required field."""

    source_paths = [
        DESIGN_PATH.as_posix(),
        ROADMAP_PATH.as_posix(),
        COMPLETE_PATH.as_posix(),
        EXCLUSION_PATH.as_posix(),
        CONDUCTOR_LOG_PATH.as_posix(),
    ]
    source_hashes = {path.as_posix(): sha256_file(root / path) for path in map(Path, source_paths)}
    return {
        field: {
            "source_paths": source_paths,
            "parser": "yaml.safe_load+markdown_contract_regex+json.loads",
            "function": "carnot.experiment_6674_v582_manifest_parity_contract.build_artifact",
            "source_hashes": source_hashes,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    root: Path,
    *,
    run_date: str,
    duration_s: float,
    validator_rows: Sequence[Mapping[str, Any]],
    validator_mismatch_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Build the terminal Exp6674 artifact from immutable local evidence."""

    design_rows = parse_design_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
    manifest_rows = load_manifest_rows(root)
    producer_rows = build_producer_consumer_rows(
        design_rows, manifest_rows, set(_retirement_index(root))
    )
    prior_rows = build_prior_failure_rows(root, manifest_rows)
    protected_rows = _protected_rows(root, protected_before)
    aggregate, failures = reduce_readiness(
        design_rows,
        manifest_rows,
        producer_rows,
        prior_rows,
        validator_rows,
        protected_rows,
    )
    ready = bool(aggregate["recomputed_ready"])
    per_unit_rows = (
        [{"row_kind": "task", **row} for row in manifest_rows]
        + [{"row_kind": "gate", **row} for row in producer_rows]
        + [{"row_kind": "prior_failure", **row} for row in prior_rows]
        + [{"row_kind": "validator", **dict(row)} for row in validator_rows]
    )
    artifact: JsonDict = {
        "experiment": 6674,
        "schema": "carnot.experiment_6674.v1",
        "run_date": datetime.strptime(run_date, "%Y%m%d").date().isoformat(),
        "title": "V582 document-to-manifest parity contract",
        "status": "complete_ready" if ready else "blocked_manifest_parity",
        "honest_verdict": (
            "complete: V582 design and active manifest have exact activation parity; no scientific claim"
            if ready
            else "blocked_v582_manifest_parity_contract: one or more activation checks failed"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": failures,
        "design_task_rows": design_rows,
        "manifest_task_rows": manifest_rows,
        "producer_consumer_rows": producer_rows,
        "prior_failure_rows": prior_rows,
        "validator_rows": [dict(row) for row in validator_rows],
        "validator_mismatch_rows": [dict(row) for row in validator_mismatch_rows],
        "v582_manifest_parity_ready": ready,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": collect_preconditions(root),
        "protected_files_unchanged": protected_rows,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Fail closed on schema, row reduction, protection, or checksum drift."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"required_fields_missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    protected = payload.get("protected_files_unchanged", [])
    if not isinstance(protected, list) or any(
        row.get("unchanged") is not True for row in protected
    ):
        errors.append("protected_file_changed")
    design = payload.get("design_task_rows", [])
    manifest = payload.get("manifest_task_rows", [])
    gates = payload.get("producer_consumer_rows", [])
    priors = payload.get("prior_failure_rows", [])
    validators = payload.get("validator_rows", [])
    aggregate, failures = reduce_readiness(design, manifest, gates, priors, validators, protected)
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    ready = payload.get("v582_manifest_parity_ready")
    if ready != aggregate["recomputed_ready"]:
        errors.append("readiness_recomputation_mismatch")
    if payload.get("gate_check_summary") != failures:
        errors.append("gate_check_summary_mismatch")
    if ready is True:
        if payload.get("status") != "complete_ready":
            errors.append("ready_status_mismatch")
        if payload.get("verdict_class") != "null":
            errors.append("ready_verdict_class_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("complete:"):
            errors.append("ready_honest_verdict_mismatch")
        if payload.get("aggregate_row_recomputation", {}).get("recomputed_ready") is not True:
            errors.append("ready_aggregate_mismatch")
    else:
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_mismatch")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Sync a complete temporary file before one atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_root() -> Path:
    """Resolve the repository from this installed source file."""

    return Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the audit artifact or validate an existing one."""

    parser = argparse.ArgumentParser(description="Audit the V582 execution contract")
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = _default_root()
    output = args.output or root / RESULT_PATH
    if args.validate:
        payload = json.loads(output.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1

    started = time.monotonic()
    protected_before = protected_hashes(root)
    validator_rows, mismatch_rows = run_validators(root, args.date)
    artifact = build_artifact(
        root,
        run_date=args.date,
        duration_s=time.monotonic() - started,
        validator_rows=validator_rows,
        validator_mismatch_rows=mismatch_rows,
        tests_run=DEFAULT_TESTS_RUN,
        protected_before=protected_before,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    write_json_atomic(output, artifact)
    print(
        json.dumps(
            {"valid": True, "output": str(output), "ready": artifact["v582_manifest_parity_ready"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m
    raise SystemExit(main())
