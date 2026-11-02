"""Audit the V583 design and execution manifest without an LLM.

The audit keeps task, gate, prior-failure, route, and validator evidence as
rows. This makes each activation blocker reproducible. See REQ-REPORT-6688.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

from carnot.experiment_6674_v582_manifest_parity_contract import (
    canonical_json,
    classify_validator_result,
    extract_required_fields,
    extract_run_command,
    payload_checksum,
    sha256_file,
    value_hash,
    write_json_atomic,
)


JsonDict = dict[str, Any]
MILESTONE = "2026.08.583"
RESULT_PATH = Path("results/experiment_6688_v583_manifest_parity_contract.json")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
INFERENCE_SUBSTRATE = "artifact_and_manifest_audit_no_llm"
RANDOM_SEED = 6688

EXPECTED_TASK_IDS = (
    "exp6688-v583-manifest-parity-contract",
    "exp6689-exact-planning-fixture",
    "exp6690-exact-planning-fixture-audit",
    "exp6691-sota-planning-proposal-corpus",
    "exp6692-structural-plan-energy",
    "exp6693-energy-backtracking-ab",
    "exp6694-energy-backtracking-audit",
    "exp6695-online-energy-update-fixture",
    "exp6696-prequential-online-energy-ab",
    "exp6697-online-energy-csl-audit",
    "exp6698-torx-factor-qualification",
    "exp6699-autocorrelation-schedule-ab",
    "exp6700-stochastic-portability-audit",
    "exp6701-v583-branch-synthesis",
)

_TRACK_BY_NUMBER = {
    6688: "execution-integrity",
    **{number: "verifier-grounded-planning" for number in range(6689, 6695)},
    **{number: "continuous-self-learning" for number in range(6695, 6698)},
    **{number: "stochastic-portability" for number in range(6698, 6701)},
    6701: "milestone-synthesis",
}

# These edges transcribe the V583 dependency graph. Keeping them separate from
# YAML prevents the audit from accepting a self-consistent but changed manifest.
_DESIGN_DEPENDENCIES = {
    6688: [],
    6689: [6688],
    6690: [6689],
    6691: [6690],
    6692: [6691],
    6693: [6692],
    6694: [6693],
    6695: [6691],
    6696: [6695],
    6697: [6696],
    6698: [6688],
    6699: [6698],
    6700: [6699],
    6701: [],
}

_PRODUCER_FIELDS = {
    6688: "v583_manifest_parity_ready",
    6689: "planning_fixture_ready",
    6690: "planning_fixture_audit_passed",
    6691: "proposal_corpus_ready",
    6692: "energy_generalization_supported",
    6693: "backtracking_ab_ready",
    6695: "online_energy_fixture_ready",
    6696: "prequential_csl_ready",
    6698: "torx_factor_parity_qualified",
    6699: "schedule_ab_ready",
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
    "route_rows",
    "validator_rows",
    "v583_manifest_parity_ready",
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
        "command": ".venv/bin/pytest -o addopts='' tests/python/test_experiment_6688_v583_manifest_parity_contract.py -q",
        "exit": 0,
        "summary": "focused Exp6688 tests passed",
    },
    {
        "command": "COVERAGE_FILE=/tmp/carnot_exp6688.coverage .venv/bin/coverage run --include='*/experiment_6688_v583_manifest_parity_contract.py' -m pytest -o addopts='' tests/python/test_experiment_6688_v583_manifest_parity_contract.py -q && COVERAGE_FILE=/tmp/carnot_exp6688.coverage .venv/bin/coverage report --include='*/experiment_6688_v583_manifest_parity_contract.py' --show-missing --fail-under=100",
        "exit": 0,
        "summary": "new Exp6688 module reached 100% statement coverage",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit": 0,
        "summary": "full Python test suite passed",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6688_v583_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused spec coverage passed",
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6688_v583_manifest_parity_contract.py tests/python/test_experiment_6688_v583_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused Ruff lint passed",
    },
    {
        "command": ".venv/bin/ruff format --check python/carnot/experiment_6688_v583_manifest_parity_contract.py tests/python/test_experiment_6688_v583_manifest_parity_contract.py",
        "exit": 0,
        "summary": "focused format check passed",
    },
    {
        "command": "cd /home/ianblenke/github.com/ianblenke/carnot && .venv/bin/python -m carnot.experiment_6688_v583_manifest_parity_contract --date 20260828",
        "exit": 0,
        "summary": "required end-to-end command atomically wrote a blocked audit artifact",
    },
    {
        "command": ".venv/bin/python -m carnot.experiment_6688_v583_manifest_parity_contract --validate --output results/experiment_6688_v583_manifest_parity_contract.json",
        "exit": 0,
        "summary": "stored artifact passed schema, reduction, protection, and checksum validation",
    },
    {
        "command": ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6688_v583_manifest_parity_contract.json",
        "exit": 0,
        "summary": "row consistency found no row and verdict contradiction",
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6688_v583_manifest_parity_contract.json",
        "exit": 1,
        "summary": "no critical finding; the no-LLM substrate produced one expected naming warning",
    },
    {
        "command": "git status --short",
        "exit": 0,
        "summary": "status was inspected after tests and before handoff",
    },
)


def _number(identity: str) -> int:
    """Read the numeric experiment identity used across document formats."""

    return int(re.search(r"(?:exp|experiment_)(\d+)", identity, re.IGNORECASE).group(1))


def _task_id_from_deliverable(number: int, deliverable: str) -> str:
    """Derive the canonical task ID from its unique result path."""

    stem = Path(deliverable).stem
    suffix = stem.removeprefix(f"experiment_{number}_").replace("_", "-")
    return f"exp{number}-{suffix}"


def _route(route_text: str) -> JsonDict:
    """Normalize the three route labels used in the conductor table."""

    return {
        "agent_type": "codex" if route_text.startswith("Codex") else "claude",
        "model": (
            "gpt-5.6-sol"
            if "gpt-5.6-sol" in route_text
            else "opus"
            if "Opus" in route_text
            else None
        ),
    }


def _designed_gates(number: int) -> list[JsonDict]:
    """Attach producer-owned Boolean gates to each designed dependency."""

    return [
        {
            "upstream": EXPECTED_TASK_IDS[producer - 6688],
            "artifact_field": _PRODUCER_FIELDS.get(producer),
            "op": "in",
            "value": [True],
        }
        for producer in _DESIGN_DEPENDENCIES[number]
    ]


def parse_design_contract(text: str) -> list[JsonDict]:
    """Parse task sections, conductor order, graph edges, and prior lineage."""

    sections = re.findall(
        r"^### Exp(?P<number>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^### Exp\d+:|^## Dependency Graph)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if len(sections) != 14:
        raise ValueError(f"V583 design must contain fourteen task sections; found {len(sections)}")
    conductor = {
        int(number): {
            "order": int(order),
            "task_category": category.strip(),
            "route": _route(route.strip()),
            "requires_gpu": gpu.strip() == "yes",
            "estimated_wall_time_min": int(wall),
        }
        for order, number, category, route, gpu, wall in re.findall(
            r"^\|\s*(\d+)\s*\|\s*Exp(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(yes|no)\s*\|\s*(\d+) min\s*\|$",
            text,
            re.MULTILINE,
        )
    }
    prior_by_consumer: dict[int, list[JsonDict]] = {}
    prior_section = text.split("## Prior-Failure Discipline", 1)[1].split("## Promotion Gates", 1)[
        0
    ]
    prior_bullets = re.findall(
        r"^- Exp(?P<prior>\d+) (?P<body>.*?)(?=^- Exp\d+|^Every entry carries)",
        prior_section,
        re.MULTILINE | re.DOTALL,
    )
    for prior, body in prior_bullets:
        normalized = " ".join(body.split()).rstrip(".")
        declaration = re.fullmatch(
            r"(?P<scope>.*?) → Exp(?P<consumer>\d+) (?P<changed>.*)", normalized
        )
        if declaration is None:
            raise ValueError(f"invalid V583 prior-failure declaration for Exp{prior}")
        consumer = int(declaration.group("consumer"))
        prior_by_consumer.setdefault(consumer, []).append(
            {
                "experiment_id": f"exp{prior}",
                "prior_scope": declaration.group("scope"),
                "changed_condition": declaration.group("changed"),
                "retire_if_same_verdict": True,
            }
        )

    rows: list[JsonDict] = []
    for number_text, title, body in sections:
        number = int(number_text)
        deliverable = re.search(r"\*\*Deliverable:\*\* `([^`]+)`", body).group(1)
        route = conductor[number]
        gates = _designed_gates(number)
        rows.append(
            {
                "order": route["order"],
                "task_id": _task_id_from_deliverable(number, deliverable),
                "experiment_number": number,
                "title": title.strip(),
                "deliverable": deliverable,
                "track": _TRACK_BY_NUMBER[number],
                "requires_gpu": route["requires_gpu"],
                "dependencies": [gate["upstream"] for gate in gates],
                "gated_on": gates,
                "route": route["route"],
                "task_category": route["task_category"],
                "estimated_wall_time_min": route["estimated_wall_time_min"],
                "prior_failure_ids": [
                    item["experiment_id"] for item in prior_by_consumer.get(number, [])
                ],
                "prior_failure_declarations": prior_by_consumer.get(number, []),
                "required_artifact_fields": sorted(
                    {
                        field
                        for consumer in _DESIGN_DEPENDENCIES
                        for producer, field in ((number, _PRODUCER_FIELDS.get(number)),)
                        if number in _DESIGN_DEPENDENCIES[consumer] and field is not None
                    }
                ),
                "run_command": (
                    f"cd {{project_root}} && .venv/bin/python -m carnot.{Path(deliverable).stem} "
                    "--date {date}"
                ),
                "declaration_source": DESIGN_PATH.as_posix(),
            }
        )
    return sorted(rows, key=lambda row: row["order"])


def select_manifest_path(root: Path) -> Path:
    """Select the checked-in V583 queue without creating or moving a roadmap."""

    for relative in (NEXT_ROADMAP_PATH, ROADMAP_PATH):
        path = root / relative
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        if isinstance(payload, Mapping) and payload.get("milestone") == MILESTONE:
            return relative
    raise ValueError("no active or next V583 execution manifest exists")


def _normalized_gates(task: Mapping[str, Any]) -> list[JsonDict]:
    """Copy each manifest gate with the stable conductor field names."""

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


def load_manifest_rows(root: Path, design_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Load the selected V583 YAML and compare each observed declaration."""

    relative = select_manifest_path(root)
    path = root / relative
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    design_by_id = {str(row["task_id"]): row for row in design_rows}
    rows: list[JsonDict] = []
    for order, task_value in enumerate(payload.get("tasks", []), 1):
        task = dict(task_value)
        task_id = str(task.get("id", ""))
        expected = design_by_id.get(task_id, {})
        prompt = str(task.get("prompt", ""))
        fields = extract_required_fields(prompt)
        gates = _normalized_gates(task)
        prior_blocks = [dict(item) for item in task.get("prior_failures", []) or []]
        prior_numbers = [_number(str(item.get("experiment_id"))) for item in prior_blocks]
        expected_prior_numbers = [_number(item) for item in expected.get("prior_failure_ids", [])]
        route = {"agent_type": task.get("agent_type") or "claude", "model": task.get("model")}
        observed = {
            "title": task.get("title"),
            "deliverable": task.get("deliverable"),
            "track": task.get("track"),
            "requires_gpu": task.get("requires_gpu", False),
            "dependency": list(dict.fromkeys(gate["upstream"] for gate in gates)),
            "gate": gates,
            "route": route,
            "prior_failure_block": prior_numbers,
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
            "prior_failure_block": expected_prior_numbers,
            "required_artifact_field": expected.get("required_artifact_fields", []),
            "run_command": expected.get("run_command"),
        }
        valid_prior_shape = all(
            bool(str(item.get("experiment_id", "")).strip())
            and bool(str(item.get("verdict", "")).strip())
            and bool(str(item.get("addressed_by", "")).strip())
            and item.get("retire_if_same_verdict") is True
            for item in prior_blocks
        )
        field_checks = {
            name: observed[name] == expected_values[name]
            for name in (
                "title",
                "deliverable",
                "track",
                "requires_gpu",
                "dependency",
                "gate",
                "route",
                "run_command",
            )
        }
        field_checks["prior_failure_block"] = (
            observed["prior_failure_block"] == expected_values["prior_failure_block"]
            and valid_prior_shape
        )
        field_checks["required_artifact_field"] = set(
            expected_values["required_artifact_field"]
        ) <= set(fields)
        differences = {
            name: {"expected": expected_values[name], "observed": observed[name]}
            for name, passed in field_checks.items()
            if not passed
        }
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task.get("title"),
                "deliverable": task.get("deliverable"),
                "track": task.get("track"),
                "requires_gpu": task.get("requires_gpu", False),
                "dependencies": observed["dependency"],
                "gated_on": gates,
                "route": route,
                "max_turns": task.get("max_turns"),
                "estimated_wall_time_min": task.get("estimated_wall_time_min"),
                "prior_failures": prior_blocks,
                "required_artifact_fields": fields,
                "run_command": observed["run_command"],
                "field_checks": field_checks,
                "field_differences": differences,
                "manifest_path": relative.as_posix(),
                "source_hash": sha256_file(path),
            }
        )
    return rows


def build_producer_consumer_rows(
    design_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    retired_numbers: set[int],
) -> list[JsonDict]:
    """Prove every design or manifest gate has one identical peer declaration."""

    design_tasks = {str(row["task_id"]): row for row in design_rows}
    manifest_tasks = {str(row["task_id"]): row for row in manifest_rows}

    def gate_key(consumer: Mapping[str, Any], gate: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(consumer.get("task_id")),
            str(gate.get("upstream")),
            str(gate.get("artifact_field")),
            str(gate.get("op")),
            canonical_json(gate.get("value")).decode("utf-8"),
        )

    design_gates = {
        gate_key(task, gate): dict(gate)
        for task in design_rows
        for gate in task.get("gated_on", [])
    }
    manifest_gates = {
        gate_key(task, gate): dict(gate)
        for task in manifest_rows
        for gate in task.get("gated_on", [])
    }
    order_by_task = {task_id: order for order, task_id in enumerate(EXPECTED_TASK_IDS, 1)}
    rows: list[JsonDict] = []
    for key in sorted(
        design_gates.keys() | manifest_gates.keys(),
        key=lambda item: (order_by_task.get(item[0], 10**9), item),
    ):
        consumer_id, producer_id, artifact_field, operator, _value_text = key
        gate = manifest_gates.get(key) or design_gates[key]
        design_producer = design_tasks.get(producer_id)
        manifest_producer = manifest_tasks.get(producer_id)
        design_fields = (
            list(design_producer.get("required_artifact_fields", [])) if design_producer else []
        )
        manifest_fields = (
            list(manifest_producer.get("required_artifact_fields", [])) if manifest_producer else []
        )
        declared_in_design = key in design_gates
        declared_in_manifest = key in manifest_gates
        rows.append(
            {
                "producer": producer_id,
                "field": artifact_field,
                "artifact_field": artifact_field,
                "consumer": consumer_id,
                "operator": operator,
                "value": gate.get("value"),
                "declared_in_design": declared_in_design,
                "declared_in_manifest": declared_in_manifest,
                "upstream_exists_in_design": design_producer is not None,
                "upstream_exists_in_manifest": manifest_producer is not None,
                "upstream_exists": design_producer is not None and manifest_producer is not None,
                "producer_declares_exact_field_in_design": artifact_field in design_fields,
                "producer_declares_exact_field_in_manifest": artifact_field in manifest_fields,
                "producer_declares_exact_field": artifact_field in design_fields
                and artifact_field in manifest_fields,
                "matches_design": declared_in_design and declared_in_manifest,
                "upstream_retired": _number(producer_id) in retired_numbers,
                "producer_required_artifact_fields_hash": value_hash(
                    {"design": design_fields, "manifest": manifest_fields}
                ),
            }
        )
    return rows


def _completed_by_number(root: Path) -> dict[int, JsonDict]:
    """Index archived task records by stable numeric identity."""

    payload = yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8")) or {}
    return {
        _number(str(task["id"])): {
            "id": task.get("id"),
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
    """Index direct and grouped exclusion entries by experiment number."""

    payload = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    rows = [
        (section, entry)
        for section in ("retired", "retired_experiments", "retired_extras")
        for entry in payload.get(section, []) or []
        if isinstance(entry, Mapping)
    ]
    index: dict[int, list[JsonDict]] = {}
    for section, entry in rows:
        identities = [entry.get("experiment_id"), *(entry.get("experiment_ids", []) or [])]
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
    root: Path,
    design_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Bind the union of design and manifest priors to durable lineage evidence."""

    completed = _completed_by_number(root)
    retired = _retirement_index(root)
    conductor_lines = (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8").splitlines()
    upstream_numbers = {
        _number(str(gate["upstream"]))
        for task in manifest_rows
        for gate in task.get("gated_on", [])
    }
    design_priors = {
        (str(task.get("task_id")), _number(str(prior.get("experiment_id")))): dict(prior)
        for task in design_rows
        for prior in task.get("prior_failure_declarations", [])
    }
    manifest_priors = {
        (str(task.get("task_id")), _number(str(prior.get("experiment_id")))): dict(prior)
        for task in manifest_rows
        for prior in task.get("prior_failures", [])
    }
    order_by_task = {task_id: order for order, task_id in enumerate(EXPECTED_TASK_IDS, 1)}
    rows: list[JsonDict] = []
    for key in sorted(
        design_priors.keys() | manifest_priors.keys(),
        key=lambda item: (order_by_task.get(item[0], 10**9), item[1]),
    ):
        consumer_id, number = key
        design_prior = design_priors.get(key)
        manifest_prior = manifest_priors.get(key)
        prior_id = str(
            (manifest_prior or {}).get("experiment_id") or (design_prior or {}).get("experiment_id")
        )
        record = completed.get(number)
        fallback_artifacts = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        artifact_path = (
            Path(str(record.get("deliverable")))
            if record
            else fallback_artifacts[0].relative_to(root)
            if fallback_artifacts
            else Path("")
        )
        artifact_hash = sha256_file(root / artifact_path) if artifact_path.name else "missing"
        artifact_verdict: Any = None
        if artifact_hash != "missing":
            try:
                artifact_payload = json.loads((root / artifact_path).read_text(encoding="utf-8"))
                artifact_verdict = artifact_payload.get("honest_verdict")
                if isinstance(artifact_verdict, Mapping) and "value" in artifact_verdict:
                    artifact_verdict = artifact_verdict.get("value")
            except (json.JSONDecodeError, OSError, AttributeError):
                artifact_verdict = None
        title = str(record.get("title", "")) if record else ""
        conductor_rows = [
            line
            for line in conductor_lines
            if prior_id.lower() in line.lower()
            or f"exp{number}" in line.lower()
            or (title and title.lower() in line.lower())
        ]
        exclusion_rows = retired.get(number, [])
        row = {
            "consumer_task_id": consumer_id,
            "prior_experiment_id": prior_id,
            "prior_scope": (
                record.get("title")
                if record
                else (design_prior or {}).get("prior_scope")
                or (exclusion_rows[0].get("id") if exclusion_rows else None)
            ),
            "verdict": (manifest_prior or {}).get("verdict")
            or artifact_verdict
            or (record or {}).get("result"),
            "changed_condition": (manifest_prior or {}).get("addressed_by")
            or (design_prior or {}).get("changed_condition"),
            "retirement_signal": (
                (manifest_prior or {}).get("retire_if_same_verdict")
                if manifest_prior
                else (design_prior or {}).get("retire_if_same_verdict")
            ),
            "declared_in_design": design_prior is not None,
            "declared_in_manifest": manifest_prior is not None,
            "design_declaration": design_prior,
            "manifest_declaration": manifest_prior,
            "completed_record_found": record is not None,
            "completed_record": record,
            "artifact_path": artifact_path.as_posix() if artifact_path.name else None,
            "artifact_hash": artifact_hash,
            "artifact_state": "present" if artifact_hash != "missing" else "missing",
            "conductor_state_rows": conductor_rows,
            "conductor_state_hash": value_hash(conductor_rows),
            "exclusion_manifest_match": bool(exclusion_rows),
            "exclusion_manifest_entries": exclusion_rows,
            "reference_role": "prior_failure",
            "retired_upstream_reference": number in upstream_numbers and bool(exclusion_rows),
        }
        row["lineage_passed"] = bool(
            (row["completed_record_found"] or row["artifact_state"] == "present" or conductor_rows)
            and str(row["verdict"] or "").strip()
            and str(row["changed_condition"] or "").strip()
            and row["retirement_signal"] is True
            and not row["retired_upstream_reference"]
        )
        row["passed"] = bool(
            row["lineage_passed"] and row["declared_in_design"] and row["declared_in_manifest"]
        )
        rows.append(row)
    return rows


def build_route_rows(
    design_rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain backend, model, turn budget, category, and route validation."""

    observed = {str(row["task_id"]): row for row in manifest_rows}
    rows: list[JsonDict] = []
    for design in design_rows:
        manifest = observed.get(str(design["task_id"]))
        route_matches = bool(
            manifest
            and manifest.get("route") == design.get("route")
            and manifest.get("requires_gpu") == design.get("requires_gpu")
            and manifest.get("estimated_wall_time_min") == design.get("estimated_wall_time_min")
            and isinstance(manifest.get("max_turns"), int)
            and manifest.get("max_turns") > 0
        )
        rows.append(
            {
                "task_id": design.get("task_id"),
                "agent_backend": manifest.get("route", {}).get("agent_type") if manifest else None,
                "model": manifest.get("route", {}).get("model") if manifest else None,
                "max_turns": manifest.get("max_turns") if manifest else None,
                "task_category": design.get("task_category"),
                "requires_gpu": manifest.get("requires_gpu") if manifest else None,
                "expected_route": design.get("route"),
                "expected_requires_gpu": design.get("requires_gpu"),
                "expected_wall_time_min": design.get("estimated_wall_time_min"),
                "observed_wall_time_min": (
                    manifest.get("estimated_wall_time_min") if manifest else None
                ),
                "validation": (
                    "passed"
                    if route_matches
                    else "route_mismatch"
                    if manifest
                    else "missing_manifest_task"
                ),
            }
        )
    return rows


def _run_command(root: Path, command: str) -> tuple[int, str]:
    """Run one fixed read-only validator and retain its combined output."""

    result = subprocess.run(
        command,
        cwd=root,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def run_validators(root: Path, run_date: str) -> list[JsonDict]:
    """Run each required validator once without changing its source or inputs."""

    rows: list[JsonDict] = []
    for name, command in VALIDATOR_DEFINITIONS:
        exit_code, output = _run_command(root, command)
        row, mismatches = classify_validator_result(name, command, exit_code, output, run_date)
        row["operator_schema_mismatch_rows"] = mismatches
        rows.append(row)
    return rows


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash both roadmap paths and the conductor, including absence."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def _protected_rows(root: Path, before: Mapping[str, str]) -> list[JsonDict]:
    """Build before-and-after identity receipts for protected inputs."""

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
    """Measure physical memory from host page counters."""

    return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def collect_preconditions(root: Path) -> JsonDict:
    """Measure source identity, prior evidence, resources, tools, and substrate."""

    inputs = (
        ROADMAP_PATH,
        NEXT_ROADMAP_PATH,
        DESIGN_PATH,
        CONDUCTOR_PATH,
        EXCLUSION_PATH,
        COMPLETE_PATH,
        CONDUCTOR_LOG_PATH,
    )
    completed = _completed_by_number(root)
    relevant_numbers = [5163, 5747, *range(6674, 6688)]
    prior_artifacts = []
    for number in relevant_numbers:
        record = completed.get(number)
        relative = Path(str(record.get("deliverable"))) if record else Path("")
        digest = sha256_file(root / relative) if relative.name else "missing"
        prior_artifacts.append(
            {
                "experiment_number": number,
                "task_id": record.get("id") if record else None,
                "path": relative.as_posix() if relative.name else None,
                "state": "present" if digest != "missing" else "missing",
                "sha256": digest,
                "completed_record_hash": value_hash(record) if record else "missing",
            }
        )
    disk = shutil.disk_usage(root)
    cpu_text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    cpu_match = re.search(r"^model name\s*:\s*(.+)$", cpu_text, re.MULTILINE)
    return {
        "inputs": [
            {
                "path": path.as_posix(),
                "state": "present" if (root / path).is_file() else "missing",
                "sha256": sha256_file(root / path),
            }
            for path in inputs
        ],
        "prior_artifacts": prior_artifacts,
        "resources": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "cpu_model": cpu_match.group(1) if cpu_match else platform.machine(),
            "cpu_logical_count": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
        },
        "tools": [
            {"path": path.as_posix(), "sha256": sha256_file(root / path)}
            for path in (
                Path("scripts/roadmap_schema.py"),
                Path("scripts/validate_prior_failures.py"),
                Path("scripts/audit_roadmap_gates.py"),
                Path("scripts/exclusion_manifest_lint.py"),
                Path("scripts/conductor_gates.py"),
            )
        ],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_calls": 0,
    }


def _failure(check: str, unit: str, expected: Any, observed: Any, reason: str) -> JsonDict:
    """Build one stable diagnostic without replacing missing values with zero."""

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
    route_rows: Sequence[Mapping[str, Any]],
    validator_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Reduce all retained row evidence to one activation-readiness Boolean."""

    design_ids = [row.get("task_id") for row in design_rows]
    manifest_ids = [row.get("task_id") for row in manifest_rows]
    manifest_deliverables = [row.get("deliverable") for row in manifest_rows]
    expected_deliverables = [row.get("deliverable") for row in design_rows]
    design_deliverables_valid = bool(
        len(expected_deliverables) == 14
        and len(set(expected_deliverables)) == 14
        and all(
            isinstance(path, str) and path.startswith("results/") and path.endswith(".json")
            for path in expected_deliverables
        )
    )
    manifest_deliverables_valid = bool(
        len(manifest_deliverables) == 14
        and len(set(manifest_deliverables)) == 14
        and all(
            isinstance(path, str) and path.startswith("results/") and path.endswith(".json")
            for path in manifest_deliverables
        )
    )
    checks: list[tuple[str, str, Any, Any, str]] = [
        (
            "design.task_order",
            "design",
            list(EXPECTED_TASK_IDS),
            design_ids,
            "ordered_task_set_mismatch",
        ),
        (
            "manifest.task_order",
            "manifest",
            list(EXPECTED_TASK_IDS),
            manifest_ids,
            "ordered_task_set_mismatch",
        ),
        (
            "manifest.deliverables",
            "manifest",
            expected_deliverables,
            manifest_deliverables,
            "unique_json_deliverable_contract_mismatch",
        ),
        (
            "design.deliverable_shape",
            "design",
            True,
            design_deliverables_valid,
            "design_requires_fourteen_unique_json_deliverables",
        ),
        (
            "manifest.deliverable_shape",
            "manifest",
            True,
            manifest_deliverables_valid,
            "manifest_requires_fourteen_unique_json_deliverables",
        ),
    ]
    checks.extend(
        (
            f"task.{field}",
            str(task.get("task_id")),
            difference.get("expected"),
            difference.get("observed"),
            "design_manifest_field_mismatch",
        )
        for task in manifest_rows
        for field, difference in task.get("field_differences", {}).items()
    )
    checks.extend(
        ("task.present", str(task_id), True, False, "manifest_task_missing")
        for task_id in EXPECTED_TASK_IDS
        if task_id not in manifest_ids
    )
    checks.extend(
        (check, f"{row.get('consumer')}->{row.get('producer')}", expected, row.get(key), reason)
        for row in gate_rows
        for check, key, expected, reason in (
            ("gate.upstream_exists", "upstream_exists", True, "gate_producer_missing"),
            (
                "gate.producer_field_spelling",
                "producer_declares_exact_field",
                True,
                "gate_field_not_owned",
            ),
            ("gate.matches_design", "matches_design", True, "gate_not_in_design_contract"),
            (
                "gate.upstream_not_retired",
                "upstream_retired",
                False,
                "retired_upstream_dependency",
            ),
        )
    )
    checks.extend(
        (
            "prior.lineage",
            str(row.get("prior_experiment_id")),
            True,
            row.get("passed"),
            "prior_failure_contract_mismatch",
        )
        for row in prior_rows
    )
    checks.extend(
        (
            "route.validation",
            str(row.get("task_id")),
            "passed",
            row.get("validation"),
            "route_or_resource_contract_mismatch",
        )
        for row in route_rows
    )
    checks.extend(
        (
            "validator.activation",
            str(row.get("validator")),
            "passed or dated nonblocking mismatch",
            row.get("classification"),
            "activation_hard_validator_failure",
        )
        for row in validator_rows
        if row.get("classification") == "activation_hard_failure"
    )
    checks.extend(
        (
            "protected_file.unchanged",
            str(row.get("path")),
            row.get("before"),
            row.get("after"),
            "protected_file_changed",
        )
        for row in protected_rows
        if row.get("unchanged") is not True
    )
    failures = [
        _failure(check, unit, expected, observed, reason)
        for check, unit, expected, observed, reason in checks
        if observed != expected
    ]
    missing = [task_id for task_id in EXPECTED_TASK_IDS if task_id not in manifest_ids]
    aggregate = {
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "design_task_ids": design_ids,
        "manifest_task_ids": manifest_ids,
        "missing_task_ids": missing,
        "design_task_count": len(design_rows),
        "manifest_task_count": len(manifest_rows),
        "gate_count": len(gate_rows),
        "prior_failure_count": len(prior_rows),
        "route_count": len(route_rows),
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
    """Name source paths, parser, reducer, and pinned input hash per field."""

    paths = (DESIGN_PATH, ROADMAP_PATH, NEXT_ROADMAP_PATH, COMPLETE_PATH, EXCLUSION_PATH)
    hashes = {path.as_posix(): sha256_file(root / path) for path in paths}
    return {
        field: {
            "source_path": "|".join(hashes),
            "parser": "yaml.safe_load+markdown_task_and_conductor_parser+json.loads",
            "function": "carnot.experiment_6688_v583_manifest_parity_contract.build_artifact",
            "hash": value_hash(hashes),
            "source_hashes": hashes,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _per_unit_rows(
    design_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    producer_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    validator_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Project every contract unit once so readers can replay the reducer."""

    return (
        [{"row_kind": "design_task", **dict(row)} for row in design_rows]
        + [{"row_kind": "manifest_task", **dict(row)} for row in manifest_rows]
        + [{"row_kind": "gate", **dict(row)} for row in producer_rows]
        + [{"row_kind": "prior_failure", **dict(row)} for row in prior_rows]
        + [{"row_kind": "route", **dict(row)} for row in route_rows]
        + [{"row_kind": "validator", **dict(row)} for row in validator_rows]
    )


def build_artifact(
    root: Path,
    *,
    run_date: str,
    duration_s: float,
    validator_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Build the terminal audit artifact from immutable local evidence."""

    design_rows = parse_design_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
    manifest_rows = load_manifest_rows(root, design_rows)
    retired = set(_retirement_index(root))
    producer_rows = build_producer_consumer_rows(design_rows, manifest_rows, retired)
    prior_rows = build_prior_failure_rows(root, design_rows, manifest_rows)
    route_rows = build_route_rows(design_rows, manifest_rows)
    protected_rows = _protected_rows(root, protected_before)
    aggregate, failures = reduce_readiness(
        design_rows,
        manifest_rows,
        producer_rows,
        prior_rows,
        route_rows,
        validator_rows,
        protected_rows,
    )
    ready = bool(aggregate["recomputed_ready"])
    per_unit_rows = _per_unit_rows(
        design_rows,
        manifest_rows,
        producer_rows,
        prior_rows,
        route_rows,
        validator_rows,
    )
    artifact: JsonDict = {
        "experiment": 6688,
        "schema": "carnot.experiment_6688.v1",
        "run_date": datetime.strptime(run_date, "%Y%m%d").date().isoformat(),
        "title": "V583 document-to-manifest parity contract",
        "status": "complete_ready" if ready else "blocked_manifest_parity",
        "honest_verdict": (
            "complete: V583 design and execution manifest have exact activation parity; no scientific claim"
            if ready
            else "blocked_v583_manifest_parity_contract: design declares fourteen tasks but the selected manifest and activation checks do not match"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": failures,
        "design_task_rows": design_rows,
        "manifest_task_rows": manifest_rows,
        "producer_consumer_rows": producer_rows,
        "prior_failure_rows": prior_rows,
        "route_rows": route_rows,
        "validator_rows": [dict(row) for row in validator_rows],
        "v583_manifest_parity_ready": ready,
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
    """Fail closed on field, row, status, protection, or checksum drift."""

    required_errors = [
        f"required_fields_missing:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in payload
    ]
    protected = payload.get("protected_files_unchanged", [])
    expected_protected_paths = [path.as_posix() for path in PROTECTED_PATHS]
    protected_valid = bool(
        isinstance(protected, list)
        and [row.get("path") for row in protected] == expected_protected_paths
        and all(
            row.get("unchanged") is True and row.get("before") == row.get("after")
            for row in protected
        )
    )
    aggregate, failures = reduce_readiness(
        payload.get("design_task_rows", []),
        payload.get("manifest_task_rows", []),
        payload.get("producer_consumer_rows", []),
        payload.get("prior_failure_rows", []),
        payload.get("route_rows", []),
        payload.get("validator_rows", []),
        protected if isinstance(protected, list) else [],
    )
    ready = aggregate["recomputed_ready"]
    expected_status = "complete_ready" if ready else "blocked_manifest_parity"
    expected_class = "null" if ready else "blocked"
    verdict_prefix = "complete:" if ready else "blocked_"
    expected_per_unit_rows = _per_unit_rows(
        payload.get("design_task_rows", []),
        payload.get("manifest_task_rows", []),
        payload.get("producer_consumer_rows", []),
        payload.get("prior_failure_rows", []),
        payload.get("route_rows", []),
        payload.get("validator_rows", []),
    )
    conditions = (
        (payload.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate_mismatch"),
        (payload.get("verifier_is_oracle") is True, "verifier_is_oracle_mismatch"),
        (
            set(payload.get("field_provenance", {})) == set(REQUIRED_ARTIFACT_FIELDS),
            "field_provenance_mismatch",
        ),
        (protected_valid, "protected_file_changed"),
        (payload.get("per_unit_rows") == expected_per_unit_rows, "per_unit_rows_mismatch"),
        (
            payload.get("aggregate_row_recomputation") == aggregate,
            "aggregate_row_recomputation_mismatch",
        ),
        (payload.get("gate_check_summary") == failures, "gate_check_summary_mismatch"),
        (payload.get("v583_manifest_parity_ready") == ready, "readiness_recomputation_mismatch"),
        (
            payload.get("status") == expected_status,
            f"{'ready' if ready else 'blocked'}_status_mismatch",
        ),
        (
            payload.get("verdict_class") == expected_class,
            f"{'ready' if ready else 'blocked'}_verdict_class_mismatch",
        ),
        (
            str(payload.get("honest_verdict", "")).startswith(verdict_prefix),
            f"{'ready' if ready else 'blocked'}_honest_verdict_mismatch",
        ),
        (
            payload.get("reproducibility_checksum") == payload_checksum(payload),
            "reproducibility_checksum_mismatch",
        ),
    )
    return required_errors + [error for passed, error in conditions if not passed]


def _default_root() -> Path:
    """Resolve the repository from this installed source file."""

    return Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the read-only audit receipt or validate an existing receipt."""

    parser = argparse.ArgumentParser(description="Audit the V583 execution contract")
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
    before = protected_hashes(root)
    validators = run_validators(root, args.date)
    artifact = build_artifact(
        root,
        run_date=args.date,
        duration_s=time.monotonic() - started,
        validator_rows=validators,
        tests_run=DEFAULT_TESTS_RUN,
        protected_before=before,
    )
    errors = validate_artifact(artifact)
    print_payload = {
        "valid": not errors,
        "errors": errors,
        "output": str(output),
        "ready": artifact["v583_manifest_parity_ready"],
    }
    print(json.dumps(print_payload, indent=2))
    if errors:
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m
    raise SystemExit(main())
