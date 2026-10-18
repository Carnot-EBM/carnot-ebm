"""Build the V633 Markdown/YAML contract receipt.

The receipt answers one narrow question: do two independent planning sources
describe the same 13 executable tasks? It records a readable mismatch as a
result. It does not repair either source or judge the planned science.

Spec refs: REQ-REPORT-7179 and SCENARIO-REPORT-7179-*.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from carnot.experiment_7151_v629_contract_preflight import task_number


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.633"
RUN_DATE = "20260910"
FIRST_TASK_ID = "exp7179-contract-receipt"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7180-symbolic-edit-fixture",
    "exp7181-qwen38-symbolic-traces",
    "exp7182-grounding-energy-audit",
    "exp7183-supersession-stream",
    "exp7184-revocable-template-csl",
    "exp7185-memory-cold-audit",
    "exp7186-arc-withheld-transfer",
    "exp7187-slice-sampler",
    "exp7188-quantized-transition-audit",
    "exp7189-rust-slice-parity",
    "exp7190-board-placement-receipt",
    "exp7191-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
RANDOM_SEED = 717920260910

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
PRIOR_ARTIFACT_PATH = Path("results/experiment_7166_v632_contract_preflight.json")
MODULE_PATH = Path("python/carnot/experiment_7179_v633_contract_receipt.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7179_v633_contract_receipt.py")
TEST_PATH = Path("tests/python/test_experiment_7179_v633_contract_receipt.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7179_v633_contract_receipt.json")
RAW_DIR = Path("results/raw/experiment_7179_v633_contract_receipt")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7179_v633_contract_receipt.json")
RAW_MARKDOWN_NAME = "research-roadmap-vNEXT.md"
RAW_YAML_NAME = "selected-roadmap.yaml"

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
ARC_LINT_PATH = Path("scripts/arc_levelup_guarantee_lint.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")
ROOT_CLUTTER_PATH = Path("scripts/root_clutter_sweep.py")
TOOL_PATHS = (
    ROADMAP_SCHEMA_PATH,
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    ARC_LINT_PATH,
    GATE_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
    ROOT_CLUTTER_PATH,
)
SOURCE_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    DESIGN_PATH,
    EXCLUSION_PATH,
    PRIOR_ARTIFACT_PATH,
    SPEC_PATH,
    *TOOL_PATHS,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts: independent V633 Markdown and YAML contract parsing"
)
POSITIVE_VERDICT = "complete_positive_v633_task_contract_exact_agreement"
DISQUALIFIED_VERDICT = "complete_disqualified_v633_markdown_yaml_contract_mismatch"
BLOCKED_PREREQUISITE_VERDICT = "blocked_v633_contract_receipt_prerequisite_missing"
BLOCKED_SOURCE_VERDICT = "blocked_contract_source"

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use aggregation when the declared work runs; use blocked_no_run only before any qualifying work.",
    "contract_complete_score": "One records exact agreement; it does not certify the science.",
    "markdown_task_rows": "Independent rows expose a stale design.",
    "yaml_task_rows": "Independent rows expose partial YAML emission.",
    "gate_contract_rows": "Each gate must name an earlier producer and its declared field.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "arc_generalization",
    "gate_audit",
    "artifact",
    "adversarial",
    "row_consistency",
    "scoped_spec_coverage",
    "ruff",
    "mypy",
    "root_clutter",
)


def _progress(phase: int, state: str, detail: str) -> None:
    """Flush phase state so an operator can distinguish work from a stall."""

    print(f"[exp7179] phase {phase} {state}: {detail}", flush=True)


def _sha256_bytes(content: bytes) -> str:
    """Bind a byte sequence without decoding or normalizing its contents."""

    return "sha256:" + hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    """Hash the exact file bytes used by the receipt."""

    return _sha256_bytes(path.read_bytes())


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Replace one file atomically so readers cannot observe partial evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}-", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write stable JSON through the same atomic evidence path."""

    content = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, content)


def _load_yaml_bytes(content: bytes, path: Path) -> dict[str, Any]:
    """Decode a non-empty roadmap mapping and name its source on failure."""

    try:
        value = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML at {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"YAML mapping required at {path}")
    return value


def _required_block(prompt: str) -> str:
    """Isolate field declarations so ordinary prose cannot declare a gate field."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _field_declared(block: str, field: object) -> bool:
    """Accept a gate field only as one bare declaration line."""

    return isinstance(field, str) and bool(re.search(rf"(?m)^\s*-\s*{re.escape(field)}\s*:", block))


def _substrate_class(prompt: str) -> str | None:
    """Read the planned class from the required artifact declaration."""

    block = _required_block(prompt)
    field = block.find("inference_substrate_class")
    if field < 0:
        return None
    match = re.search(r"\bUse\s+([a-z_]+)", block[field : field + 320], re.I)
    return match.group(1).lower() if match else None


def _parse_scalar(text: str) -> Any:
    """Parse one gate value while rejecting hidden containers."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def _parse_markdown_gate(cell: str, ids_by_number: Mapping[int, str]) -> list[dict[str, Any]]:
    """Parse gate expressions without reading any YAML-derived value."""

    if cell.strip().lower() == "none":
        return []
    pattern = re.compile(
        r"^(exp\d+(?:-[A-Za-z0-9_-]+)?)\.([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(==|!=|>=|<=|>|<)\s*(.+)$",
        re.I,
    )
    gates = []
    for expression in re.split(r"\s*(?:;|\s+AND\s+)\s*", cell, flags=re.I):
        match = pattern.fullmatch(expression.strip().strip("`"))
        if match is None:
            raise ValueError(f"malformed Markdown structured gate: {cell}")
        number = task_number(match.group(1))
        gates.append(
            {
                "upstream": ids_by_number.get(number, match.group(1)),
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4).strip().strip("`")),
            }
        )
    return gates


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Parse the Markdown milestone and table without consulting the YAML."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section = re.search(r"^## Exact Task Contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.I | re.M)
    if section is None:
        raise ValueError("Markdown task contract section is missing")
    header: list[str] | None = None
    pending: list[tuple[dict[str, Any], str]] = []
    for line in section.group(1).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if header is None:
            header = [re.sub(r"\s+", " ", cell.lower()) for cell in cells]
            continue
        columns = dict(zip(header, cells, strict=False))
        order = columns.get("order", "")
        task_id = columns.get("task id", "").strip("`")
        title = columns.get("exact title", "")
        deliverable = columns.get("deliverable", "").strip("`")
        number = task_number(task_id)
        if not order.isdigit() or number is None or not title or not deliverable:
            raise ValueError(f"malformed Markdown task row: {line}")
        pending.append(
            (
                {
                    "order": int(order),
                    "id": task_id,
                    "number": number,
                    "title": title,
                    "deliverable": deliverable,
                    "phase": columns.get("phase"),
                    "substrate": columns.get("substrate class", "").strip("`") or None,
                    "milestone": milestone_match.group(1),
                },
                columns.get("structured gate", "none"),
            )
        )
    if not pending:
        raise ValueError("Markdown task table is missing")
    ids_by_number = {row["number"]: row["id"] for row, _gate in pending}
    tasks = []
    for row, gate_cell in pending:
        row["gates"] = _parse_markdown_gate(gate_cell, ids_by_number)
        tasks.append(row)
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def parse_yaml_contract(document: object) -> dict[str, Any]:
    """Parse executable task rows without receiving the Markdown result."""

    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        raise ValueError("YAML roadmap mapping with tasks is required")
    tasks = []
    for order, raw in enumerate(document["tasks"], 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"YAML task {order} must be a mapping")
        prompt = raw.get("prompt")
        task_id = raw.get("id")
        if not isinstance(prompt, str) or not isinstance(task_id, str):
            raise ValueError(f"YAML task {order} has malformed id or prompt")
        raw_gates = raw.get("gated_on") or []
        if not isinstance(raw_gates, list) or not all(
            isinstance(gate, Mapping) for gate in raw_gates
        ):
            raise ValueError(f"YAML task {order} has malformed gates")
        gates = [
            {
                "upstream": gate.get("upstream"),
                "artifact_field": gate.get("artifact_field"),
                "op": gate.get("op"),
                "value": gate.get("value"),
            }
            for gate in raw_gates
        ]
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": task_number(task_id),
                "title": raw.get("title"),
                "deliverable": raw.get("deliverable"),
                "phase": raw.get("track"),
                "substrate": _substrate_class(prompt),
                "milestone": raw.get("milestone", document.get("milestone")),
                "gates": gates,
                "prompt": prompt,
                "required_block": _required_block(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only independently recheckable contract values in source rows."""

    return {
        key: task.get(key)
        for key in (
            "order",
            "id",
            "number",
            "title",
            "deliverable",
            "phase",
            "substrate",
            "milestone",
            "gates",
        )
    }


def _canonical_gates(gates: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Make gate comparisons independent of dictionary insertion order."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _gate_producer_rows(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Check that each YAML gate points backward to one declared field."""

    by_id = {task.get("id"): task for task in tasks}
    rows = []
    for consumer in tasks:
        for gate in consumer.get("gates", []):
            upstream = gate.get("upstream")
            field = gate.get("artifact_field")
            producer = by_id.get(upstream)
            checks = {
                "producer_exists": producer is not None,
                "producer_is_earlier": bool(
                    producer and int(producer["order"]) < int(consumer["order"])
                ),
                "same_milestone": bool(
                    producer and producer.get("milestone") == consumer.get("milestone") == MILESTONE
                ),
                "bare_field_declared": bool(
                    producer and _field_declared(str(producer.get("required_block", "")), field)
                ),
                "receipt_is_not_upstream": upstream != FIRST_TASK_ID,
            }
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "field": field,
                    "operator": gate.get("op"),
                    "value": gate.get("value"),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def evaluate_contract(markdown_text: str, yaml_document: object) -> dict[str, Any]:
    """Compare independent V633 sources and retain every local decision."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))
    task_rows = []
    gate_rows = []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        checks = {
            "id": bool(expected and observed and expected["id"] == observed["id"]),
            "title": bool(expected and observed and expected["title"] == observed["title"]),
            "deliverable": bool(
                expected and observed and expected["deliverable"] == observed["deliverable"]
            ),
            "phase": bool(expected and observed and expected["phase"] == observed["phase"]),
            "substrate": bool(
                expected and observed and expected["substrate"] == observed["substrate"]
            ),
            "gates": bool(
                expected
                and observed
                and _canonical_gates(expected["gates"]) == _canonical_gates(observed["gates"])
            ),
        }
        task_rows.append(
            {
                "order": index + 1,
                "task_id": (observed or expected or {}).get("id"),
                "markdown": _public_task(expected) if expected else None,
                "yaml": _public_task(observed) if observed else None,
                "checks": checks,
                "errors": [name for name, passed in checks.items() if not passed],
                "passed": all(checks.values()),
            }
        )
        gate_rows.append(
            {
                "order": index + 1,
                "task_id": (observed or expected or {}).get("id"),
                "expected": expected.get("gates") if expected else None,
                "observed": observed.get("gates") if observed else None,
                "passed": checks["id"] and checks["gates"],
            }
        )
    markdown_ids = [task["id"] for task in markdown_tasks]
    yaml_ids = [task["id"] for task in yaml_tasks]
    producer_rows = _gate_producer_rows(yaml_tasks)
    consumers = [
        task["id"]
        for task in yaml_tasks
        if any(gate.get("upstream") == FIRST_TASK_ID for gate in task["gates"])
    ]
    dependency_rows = [
        {
            "upstream": FIRST_TASK_ID,
            "expected_consumers": [],
            "observed_consumers": consumers,
            "passed": not consumers,
        }
    ]
    passed = all(
        (
            markdown["milestone"] == MILESTONE,
            roadmap["milestone"] == MILESTONE,
            len(markdown_tasks) == EXPECTED_TASK_COUNT,
            len(yaml_tasks) == EXPECTED_TASK_COUNT,
            markdown_ids == list(EXPECTED_ID_ORDER),
            yaml_ids == list(EXPECTED_ID_ORDER),
            all(row["passed"] for row in task_rows),
            all(row["passed"] for row in producer_rows),
            all(row["passed"] for row in dependency_rows),
        )
    )
    return {
        "passed": passed,
        "markdown_milestone": markdown["milestone"],
        "yaml_milestone": roadmap["milestone"],
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "task_contract_rows": task_rows,
        "gate_contract_rows": gate_rows,
        "gate_producer_rows": producer_rows,
        "receipt_dependency_rows": dependency_rows,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": markdown_ids,
        "observed_id_order": yaml_ids,
    }


def _select_yaml_authority(
    root: Path,
) -> tuple[Path | None, dict[str, Any] | None, bytes | None, list[dict[str, Any]]]:
    """Prefer a matching active roadmap, then a matching next roadmap."""

    candidates = []
    selected: tuple[Path, dict[str, Any], bytes] | None = None
    for relative in (ACTIVE_ROADMAP_PATH, NEXT_ROADMAP_PATH):
        path = root / relative
        try:
            content = path.read_bytes()
            document = _load_yaml_bytes(content, path)
            observed = document.get("milestone")
            matches = observed == MILESTONE
            error = None
        except (OSError, ValueError) as exc:
            content, document, observed, matches = b"", None, None, False
            error = f"{type(exc).__name__}: {exc}"
        candidates.append(
            {
                "check": f"yaml_candidate:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "field": "milestone",
                "expected_value": MILESTONE,
                "observed_value": observed if error is None else error,
                "available": matches,
                "blocking_external": False,
            }
        )
        if selected is None and matches and document is not None:
            selected = (relative, document, content)
    if selected is None:
        return None, None, None, candidates
    return *selected, candidates


def _check_writable_directory(path: Path) -> tuple[bool, str]:
    """Probe one output directory without leaving a scratch file."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".exp7179-write-"):
            pass
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "temporary_write_succeeded"


def _source_hashes(root: Path, authority: Path | None) -> dict[str, str | None]:
    """Hash contracts, code, tests, policy inputs, and the selected roadmap."""

    paths = list(SOURCE_PATHS)
    if authority is not None:
        paths.append(authority)
    hashes: dict[str, str | None] = {}
    for relative in dict.fromkeys(paths):
        try:
            hashes[str(relative)] = _sha256(root / relative)
        except OSError:
            hashes[str(relative)] = None
    return hashes


def _preconditions(
    root: Path,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], Path | None, dict[str, Any] | None, bytes | None]:
    """Record source, tool, directory, requirement, hash, and gate availability."""

    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        path: Path,
        expected: Any,
        observed: Any,
        available: bool,
        *,
        upstream: str | None = None,
        field: str | None = None,
        blocking_external: bool = True,
    ) -> None:
        rows.append(
            {
                "check": check,
                "path": str(path),
                "upstream": upstream or str(path),
                "field": field,
                "expected_value": expected,
                "observed_value": observed,
                "available": available,
                "blocking_external": blocking_external,
            }
        )

    authority, roadmap, yaml_bytes, candidate_rows = _select_yaml_authority(root)
    rows.extend(candidate_rows)
    observations = {row["path"]: row["observed_value"] for row in candidate_rows}
    add(
        "yaml_authority",
        root / (authority or ACTIVE_ROADMAP_PATH),
        f"one roadmap with milestone {MILESTONE}",
        str(authority) if authority else observations,
        authority is not None,
        upstream="research-roadmap.yaml|research-roadmap-next.yaml",
        field="milestone",
    )
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        add(
            f"source_bytes:{relative}",
            path,
            "readable_nonempty_bytes",
            path.stat().st_size if available else "missing_or_empty",
            available,
            field="bytes",
        )
    spec_path = root / SPEC_PATH
    try:
        spec_has_req = "REQ-REPORT-7179" in spec_path.read_text(encoding="utf-8")
    except OSError:
        spec_has_req = False
    add(
        "driving_requirement",
        spec_path,
        "REQ-REPORT-7179",
        "REQ-REPORT-7179" if spec_has_req else "missing",
        spec_has_req,
        field="REQ-*",
    )
    python_path = Path(sys.executable)
    add(
        "python_tool",
        python_path,
        "executable_file",
        str(python_path) if python_path.is_file() else "missing",
        python_path.is_file(),
        field="tool",
    )
    for label, directory in (
        ("result_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        available, observed = _check_writable_directory(directory)
        add(label, directory, "temporary_write_succeeds", observed, available, field="writable")
    hashes = _source_hashes(root, authority)
    add(
        "source_hashes",
        root,
        "sha256_for_every_available_input",
        {key: value for key, value in hashes.items() if value is None},
        all(value is not None for value in hashes.values()),
        field="sha256",
    )
    if roadmap is not None:
        yaml_tasks = parse_yaml_contract(roadmap)["tasks"]
        gate_rows = _gate_producer_rows(yaml_tasks)
        gate_passed = all(row["passed"] for row in gate_rows)
        add(
            "same_milestone_gate_fields_checked",
            root / authority if authority else root,
            "all gates name earlier same-milestone producers and bare fields",
            {"gate_count": len(gate_rows), "all_passed": gate_passed},
            True,
            field="gated_on",
            blocking_external=False,
        )
    return rows, authority, roadmap, yaml_bytes


def _raw_source_rows(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[dict[str, Any]]:
    """Preserve exact source bytes and prove each copy matches its source."""

    sources = (
        ("markdown", DESIGN_PATH, root / DESIGN_PATH, raw_dir / RAW_MARKDOWN_NAME),
        ("yaml", authority, root / authority, raw_dir / RAW_YAML_NAME),
    )
    rows = []
    for source_type, relative, source, destination in sources:
        content = yaml_bytes if source_type == "yaml" else source.read_bytes()
        _atomic_write_bytes(destination, content)
        source_hash = _sha256_bytes(content)
        raw_hash = _sha256(destination)
        rows.append(
            {
                "source_type": source_type,
                "source_path": str(relative),
                "raw_path": str(destination.relative_to(root)),
                "source_sha256": source_hash,
                "raw_sha256": raw_hash,
                "hash_matches": source_hash == raw_hash,
            }
        )
    return rows


def _v632_history(root: Path) -> dict[str, Any]:
    """Retain the prior mismatch without using it as V633 contract evidence."""

    path = root / PRIOR_ARTIFACT_PATH
    prior = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source_path": str(PRIOR_ARTIFACT_PATH),
        "source_sha256": _sha256(path),
        "honest_verdict": prior.get("honest_verdict"),
        "contract_complete_score": prior.get("v632_task_contract_conforms_score"),
        "expected_task_count": prior.get("expected_task_count"),
        "observed_task_count": prior.get("observed_task_count"),
        "expected_id_order": prior.get("expected_id_order"),
        "observed_id_order": prior.get("observed_id_order"),
        "historical_only": True,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every artifact field and preserve the prompt's exact principles."""

    principles = {
        field: f"The {field} evidence lets an auditor recheck the V633 contract receipt."
        for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every stored value except the checksum's own location."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _base_artifact(run_date: str, checkpoint_path: Path) -> dict[str, Any]:
    """Create a complete running checkpoint before checking prerequisites."""

    artifact: dict[str, Any] = {
        "schema": "carnot.exp7179.v633_contract_receipt.v1",
        "experiment_id": FIRST_TASK_ID,
        "status": "running",
        "field_principles": {},
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "host_identity": platform.node() or "unknown",
        "duration_s": 0.000001,
        "source_artifact_hashes": {},
        "rows": [],
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "task_contract_rows": [],
        "gate_contract_rows": [],
        "gate_producer_rows": [],
        "receipt_dependency_rows": [],
        "raw_source_rows": [],
        "validation_command_rows": [],
        "v632_mismatch_history": {},
        "yaml_authority_path": None,
        "checkpoint_path": str(checkpoint_path),
        "markdown_milestone": None,
        "yaml_milestone": None,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": [],
        "observed_id_order": [],
        "random_seed": RANDOM_SEED,
        "contract_complete_score": 0,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "preconditions_not_checked",
            "upstream": None,
            "field": None,
            "expected_value": "all_external_inputs_available",
            "observed_value": "not_checked",
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_v633_contract_receipt",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _set_time_and_checksum(artifact: dict[str, Any], started: float) -> None:
    """Measure elapsed time once and bind that measured value."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: dict[str, Any], started: float) -> None:
    """Persist live state only in the checkpoint tree."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def _summary(
    failed_check: str,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    field: Any = None,
) -> dict[str, Any]:
    """Use one diagnostic shape for blocked and disqualified results."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def _failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while local rows keep all failures."""

    if contract.get("markdown_milestone") != MILESTONE:
        return _summary(
            "markdown_milestone",
            MILESTONE,
            contract.get("markdown_milestone"),
            upstream=str(DESIGN_PATH),
            field="milestone",
        )
    if contract.get("yaml_milestone") != MILESTONE:
        return _summary(
            "yaml_milestone",
            MILESTONE,
            contract.get("yaml_milestone"),
            upstream=contract.get("yaml_authority_path"),
            field="milestone",
        )
    markdown_rows = contract.get("markdown_task_rows", [])
    yaml_rows = contract.get("yaml_task_rows", [])
    if len(markdown_rows) != EXPECTED_TASK_COUNT:
        return _summary(
            "markdown_task_count", EXPECTED_TASK_COUNT, len(markdown_rows), field="tasks"
        )
    if len(yaml_rows) != EXPECTED_TASK_COUNT:
        return _summary("yaml_task_count", EXPECTED_TASK_COUNT, len(yaml_rows), field="tasks")
    if contract.get("markdown_id_order") != list(EXPECTED_ID_ORDER):
        return _summary(
            "markdown_id_order",
            list(EXPECTED_ID_ORDER),
            contract.get("markdown_id_order"),
            field="id",
        )
    if contract.get("observed_id_order") != list(EXPECTED_ID_ORDER):
        return _summary(
            "yaml_id_order",
            list(EXPECTED_ID_ORDER),
            contract.get("observed_id_order"),
            field="id",
        )
    producer = next(
        (row for row in contract.get("gate_producer_rows", []) if not row.get("passed")), None
    )
    if producer:
        return _summary(
            "gate_producer_contract",
            True,
            producer.get("checks"),
            upstream=producer.get("upstream"),
            field=producer.get("field"),
        )
    dependency = next(
        (row for row in contract.get("receipt_dependency_rows", []) if not row.get("passed")), None
    )
    if dependency:
        return _summary(
            "receipt_has_downstream_gate",
            [],
            dependency.get("observed_consumers"),
            upstream=FIRST_TASK_ID,
            field="gated_on.upstream",
        )
    row = next(
        (row for row in contract.get("task_contract_rows", []) if not row.get("passed")),
        None,
    )
    if row is None:
        return _summary(
            "stored_contract_evidence",
            "one explicit failed contract row",
            "score failed without a failed row",
            field="task_contract_rows",
        )
    return _summary(
        f"task_contract_order_{row['order']}",
        row.get("markdown"),
        row.get("yaml"),
        upstream=row.get("task_id"),
        field=",".join(row.get("errors", [])),
    )


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute exact agreement only from stored contract evidence."""

    markdown = artifact.get("markdown_task_rows")
    yaml_rows = artifact.get("yaml_task_rows")
    task_rows = artifact.get("task_contract_rows")
    gate_rows = artifact.get("gate_contract_rows")
    producer_rows = artifact.get("gate_producer_rows")
    dependency_rows = artifact.get("receipt_dependency_rows")
    lists = (markdown, yaml_rows, task_rows, gate_rows, producer_rows, dependency_rows)
    if not all(isinstance(rows, list) for rows in lists):
        return 0
    if (
        artifact.get("markdown_milestone") != MILESTONE
        or artifact.get("yaml_milestone") != MILESTONE
    ):
        return 0
    if not all(
        len(rows) == EXPECTED_TASK_COUNT for rows in (markdown, yaml_rows, task_rows, gate_rows)
    ):
        return 0
    if [row.get("id") for row in markdown] != list(EXPECTED_ID_ORDER):
        return 0
    if [row.get("id") for row in yaml_rows] != list(EXPECTED_ID_ORDER):
        return 0
    if not all(
        row.get("passed") is True for row in task_rows + gate_rows + producer_rows + dependency_rows
    ):
        return 0
    return 1


def validate_artifact(artifact: object) -> list[str]:
    """Recompute fields, lifecycle state, verdict, diagnostics, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != value for field, value in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if artifact.get("status") != "complete":
        errors.append("status_not_complete")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if not isinstance(artifact.get("host_identity"), str) or not artifact["host_identity"]:
        errors.append("host_identity_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_s_invalid")
    precondition_rows = artifact.get("preconditions_checked")
    if not isinstance(precondition_rows, list):
        errors.append("preconditions_checked_invalid")
        precondition_rows = []
    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        errors.append("source_artifact_hashes_invalid")
    if artifact.get("expected_task_count") != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_invalid")
    if artifact.get("expected_id_order") != list(EXPECTED_ID_ORDER):
        errors.append("expected_id_order_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("rows") != artifact.get("task_contract_rows"):
        errors.append("rows_invalid")
    raw_rows = artifact.get("raw_source_rows")
    if raw_rows and (
        not isinstance(raw_rows, list)
        or any(
            row.get("hash_matches") is not True or row.get("source_sha256") != row.get("raw_sha256")
            for row in raw_rows
        )
    ):
        errors.append("raw_source_rows_invalid")
    history = artifact.get("v632_mismatch_history")
    if history and (
        not isinstance(history, Mapping)
        or history.get("historical_only") is not True
        or history.get("honest_verdict")
        != "complete_disqualified_v632_markdown_yaml_contract_mismatch"
    ):
        errors.append("v632_mismatch_history_invalid")
    command_rows = artifact.get("validation_command_rows")
    if not isinstance(command_rows, list) or [row.get("name") for row in command_rows] != list(
        VALIDATION_COMMAND_NAMES[: len(command_rows)]
    ):
        errors.append("validation_command_rows_invalid")
    score = _score_from_artifact(artifact)
    if artifact.get("contract_complete_score") != score:
        errors.append("contract_complete_score_invalid")
    failed_precondition = next(
        (
            row
            for row in precondition_rows
            if row.get("blocking_external") is True and row.get("available") is not True
        ),
        None,
    )
    if failed_precondition:
        expected_class = "blocked"
        expected_substrate = "blocked_no_run"
        expected_verdict = (
            BLOCKED_SOURCE_VERDICT
            if failed_precondition.get("check") == "yaml_authority"
            else BLOCKED_PREREQUISITE_VERDICT
        )
        expected_summary = _summary(
            failed_precondition.get("check"),
            failed_precondition.get("expected_value"),
            failed_precondition.get("observed_value"),
            upstream=failed_precondition.get("upstream"),
            field=failed_precondition.get("field"),
        )
    elif score:
        expected_class = "positive"
        expected_substrate = "aggregation"
        expected_verdict = POSITIVE_VERDICT
        expected_summary = {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_13_v633_task_contracts_exactly_agree",
            "observed_value": "all_13_v633_task_contracts_exactly_agree",
            "passed": True,
        }
    else:
        expected_class = "disqualified"
        expected_substrate = "aggregation"
        expected_verdict = DISQUALIFIED_VERDICT
        stored_summary = artifact.get("gate_check_summary")
        expected_summary = (
            stored_summary
            if not artifact.get("task_contract_rows")
            and isinstance(stored_summary, Mapping)
            and stored_summary.get("failed_check") == "contract_parse"
            else _failure_summary(artifact)
        )
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_invalid")
    if artifact.get("inference_substrate_class") != expected_substrate:
        errors.append("inference_substrate_class_invalid")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_invalid")
    if artifact.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _run_streaming_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    heartbeat_s: float,
    operation: str,
) -> dict[str, Any]:
    """Stream one child, emit heartbeats, and preserve a real timeout."""

    started = time.monotonic()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output: list[str] = []
    next_heartbeat = started + heartbeat_s
    timed_out = False
    while process.poll() is None:
        now = time.monotonic()
        if now - started >= timeout_s:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive native-call fallback.
                process.kill()
                process.wait()
            break
        events = selector.select(timeout=min(0.25, max(next_heartbeat - now, 0.001)))
        for key, _mask in events:
            line = key.fileobj.readline()
            if line:
                output.append(line)
                print(line, end="", flush=True)
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"[exp7179] heartbeat elapsed_s={now - started:.1f} completed_units=0 "
                f"operation={operation}",
                flush=True,
            )
            next_heartbeat = now + heartbeat_s
    remainder = process.stdout.read()
    if remainder:
        output.append(remainder)
        print(remainder, end="", flush=True)
    selector.close()
    exit_code = 124 if timed_out else int(process.returncode or 0)
    return {
        "exit_code": exit_code,
        "duration_s": max(round(time.monotonic() - started, 6), 0.000001),
        "output": "".join(output)[-12000:],
        "timed_out": timed_out,
    }


def _validation_commands(
    root: Path, artifact_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Use shipped validators and changed-code checks without changing them."""

    python = str(Path(sys.executable))
    roadmap = str(root / authority)
    artifact = str(artifact_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    artifact_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7179_v633_contract_receipt import validate_artifact;"
        f"errors=validate_artifact(json.loads(pathlib.Path({artifact!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        ("arc_generalization", [python, "-u", str(root / ARC_LINT_PATH), roadmap]),
        ("gate_audit", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        ("artifact", [python, "-u", "-c", artifact_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), artifact]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), artifact]),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        (
            "ruff",
            [
                str(root / ".venv/bin/ruff"),
                "check",
                str(root / MODULE_PATH),
                str(root / WRAPPER_PATH),
                str(root / TEST_PATH),
            ],
        ),
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        ("root_clutter", [python, "-u", str(root / ROOT_CLUTTER_PATH)]),
    )


def run_validation_commands(
    root: Path,
    checkpoint_path: Path,
    artifact: dict[str, Any],
    started: float,
) -> list[dict[str, Any]]:
    """Run bounded checks with streaming output and durable receipts."""

    rows = []
    authority = Path(str(artifact["yaml_authority_path"]))
    commands = _validation_commands(root, checkpoint_path, authority)
    for index, (name, argv) in enumerate(commands, 1):
        _checkpoint(checkpoint_path, artifact, started)
        _progress(3, "subprocess start", f"{index}/{len(commands)} {name}")
        receipt = _run_streaming_command(
            argv,
            cwd=root,
            timeout_s=300,
            heartbeat_s=60,
            operation=name,
        )
        row = {
            "name": name,
            "command": shlex.join(argv),
            **receipt,
            "passed": receipt["exit_code"] == 0,
        }
        rows.append(row)
        artifact["validation_command_rows"] = list(rows)
        _checkpoint(checkpoint_path, artifact, started)
        _progress(
            3, "subprocess end", f"{index}/{len(commands)} {name} exit_code={row['exit_code']}"
        )
    return rows


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    run_commands: bool = True,
) -> dict[str, Any]:
    """Build one positive, disqualified, or externally blocked receipt."""

    started = time.monotonic()
    _progress(0, "start", "initialize the checkpoint before prerequisite checks")
    artifact = _base_artifact(run_date, checkpoint_path)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", "running checkpoint persisted outside the terminal result path")

    _progress(1, "start", "check sources, milestone authority, tools, outputs, gates, and hashes")
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _source_hashes(root, authority)
    failed = next(
        (
            row
            for row in preconditions
            if row.get("blocking_external") is True and row.get("available") is not True
        ),
        None,
    )
    if failed is not None:
        artifact["status"] = "complete"
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = (
            BLOCKED_SOURCE_VERDICT
            if failed["check"] == "yaml_authority"
            else BLOCKED_PREREQUISITE_VERDICT
        )
        artifact["gate_check_summary"] = _summary(
            failed["check"],
            failed["expected_value"],
            failed["observed_value"],
            upstream=failed["upstream"],
            field=failed["field"],
        )
        _checkpoint(checkpoint_path, artifact, started)
        _progress(1, "end", f"externally blocked on {failed['check']}")
    else:
        assert authority is not None and roadmap is not None and yaml_bytes is not None
        artifact["raw_source_rows"] = _raw_source_rows(root, authority, yaml_bytes, raw_dir)
        artifact["v632_mismatch_history"] = _v632_history(root)
        artifact["inference_substrate_class"] = "aggregation"
        _checkpoint(checkpoint_path, artifact, started)
        _progress(1, "end", "all required inputs are hashed and raw source bytes are preserved")

        _progress(2, "start", "parse independent sources and compare 13 task contracts")
        try:
            contract = evaluate_contract((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            artifact["gate_check_summary"] = _summary(
                "contract_parse",
                "independently_parseable_markdown_and_yaml",
                f"{type(exc).__name__}: {exc}",
                upstream=f"{DESIGN_PATH}|{authority}",
                field="task_contract",
            )
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        else:
            for field in (
                "markdown_milestone",
                "yaml_milestone",
                "markdown_task_rows",
                "yaml_task_rows",
                "task_contract_rows",
                "gate_contract_rows",
                "gate_producer_rows",
                "receipt_dependency_rows",
                "expected_id_order",
                "markdown_id_order",
                "observed_id_order",
            ):
                artifact[field] = contract[field]
            artifact["rows"] = artifact["task_contract_rows"]
            artifact["observed_task_count"] = len(artifact["yaml_task_rows"])
            artifact["contract_complete_score"] = int(contract["passed"])
            if contract["passed"]:
                artifact["verdict_class"] = "positive"
                artifact["honest_verdict"] = POSITIVE_VERDICT
                artifact["gate_check_summary"] = {
                    "failed_check": None,
                    "upstream": None,
                    "field": None,
                    "expected_value": "all_13_v633_task_contracts_exactly_agree",
                    "observed_value": "all_13_v633_task_contracts_exactly_agree",
                    "passed": True,
                }
            else:
                artifact["verdict_class"] = "disqualified"
                artifact["honest_verdict"] = DISQUALIFIED_VERDICT
                artifact["gate_check_summary"] = _failure_summary(artifact)
        artifact["status"] = "complete"
        _checkpoint(checkpoint_path, artifact, started)
        _progress(2, "end", f"contract verdict is {artifact['verdict_class']}")

        if run_commands:
            _progress(3, "start", "run unchanged validators and changed-code checks")
            run_validation_commands(root, checkpoint_path, artifact, started)
            _progress(3, "end", "validation subprocess receipts are complete")

    _progress(4, "validation start", "recompute the terminal artifact")
    _set_time_and_checksum(artifact, started)
    errors = validate_artifact(artifact)
    _progress(4, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - the CLI must fail closed if its own validator rejects output.
        raise ValueError(f"invalid Exp7179 artifact: {errors}")
    _progress(5, "write start", "write the final artifact atomically")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    _progress(5, "write end", f"{artifact['verdict_class']} artifact complete")
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the fixed real execution date for this contract."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the receipt and return success for every valid terminal class."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    artifact = build_artifact(
        REPO_ROOT,
        args.date,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7179] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7179] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the required wrapper.
    raise SystemExit(main())
