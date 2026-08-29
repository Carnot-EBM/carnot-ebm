"""Build the V587 activation evidence contract without running an LLM.

The module treats the roadmap as data. It records every source, task, gate,
prior-failure, and model-policy check as a row so a blocked activation names
the exact defect. See REQ-REPORT-6729 and SCENARIO-REPORT-6729-*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
MILESTONE = "2026.08.587"
SOURCE_CUTOFF = "2026-08-29"
RANDOM_SEED = 6729
INFERENCE_SUBSTRATE = "source_receipts_and_local_method_preregistration_no_llm"
RESULT_PATH = Path("results/experiment_6729_v587_activation_evidence_contract.json")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
REFERENCES_PATH = Path("research-references.md")
COMPLETE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_GATES_PATH = Path("scripts/conductor_gates.py")
VERDICT_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
ADVERSARIAL_VERIFY_PATH = Path("scripts/adversarial_verify.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

LOCAL_INPUT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    DESIGN_PATH,
    NEXT_ROADMAP_PATH,
    ACTIVE_ROADMAP_PATH,
    REFERENCES_PATH,
    COMPLETE_PATH,
    CONDUCTOR_LOG_PATH,
    EXCLUSION_PATH,
    CONDUCTOR_GATES_PATH,
    VERDICT_LINT_PATH,
    ADVERSARIAL_VERIFY_PATH,
)

PRIMARY_SOURCE_IDS = (
    "2608.08786",
    "2607.17047",
    "2608.24082",
    "2607.20792",
    "2608.00220",
    "2608.01615",
    "2608.01612",
    "2608.03065",
)
SOURCE_TITLES = {
    "2608.08786": "SymDiag",
    "2607.17047": "Solver-Hard Is Not Model-Hard",
    "2608.24082": "PARTAB",
    "2607.20792": "Memoir",
    "2608.00220": "Verifier-Induced Support Reshaping",
    "2608.01615": "Thermalizing Stochastic Programs",
    "2608.01612": "A Framework for Stochastic Differentiable Programming",
    "2608.03065": "Parser Stack Classification",
}
SOURCE_EXPERIMENT_MAPPING = {
    "2608.08786": (
        "exp6734-sota-dual-encoding-proposal-corpus",
        "exp6735-oracle-distinct-diagnostic-energy",
        "exp6736-diagnostic-energy-localized-repair-ab",
    ),
    "2607.17047": (
        "exp6733-hardness-controlled-certificate-stream",
        "exp6734-sota-dual-encoding-proposal-corpus",
        "exp6735-oracle-distinct-diagnostic-energy",
    ),
    "2608.24082": (
        "exp6731-object-table-fetch-on-demand-ab",
        "exp6732-object-table-ab-cold-audit",
    ),
    "2607.20792": (
        "exp6737-transactional-constraint-memory-fixture",
        "exp6738-prospective-support-preserving-csl-ab",
        "exp6739-csl-support-durability-audit",
    ),
    "2608.00220": (
        "exp6738-prospective-support-preserving-csl-ab",
        "exp6739-csl-support-durability-audit",
    ),
    "2608.01615": ("exp6740-thermalizer-compiler-fidelity",),
    "2608.01612": ("exp6740-thermalizer-compiler-fidelity",),
    "2608.03065": ("exp6734-sota-dual-encoding-proposal-corpus",),
}

EXPECTED_TASK_IDS = (
    "exp6729-v587-activation-evidence-contract",
    "exp6730-arc-context-tool-preflight",
    "exp6731-object-table-fetch-on-demand-ab",
    "exp6732-object-table-ab-cold-audit",
    "exp6733-hardness-controlled-certificate-stream",
    "exp6734-sota-dual-encoding-proposal-corpus",
    "exp6735-oracle-distinct-diagnostic-energy",
    "exp6736-diagnostic-energy-localized-repair-ab",
    "exp6737-transactional-constraint-memory-fixture",
    "exp6738-prospective-support-preserving-csl-ab",
    "exp6739-csl-support-durability-audit",
    "exp6740-thermalizer-compiler-fidelity",
    "exp6741-v587-branch-disposition",
)
CAPSTONE_TASK_ID = "exp6741-v587-branch-disposition"
INFRASTRUCTURE_TASK_IDS = {
    "exp6729-v587-activation-evidence-contract",
    "exp6732-object-table-ab-cold-audit",
}
COMPARISON_TASK_IDS = {
    "exp6731-object-table-fetch-on-demand-ab",
    "exp6732-object-table-ab-cold-audit",
    "exp6734-sota-dual-encoding-proposal-corpus",
    "exp6735-oracle-distinct-diagnostic-energy",
    "exp6736-diagnostic-energy-localized-repair-ab",
    "exp6738-prospective-support-preserving-csl-ab",
    "exp6739-csl-support-durability-audit",
    "exp6740-thermalizer-compiler-fidelity",
    "exp6741-v587-branch-disposition",
}
LLM_TASK_IDS = {
    "exp6730-arc-context-tool-preflight",
    "exp6731-object-table-fetch-on-demand-ab",
    "exp6734-sota-dual-encoding-proposal-corpus",
    "exp6736-diagnostic-energy-localized-repair-ab",
    "exp6738-prospective-support-preserving-csl-ab",
}
ARC_TASK_IDS = {
    "exp6730-arc-context-tool-preflight",
    "exp6731-object-table-fetch-on-demand-ab",
    "exp6732-object-table-ab-cold-audit",
}
MANDATED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.8-27B-GGUF",
)
ARC_GENERATOR_MODEL = "unsloth/Qwen3.8-27B-GGUF"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

EXPECTED_GATES = {
    "exp6730-arc-context-tool-preflight": (
        ("exp6729-v587-activation-evidence-contract", "v587_contract_ready", "==", True),
    ),
    "exp6731-object-table-fetch-on-demand-ab": (
        ("exp6730-arc-context-tool-preflight", "arc_context_tool_preflight_ready", "==", True),
    ),
    "exp6732-object-table-ab-cold-audit": (
        ("exp6731-object-table-fetch-on-demand-ab", "object_table_ab_completed", "==", True),
    ),
    "exp6733-hardness-controlled-certificate-stream": (
        ("exp6729-v587-activation-evidence-contract", "v587_contract_ready", "==", True),
    ),
    "exp6734-sota-dual-encoding-proposal-corpus": (
        ("exp6733-hardness-controlled-certificate-stream", "hardness_stream_ready", "==", True),
    ),
    "exp6735-oracle-distinct-diagnostic-energy": (
        ("exp6734-sota-dual-encoding-proposal-corpus", "dual_encoding_corpus_ready", "==", True),
    ),
    "exp6736-diagnostic-energy-localized-repair-ab": (
        ("exp6735-oracle-distinct-diagnostic-energy", "heldout_reasoning_error_auroc", ">=", 0.65),
        ("exp6735-oracle-distinct-diagnostic-energy", "oracle_leakage_detected", "==", False),
    ),
    "exp6738-prospective-support-preserving-csl-ab": (
        ("exp6737-transactional-constraint-memory-fixture", "transaction_stream_ready", "==", True),
    ),
    "exp6739-csl-support-durability-audit": (
        ("exp6738-prospective-support-preserving-csl-ab", "csl_run_completed", "==", True),
    ),
    "exp6740-thermalizer-compiler-fidelity": (
        ("exp6729-v587-activation-evidence-contract", "v587_contract_ready", "==", True),
    ),
}
SUPPORTED_GATE_OPS = {"==", "!=", ">", "<", ">=", "<=", "exists", "in"}

REQUIRED_ARTIFACT_FIELDS = (
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_cutoff",
    "source_receipts",
    "task_contract_rows",
    "gate_contract_rows",
    "prior_failure_rows",
    "model_policy_rows",
    "v587_contract_ready",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
    "field_principles",
)
FIELD_PRINCIPLES = {
    "status": "The terminal state makes blocked activation visible to the conductor.",
    "inference_substrate": "The source-only substrate prevents false claims that a model was run.",
    "duration_s": "Monotonic wall time lets readers reject zero-time fabricated checks.",
    "random_seed": "An explicit unused seed keeps deterministic replay policy visible.",
    "reproducibility_checksum": "The checksum binds the checked inputs and derived contract rows.",
    "source_cutoff": "The dated cutoff prevents later source drift from changing this receipt.",
    "source_receipts": "One row per primary source ties each adopted idea to local experiments.",
    "task_contract_rows": "One row per planned task exposes missing, duplicate, or renamed work.",
    "gate_contract_rows": "Each gate must cite an upstream field with exact producer spelling.",
    "prior_failure_rows": "Reruns must name the old verdict, changed condition, and retirement rule.",
    "model_policy_rows": "LLM tasks must declare a current mandated local GGUF identity.",
    "v587_contract_ready": "This gate opens downstream work only after every contract check passes.",
    "gate_check_summary": "Blocked paths must name the failing check and observed value.",
    "verdict_class": "The closed class keeps readiness separate from a scientific positive.",
    "honest_verdict": "The terminal prefix makes blocked or null infrastructure status explicit.",
    "field_principles": "Principles document why each artifact field is required.",
    "preconditions_checked": "Input existence, YAML parse state, and source checks remain auditable.",
    "field_provenance": "Each field names the parser or reducer that produced it.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_hash(value: Any) -> str:
    """Hash one JSON-compatible value using the canonical encoding."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file or record its absence without raising."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _task_number(task_id: str) -> int:
    match = re.match(r"exp(\d+)-", task_id)
    if match is None:
        raise ValueError(f"invalid task id: {task_id}")  # pragma: no cover
    return int(match.group(1))


def _v587_section(text: str) -> str:
    match = re.search(
        r"## V587 Planner Refresh - 2026-08-29\n(?P<body>.*?)(?=\n## V\d+ Planner Refresh|\Z)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("V587 planner refresh section missing")  # pragma: no cover
    return match.group("body")


def parse_design_contract(text: str) -> JsonDict:
    """Parse the V587 proposal into phase, PRD-gap, and task rows."""

    tasks: list[JsonDict] = []
    phases: list[str] = []
    current_phase: str | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        phase_match = re.match(r"## (Phase \d+):\s*(.+)$", line)
        if phase_match:
            current_phase = phase_match.group(1)
            phases.append(current_phase)
            continue
        task_match = re.match(r"### Exp (\d+):\s*(.+)$", line)
        if task_match:
            number = int(task_match.group(1))
            task_id = f"exp{number}-{_slug(task_match.group(2))}"
            deliverable = _next_deliverable(lines, index)
            tasks.append(
                {
                    "order": len(tasks) + 1,
                    "number": number,
                    "task_id": task_id,
                    "title": task_match.group(2).strip(),
                    "phase": current_phase,
                    "deliverable": deliverable,
                    "is_capstone": number == 6741,
                    "is_infrastructure": number in (6729, 6732),
                }
            )
    prd_gaps = re.findall(r"^\|\s*\d+\s*\|\s*\*\*(.*?)\*\*\s*\|", text, re.MULTILINE)
    return {
        "tasks": tasks,
        "phases": list(dict.fromkeys(phases)),
        "prd_gaps": prd_gaps,
        "design_sha256": value_hash(text),
    }


def _slug(title: str) -> str:
    """Convert a proposal heading to the roadmap task-id suffix."""

    replacements = {
        "V587 activation and evidence contract": "v587-activation-evidence-contract",
        "Owned 32K context and code-carrying selfparse preflight": "arc-context-tool-preflight",
        "Object-table fetch-on-demand A/B": "object-table-fetch-on-demand-ab",
        "Cold object-table row and provenance audit": "object-table-ab-cold-audit",
        "Hardness-controlled exact certificate stream": "hardness-controlled-certificate-stream",
        "Three-family SOTA dual-encoding proposal corpus": "sota-dual-encoding-proposal-corpus",
        "Oracle-distinct diagnostic energy": "oracle-distinct-diagnostic-energy",
        "Diagnostic-energy localized repair A/B": "diagnostic-energy-localized-repair-ab",
        "Read-only episode and atomic commit fixture": "transactional-constraint-memory-fixture",
        "Prospective support-preserving self-learning A/B": "prospective-support-preserving-csl-ab",
        "Cold self-learning durability and poison audit": "csl-support-durability-audit",
        "Thermalizers-style factor-to-EBM compiler fidelity": "thermalizer-compiler-fidelity",
        "V587 branch disposition and PRD gap update": "v587-branch-disposition",
    }
    if title in replacements:
        return replacements[title]
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")  # pragma: no cover


def _next_deliverable(lines: Sequence[str], start: int) -> str:
    for line in lines[start + 1 :]:
        match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", line)
        if match:
            return match.group(1)
        if line.startswith("### Exp "):
            break
    raise ValueError("task deliverable missing")  # pragma: no cover


def _read_yaml_payload(path: Path) -> tuple[Any | None, JsonDict]:
    """Parse a YAML document and keep failures as row data."""

    if not path.is_file():
        return None, _failure(
            "precondition.yaml_parse", path.as_posix(), "parseable YAML", "missing", "file_missing"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover
        return None, _failure(
            "precondition.yaml_parse", path.as_posix(), "parseable YAML", str(exc), "yaml_error"
        )
    return payload, _failure(
        "precondition.yaml_parse", path.as_posix(), "parseable YAML", True, "parsed"
    )


def collect_precondition_rows(root: Path, require_local_files: bool) -> list[JsonDict]:
    """Record input-file existence and parse state before reduction."""

    rows: list[JsonDict] = []
    for relative in LOCAL_INPUT_PATHS:
        path = root / relative
        present = path.is_file()
        row = _failure(
            "precondition.local_file",
            relative.as_posix(),
            "present",
            "present" if present else "missing",
            "listed_input_file",
        )
        row["sha256"] = sha256_file(path)
        row["passed"] = present or not require_local_files
        rows.append(row)
    for relative in (ACTIVE_ROADMAP_PATH, NEXT_ROADMAP_PATH):
        _payload, row = _read_yaml_payload(root / relative)
        row["passed"] = row["observed_value"] is True or not require_local_files
        rows.append(row)
    rows.append(
        _failure(
            "precondition.no_source_lookup",
            "runtime",
            False,
            False,
            "local_receipts_only",
        )
    )
    rows[-1]["passed"] = True
    rows.append(
        _failure(
            "precondition.model_cache_required",
            "exp6729",
            False,
            False,
            "source_manifest_only",
        )
    )
    rows[-1]["passed"] = True
    return rows


def load_yaml_if_present(root: Path, relative: Path) -> Any | None:
    """Load YAML when available; missing files are represented by None."""

    path = root / relative
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def collect_source_receipts(root: Path, design: Mapping[str, Any]) -> list[JsonDict]:
    """Build one local receipt row for each V587 primary source."""

    section = _v587_section((root / REFERENCES_PATH).read_text(encoding="utf-8"))
    design_ids = {str(row["task_id"]) for row in design["tasks"]}
    rows: list[JsonDict] = []
    for arxiv_id in PRIMARY_SOURCE_IDS:
        block = _source_block(section, arxiv_id)
        mapping = list(SOURCE_EXPERIMENT_MAPPING[arxiv_id])
        adoption = _adoption_text(block)
        row = {
            "arxiv_id": arxiv_id,
            "title": SOURCE_TITLES[arxiv_id],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source_cutoff": SOURCE_CUTOFF,
            "date_or_cutoff": SOURCE_CUTOFF,
            "local_adoption_decision": adoption,
            "experiment_mapping": mapping,
            "source_path": REFERENCES_PATH.as_posix(),
            "section_sha256": value_hash(block),
            "url_present": f"https://arxiv.org/abs/{arxiv_id}" in block,
            "date_or_cutoff_present": True,
            "adoption_decision_present": bool(adoption),
            "experiment_mapping_present": bool(mapping)
            and all(task in design_ids for task in mapping),
        }
        row["passed"] = all(
            row[key]
            for key in (
                "url_present",
                "date_or_cutoff_present",
                "adoption_decision_present",
                "experiment_mapping_present",
            )
        )
        rows.append(row)
    return rows


def _source_block(section: str, arxiv_id: str) -> str:
    index = section.find(arxiv_id)
    if index < 0:
        return ""  # pragma: no cover
    start = section.rfind("\n- ", 0, index)
    end = section.find("\n- ", index + 1)
    if start < 0:
        start = 0
    if end < 0:
        end = len(section)
    return section[start:end]


def _adoption_text(block: str) -> str:
    match = re.search(r"Carnot hook:\s*(.+?)(?=\n-|$)", block, re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def extract_required_artifact_fields(prompt: str) -> list[str]:
    """Extract backtick-declared field names from the required field block."""

    if "REQUIRED ARTIFACT FIELDS" not in prompt:
        return []
    block = prompt.split("REQUIRED ARTIFACT FIELDS", 1)[1].split("Run command:", 1)[0]
    return list(dict.fromkeys(re.findall(r"`([a-z][a-z0-9_]*)`", block)))


def extract_model_specs(prompt: str) -> list[str]:
    """Read MODEL_SPECS GGUF identifiers without accepting prose elsewhere."""

    if "MODEL_SPECS:" not in prompt:
        return []
    block = prompt.split("MODEL_SPECS:", 1)[1].split("CONCRETE STEPS:", 1)[0]
    return list(dict.fromkeys(re.findall(r"`([^`]*-GGUF)`", block)))


def build_task_contract_rows(
    design: Mapping[str, Any],
    manifest_payload: Mapping[str, Any] | None,
    retired_ids: set[str],
) -> list[JsonDict]:
    """Create one V587 row per planned task and compare manifest details."""

    tasks = manifest_payload.get("tasks", []) if isinstance(manifest_payload, Mapping) else []
    manifest_by_id = {
        str(task.get("id")): dict(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") is not None
    }
    rows: list[JsonDict] = []
    for design_row in design["tasks"]:
        task_id = str(design_row["task_id"])
        task = manifest_by_id.get(task_id)
        prompt = str(task.get("prompt", "")) if task else ""
        required_fields = extract_required_artifact_fields(prompt)
        gates = task.get("gated_on", []) if task else []
        requires_edges = _as_list(task.get("requires") if task else [])
        checks = {
            "manifest_present": task is not None,
            "milestone": task.get("milestone") == MILESTONE if task else False,
            "deliverable": task.get("deliverable") == design_row["deliverable"] if task else False,
            "required_fields_include_verdict_class": "verdict_class" in required_fields,
            "required_fields_include_gate_check_summary": "gate_check_summary" in required_fields,
            "closed_verdict_class_declared": _closed_verdict_declared(prompt),
            "comparison_per_unit_rows": (
                task_id not in COMPARISON_TASK_IDS or (task or {}).get("per_unit_rows") is True
            ),
            "capstone_ungated": task_id != CAPSTONE_TASK_ID or not gates,
            "arc_boundary": _arc_boundary_ok(task_id, prompt),
            "task_id_not_retired": task_id not in retired_ids,
            "requires_no_retired_task": not any(edge in retired_ids for edge in requires_edges),
        }
        row = {
            "order": design_row["order"],
            "task_id": task_id,
            "phase": design_row["phase"],
            "track": task.get("track") if task else None,
            "design_deliverable": design_row["deliverable"],
            "manifest_deliverable": task.get("deliverable") if task else None,
            "required_artifact_fields": required_fields,
            "requires_edges": requires_edges,
            "gated_on_count": len(gates) if isinstance(gates, list) else None,
            "is_infrastructure": bool(design_row["is_infrastructure"]),
            "is_capstone": bool(design_row["is_capstone"]),
            "checks": checks,
        }
        row["passed"] = all(checks.values())
        rows.append(row)
    return rows


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _closed_verdict_declared(prompt: str) -> bool:
    return "verdict_class" in prompt and all(verdict in prompt for verdict in VERDICT_CLASSES)


def _arc_boundary_ok(task_id: str, prompt: str) -> bool:
    if task_id not in ARC_TASK_IDS:
        return True
    lower = prompt.lower()
    positive_forbidden = (
        "uses game source",
        "use game source",
        "uses source/bfs",
        "bfs per-game",
        "claims a level solve",
        "solved level",
        "solve_claim` (true",
    )
    return not any(marker in lower for marker in positive_forbidden)


def build_gate_contract_rows(manifest_payload: Mapping[str, Any] | None) -> list[JsonDict]:
    """Validate every manifest gate against the producer's exact field block."""

    tasks = manifest_payload.get("tasks", []) if isinstance(manifest_payload, Mapping) else []
    task_by_id = {
        str(task.get("id")): dict(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") is not None
    }
    rows: list[JsonDict] = []
    for task in task_by_id.values():
        consumer = str(task.get("id"))
        for gate in task.get("gated_on", []) or []:
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream"))
            producer = task_by_id.get(upstream)
            producer_fields = (
                extract_required_artifact_fields(str(producer.get("prompt", "")))
                if producer
                else []
            )
            expected = EXPECTED_GATES.get(consumer, ())
            gate_tuple = (
                upstream,
                str(gate.get("artifact_field")),
                str(gate.get("op")),
                gate.get("value"),
            )
            checks = {
                "upstream_exists": producer is not None,
                "producer_declares_exact_field": gate.get("artifact_field") in producer_fields,
                "operator_supported": gate.get("op") in SUPPORTED_GATE_OPS,
                "matches_expected_gate": gate_tuple in expected,
            }
            row = {
                "consumer_task_id": consumer,
                "upstream_task_id": upstream,
                "artifact_field": gate.get("artifact_field"),
                "op": gate.get("op"),
                "value": gate.get("value"),
                "producer_required_artifact_fields": producer_fields,
                "checks": checks,
            }
            row["passed"] = all(checks.values())
            rows.append(row)
    return rows


def build_prior_failure_rows(manifest_payload: Mapping[str, Any] | None) -> list[JsonDict]:
    """Return one row for each declared prior-failure rerun block."""

    tasks = manifest_payload.get("tasks", []) if isinstance(manifest_payload, Mapping) else []
    rows: list[JsonDict] = []
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, Mapping):
            continue
        for prior in task.get("prior_failures", []) or []:
            prior_map = prior if isinstance(prior, Mapping) else {}
            checks = {
                "experiment_id": bool(str(prior_map.get("experiment_id", "")).strip()),
                "verdict": bool(str(prior_map.get("verdict", "")).strip()),
                "addressed_by": bool(str(prior_map.get("addressed_by", "")).strip()),
                "retire_if_same_verdict": prior_map.get("retire_if_same_verdict") is True,
            }
            row = {
                "consumer_task_id": task.get("id"),
                "experiment_id": prior_map.get("experiment_id"),
                "verdict": prior_map.get("verdict"),
                "addressed_by": prior_map.get("addressed_by"),
                "retire_if_same_verdict": prior_map.get("retire_if_same_verdict"),
                "checks": checks,
            }
            row["passed"] = all(checks.values())
            rows.append(row)
    return rows


def build_model_policy_rows(manifest_payload: Mapping[str, Any] | None) -> list[JsonDict]:
    """Check each LLM task has a mandated model and ARC keeps Qwen3.8."""

    tasks = manifest_payload.get("tasks", []) if isinstance(manifest_payload, Mapping) else []
    task_by_id = {
        str(task.get("id")): dict(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") is not None
    }
    rows: list[JsonDict] = []
    for task_id in sorted(LLM_TASK_IDS, key=_task_number):
        task = task_by_id.get(task_id)
        specs = extract_model_specs(str(task.get("prompt", ""))) if task else []
        checks = {
            "task_present": task is not None,
            "model_specs_declared": bool(specs),
            "mandated_model_present": any(model in MANDATED_MODELS for model in specs),
            "arc_generator_pinned": task_id not in ARC_TASK_IDS or ARC_GENERATOR_MODEL in specs,
            "model_cache_required_for_exp6729": False,
        }
        row = {
            "task_id": task_id,
            "models_declared": specs,
            "mandated_models_present": [model for model in specs if model in MANDATED_MODELS],
            "arc_scored_generator": ARC_GENERATOR_MODEL if task_id in ARC_TASK_IDS else None,
            "model_cache_required": False,
            "checks": checks,
        }
        row["passed"] = all(
            value is False if key == "model_cache_required_for_exp6729" else value
            for key, value in checks.items()
        )
        rows.append(row)
    return rows


def retired_task_ids(root: Path) -> set[str]:
    """Extract retired experiment ids from the exclusion manifest."""

    payload = load_yaml_if_present(root, EXCLUSION_PATH) or {}
    retired: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for entry in payload.get(section, []) or []:
            if not isinstance(entry, Mapping):
                continue
            values: list[Any] = [entry.get("experiment_id"), entry.get("id")]
            values.extend(entry.get("experiment_ids", []) or [])
            for value in values:
                match = re.search(r"(exp\d+-[a-z0-9-]+)", str(value))
                if match:
                    retired.add(match.group(1))
    return retired


def audit_contract(
    root: Path,
    design: Mapping[str, Any],
    active_payload: Mapping[str, Any] | None,
    next_payload: Mapping[str, Any] | None,
    source_receipts: Sequence[Mapping[str, Any]],
    retired_ids: set[str] | None = None,
    precondition_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Reduce V587 activation evidence to rows and named failures."""

    del root
    retired_ids = retired_ids or set()
    task_rows = build_task_contract_rows(design, active_payload, retired_ids)
    gate_rows = build_gate_contract_rows(active_payload)
    prior_rows = build_prior_failure_rows(active_payload)
    model_rows = build_model_policy_rows(active_payload)
    failures: list[JsonDict] = []
    failures.extend(dict(row) for row in precondition_rows if row.get("passed") is not True)
    failures.extend(
        _failure(
            "source.receipt",
            str(row.get("arxiv_id")),
            True,
            row.get("passed"),
            "source_receipt_incomplete",
        )
        for row in source_receipts
        if row.get("passed") is not True
    )
    failures.extend(_design_failures(design))
    failures.extend(_manifest_identity_failures(active_payload, ACTIVE_ROADMAP_PATH))
    failures.extend(_manifest_identity_failures(next_payload, NEXT_ROADMAP_PATH))
    for row in task_rows:
        for name, passed in row["checks"].items():
            if passed is not True:
                check = (
                    "task.per_unit_rows" if name == "comparison_per_unit_rows" else f"task.{name}"
                )
                failures.append(
                    _failure(check, str(row["task_id"]), True, passed, "task_contract_mismatch")
                )
    for row in gate_rows:
        for name, passed in row["checks"].items():
            if passed is not True:
                check = (
                    "gate.producer_field"
                    if name == "producer_declares_exact_field"
                    else f"gate.{name}"
                )
                failures.append(
                    _failure(
                        check,
                        f"{row['consumer_task_id']}->{row['upstream_task_id']}",
                        True,
                        passed,
                        "gate_contract_mismatch",
                    )
                )
    for row in prior_rows:
        if row["passed"] is not True:
            failures.append(
                _failure(
                    "prior.failure_contract",
                    str(row["consumer_task_id"]),
                    True,
                    row["checks"],
                    "prior_failure_subfield_mismatch",
                )
            )
    for row in model_rows:
        if row["passed"] is not True:
            failures.append(
                _failure(
                    "model.policy",
                    str(row["task_id"]),
                    True,
                    row["checks"],
                    "model_policy_mismatch",
                )
            )
    return {
        "task_contract_rows": task_rows,
        "gate_contract_rows": gate_rows,
        "prior_failure_rows": prior_rows,
        "model_policy_rows": model_rows,
        "failures": failures,
        "passed": not failures,
    }


def _design_failures(design: Mapping[str, Any]) -> list[JsonDict]:
    tasks = design["tasks"]
    failures: list[JsonDict] = []
    if len(tasks) != 13:
        failures.append(
            _failure(
                "design.task_count",
                DESIGN_PATH.as_posix(),
                13,
                len(tasks),
                "wrong_design_task_count",
            )
        )
    if len(set(row["task_id"] for row in tasks)) != 13:
        failures.append(
            _failure(
                "design.task_ids_unique",
                DESIGN_PATH.as_posix(),
                13,
                len(set(row["task_id"] for row in tasks)),
                "duplicate_design_task_id",
            )
        )
    if len(set(row["deliverable"] for row in tasks)) != 13:
        failures.append(
            _failure(
                "design.deliverables_unique",
                DESIGN_PATH.as_posix(),
                13,
                len(set(row["deliverable"] for row in tasks)),
                "duplicate_design_deliverable",
            )
        )
    if len(design["phases"]) != 4:
        failures.append(
            _failure(
                "design.phase_count",
                DESIGN_PATH.as_posix(),
                4,
                len(design["phases"]),
                "phase_count_mismatch",
            )
        )
    if len(design["prd_gaps"]) != 3:
        failures.append(
            _failure(
                "design.prd_gap_count",
                DESIGN_PATH.as_posix(),
                3,
                len(design["prd_gaps"]),
                "prd_gap_count_mismatch",
            )
        )
    return failures


def _manifest_identity_failures(payload: Mapping[str, Any] | None, path: Path) -> list[JsonDict]:
    tasks = payload.get("tasks", []) if isinstance(payload, Mapping) else []
    ids = [str(task.get("id")) for task in tasks if isinstance(task, Mapping)]
    deliverables = [str(task.get("deliverable")) for task in tasks if isinstance(task, Mapping)]
    failures: list[JsonDict] = []
    if len(ids) != 13:
        failures.append(
            _failure(
                "manifest.task_count", path.as_posix(), 13, len(ids), "wrong_manifest_task_count"
            )
        )
    if len(set(ids)) != len(ids) or len(ids) != 13:
        failures.append(
            _failure(
                "manifest.task_ids_unique",
                path.as_posix(),
                13,
                len(set(ids)),
                "manifest_task_ids_not_unique_or_complete",
            )
        )
    if len(set(deliverables)) != len(deliverables) or len(deliverables) != 13:
        failures.append(
            _failure(
                "manifest.deliverables_unique",
                path.as_posix(),
                13,
                len(set(deliverables)),
                "manifest_deliverables_not_unique_or_complete",
            )
        )
    infra_count = sum(
        1
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") in INFRASTRUCTURE_TASK_IDS
    )
    if infra_count < 2:
        failures.append(
            _failure(
                "manifest.infrastructure_count",
                path.as_posix(),
                ">=2",
                infra_count,
                "too_few_infrastructure_tasks",
            )
        )
    return failures


def _failure(check: str, unit: str, expected: Any, observed: Any, reason: str) -> JsonDict:
    return {
        "check": check,
        "unit": unit,
        "expected_value": expected,
        "observed_value": observed,
        "reason": reason,
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash checked inputs and derived rows, excluding timing and prose."""

    stable_keys = (
        "status",
        "inference_substrate",
        "random_seed",
        "source_cutoff",
        "source_receipts",
        "task_contract_rows",
        "gate_contract_rows",
        "prior_failure_rows",
        "model_policy_rows",
        "v587_contract_ready",
        "gate_check_summary",
        "verdict_class",
        "field_principles",
        "preconditions_checked",
        "field_provenance",
    )
    return value_hash({key: payload.get(key) for key in stable_keys})


def _field_provenance() -> JsonDict:
    return {
        "inference_substrate": "constant from Exp6729 prompt",
        "duration_s": "time.monotonic elapsed seconds supplied by caller",
        "random_seed": "constant RANDOM_SEED; no random sampling performed",
        "reproducibility_checksum": "canonical hash from reproducibility_checksum()",
        "source_cutoff": "constant SOURCE_CUTOFF from V587 planning date",
        "source_receipts": "collect_source_receipts() over research-references.md",
        "task_contract_rows": "build_task_contract_rows() over design and active YAML",
        "gate_contract_rows": "build_gate_contract_rows() over manifest gated_on entries",
        "prior_failure_rows": "build_prior_failure_rows() over manifest prior_failures",
        "model_policy_rows": "build_model_policy_rows() over manifest MODEL_SPECS",
        "v587_contract_ready": "audit_contract() failure reduction",
        "gate_check_summary": "audit_contract() named failure rows",
        "verdict_class": "derived from v587_contract_ready",
        "honest_verdict": "derived terminal-prefixed summary",
        "field_principles": "constant FIELD_PRINCIPLES",
    }


def build_artifact(
    repo_root: Path,
    duration_s: float,
    active_payload: Mapping[str, Any] | None = None,
    next_payload: Mapping[str, Any] | None = None,
    require_local_files: bool = True,
) -> JsonDict:
    """Assemble the terminal V587 activation artifact."""

    design = parse_design_contract((repo_root / DESIGN_PATH).read_text(encoding="utf-8"))
    active = (
        active_payload
        if active_payload is not None
        else load_yaml_if_present(repo_root, ACTIVE_ROADMAP_PATH)
    )
    staged = (
        next_payload
        if next_payload is not None
        else load_yaml_if_present(repo_root, NEXT_ROADMAP_PATH)
    )
    preconditions = collect_precondition_rows(repo_root, require_local_files=require_local_files)
    source_receipts = collect_source_receipts(repo_root, design)
    audit = audit_contract(
        repo_root,
        design,
        active if isinstance(active, Mapping) else None,
        staged if isinstance(staged, Mapping) else None,
        source_receipts,
        retired_ids=retired_task_ids(repo_root),
        precondition_rows=preconditions,
    )
    ready = bool(audit["passed"])
    artifact: JsonDict = {
        "status": "complete_v587_activation_contract_ready"
        if ready
        else "blocked_v587_activation_contract",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_cutoff": SOURCE_CUTOFF,
        "source_receipts": source_receipts,
        "task_contract_rows": audit["task_contract_rows"],
        "gate_contract_rows": audit["gate_contract_rows"],
        "prior_failure_rows": audit["prior_failure_rows"],
        "model_policy_rows": audit["model_policy_rows"],
        "v587_contract_ready": ready,
        "gate_check_summary": audit["failures"],
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": (
            "complete_null: V587 activation contract is ready; this is source and manifest evidence only."
            if ready
            else "blocked_v587_activation_contract: one or more source, manifest, gate, prior, or model checks failed."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "field_provenance": _field_provenance(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return validation errors for a stored Exp6729 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("source_cutoff") != SOURCE_CUTOFF:
        errors.append("source_cutoff_mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if payload.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_unknown")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_missing")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    source_rows = payload.get("source_receipts")
    if not isinstance(source_rows, list) or len(source_rows) != len(PRIMARY_SOURCE_IDS):
        errors.append("source_receipt_count_mismatch")
    task_rows = payload.get("task_contract_rows")
    if not isinstance(task_rows, list) or len(task_rows) != len(EXPECTED_TASK_IDS):
        errors.append("task_row_count_mismatch")
    ready = payload.get("v587_contract_ready") is True
    gate_summary = payload.get("gate_check_summary")
    if ready:
        if gate_summary != []:
            errors.append("ready_gate_summary_nonempty")
        if payload.get("verdict_class") != "null":
            errors.append("ready_verdict_class_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("complete_null:"):
            errors.append("ready_honest_verdict_mismatch")
    else:
        if not gate_summary:
            errors.append("blocked_gate_summary_missing")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete JSON object and remove temporary files on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def synthetic_ready_manifest(repo_root: Path) -> JsonDict:
    """Build a complete in-memory manifest for fail-closed regression tests."""

    design = parse_design_contract((repo_root / DESIGN_PATH).read_text(encoding="utf-8"))
    tasks: list[JsonDict] = []
    for design_row in design["tasks"]:
        task_id = str(design_row["task_id"])
        fields = _synthetic_required_fields(task_id)
        task = {
            "id": task_id,
            "title": design_row["title"],
            "track": "infrastructure"
            if task_id in INFRASTRUCTURE_TASK_IDS
            else _track_for(task_id),
            "priority": "critical",
            "agent_type": "codex",
            "model": "gpt-5.5",
            "requires_gpu": task_id in LLM_TASK_IDS,
            "max_turns": 30,
            "estimated_wall_time_min": 30,
            "per_unit_rows": task_id in COMPARISON_TASK_IDS,
            "milestone": MILESTONE,
            "deliverable": design_row["deliverable"],
            "prompt": _synthetic_prompt(task_id, fields),
        }
        if task_id in EXPECTED_GATES:
            task["gated_on"] = [
                {"upstream": upstream, "artifact_field": field, "op": op, "value": value}
                for upstream, field, op, value in EXPECTED_GATES[task_id]
            ]
        if task_id != "exp6729-v587-activation-evidence-contract":
            task["prior_failures"] = [
                {
                    "experiment_id": f"exp{_task_number(task_id) - 10}-prior-scope",
                    "verdict": "blocked_previous_scope",
                    "addressed_by": "The V587 contract changes the upstream gate, corpus, or local method.",
                    "retire_if_same_verdict": True,
                }
            ]
        tasks.append(task)
    return {
        "milestone": MILESTONE,
        "milestone_title": "synthetic V587",
        "milestone_doc": DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _synthetic_required_fields(task_id: str) -> list[str]:
    common = [
        "inference_substrate",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verdict_class",
        "honest_verdict",
    ]
    produced = {
        "exp6729-v587-activation-evidence-contract": ["v587_contract_ready"],
        "exp6730-arc-context-tool-preflight": ["arc_context_tool_preflight_ready", "solve_claim"],
        "exp6731-object-table-fetch-on-demand-ab": [
            "object_table_ab_completed",
            "solve_claim",
            "rows",
        ],
        "exp6733-hardness-controlled-certificate-stream": ["hardness_stream_ready", "rows"],
        "exp6734-sota-dual-encoding-proposal-corpus": ["dual_encoding_corpus_ready", "rows"],
        "exp6735-oracle-distinct-diagnostic-energy": [
            "heldout_reasoning_error_auroc",
            "oracle_leakage_detected",
            "rows",
        ],
        "exp6737-transactional-constraint-memory-fixture": ["transaction_stream_ready"],
        "exp6738-prospective-support-preserving-csl-ab": ["csl_run_completed", "rows"],
    }
    return common + produced.get(task_id, ["rows"] if task_id in COMPARISON_TASK_IDS else [])


def _synthetic_prompt(task_id: str, fields: Sequence[str]) -> str:
    lines: list[str] = []
    if task_id in LLM_TASK_IDS:
        lines.append("MODEL_SPECS:")
        for model in _models_for_task(task_id):
            lines.append(f"- `{model}`: mandated local GGUF.")
    if task_id in ARC_TASK_IDS:
        lines.append("This ARC task makes no level solve claim and does not inspect game source.")
        lines.append("It forbids offline ground-truth BFS and per-game adapters.")
    lines.append("REQUIRED ARTIFACT FIELDS:")
    for field in fields:
        lines.append(f"`{field}`: principle-backed field.")
    lines.append(
        "verdict_class uses positive | circular_positive | null | blocked | disqualified | partial."
    )
    lines.append("Blocked paths require gate_check_summary.")
    lines.append(
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/synthetic.py"
    )
    return "\n".join(lines)


def _models_for_task(task_id: str) -> tuple[str, ...]:
    if task_id in ARC_TASK_IDS:
        return (ARC_GENERATOR_MODEL, "unsloth/Qwen3.6-35B-A3B-GGUF")
    if task_id == "exp6734-sota-dual-encoding-proposal-corpus":
        return (
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/gemma-4-26B-A4B-it-GGUF",
        )
    if task_id == "exp6738-prospective-support-preserving-csl-ab":
        return ("unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF")
    return ("unsloth/Qwen3.6-35B-A3B-GGUF",)


def _track_for(task_id: str) -> str:
    number = _task_number(task_id)
    if number in (6730, 6731):
        return "arc-agi3"
    if number in (6733, 6734):
        return "verifiable-reasoning"
    if number in (6735, 6736):
        return "energy-verification"
    if number in (6737, 6738, 6739):
        return "continuous-self-learning"
    if number == 6740:
        return "hardware-preparation"
    return "synthesis"


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be an object")  # pragma: no cover
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    if args.validate:
        errors = validate_artifact(_read_json(args.output))
        if errors:
            print(json.dumps({"status": "invalid", "errors": errors}, indent=2, sort_keys=True))
            return 1
        print(
            json.dumps(
                {"status": "valid", "path": args.output.as_posix()}, indent=2, sort_keys=True
            )
        )
        return 0

    start = time.monotonic()
    artifact = build_artifact(repo_root, duration_s=0.0)
    artifact["duration_s"] = time.monotonic() - start
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"status": "invalid", "errors": errors}, indent=2, sort_keys=True))
        return 1
    write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {"status": artifact["status"], "path": args.output.as_posix()}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
