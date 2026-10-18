"""Audit the V588 handoff contract without running a model.

The audit treats the active roadmap and design as data. It keeps binding state
in typed rows so a reader can see whether each prerequisite still controls
execution, not only whether matching words survived in prose.

Spec refs: REQ-REPORT-6742, REQ-HARNESS-008, SCENARIO-REPORT-6742-*.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
MILESTONE = "2026.08.588"
PLANNING_DATE = "20260829"
SOURCE_CUTOFF = "2026-08-29"
PROJECT_ROOT_LITERAL = "/home/ianblenke/github.com/ianblenke/carnot"
RANDOM_SEED = 6742
INFERENCE_SUBSTRATE = "activated_manifest_binding_contract_static_audit_no_llm"

RESULT_PATH = Path("results/experiment_6742_v588_handoff_contract_audit.json")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REFERENCES_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
AUDIT_GATES_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

PRECONDITION_PATHS = (
    ACTIVE_ROADMAP_PATH,
    DESIGN_PATH,
    REFERENCES_PATH,
    ROADMAP_SCHEMA_PATH,
    EXCLUSION_PATH,
)

EXPECTED_TASK_IDS = (
    "exp6742-v588-handoff-contract-audit",
    "exp6743-task-owned-phase-accelerator-canary",
    "exp6744-hardness-controlled-certificate-stream",
    "exp6745-sota-dual-encoding-proposal-corpus",
    "exp6746-oracle-distinct-diagnostic-energy",
    "exp6747-diagnostic-energy-localized-repair-ab",
    "exp6748-transactional-constraint-memory-fixture",
    "exp6749-prospective-support-preserving-csl-ab",
    "exp6750-csl-durability-support-poison-audit",
    "exp6751-thermalizer-factor-trajectory-fidelity",
    "exp6752-arc-code-carrying-tool-preflight",
    "exp6753-object-table-fetch-on-demand-ab",
    "exp6754-v588-branch-disposition",
)
CAPSTONE_TASK_ID = "exp6754-v588-branch-disposition"
HANDOFF_AUDIT_IDS = {
    "exp6742-v588-handoff-contract-audit",
    "exp6743-task-owned-phase-accelerator-canary",
}
INFRASTRUCTURE_MINIMUM = 2
MANDATED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.8-27B-GGUF",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
VERDICT_ENUM_TEXT = "positive | circular_positive | null | blocked | disqualified | partial"
SUPPORTED_GATE_OPS = {"==", "!=", ">", "<", ">=", "<=", "exists", "in"}

EXPECTED_GATES: dict[str, tuple[tuple[str, str, str, Any], ...]] = {
    "exp6745-sota-dual-encoding-proposal-corpus": (
        ("exp6744-hardness-controlled-certificate-stream", "hardness_stream_ready", "==", True),
    ),
    "exp6746-oracle-distinct-diagnostic-energy": (
        ("exp6745-sota-dual-encoding-proposal-corpus", "dual_encoding_corpus_ready", "==", True),
    ),
    "exp6747-diagnostic-energy-localized-repair-ab": (
        ("exp6746-oracle-distinct-diagnostic-energy", "heldout_reasoning_error_auroc", ">=", 0.65),
        ("exp6746-oracle-distinct-diagnostic-energy", "oracle_leakage_detected", "==", False),
    ),
    "exp6749-prospective-support-preserving-csl-ab": (
        ("exp6748-transactional-constraint-memory-fixture", "transaction_memory_ready", "==", True),
    ),
    "exp6750-csl-durability-support-poison-audit": (
        ("exp6749-prospective-support-preserving-csl-ab", "prospective_csl_completed", "==", True),
    ),
    "exp6753-object-table-fetch-on-demand-ab": (
        ("exp6752-arc-code-carrying-tool-preflight", "arc_context_tool_preflight_ready", "==", True),
    ),
}
TASK_ID_OVERRIDES = {
    "Activated handoff and binding-contract audit": "v588-handoff-contract-audit",
    "Task-owned phase and accelerator canary": "task-owned-phase-accelerator-canary",
    "Hardness-controlled exact certificate stream": "hardness-controlled-certificate-stream",
    "Three-family SOTA dual-encoding proposal corpus": "sota-dual-encoding-proposal-corpus",
    "Held-family oracle-distinct diagnostic energy": "oracle-distinct-diagnostic-energy",
    "Diagnostic-energy localized repair A/B": "diagnostic-energy-localized-repair-ab",
    "Read-only episode and atomic commit fixture": "transactional-constraint-memory-fixture",
    "Prospective support-preserving self-learning A/B": "prospective-support-preserving-csl-ab",
    "Cold durability, support, and poison audit": "csl-durability-support-poison-audit",
    "Thermalizers-style factor-to-trajectory compiler fidelity": "thermalizer-factor-trajectory-fidelity",
    "Owned 32K code-carrying ARC tool preflight": "arc-code-carrying-tool-preflight",
    "Live object-table fetch-on-demand A/B": "object-table-fetch-on-demand-ab",
    "V588 branch disposition and PRD gap update": "v588-branch-disposition",
}
PRIMARY_SOURCE_IDS = ("2608.27311", "2608.26753", "2608.24569")
SOURCE_TITLES = {
    "2608.27311": "HarnessLens",
    "2608.26753": "ABE-Ralph",
    "2608.24569": 'When "Must" Becomes "Maybe"',
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_cutoff",
    "rows",
    "binding_contract_rows",
    "task_count",
    "handoff_contract_preserved",
    "science_branches_independent_of_handoff_audit",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Explains why every required field exists, including this field.",
    "inference_substrate": "Declares that this is a static audit and not a model run.",
    "duration_s": "Records monotonic wall time so a zero-duration artifact is visible.",
    "random_seed": "Keeps reproducibility explicit even though this audit uses no sampling.",
    "reproducibility_checksum": "Binds input hashes and derived rows with canonical JSON.",
    "source_cutoff": "Freezes the dated planning window used by the V588 source refresh.",
    "rows": "Provides one auditable row for every task and every gate.",
    "binding_contract_rows": "Carries the typed binding tuple for every task and gate.",
    "task_count": "Records the observed active task count before reductions.",
    "handoff_contract_preserved": "States whether all hard handoff checks passed.",
    "science_branches_independent_of_handoff_audit": "Shows the audit is not a science gate.",
    "gate_check_summary": "Names failed checks and observed values for blocked outcomes.",
    "verdict_class": "Uses the closed outcome vocabulary required by the prompt.",
    "honest_verdict": "Gives a terminal-prefixed human-readable outcome.",
}
FIELD_RE = re.compile(r"(?:REQUIRED ARTIFACT FIELDS:\s*|,\s*|\band\s+)([a-z][a-z0-9_]+)\s*\(")
MODEL_RE = re.compile(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+-GGUF)")


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for replayable checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_hash(value: Any) -> str:
    """Hash one JSON-compatible value using canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file while representing absence as data."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _check(
    check: str,
    unit: str,
    expected: Any,
    observed: Any,
    reason: str,
    passed: bool,
) -> JsonDict:
    return {
        "check": check,
        "unit": unit,
        "expected_value": expected,
        "observed_value": observed,
        "reason": reason,
        "passed": passed,
    }


def _failure(check: str, unit: str, expected: Any, observed: Any, reason: str) -> JsonDict:
    return _check(check, unit, expected, observed, reason, False)


def _slug(title: str) -> str:
    return TASK_ID_OVERRIDES.get(title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"))


def _task_number(task_id: str) -> int:
    match = re.match(r"exp(\d+)-", task_id)
    if match is None:
        raise ValueError(f"invalid task id: {task_id}")
    return int(match.group(1))


def _v588_section(text: str) -> str:
    match = re.search(
        r"## V588 Planner Refresh - 2026-08-29\n(?P<body>.*?)(?=\n## V\d+ Planner Refresh|\Z)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("V588 planner refresh section missing")
    return match.group("body")


def _next_deliverable(lines: Sequence[str], start: int) -> str:
    for line in lines[start + 1 :]:
        match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", line)
        if match:
            return match.group(1)
        if line.startswith("### Exp "):
            break
    raise ValueError("task deliverable missing")


def parse_design_contract(text: str) -> JsonDict:
    """Parse the V588 proposal into milestone, phase, and task rows."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if milestone_match is None:
        raise ValueError("V588 design milestone missing")
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
            title = task_match.group(2).strip()
            tasks.append(
                {
                    "order": len(tasks) + 1,
                    "number": number,
                    "task_id": f"exp{number}-{_slug(title)}",
                    "title": title,
                    "phase": current_phase,
                    "deliverable": _next_deliverable(lines, index),
                    "is_capstone": number == 6754,
                }
            )
    return {
        "milestone": milestone_match.group(1),
        "tasks": tasks,
        "phases": list(dict.fromkeys(phases)),
        "design_sha256": value_hash(text),
    }


def _read_yaml_mapping(path: Path) -> tuple[Mapping[str, Any] | None, JsonDict]:
    if not path.is_file():
        return None, _failure(
            "precondition.yaml_parse",
            path.as_posix(),
            "parseable YAML mapping",
            "missing",
            "file_missing",
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, _failure(
            "precondition.yaml_parse",
            path.as_posix(),
            "parseable YAML mapping",
            str(exc),
            "yaml_error",
        )
    passed = isinstance(payload, Mapping)
    return (
        payload if passed else None,
        _check(
            "precondition.yaml_parse",
            path.as_posix(),
            "parseable YAML mapping",
            type(payload).__name__,
            "parsed" if passed else "top_level_not_mapping",
            passed,
        ),
    )


def _load_schema_module(root: Path) -> Any:
    path = root / ROADMAP_SCHEMA_PATH
    spec = importlib.util.spec_from_file_location("_exp6742_roadmap_schema", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _roadmap_schema_row(root: Path, active_payload: Mapping[str, Any] | None) -> JsonDict:
    try:
        module = _load_schema_module(root)
        module.Roadmap.model_validate(active_payload)
    except Exception as exc:
        return _failure(
            "precondition.roadmap_schema_parse",
            ROADMAP_SCHEMA_PATH.as_posix(),
            "Roadmap validates active manifest",
            str(exc),
            "schema_validation_failed",
        )
    return _check(
        "precondition.roadmap_schema_parse",
        ROADMAP_SCHEMA_PATH.as_posix(),
        "Roadmap validates active manifest",
        True,
        "schema_valid",
        True,
    )


def collect_precondition_rows(
    root: Path,
) -> tuple[list[JsonDict], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Read required static inputs and record any missing or bad parse."""

    rows: list[JsonDict] = []
    for relative in PRECONDITION_PATHS:
        path = root / relative
        present = path.is_file()
        row = _check(
            "precondition.local_file",
            relative.as_posix(),
            "present",
            "present" if present else "missing",
            "required_handoff_input",
            present,
        )
        row["sha256"] = sha256_file(path)
        rows.append(row)

    active_payload, active_row = _read_yaml_mapping(root / ACTIVE_ROADMAP_PATH)
    rows.append(active_row)
    rows.append(_roadmap_schema_row(root, active_payload))

    design_payload: Mapping[str, Any] | None = None
    try:
        design_payload = parse_design_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
        rows.append(
            _check(
                "precondition.design_parse",
                DESIGN_PATH.as_posix(),
                "parseable V588 design",
                True,
                "design_parsed",
                True,
            )
        )
    except Exception as exc:
        rows.append(
            _failure(
                "precondition.design_parse",
                DESIGN_PATH.as_posix(),
                "parseable V588 design",
                str(exc),
                "design_parse_failed",
            )
        )

    try:
        section = _v588_section((root / REFERENCES_PATH).read_text(encoding="utf-8"))
        rows.append(
            _check(
                "precondition.source_refresh",
                REFERENCES_PATH.as_posix(),
                "V588 Planner Refresh - 2026-08-29",
                bool(section),
                "source_refresh_found",
                True,
            )
        )
    except Exception as exc:
        rows.append(
            _failure(
                "precondition.source_refresh",
                REFERENCES_PATH.as_posix(),
                "V588 Planner Refresh - 2026-08-29",
                str(exc),
                "source_refresh_missing",
            )
        )

    _exclusion_payload, exclusion_row = _read_yaml_mapping(root / EXCLUSION_PATH)
    exclusion_row["check"] = "precondition.exclusion_manifest_parse"
    rows.append(exclusion_row)
    rows.append(
        _check(
            "precondition.no_next_roadmap_fallback",
            "research-roadmap-next.yaml",
            "not consulted",
            "not consulted",
            "active_manifest_only",
            True,
        )
    )
    return rows, active_payload, design_payload


def _source_block(section: str, arxiv_id: str) -> str:
    index = section.find(arxiv_id)
    if index < 0:
        return ""
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


def collect_source_receipts(root: Path, design: Mapping[str, Any]) -> list[JsonDict]:
    """Build one source row for the three V588 handoff papers."""

    section = _v588_section((root / REFERENCES_PATH).read_text(encoding="utf-8"))
    rows: list[JsonDict] = []
    for arxiv_id in PRIMARY_SOURCE_IDS:
        block = _source_block(section, arxiv_id)
        adoption = _adoption_text(block)
        row = {
            "arxiv_id": arxiv_id,
            "title": SOURCE_TITLES[arxiv_id],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source_cutoff": SOURCE_CUTOFF,
            "source_path": REFERENCES_PATH.as_posix(),
            "section_sha256": value_hash(block),
            "url_present": f"https://arxiv.org/abs/{arxiv_id}" in block,
            "adoption_decision_present": bool(adoption),
            "adoption_decision": adoption,
            "mapped_to_v588": "Carnot hook:" in block,
            "design_milestone": design.get("milestone"),
        }
        row["passed"] = all(
            row[key] for key in ("url_present", "adoption_decision_present", "mapped_to_v588")
        )
        rows.append(row)
    return rows


def extract_required_artifact_fields(prompt: str) -> list[str]:
    """Extract prose-declared field names from a REQUIRED ARTIFACT FIELDS block."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return []
    block = prompt.split(marker, 1)[1].split("Run command:", 1)[0]
    return list(dict.fromkeys(FIELD_RE.findall(marker + block)))


def extract_model_specs(prompt: str) -> list[str]:
    """Read local GGUF model identifiers only from the MODEL_SPECS block."""

    marker = "MODEL_SPECS:"
    if marker not in prompt:
        return []
    block = prompt.split(marker, 1)[1].split("CONCRETE STEPS:", 1)[0]
    return list(dict.fromkeys(MODEL_RE.findall(block)))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _observed_literal(prompt: str, literal: str, placeholder: str) -> str:
    if literal in prompt:
        return literal
    if placeholder in prompt:
        return placeholder
    return "missing"


def _prompt_contract(task_id: str, prompt: str) -> tuple[JsonDict, list[JsonDict]]:
    checks = {
        "context_section": "CONTEXT:" in prompt,
        "existing_code_section": "EXISTING CODE TO READ FIRST:" in prompt,
        "task_section": "TASK:" in prompt,
        "concrete_steps_section": "CONCRETE STEPS:" in prompt,
        "project_root_literal": PROJECT_ROOT_LITERAL in prompt,
        "planning_date_literal": PLANNING_DATE in prompt,
        "run_command": "Run command:" in prompt,
        "prohibition_no_push": "Do NOT push." in prompt,
        "prohibition_no_research_conductor": "Do NOT modify scripts/research_conductor.py." in prompt,
        "verdict_class_enum_exact": VERDICT_ENUM_TEXT in _normalize_text(prompt),
    }
    observed = {
        "project_root_literal": _observed_literal(prompt, PROJECT_ROOT_LITERAL, "{project_root}"),
        "planning_date_literal": _observed_literal(prompt, PLANNING_DATE, "{date}"),
    }
    failures = [
        _failure(
            f"prompt.{name}",
            task_id,
            expected,
            observed.get(name, checks[name]),
            "prompt_contract_mismatch",
        )
        for name, expected in (
            ("context_section", True),
            ("existing_code_section", True),
            ("task_section", True),
            ("concrete_steps_section", True),
            ("project_root_literal", PROJECT_ROOT_LITERAL),
            ("planning_date_literal", PLANNING_DATE),
            ("run_command", True),
            ("prohibition_no_push", True),
            ("prohibition_no_research_conductor", True),
            ("verdict_class_enum_exact", VERDICT_ENUM_TEXT),
        )
        if checks[name] is not True
    ]
    return checks, failures


def _block_after(prompt: str, start_marker: str, end_markers: Sequence[str]) -> str:
    if start_marker not in prompt:
        return ""
    text = prompt.split(start_marker, 1)[1]
    positions = [text.find(marker) for marker in end_markers if text.find(marker) >= 0]
    if positions:
        text = text[: min(positions)]
    return _normalize_text(text)


def _first_sentence_with(prompt: str, words: Sequence[str]) -> str:
    normalized = _normalize_text(prompt)
    sentences = re.split(r"(?<=[.])\s+", normalized)
    for sentence in sentences:
        lower = sentence.lower()
        if any(word in lower for word in words):
            return sentence[:300]
    return ""


def _binding_task_fields(task: Mapping[str, Any]) -> JsonDict:
    prompt = str(task.get("prompt") or "")
    gates = task.get("gated_on") or []
    prerequisite_value = (
        [
            {
                "upstream": gate.get("upstream"),
                "artifact_field": gate.get("artifact_field"),
                "op": gate.get("op"),
                "value": gate.get("value"),
            }
            for gate in gates
            if isinstance(gate, Mapping)
        ]
        if gates
        else _block_after(prompt, "0. PRECONDITIONS:", ("1.", "2."))
    )
    consequence = _first_sentence_with(
        prompt,
        (
            "only when",
            "only if",
            "does not gate",
            "requires",
            "adoption requires",
            "positive credit",
        ),
    )
    if not consequence:
        consequence = (
            "consumer waits on its declared conductor gates"
            if gates
            else "ungated task executes under its own preconditions"
        )
    return {
        "prerequisite": {"type": "task_precondition_or_gate", "value": prerequisite_value},
        "authority": {
            "type": "declared_verifier_or_schema_authority",
            "value": _first_sentence_with(
                prompt,
                ("authority", "exact", "recompute", "verify", "validate", "certificate"),
            ),
        },
        "fallback": {
            "type": "task_owned_blocked_artifact",
            "value": _first_sentence_with(prompt, ("complete_blocked", "on failure", "if unavailable")),
        },
        "execution_consequence": {
            "type": "readiness_or_gate_consequence",
            "value": consequence,
        },
        "blocked_artifact_behavior": {
            "type": "failed_check_and_observed_value",
            "value": _first_sentence_with(prompt, ("gate_check_summary", "observed value")),
        },
        "model_role": {
            "agent_type": task.get("agent_type"),
            "model": task.get("model"),
            "requires_gpu": task.get("requires_gpu"),
            "model_specs": extract_model_specs(prompt),
        },
        "claim_boundary": {
            "type": "declared_scope_limit",
            "value": _first_sentence_with(
                prompt,
                ("claim", "does not", "do not", "not a", "not infer", "no "),
            ),
        },
    }


def _binding_fields_present(binding: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "prerequisite_preserved": bool(binding["prerequisite"]["value"]),
        "authority_preserved": bool(binding["authority"]["value"]),
        "fallback_preserved": bool(binding["fallback"]["value"]),
        "execution_consequence_preserved": bool(binding["execution_consequence"]["value"]),
        "blocked_artifact_behavior_preserved": bool(binding["blocked_artifact_behavior"]["value"]),
        "model_role_preserved": bool(binding["model_role"]["agent_type"])
        and bool(binding["model_role"]["model"]),
        "claim_boundary_preserved": bool(binding["claim_boundary"]["value"]),
    }


def _task_by_id(payload: Mapping[str, Any] | None) -> dict[str, JsonDict]:
    tasks = payload.get("tasks", []) if isinstance(payload, Mapping) else []
    return {
        str(task.get("id")): dict(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") is not None
    }


def build_task_binding_rows(
    design: Mapping[str, Any],
    manifest_payload: Mapping[str, Any] | None,
    retired_ids: set[str],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Create one task binding row for every designed V588 task."""

    by_id = _task_by_id(manifest_payload)
    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    minimal_fields = {
        "field_principles",
        "inference_substrate",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verdict_class",
        "honest_verdict",
    }
    for design_row in design["tasks"]:
        task_id = str(design_row["task_id"])
        task = by_id.get(task_id)
        prompt = str(task.get("prompt", "")) if task else ""
        required_fields = extract_required_artifact_fields(prompt)
        prompt_checks, prompt_failures = _prompt_contract(task_id, prompt)
        gates = task.get("gated_on", []) if task else []
        expected_gates = EXPECTED_GATES.get(task_id, ())
        prior_failures = task.get("prior_failures", []) if task else []
        binding = _binding_task_fields(task or {})
        binding_checks = _binding_fields_present(binding)
        is_llm = _is_llm_task(task or {})
        model_specs = binding["model_role"]["model_specs"]
        checks = {
            "manifest_present": task is not None,
            "milestone": task.get("milestone") == MILESTONE if task else False,
            "deliverable": task.get("deliverable") == design_row["deliverable"] if task else False,
            "deliverable_json": str(task.get("deliverable", "")).startswith("results/")
            and str(task.get("deliverable", "")).endswith(".json")
            if task
            else False,
            "required_fields_present": minimal_fields <= set(required_fields),
            "prompt_contract": all(prompt_checks.values()),
            "expected_gate_count": len(gates) == len(expected_gates) if isinstance(gates, list) else False,
            "capstone_ungated": task_id != CAPSTONE_TASK_ID or not gates,
            "prior_failure_present": isinstance(prior_failures, list) and bool(prior_failures),
            "task_id_not_retired": task_id not in retired_ids,
            "requires_no_retired_task": not any(
                edge in retired_ids for edge in _as_list(task.get("requires") if task else None)
            ),
            "per_unit_rows": task.get("per_unit_rows") is True if task else False,
            "model_role": (not is_llm) or any(model in MANDATED_MODELS for model in model_specs),
            **binding_checks,
        }
        row = {
            "row_type": "task",
            "item_id": task_id,
            "task_id": task_id,
            "order": design_row["order"],
            "phase": design_row["phase"],
            "track": task.get("track") if task else None,
            "design_deliverable": design_row["deliverable"],
            "manifest_deliverable": task.get("deliverable") if task else None,
            "required_artifact_fields": required_fields,
            "gated_on_count": len(gates) if isinstance(gates, list) else None,
            "checks": checks,
            **binding,
        }
        row["operationally_preserved"] = all(checks.values())
        rows.append(row)
        failures.extend(prompt_failures)
        for name, passed in checks.items():
            if passed is not True and name not in prompt_checks:
                check = "task.per_unit_rows" if name == "per_unit_rows" else f"task.{name}"
                failures.append(
                    _failure(check, task_id, True, passed, "task_binding_contract_mismatch")
                )
    return rows, failures


def _is_llm_task(task: Mapping[str, Any]) -> bool:
    prompt = str(task.get("prompt") or "")
    return bool(task.get("requires_gpu")) or "MODEL_SPECS:" in prompt


def build_gate_binding_rows(
    manifest_payload: Mapping[str, Any] | None,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Validate every active gate against exact upstream field declarations."""

    by_id = _task_by_id(manifest_payload)
    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    for consumer_id in sorted(by_id, key=_task_number):
        task = by_id[consumer_id]
        for gate in task.get("gated_on", []) or []:
            if not isinstance(gate, Mapping):
                continue
            upstream_id = str(gate.get("upstream"))
            producer = by_id.get(upstream_id)
            producer_fields = (
                extract_required_artifact_fields(str(producer.get("prompt", ""))) if producer else []
            )
            gate_tuple = (
                upstream_id,
                str(gate.get("artifact_field")),
                str(gate.get("op")),
                gate.get("value"),
            )
            expected_gates = EXPECTED_GATES.get(consumer_id, ())
            consumer_prompt_checks, _prompt_failures = _prompt_contract(
                consumer_id, str(task.get("prompt") or "")
            )
            binding = {
                "prerequisite": {
                    "type": "upstream_artifact_field",
                    "upstream": upstream_id,
                    "artifact_field": gate.get("artifact_field"),
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                },
                "authority": {
                    "type": "upstream_required_artifact_field",
                    "producer_required_artifact_fields": producer_fields,
                },
                "fallback": {
                    "type": "conductor_gate_block",
                    "value": "write blocked artifact with exact failed gate",
                },
                "execution_consequence": {
                    "type": "consumer_dispatch_block",
                    "value": "consumer cannot run until the gate predicate passes",
                },
                "blocked_artifact_behavior": {
                    "type": "failed_check_and_observed_value",
                    "value": "gate_check_summary records failed upstream field and observed value",
                },
                "model_role": {
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "requires_gpu": task.get("requires_gpu"),
                    "model_specs": extract_model_specs(str(task.get("prompt") or "")),
                },
                "claim_boundary": _binding_task_fields(task)["claim_boundary"],
            }
            checks = {
                "upstream_exists": producer is not None,
                "producer_declares_exact_field": gate.get("artifact_field") in producer_fields,
                "operator_supported": gate.get("op") in SUPPORTED_GATE_OPS,
                "matches_design_gate": gate_tuple in expected_gates,
                "consumer_prompt_contract": all(consumer_prompt_checks.values()),
            }
            row = {
                "row_type": "gate",
                "item_id": f"{consumer_id}->{upstream_id}.{gate.get('artifact_field')}",
                "consumer_task_id": consumer_id,
                "upstream_task_id": upstream_id,
                "artifact_field": gate.get("artifact_field"),
                "op": gate.get("op"),
                "value": gate.get("value"),
                "checks": checks,
                **binding,
            }
            row["operationally_preserved"] = all(checks.values())
            rows.append(row)
            for name, passed in checks.items():
                if passed is not True:
                    check = (
                        "gate.producer_field"
                        if name == "producer_declares_exact_field"
                        else f"gate.{name}"
                    )
                    failures.append(
                        _failure(
                            check,
                            f"{consumer_id}->{upstream_id}",
                            True,
                            passed,
                            "gate_binding_contract_mismatch",
                        )
                    )
    return rows, failures


def build_prior_failure_rows(manifest_payload: Mapping[str, Any] | None) -> tuple[list[JsonDict], list[JsonDict]]:
    """Return one row for each prior-failure block and its four required fields."""

    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    tasks = manifest_payload.get("tasks", []) if isinstance(manifest_payload, Mapping) else []
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
                "task_id": task.get("id"),
                "experiment_id": prior_map.get("experiment_id"),
                "verdict": prior_map.get("verdict"),
                "addressed_by": prior_map.get("addressed_by"),
                "retire_if_same_verdict": prior_map.get("retire_if_same_verdict"),
                "checks": checks,
            }
            row["passed"] = all(checks.values())
            rows.append(row)
            if row["passed"] is not True:
                failures.append(
                    _failure(
                        "prior.failure_contract",
                        str(task.get("id")),
                        True,
                        checks,
                        "prior_failure_subfield_mismatch",
                    )
                )
    return rows, failures


def build_model_policy_rows(manifest_payload: Mapping[str, Any] | None) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check local GGUF declarations and Codex formulaic routing."""

    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    for task_id, task in sorted(_task_by_id(manifest_payload).items(), key=lambda item: _task_number(item[0])):
        if not _is_llm_task(task):
            continue
        prompt = str(task.get("prompt") or "")
        specs = extract_model_specs(prompt)
        checks = {
            "agent_type_codex": task.get("agent_type") == "codex",
            "formulaic_codex_model_allowed": task.get("model") in {"gpt-5.5", "gpt-5.6-sol"},
            "model_specs_declared": bool(specs),
            "mandated_local_gguf": any(model in MANDATED_MODELS for model in specs),
        }
        row = {
            "task_id": task_id,
            "agent_type": task.get("agent_type"),
            "model": task.get("model"),
            "models_declared": specs,
            "mandated_models_present": [model for model in specs if model in MANDATED_MODELS],
            "checks": checks,
        }
        row["passed"] = all(checks.values())
        rows.append(row)
        if row["passed"] is not True:
            failures.append(_failure("model.policy", task_id, True, checks, "model_policy_mismatch"))
    return rows, failures


def retired_task_ids(root: Path) -> set[str]:
    """Extract retired experiment IDs as active-roadmap task IDs when possible."""

    payload = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    active_by_number = {str(_task_number(task_id)): task_id for task_id in EXPECTED_TASK_IDS}
    retired: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for entry in payload.get(section, []) or []:
            if not isinstance(entry, Mapping):
                continue
            values: list[Any] = [entry.get("experiment_id"), entry.get("id")]
            values.extend(entry.get("experiment_ids", []) or [])
            for value in values:
                text = str(value)
                if text.isdigit() and text in active_by_number:
                    retired.add(active_by_number[text])
                match = re.search(r"(exp\d+-[a-z0-9-]+)", text)
                if match:
                    retired.add(match.group(1))
    return retired


def _manifest_identity_failures(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    tasks = payload.get("tasks", []) if isinstance(payload, Mapping) else []
    ids = [str(task.get("id")) for task in tasks if isinstance(task, Mapping)]
    deliverables = [str(task.get("deliverable")) for task in tasks if isinstance(task, Mapping)]
    tracks = [str(task.get("track")) for task in tasks if isinstance(task, Mapping)]
    by_id = _task_by_id(payload)
    failures: list[JsonDict] = []
    checks = (
        ("manifest.task_count", 13, len(ids), len(ids) == 13, "wrong_manifest_task_count"),
        (
            "manifest.task_ids_unique",
            13,
            len(set(ids)),
            len(set(ids)) == len(ids) == 13,
            "manifest_task_ids_not_unique_or_complete",
        ),
        (
            "manifest.deliverables_unique",
            13,
            len(set(deliverables)),
            len(set(deliverables)) == len(deliverables) == 13
            and all(path.startswith("results/") and path.endswith(".json") for path in deliverables),
            "manifest_deliverables_not_unique_json",
        ),
        (
            "manifest.milestone",
            MILESTONE,
            payload.get("milestone") if isinstance(payload, Mapping) else None,
            isinstance(payload, Mapping) and payload.get("milestone") == MILESTONE,
            "milestone_mismatch",
        ),
        (
            "manifest.infrastructure_count",
            f">={INFRASTRUCTURE_MINIMUM}",
            tracks.count("infrastructure"),
            tracks.count("infrastructure") >= INFRASTRUCTURE_MINIMUM,
            "too_few_infrastructure_tasks",
        ),
        (
            "manifest.exp6754_ungated",
            True,
            not bool(by_id.get(CAPSTONE_TASK_ID, {}).get("gated_on")),
            not bool(by_id.get(CAPSTONE_TASK_ID, {}).get("gated_on")),
            "capstone_gated",
        ),
    )
    for check, expected, observed, passed, reason in checks:
        if not passed:
            failures.append(_failure(check, ACTIVE_ROADMAP_PATH.as_posix(), expected, observed, reason))
    return failures


def _design_failures(design: Mapping[str, Any]) -> list[JsonDict]:
    tasks = design["tasks"]
    failures: list[JsonDict] = []
    checks = (
        ("design.milestone", MILESTONE, design.get("milestone"), design.get("milestone") == MILESTONE),
        ("design.task_count", 13, len(tasks), len(tasks) == 13),
        (
            "design.task_ids_unique",
            13,
            len({row["task_id"] for row in tasks}),
            len({row["task_id"] for row in tasks}) == 13,
        ),
        (
            "design.deliverables_unique",
            13,
            len({row["deliverable"] for row in tasks}),
            len({row["deliverable"] for row in tasks}) == 13,
        ),
        ("design.phase_count", 4, len(design["phases"]), len(design["phases"]) == 4),
    )
    for check, expected, observed, passed in checks:
        if not passed:
            failures.append(
                _failure(check, DESIGN_PATH.as_posix(), expected, observed, "design_contract_mismatch")
            )
    return failures


def science_branches_independent(manifest_payload: Mapping[str, Any] | None) -> bool:
    """True when the handoff audit and canary are not upstream science gates."""

    for task_id, task in _task_by_id(manifest_payload).items():
        if task_id == CAPSTONE_TASK_ID:
            continue
        for gate in task.get("gated_on", []) or []:
            if isinstance(gate, Mapping) and gate.get("upstream") in HANDOFF_AUDIT_IDS:
                return False
    return True


def _run_command(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


def build_validator_rows(
    root: Path,
    manifest_payload: Mapping[str, Any] | None,
    gate_failures: Sequence[Mapping[str, Any]],
    prompt_failures: Sequence[Mapping[str, Any]],
    run_external_validators: bool,
) -> list[JsonDict]:
    """Run the focused validators and classify the legacy model-only audit."""

    schema_row = _roadmap_schema_row(root, manifest_payload)
    rows: list[JsonDict] = [
        {
            "validator": "roadmap_schema",
            "command": "Roadmap.model_validate(research-roadmap.yaml)",
            "passed": schema_row["passed"],
            "observed_value": schema_row["observed_value"],
            "authoritative_for_handoff": True,
        },
        {
            "validator": "gate_contract",
            "command": "internal exact upstream REQUIRED ARTIFACT FIELDS check",
            "passed": not gate_failures,
            "observed_value": len(gate_failures),
            "authoritative_for_handoff": True,
        },
        {
            "validator": "prompt_contract",
            "command": "internal prompt section and literal check",
            "passed": not prompt_failures,
            "observed_value": len(prompt_failures),
            "authoritative_for_handoff": True,
        },
    ]
    if not run_external_validators:
        rows.append(
            {
                "validator": "external_validators",
                "command": "not run in injected-payload unit path",
                "passed": True,
                "observed_value": "skipped_by_test_fixture",
                "authoritative_for_handoff": False,
            }
        )
        return rows

    exclusion = _run_command(root, [EXCLUSION_LINT_PATH.as_posix(), ACTIVE_ROADMAP_PATH.as_posix()])
    rows.append(
        {
            "validator": "exclusion_manifest_lint",
            "command": f"{sys.executable} {EXCLUSION_LINT_PATH} {ACTIVE_ROADMAP_PATH}",
            "passed": exclusion.returncode == 0,
            "observed_value": exclusion.stdout.strip() or exclusion.stderr.strip(),
            "authoritative_for_handoff": True,
        }
    )

    audit = _run_command(root, [AUDIT_GATES_PATH.as_posix(), ACTIVE_ROADMAP_PATH.as_posix()])
    try:
        audit_payload = json.loads(audit.stdout)
    except json.JSONDecodeError:
        audit_payload = {"failure_details": [audit.stdout.strip(), audit.stderr.strip()]}
    details = [str(item) for item in audit_payload.get("failure_details", [])]
    model_only = bool(details) and all(detail.startswith("MODEL_AGENT_COHERENCE") for detail in details)
    compatibility = model_only and audit_payload.get("n_gate_upstream_failures") == 0
    compatibility = compatibility and audit_payload.get("n_gate_field_cross_ref_failures") == 0
    compatibility = compatibility and audit_payload.get("n_prior_failures_missing") == 0
    rows.append(
        {
            "validator": "audit_roadmap_gates_legacy",
            "command": f"{sys.executable} {AUDIT_GATES_PATH} {ACTIVE_ROADMAP_PATH}",
            "passed": audit.returncode == 0,
            "observed_value": audit_payload,
            "model_only_findings": model_only,
            "compatibility_accepted": compatibility,
            "authoritative_for_handoff": not compatibility,
        }
    )
    return rows


def audit_contract(
    root: Path,
    design: Mapping[str, Any],
    manifest_payload: Mapping[str, Any] | None,
    source_receipts: Sequence[Mapping[str, Any]],
    retired_ids: set[str] | None = None,
    precondition_rows: Sequence[Mapping[str, Any]] = (),
    validator_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Reduce V588 inputs to binding rows and hard failure rows."""

    del root
    retired_ids = retired_ids or set()
    task_rows, task_failures = build_task_binding_rows(design, manifest_payload, retired_ids)
    gate_rows, gate_failures = build_gate_binding_rows(manifest_payload)
    prior_rows, prior_failures = build_prior_failure_rows(manifest_payload)
    model_rows, model_failures = build_model_policy_rows(manifest_payload)
    binding_rows = task_rows + gate_rows
    failures: list[JsonDict] = []
    failures.extend(dict(row) for row in precondition_rows if row.get("passed") is not True)
    failures.extend(
        _failure("source.receipt", str(row.get("arxiv_id")), True, row.get("passed"), "source_receipt")
        for row in source_receipts
        if row.get("passed") is not True
    )
    failures.extend(_design_failures(design))
    failures.extend(_manifest_identity_failures(manifest_payload))
    failures.extend(task_failures)
    failures.extend(gate_failures)
    failures.extend(prior_failures)
    failures.extend(model_failures)
    failures.extend(
        _failure(
            f"validator.{row.get('validator')}",
            str(row.get("command")),
            True,
            row.get("observed_value"),
            "focused_validator_failed",
        )
        for row in validator_rows
        if row.get("passed") is not True and row.get("authoritative_for_handoff") is True
    )
    return {
        "rows": binding_rows,
        "binding_contract_rows": binding_rows,
        "prior_failure_rows": prior_rows,
        "model_policy_rows": model_rows,
        "failures": failures,
        "passed": not failures,
        "science_branches_independent_of_handoff_audit": science_branches_independent(manifest_payload),
    }


def _input_receipts(root: Path) -> list[JsonDict]:
    return [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "exists": (root / relative).is_file(),
        }
        for relative in (
            ACTIVE_ROADMAP_PATH,
            DESIGN_PATH,
            REFERENCES_PATH,
            EXCLUSION_PATH,
            ROADMAP_SCHEMA_PATH,
            AUDIT_GATES_PATH,
            EXCLUSION_LINT_PATH,
            SPEC_PATH,
            HARNESS_SPEC_PATH,
        )
    ]


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash stable inputs and reductions, excluding timing and verdict prose."""

    stable_keys = (
        "status",
        "inference_substrate",
        "random_seed",
        "source_cutoff",
        "input_receipts",
        "source_receipts",
        "validator_rows",
        "rows",
        "binding_contract_rows",
        "task_count",
        "handoff_contract_preserved",
        "science_branches_independent_of_handoff_audit",
        "gate_check_summary",
        "verdict_class",
        "field_principles",
        "preconditions_checked",
        "prior_failure_rows",
        "model_policy_rows",
    )
    return value_hash({key: payload.get(key) for key in stable_keys})


def _blocked_input_artifact(
    repo_root: Path,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    failures = [dict(row) for row in preconditions if row.get("passed") is not True]
    artifact: JsonDict = {
        "status": "complete_blocked_handoff_input",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_cutoff": SOURCE_CUTOFF,
        "input_receipts": _input_receipts(repo_root),
        "source_receipts": [],
        "validator_rows": [],
        "rows": [],
        "binding_contract_rows": [],
        "task_count": 0,
        "handoff_contract_preserved": False,
        "science_branches_independent_of_handoff_audit": False,
        "gate_check_summary": failures,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_handoff_input: required active roadmap, design, "
            "source refresh, schema, or exclusion manifest input failed."
        ),
        "preconditions_checked": list(preconditions),
        "prior_failure_rows": [],
        "model_policy_rows": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    repo_root: Path,
    duration_s: float,
    active_payload: Mapping[str, Any] | None = None,
    run_external_validators: bool = True,
    retired_ids: set[str] | None = None,
) -> JsonDict:
    """Assemble the terminal Exp6742 artifact."""

    preconditions, active_from_file, design = collect_precondition_rows(repo_root)
    if any(row.get("passed") is not True for row in preconditions) or design is None:
        return _blocked_input_artifact(repo_root, duration_s, preconditions)

    active = active_payload if active_payload is not None else active_from_file
    source_receipts = collect_source_receipts(repo_root, design)
    retired = retired_ids if retired_ids is not None else retired_task_ids(repo_root)

    provisional_task_rows, provisional_task_failures = build_task_binding_rows(
        design, active, retired
    )
    provisional_gate_rows, provisional_gate_failures = build_gate_binding_rows(active)
    prompt_failures = [row for row in provisional_task_failures if row["check"].startswith("prompt.")]
    validator_rows = build_validator_rows(
        repo_root,
        active,
        [
            row
            for row in provisional_gate_failures
            if row.get("check") != "gate.consumer_prompt_contract"
        ],
        prompt_failures,
        run_external_validators=run_external_validators,
    )
    audit = audit_contract(
        repo_root,
        design,
        active,
        source_receipts,
        retired_ids=retired,
        precondition_rows=preconditions,
        validator_rows=validator_rows,
    )
    handoff_preserved = bool(audit["passed"])
    task_count = len(active.get("tasks", [])) if isinstance(active, Mapping) else 0
    artifact: JsonDict = {
        "status": "complete_v588_handoff_contract_preserved"
        if handoff_preserved
        else "complete_blocked_handoff_contract",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_cutoff": SOURCE_CUTOFF,
        "input_receipts": _input_receipts(repo_root),
        "source_receipts": source_receipts,
        "validator_rows": validator_rows,
        "rows": audit["rows"],
        "binding_contract_rows": audit["binding_contract_rows"],
        "task_count": task_count,
        "handoff_contract_preserved": handoff_preserved,
        "science_branches_independent_of_handoff_audit": audit[
            "science_branches_independent_of_handoff_audit"
        ],
        "gate_check_summary": audit["failures"],
        "verdict_class": "null" if handoff_preserved else "blocked",
        "honest_verdict": (
            "complete_null: V588 handoff contract is preserved; this is a static audit only."
            if handoff_preserved
            else "complete_blocked_handoff_contract: one or more V588 handoff checks failed."
        ),
        "preconditions_checked": preconditions,
        "prior_failure_rows": audit["prior_failure_rows"],
        "model_policy_rows": audit["model_policy_rows"],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-level errors for a stored Exp6742 artifact."""

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
    if payload.get("rows") != payload.get("binding_contract_rows"):
        errors.append("rows_binding_contract_rows_mismatch")
    handoff = payload.get("handoff_contract_preserved") is True
    gate_summary = payload.get("gate_check_summary")
    if handoff:
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
        if not str(payload.get("honest_verdict", "")).startswith("complete_blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete JSON object and remove temp files on failure."""

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Exp6742 V588 handoff audit artifact.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or args.repo_root / RESULT_PATH
    if args.validate:
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 1
        errors = validate_artifact(payload)
        if errors:
            print(json.dumps({"errors": errors}, indent=2, sort_keys=True))
            return 1
        return 0

    start = time.monotonic()
    artifact = build_artifact(args.repo_root, duration_s=time.monotonic() - start)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, indent=2, sort_keys=True))
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
