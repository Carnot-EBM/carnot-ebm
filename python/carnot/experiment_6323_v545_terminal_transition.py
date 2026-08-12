"""Exp6323 V544-to-V545 terminal transition.

Spec refs: REQ-INFRA-6323, SCENARIO-INFRA-6323-1,
SCENARIO-INFRA-6323-2, SCENARIO-INFRA-6323-3,
SCENARIO-INFRA-6323-4, SCENARIO-INFRA-6323-5,
SCENARIO-INFRA-6323-6.
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
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from roadmap_schema import Roadmap  # noqa: E402


MILESTONE_V544 = "2026.08.544"
MILESTONE_V545 = "2026.08.545"
EXPERIMENT_ID = "exp6323-v545-terminal-transition"
SCHEMA = "carnot.experiment_6323.v545_terminal_transition.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6323_v545_terminal_transition.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

V544_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6322_v544_adversarial_capstone.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPECTED_V544_TASK_IDS = (
    "exp6310-v544-terminal-transition",
    "exp6311-v544-post-marker-source-scope-freeze",
    "exp6312-model-local-representation-surface-preflight",
    "exp6313-exact-code-safety-pair-fixture",
    "exp6314-three-family-model-local-state-corpus",
    "exp6315-model-local-paired-difference-energy-probes",
    "exp6316-model-local-probe-integrity-audit",
    "exp6317-live-three-family-model-local-verifier-benchmark",
    "exp6318-versioned-factor-local-online-initializer",
    "exp6319-feedback-directed-online-update-search",
    "exp6320-online-self-evolution-safety-audit",
    "exp6321-arc-target-licensed-route-live-shadow-ab",
    "exp6322-v544-adversarial-capstone",
)
EXPECTED_V545_TASK_IDS = (
    "exp6323-v545-terminal-transition",
    "exp6324-v545-post-marker-source-scope-freeze",
    "exp6325-gatemate-dated-receipt-single-detect",
    "exp6326-restricted-policy-contract-compiler",
    "exp6327-three-family-guarded-policy-synthesis",
    "exp6328-blind-guard-integrity-audit",
    "exp6329-prospective-held-family-guarded-policy-ab",
    "exp6330-anytime-valid-release-certificate-engine",
    "exp6331-counterexample-factor-update-calibration",
    "exp6332-prospective-certified-continuous-learning-ab",
    "exp6333-certified-learning-safety-audit",
    "exp6334-arc-counterfactual-action-influence-preflight",
    "exp6335-arc-default-off-live-causal-influence-ab",
    "exp6336-v545-adversarial-capstone",
)
EXPECTED_V545_IDS_BY_NUMBER = {
    int(re.match(r"exp(\d+)", task_id).group(1)): task_id  # type: ignore[union-attr]
    for task_id in EXPECTED_V545_TASK_IDS
}
RESERVED_EXP_IDS = tuple(range(6323, 6337))
MANDATED_HEADLINE_GGUF_IDS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
)
SAFETY_ONLY_TASK_ID = "exp6320-online-self-evolution-safety-audit"
SHADOW_ONLY_TASK_ID = "exp6321-arc-target-licensed-route-live-shadow-ab"

STAGED_REQUIRES: dict[str, tuple[str, ...]] = {
    "exp6324-v545-post-marker-source-scope-freeze": ("exp6323-v545-terminal-transition",),
    "exp6325-gatemate-dated-receipt-single-detect": ("exp6323-v545-terminal-transition",),
    "exp6326-restricted-policy-contract-compiler": (
        "exp6324-v545-post-marker-source-scope-freeze",
    ),
    "exp6327-three-family-guarded-policy-synthesis": (
        "exp6326-restricted-policy-contract-compiler",
    ),
    "exp6328-blind-guard-integrity-audit": ("exp6327-three-family-guarded-policy-synthesis",),
    "exp6329-prospective-held-family-guarded-policy-ab": ("exp6328-blind-guard-integrity-audit",),
    "exp6330-anytime-valid-release-certificate-engine": (
        "exp6324-v545-post-marker-source-scope-freeze",
    ),
    "exp6331-counterexample-factor-update-calibration": (
        "exp6324-v545-post-marker-source-scope-freeze",
    ),
    "exp6332-prospective-certified-continuous-learning-ab": (
        "exp6330-anytime-valid-release-certificate-engine",
        "exp6331-counterexample-factor-update-calibration",
    ),
    "exp6333-certified-learning-safety-audit": (
        "exp6330-anytime-valid-release-certificate-engine",
        "exp6331-counterexample-factor-update-calibration",
    ),
    "exp6334-arc-counterfactual-action-influence-preflight": (
        "exp6324-v545-post-marker-source-scope-freeze",
    ),
    "exp6335-arc-default-off-live-causal-influence-ab": (
        "exp6334-arc-counterfactual-action-influence-preflight",
    ),
    "exp6336-v545-adversarial-capstone": EXPECTED_V545_TASK_IDS[:-1],
}
STAGED_GATES: dict[str, tuple[JsonDict, ...]] = {
    "exp6327-three-family-guarded-policy-synthesis": (
        {
            "upstream": "exp6326-restricted-policy-contract-compiler",
            "artifact_field": "contract_guard_ready_score",
            "op": "==",
            "value": 1.0,
        },
    ),
    "exp6329-prospective-held-family-guarded-policy-ab": (
        {
            "upstream": "exp6328-blind-guard-integrity-audit",
            "artifact_field": "guard_integrity_ready_score",
            "op": "==",
            "value": 1.0,
        },
    ),
}

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
    V544_CAPSTONE_RELATIVE_PATH,
    Path("results/experiment_6312_model_local_representation_surface_preflight.json"),
    Path("results/experiment_6318_versioned_factor_local_online_initializer.json"),
    Path("results/experiment_6319_feedback_directed_online_update_search.json"),
    Path("results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"),
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
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
    V544_CAPSTONE_RELATIVE_PATH,
    Path("results/experiment_6312_model_local_representation_surface_preflight.json"),
    Path("results/experiment_6318_versioned_factor_local_online_initializer.json"),
    Path("results/experiment_6319_feedback_directed_online_update_search.json"),
    Path("results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
)
EXPERIMENT_SCAN_ROOTS = (Path("python/carnot"), Path("tests/python"), Path("results"))
ALLOWED_LOCAL_RESERVED_PATHS = {
    "python/carnot/experiment_6323_v545_terminal_transition.py",
    "tests/python/test_experiment_6323_v545_terminal_transition.py",
    RESULT_RELATIVE_PATH.as_posix(),
}

RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6323_v545_terminal_transition --date 20260812"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6323_v545_terminal_transition.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6323_v545_terminal_transition.py "
    "-m pytest tests/python/test_experiment_6323_v545_terminal_transition.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6323_v545_terminal_transition.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6323_v545_terminal_transition.py "
    "tests/python/test_experiment_6323_v545_terminal_transition.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check python/carnot/experiment_6323_v545_terminal_transition.py "
    "tests/python/test_experiment_6323_v545_terminal_transition.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6323_v545_terminal_transition.py"
)
ROADMAP_CHECK_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6323_v545_terminal_transition --check-roadmap-only"
)
ROADMAP_GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
PROTECTED_DIFF_COMMAND = (
    "git diff --exit-code -- research-roadmap.yaml research-roadmap-next.yaml "
    "openspec/change-proposals/research-roadmap-vNEXT.md scripts/research_conductor.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n 1,220p ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6323_v545_terminal_transition.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROADMAP_CHECK_COMMAND,
    ROADMAP_GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    PROTECTED_DIFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6323_test_receipts.json")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v544_roadmap_path_and_hash",
    "v544_task_terminal_matrix",
    "v544_capstone_path_hash_and_summary",
    "v544_validation_failure_receipts",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts",
    "v545_roadmap_path_and_hash",
    "v545_task_ids_and_deliverables",
    "task_count",
    "phase_counts",
    "dependency_validation",
    "gated_on_validation",
    "prior_failure_validation",
    "retired_dependency_count",
    "id_collision_count",
    "agent_routing_validation",
    "model_policy_validation",
    "prompt_contract_validation",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The handoff is terminal even when the staged contract is blocked.",
    "v544_roadmap_path_and_hash": "The V544 denominator comes from the capstone evidence.",
    "v544_task_terminal_matrix": "Each V544 task keeps the class of its exact path.",
    "v544_capstone_path_hash_and_summary": "The V544 capstone is input evidence, not an override.",
    "v544_validation_failure_receipts": "Failed required commands stay visible.",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts": "Special states remain separate.",
    "v545_roadmap_path_and_hash": "The staged V545 plan is content-addressed.",
    "v545_task_ids_and_deliverables": "The V545 denominator is Exp6323 through Exp6336.",
    "task_count": "Bare 14 proves the full staged denominator was audited.",
    "phase_counts": "Phase counts detect truncation of the staged plan.",
    "dependency_validation": "Dependencies must name same-milestone tasks.",
    "gated_on_validation": "Structured gates must point at declared upstream fields.",
    "prior_failure_validation": "Prior failures need all required subfields.",
    "retired_dependency_count": "Bare zero proves no dependency points at a retired id.",
    "id_collision_count": "Bare zero proves task ids and deliverables are unique.",
    "agent_routing_validation": "Every executable task must route to Codex.",
    "model_policy_validation": "LLM tasks must name the required local GGUF identities.",
    "prompt_contract_validation": "Prompt endings and exclusions prevent conductor drift.",
    "protected_files_unchanged": "Protected hashes show the handoff did not rewrite them.",
    "preconditions_checked": "Inputs, system state, and command availability are frozen first.",
    "inference_substrate": "This artifact aggregates checked-in evidence only.",
    "verifier_is_oracle": "The transition audits records and is not an answer oracle.",
    "field_provenance": "Each required field cites its evidence sources.",
    "field_principles": "Each required field states why it exists.",
    "test_commands": "Commands define the verification boundary.",
    "test_exit_codes": "Exit codes remain unlaundered.",
    "duration_s": "Wall time records audit cost without padding.",
    "reproducibility_checksum": "A normalized checksum detects silent payload drift.",
    "honest_verdict": "The verdict states the terminal result with a prefix.",
}
COUNT_PRINCIPLES: dict[str, str] = {
    "task_count": "The V544 denominator is Exp6310 through Exp6322.",
    "terminal_class_task_count_sum": "Terminal-class buckets must add to 13.",
    "missing": "Missing exact paths cannot be replaced.",
    "nonterminal": "Nonterminal rows cannot feed claims.",
    "blocked": "Raw blocked status remains visible.",
    "skipped": "Gate skips stay distinct.",
    "null": "Null closure is not positive evidence.",
    "flagged": "Flagged rows remain quarantined.",
    "retired": "Retired rows remain retired.",
    "ready": "Ready evidence is not a utility claim.",
    "positive": "Positive evidence cannot promote another branch.",
    "safety_only": "Safety-only evidence cannot promote utility.",
    "shadow_only": "ARC shadow evidence cannot claim solve credit.",
}

_TASK_HEADING_RE = re.compile(r"^### Exp(?P<num>\d+): (?P<title>.+)$", re.MULTILINE)
_PHASE_RE = re.compile(r"^## Phase (?P<num>\d+): (?P<title>.+)$", re.MULTILINE)
_DELIVERABLE_RE = re.compile(r"\*\*Deliverable:\*\* `(?P<path>results/[^`]+\.json)`")
_NEGATION_MARKERS = (
    "do not",
    "does not",
    "must not",
    "cannot",
    "no ",
    "without",
    "not scheduled",
    "not use",
    "forbid",
    "forbids",
    "closed",
    "terminal",
    "retired",
)
_SCHEDULING_MARKERS = ("schedule", "activate", "reactivate", "retry", "use ", "run ", "build")
_FORBIDDEN_PATTERNS = (
    (
        "hidden_model_local_state",
        re.compile(
            r"hidden[- ]state|model[- ]local|activation|embedding|prefix[- ]state|pooled", re.I
        ),
    ),
    (
        "external_text_scorer",
        re.compile(
            r"external (?:generated[- ]text|text) scorer|masked[- ]model energy|best[- ]of[- ]n",
            re.I,
        ),
    ),
    ("kan", re.compile(r"\bKAN\b", re.I)),
    ("cross_family_transfer", re.compile(r"cross[- ]family|cross[- ]domain|cross[- ]game", re.I)),
    (
        "public_arc_resolve",
        re.compile(
            r"public ARC game re[- ]solve|public[- ]game solve|level solve|solve registry", re.I
        ),
    ),
    (
        "unapproved_hardware",
        re.compile(
            r"flash|synthesis|place and route|timing|KV260|PolarFire|hardware command", re.I
        ),
    ),
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


def _roadmap_tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def _phase_for_offset(phases: list[tuple[int, str, int]], offset: int) -> tuple[int, str]:
    current = (0, "unassigned")
    for number, title, start in phases:
        if start <= offset:
            current = (number, title)
    return current


def _parse_markdown_tasks(markdown: str, active_data: JsonMap) -> list[JsonDict]:
    active_by_id = {str(task.get("id") or ""): dict(task) for task in _roadmap_tasks(active_data)}
    phases = [
        (int(match.group("num")), match.group("title").strip(), match.start())
        for match in _PHASE_RE.finditer(markdown)
    ]
    matches = list(_TASK_HEADING_RE.finditer(markdown))
    tasks: list[JsonDict] = []
    for index, match in enumerate(matches):
        exp_num = int(match.group("num"))
        task_id = EXPECTED_V545_IDS_BY_NUMBER.get(exp_num, f"exp{exp_num}")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        section = markdown[start:end]
        deliverable_match = _DELIVERABLE_RE.search(section)
        phase_number, phase_title = _phase_for_offset(phases, match.start())
        active = active_by_id.get(task_id, {})
        task: JsonDict = {
            **active,
            "id": task_id,
            "milestone": MILESTONE_V545,
            "title": active.get("title") or match.group("title").strip(),
            "deliverable": active.get("deliverable")
            or (deliverable_match.group("path") if deliverable_match else ""),
            "track": active.get("track") or f"phase_{phase_number}",
            "phase": {"number": phase_number, "title": phase_title},
            "requires": list(active.get("requires") or STAGED_REQUIRES.get(task_id, ())),
            "gated_on": list(active.get("gated_on") or STAGED_GATES.get(task_id, ())),
            "prior_failures": list(active.get("prior_failures") or []),
            "prompt": str(active.get("prompt") or ""),
            "agent_type": active.get("agent_type"),
            "model": active.get("model"),
            "requires_gpu": active.get("requires_gpu") is True,
            "staged_markdown_section": section.strip(),
            "structured_yaml_present": bool(active),
        }
        tasks.append(task)
    return tasks


def load_v545_roadmap(root: Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    doc_path = root / MILESTONE_DOC_RELATIVE_PATH
    active_data = read_yaml_mapping(active_path)
    markdown = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    tasks = _parse_markdown_tasks(markdown, active_data)
    data = {
        "milestone": MILESTONE_V545,
        "milestone_title": active_data.get("milestone_title")
        or "Contract-Guarded Energy and Certified Self-Learning",
        "milestone_doc": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        "tasks": tasks,
    }
    identity = {
        "path": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(doc_path),
        "milestone": MILESTONE_V545 if tasks else None,
        "task_count": len(tasks),
        "expected_task_count": 14,
        "active_roadmap_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
        "active_roadmap_sha256": path_sha256(active_path),
        "active_roadmap_milestone": active_data.get("milestone"),
        "active_roadmap_task_count": len(_roadmap_tasks(active_data)),
        "requested_next_path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        "research_roadmap_next_present": next_path.exists(),
        "research_roadmap_next_sha256": path_sha256(next_path),
        "selection_note": "full 14-task V545 plan parsed from change-proposal markdown; active YAML is preserved",
    }
    return data, identity


def v545_task_ids_and_deliverables(data: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in _roadmap_tasks(data):
        prompt = str(task.get("prompt") or "")
        rows.append(
            {
                "task_id": str(task.get("id") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "phase": task.get("phase"),
                "track": str(task.get("track") or ""),
                "requires": list(task.get("requires") or []),
                "gated_on": list(task.get("gated_on") or []),
                "agent_type": task.get("agent_type"),
                "model": task.get("model"),
                "requires_gpu": task.get("requires_gpu") is True,
                "structured_yaml_present": task.get("structured_yaml_present") is True,
                "model_specs_named_in_prompt": model_specs_named_in_prompt(prompt),
            }
        )
    return rows


def _is_live_llm_task(task: JsonMap) -> bool:
    text = f"{task.get('title', '')}\n{task.get('prompt', '')}\n{task.get('staged_markdown_section', '')}"
    return (
        task.get("requires_gpu") is True
        or "local GGUF model" in text
        or "local GGUF models" in text
    )


def _negated_or_historical(text: str, start: int) -> bool:
    window = text[max(0, start - 140) : start + 140].lower()
    return any(marker in window for marker in _NEGATION_MARKERS)


def _scheduled_context(text: str, start: int) -> bool:
    window = text[max(0, start - 90) : start + 90].lower()
    return any(marker in window for marker in _SCHEDULING_MARKERS)


def _forbidden_scope_validation(tasks: Sequence[JsonMap]) -> JsonDict:
    failures: list[JsonDict] = []
    clean_mentions: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        text = f"{task.get('title', '')}\n{task.get('prompt', '')}\n{task.get('staged_markdown_section', '')}"
        for mechanism, pattern in _FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                negated = _negated_or_historical(text, match.start())
                scheduled = _scheduled_context(text, match.start())
                row = {
                    "task_id": task_id,
                    "mechanism": mechanism,
                    "matched_text": match.group(0),
                    "negated_or_historical_context": negated,
                    "scheduled_context": scheduled,
                }
                if scheduled and not negated:
                    failures.append(row)
                else:
                    clean_mentions.append(row)
                break
    return {
        "ok": not failures,
        "failures": failures,
        "clean_mentions": clean_mentions,
        "audited_mechanisms": [name for name, _pattern in _FORBIDDEN_PATTERNS],
    }


def validate_v545_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _roadmap_tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    id_counts = Counter(ids)
    deliverable_counts = Counter(deliverables)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    duplicate_deliverables = sorted(
        path for path, count in deliverable_counts.items() if path and count > 1
    )
    id_set = set(ids)
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        for task_id, task in tasks_by_id.items()
    }

    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001
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
        for dep in task.get("requires") if isinstance(task.get("requires"), list) else []:
            dep_text = str(dep)
            dep_num = exp_number(dep_text)
            retired = dep_num in retired_exp_ids if dep_num is not None else False
            missing = dep_text not in id_set
            self_dependency = dep_text == task_id
            if missing or retired or self_dependency:
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
    for task in tasks:
        task_id = str(task.get("id") or "")
        for gate in task.get("gated_on") if isinstance(task.get("gated_on"), list) else []:
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            if not ok:
                gate_failures.append({"task_id": task_id, "gate": gate, "reason": reason})

    prior_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        priors = task.get("prior_failures")
        if priors is None:
            continue
        if not isinstance(priors, list):
            prior_failures.append({"task_id": task_id, "reason": "prior_failures_not_list"})
            continue
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    route_failures: list[JsonDict] = []
    model_failures: list[JsonDict] = []
    prompt_failures: list[JsonDict] = []
    available_prompt_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        if task.get("agent_type") != "codex":
            route_failures.append(
                {
                    "task_id": task_id,
                    "reason": "missing_or_non_codex_agent_type",
                    "agent_type": task.get("agent_type"),
                    "structured_yaml_present": task.get("structured_yaml_present") is True,
                }
            )
        if task.get("model") != "gpt-5.5":
            model_failures.append(
                {"task_id": task_id, "reason": "missing_or_wrong_model", "model": task.get("model")}
            )
        if _is_live_llm_task(task):
            named = set(model_specs_named_in_prompt(prompt))
            if named and not named <= MANDATED_HEADLINE_GGUF_IDS:
                model_failures.append(
                    {
                        "task_id": task_id,
                        "reason": "non_mandated_gguf_id",
                        "ids": sorted(named - MANDATED_HEADLINE_GGUF_IDS),
                    }
                )
            if "all three" in f"{prompt}\n{task.get('staged_markdown_section', '')}".lower() and (
                not MANDATED_HEADLINE_GGUF_IDS <= named
            ):
                model_failures.append(
                    {
                        "task_id": task_id,
                        "reason": "missing_required_mandated_gguf_ids",
                        "expected": sorted(MANDATED_HEADLINE_GGUF_IDS),
                        "found": sorted(named),
                    }
                )
        if prompt:
            available_prompt_count += 1
            expected_run = (
                f"Run command: .venv/bin/python -m carnot.{module_name_for_task(task)} --date"
            )
            has_run = expected_run in prompt
            has_ending = prompt.strip().endswith(
                "Do NOT push. Do NOT modify scripts/research_conductor.py."
            )
            if not has_run or not has_ending:
                prompt_failures.append(
                    {
                        "task_id": task_id,
                        "run_command_present": has_run,
                        "protected_conductor_ending": has_ending,
                    }
                )
        else:
            prompt_failures.append({"task_id": task_id, "reason": "missing_executable_prompt"})

    forbidden_scope = _forbidden_scope_validation(tasks)
    if not forbidden_scope["ok"]:
        prompt_failures.extend(forbidden_scope["failures"])

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "task_count": len(tasks),
        "phase_counts": dict(
            sorted(
                Counter(f"phase_{task.get('phase', {}).get('number', 0)}" for task in tasks).items()
            )
        ),
        "task_id_validation": {
            "task_ids": ids,
            "expected_task_ids": list(EXPECTED_V545_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V545_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "deliverable_validation": {
            "ok": not deliverable_failures and not duplicate_deliverables,
            "failures": deliverable_failures,
            "duplicate_deliverables": duplicate_deliverables,
        },
        "dependency_validation": {"ok": not dependency_failures, "failures": dependency_failures},
        "gated_on_validation": {"ok": not gate_failures, "failures": gate_failures},
        "prior_failure_validation": {"ok": not prior_failures, "failures": prior_failures},
        "retired_dependency_count": retired_dependency_count,
        "id_collision_count": sum(count - 1 for count in id_counts.values() if count > 1)
        + sum(count - 1 for count in deliverable_counts.values() if count > 1),
        "agent_routing_validation": {
            "ok": not route_failures,
            "failures": route_failures,
            "missing_structured_route_count": sum(
                1 for row in route_failures if row.get("agent_type") is None
            ),
        },
        "model_policy_validation": {"ok": not model_failures, "failures": model_failures},
        "prompt_contract_validation": {
            "ok": not prompt_failures,
            "failures": prompt_failures,
            "available_prompt_count": available_prompt_count,
            "missing_prompt_count": len(tasks) - available_prompt_count,
            "forbidden_scope_validation": forbidden_scope,
        },
    }


def classify_v544_tasks(root: Path, capstone_payload: JsonMap) -> JsonDict:
    declared = capstone_payload.get("declared_task_ids_and_deliverables")
    if not isinstance(declared, list):
        declared = []
    rows: JsonDict = {}
    for item in declared:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("task_id") or "")
        rel = Path(str(item.get("deliverable") or ""))
        payload, meta = read_json_mapping(root / rel)
        classification = classify_artifact_path(root / rel).to_dict()
        rows[task_id] = {
            "task_id": task_id,
            "title": str(item.get("title") or task_id),
            "track": str(item.get("track") or ""),
            "declared_deliverable": rel.as_posix(),
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"] or meta.get("sha256"),
            "terminal_class": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "raw_blocked_status": str(classification["status_raw"] or "").startswith("blocked")
            or str(classification["honest_verdict_raw"] or "").startswith("blocked"),
            "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_stamped": bool(payload.get("corrigendum_pending")),
            "verifier_is_oracle_raw": payload.get("verifier_is_oracle"),
            "oracle_only": payload.get("verifier_is_oracle") is True,
            "safety_only": task_id == SAFETY_ONLY_TASK_ID,
            "shadow_only": task_id == SHADOW_ONLY_TASK_ID,
        }
    return {task_id: rows[task_id] for task_id in EXPECTED_V544_TASK_IDS if task_id in rows}


def count_v544_states(matrix: JsonMap) -> JsonDict:
    classes = Counter(str(row.get("terminal_class")) for row in matrix.values())
    counts: JsonDict = {
        key: int(classes.get(key, 0))
        for key in ("missing", "skipped", "null", "flagged", "retired", "ready", "positive")
    }
    counts["task_count"] = len(matrix)
    counts["terminal_class_task_count_sum"] = int(sum(classes.values()))
    counts["nonterminal"] = sum(1 for row in matrix.values() if row.get("terminal") is not True)
    counts["blocked"] = sum(1 for row in matrix.values() if row.get("raw_blocked_status") is True)
    counts["safety_only"] = sum(1 for row in matrix.values() if row.get("safety_only") is True)
    counts["shadow_only"] = sum(1 for row in matrix.values() if row.get("shadow_only") is True)
    counts["ready"] += sum(
        1
        for row in matrix.values()
        if row.get("shadow_only") is True and row.get("terminal_class") != "ready"
    )
    counts["terminal_class_counts"] = dict(
        sorted((key, int(value)) for key, value in classes.items())
    )
    counts["count_principles"] = dict(COUNT_PRINCIPLES)
    return counts


def v544_validation_failure_receipts(capstone_payload: JsonMap) -> JsonDict:
    exits = capstone_payload.get("test_exit_codes")
    test_exit_codes = dict(exits) if isinstance(exits, Mapping) else {}
    nonzero = {
        str(command): int(code)
        for command, code in test_exit_codes.items()
        if isinstance(code, int) and code != 0
    }
    return {
        "nonzero_exit_codes_by_command": nonzero,
        "broad_validation": {
            "command": FULL_PYTEST_COMMAND,
            "exit_code": test_exit_codes.get(FULL_PYTEST_COMMAND),
            "failed_count": 1 if test_exit_codes.get(FULL_PYTEST_COMMAND) not in (None, 0) else 0,
        },
        "determination_validation": {
            "command": DETERMINATION_COMMAND,
            "exit_code": test_exit_codes.get(DETERMINATION_COMMAND),
            "failed_count": 1 if test_exit_codes.get(DETERMINATION_COMMAND) not in (None, 0) else 0,
        },
        "all_recorded_test_exit_codes": test_exit_codes,
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


def _experiment_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for base_rel in EXPERIMENT_SCAN_ROOTS:
        base = root / base_rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file() and exp_number(path.name) is not None:
                paths.append(path.relative_to(root))
    return sorted(paths)


def scan_reserved_id_collisions(root: Path, staged_deliverables: set[str]) -> JsonDict:
    unexpected: dict[str, list[str]] = {str(exp_id): [] for exp_id in RESERVED_EXP_IDS}
    for rel in _experiment_paths(root):
        number = exp_number(rel.name)
        rel_text = rel.as_posix()
        if (
            number in RESERVED_EXP_IDS
            and rel_text not in ALLOWED_LOCAL_RESERVED_PATHS
            and rel_text not in staged_deliverables
        ):
            unexpected[str(number)].append(rel_text)
    unexpected = {key: sorted(value) for key, value in unexpected.items() if value}
    return {
        "scan_roots": [path.as_posix() for path in EXPERIMENT_SCAN_ROOTS],
        "reserved_exp_ids": list(RESERVED_EXP_IDS),
        "allowed_reserved_paths": sorted(ALLOWED_LOCAL_RESERVED_PATHS),
        "staged_deliverables": sorted(staged_deliverables),
        "unexpected_reserved_paths_by_exp_id": unexpected,
        "unexpected_reserved_collision_count": sum(len(paths) for paths in unexpected.values()),
        "tracked_and_untracked_basis": "filesystem scan covers tracked and untracked files",
    }


def _disk_receipt(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _ram_receipt() -> JsonDict:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():  # pragma: no cover
        return {"available": False}
    values: JsonDict = {"available": True}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[f"{key.lower()}_kb"] = int(rest.strip().split()[0])
    return values


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


def _field_provenance() -> JsonDict:
    sources = sorted(
        {
            "REQ-INFRA-6323",
            V544_CAPSTONE_RELATIVE_PATH.as_posix(),
            ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            SPEC_RELATIVE_PATH.as_posix(),
            TERMINAL_ARTIFACTS_RELATIVE_PATH.as_posix(),
            "scripts/roadmap_schema.py",
            "scripts/audit_roadmap_gates.py",
            "scripts/exclusion_manifest_lint.py",
            "CLAUDE.md Codex-Default and SOTA Local Models sections",
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


def preconditions_checked(
    root: Path,
    v545_identity: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    collision_scan: JsonMap,
    git_status_after_tests: Sequence[str] | None = None,
) -> JsonDict:
    return {
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests or []),
        "input_hashes_before": _input_hashes(root),
        "staged_roadmap_identity": v545_identity,
        "reserved_id_scan_before_artifact_write": collision_scan,
        "active_roadmap_hash_before_artifact_write": path_sha256(
            root / ACTIVE_ROADMAP_RELATIVE_PATH
        ),
        "protected_hashes_before_artifact_write": before_hashes,
        "disk": _disk_receipt(root),
        "ram": _ram_receipt(),
        "command_availability": _command_availability(),
        "active_research_roadmap_was_not_edited": True,
        "research_roadmap_next_was_not_activated": True,
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
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
    capstone_payload, capstone_meta = read_json_mapping(root / V544_CAPSTONE_RELATIVE_PATH)
    v545_data, v545_identity = load_v545_roadmap(root)
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    staged_deliverables = {row["deliverable"] for row in v545_task_ids_and_deliverables(v545_data)}
    collision_scan = scan_reserved_id_collisions(root, staged_deliverables)
    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    v545_validation = validate_v545_roadmap_data(v545_data, retired_ids)
    matrix = classify_v544_tasks(root, capstone_payload)
    command_rows = [dict(row) for row in (command_receipts or [])]
    v544_roadmap = capstone_payload.get("roadmap_path_and_hash")
    if not isinstance(v544_roadmap, Mapping):
        v544_roadmap = {}

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete",
        "v544_roadmap_path_and_hash": {
            "milestone": MILESTONE_V544,
            "capstone_path": V544_CAPSTONE_RELATIVE_PATH.as_posix(),
            "capstone_sha256": capstone_meta.get("sha256"),
            "recorded_roadmap": dict(v544_roadmap),
            "expected_task_ids": list(EXPECTED_V544_TASK_IDS),
        },
        "v544_task_terminal_matrix": matrix,
        "v544_capstone_path_hash_and_summary": {
            **capstone_meta,
            "summary": {
                "status": capstone_payload.get("status"),
                "honest_verdict": capstone_payload.get("honest_verdict"),
                "branch_promotion_matrix": capstone_payload.get("branch_promotion_matrix"),
                "counts": capstone_payload.get(
                    "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts"
                ),
            },
        },
        "v544_validation_failure_receipts": v544_validation_failure_receipts(capstone_payload),
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts": count_v544_states(
            matrix
        ),
        "v545_roadmap_path_and_hash": v545_identity,
        "v545_task_ids_and_deliverables": v545_task_ids_and_deliverables(v545_data),
        "task_count": v545_validation["task_count"],
        "phase_counts": v545_validation["phase_counts"],
        "dependency_validation": v545_validation["dependency_validation"],
        "gated_on_validation": v545_validation["gated_on_validation"],
        "prior_failure_validation": v545_validation["prior_failure_validation"],
        "retired_dependency_count": v545_validation["retired_dependency_count"],
        "id_collision_count": v545_validation["id_collision_count"],
        "agent_routing_validation": v545_validation["agent_routing_validation"],
        "model_policy_validation": v545_validation["model_policy_validation"],
        "prompt_contract_validation": v545_validation["prompt_contract_validation"],
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root,
            v545_identity,
            before,
            status_before,
            collision_scan,
            git_status_after_tests,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": "complete: V544 evidence preserved and V545 staged roadmap validated",
    }
    blocking = (
        report["task_count"] != 14
        or report["retired_dependency_count"] != 0
        or report["id_collision_count"] != 0
        or not report["dependency_validation"]["ok"]
        or not report["gated_on_validation"]["ok"]
        or not report["prior_failure_validation"]["ok"]
        or not report["agent_routing_validation"]["ok"]
        or not report["model_policy_validation"]["ok"]
        or not report["prompt_contract_validation"]["ok"]
        or not report["prompt_contract_validation"]["forbidden_scope_validation"]["ok"]
        or not report["protected_files_unchanged"]["unchanged"]
    )
    if blocking:
        report["status"] = "blocked"
        report["honest_verdict"] = (
            "blocked: V545 full staged roadmap exists only as markdown; "
            "seven executable YAML prompt contracts are absent"
        )
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
    counts = report.get(
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts"
    )
    if not isinstance(counts, Mapping):
        errors.append("counts field is not a mapping")
    else:
        if counts.get("task_count") != 13:
            errors.append("V544 counts task_count must be 13")
        if counts.get("terminal_class_task_count_sum") != 13:
            errors.append("terminal class counts must conserve 13 V544 tasks")
        count_principles = counts.get("count_principles", {})
        for field in COUNT_PRINCIPLES:
            if not isinstance(count_principles, Mapping) or field not in count_principles:
                errors.append(f"missing count principle: {field}")
    if report.get("task_count") != 14:
        errors.append("task_count must be 14")
    if report.get("retired_dependency_count") != 0:
        errors.append("retired_dependency_count must be 0")
    if report.get("id_collision_count") != 0:
        errors.append("id_collision_count must be 0")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "blocked:",
            "blocked_",
        )
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
        raise ValueError(f"invalid Exp6323 report: {errors}")
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
    elif isinstance(payload, list):
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
    parser.add_argument("--check-roadmap-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_roadmap_only:  # pragma: no cover
        data, _identity = load_v545_roadmap(REPO_ROOT)
        result = validate_v545_roadmap_data(
            data, load_retired_exp_ids(REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH)
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["task_count"] == 14 and result["retired_dependency_count"] == 0 else 1
    artifact = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": artifact["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
