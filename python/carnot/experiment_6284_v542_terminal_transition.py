"""Exp6284 V541-to-V542 exact terminal transition.

Spec refs: REQ-INFRA-6284, SCENARIO-INFRA-6284-1,
SCENARIO-INFRA-6284-2, SCENARIO-INFRA-6284-3,
SCENARIO-INFRA-6284-4, SCENARIO-INFRA-6284-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_6272_v541_terminal_transition import (
    exp_number,
    focused_and_broad_validation_receipts_by_task,
    gate_ok,
    git_status_lines,
    load_retired_exp_ids,
    module_name_for_task,
    prior_ok,
    read_json_mapping,
    read_yaml_mapping,
    required_artifact_fields_from_prompt,
    same_number_aliases,
)
from carnot.experiment_artifacts import atomic_write_json
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
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from roadmap_schema import Roadmap  # noqa: E402


MILESTONE_V541 = "2026.08.541"
MILESTONE_V542 = "2026.08.542"
EXPERIMENT_ID = "exp6284-v542-terminal-transition"
SCHEMA = "carnot.experiment_6284.v542_terminal_transition.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6284_v542_terminal_transition.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

V541_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6283_v541_adversarial_capstone.json")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_541.json")
V542_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V542_NEXT_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")

EXPECTED_V541_TASK_IDS = (
    "exp6272-v541-terminal-transition",
    "exp6273-v541-post-marker-source-scope-freeze",
    "exp6274-asp-energy-semantic-compiler",
    "exp6275-flagship-asp-constraint-verification-benchmark",
    "exp6276-certified-dual-cache-admission",
    "exp6277-chronological-certified-csl-ab",
    "exp6278-model-family-task-holdout-csl-audit",
    "exp6279-certified-memory-shadow-consumer",
    "exp6280-variable-cardinality-mode-jump-backend",
    "exp6281-mode-jump-multifamily-rerun",
    "exp6282-arc-mechanic-class-live-router",
    "exp6283-v541-adversarial-capstone",
)
EXPECTED_V542_TASK_IDS = tuple(f"exp{exp_id}" for exp_id in range(6284, 6297))
EXPECTED_V542_TASK_IDS = (
    "exp6284-v542-terminal-transition",
    "exp6285-v542-post-marker-source-scope-freeze",
    "exp6286-v541-evidence-eligibility-ledger",
    "exp6287-asp-continuous-relaxation",
    "exp6288-partial-atom-evidence-adapter",
    "exp6289-flagship-exact-state-refinement-benchmark",
    "exp6290-revocable-atomic-repair-memory",
    "exp6291-chronological-memory-crystallization-ab",
    "exp6292-revocable-memory-holdout-audit",
    "exp6293-revocable-memory-shadow-consumer",
    "exp6294-arc-mechanic-router-causal-canary",
    "exp6295-arc-mechanic-router-holdout-audit",
    "exp6296-v542-adversarial-capstone",
)
RESERVED_EXP_IDS = tuple(range(6284, 6297))
GATE_OPS = frozenset({"==", "!=", ">", "<", ">=", "<=", "exists", "in"})
MANDATED_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
GGUF_ID_RE = re.compile(r"[\w.-]+/[\w.-]+-GGUF")

PROTECTED_RELATIVE_PATHS = (
    V542_ROADMAP_RELATIVE_PATH,
    V542_NEXT_ROADMAP_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    V541_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
)
INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    V542_ROADMAP_RELATIVE_PATH,
    V542_NEXT_ROADMAP_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    V541_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    Path("results/experiment_6274_asp_energy_semantic_compiler.json"),
    Path("results/experiment_6275_flagship_asp_constraint_verification_benchmark.json"),
    Path("results/experiment_6276_certified_dual_cache_admission.json"),
    Path("results/experiment_6280_variable_cardinality_mode_jump_backend.json"),
    Path("results/experiment_6281_mode_jump_multifamily_rerun.json"),
    Path("results/experiment_6282_arc_mechanic_class_live_router.json"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
)
EXPERIMENT_SCAN_ROOTS = (
    Path("python/carnot"),
    Path("scripts/experiments"),
    Path("results"),
    Path("tests/python"),
)
ALLOWED_LOCAL_RESERVED_PATHS = {
    "python/carnot/experiment_6284_v542_terminal_transition.py",
    "tests/python/test_experiment_6284_v542_terminal_transition.py",
    RESULT_RELATIVE_PATH.as_posix(),
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6284_v542_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6284_v542_terminal_transition.py -m pytest tests/python/test_experiment_6284_v542_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6284_v542_terminal_transition.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6284_v542_terminal_transition.py tests/python/test_experiment_6284_v542_terminal_transition.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6284_v542_terminal_transition.py tests/python/test_experiment_6284_v542_terminal_transition.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6284_v542_terminal_transition.py",
    ".venv/bin/python -m carnot.experiment_6284_v542_terminal_transition --check-roadmap-only",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6284_v542_terminal_transition.json",
)
COMMAND_TIMEOUTS_S = {".venv/bin/pytest tests/python -q": 3600}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v541_milestone_roadmap_and_hash",
    "v541_task_terminal_matrix",
    "v541_capstone_path_hash_and_summary",
    "operational_retro_path_hash_and_summary",
    "focused_and_broad_validation_receipts_by_task",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts",
    "v542_roadmap_path_and_hash",
    "v542_task_ids_and_deliverables",
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
    "raw_evidence_eligibility_policy",
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
    "status": "The handoff closes only after exact V541 evidence and V542 contracts are checked.",
    "v541_milestone_roadmap_and_hash": "The V541 denominator comes from the terminal capstone record.",
    "v541_task_terminal_matrix": "Every V541 task keeps the class of its exact artifact path.",
    "v541_capstone_path_hash_and_summary": "The V541 capstone is input evidence, not an override.",
    "operational_retro_path_hash_and_summary": "Runtime receipts stay separate from readiness.",
    "focused_and_broad_validation_receipts_by_task": "Focused checks and broad-suite failures stay separate.",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": "Mixed V541 states remain visible.",
    "v542_roadmap_path_and_hash": "The staged V542 roadmap identity is content-addressed.",
    "v542_task_ids_and_deliverables": "The V542 denominator is exactly Exp6284 through Exp6296.",
    "task_count": "Exactly thirteen tasks are reserved for V542.",
    "phase_counts": "Track counts keep the staged work auditable.",
    "dependency_validation": "Dependencies must point to live V542 tasks.",
    "gated_on_validation": "Gates may read only fields promised by upstream artifacts.",
    "prior_failure_validation": "Reruns need a prior, a changed mechanism, and a retirement rule.",
    "retired_dependency_count": "Bare zero proves no dependency points at a retired id.",
    "id_collision_count": "Bare zero proves reserved ids and deliverables have no unexpected file collision.",
    "agent_routing_validation": "Direct agent routes must match their model namespace.",
    "model_policy_validation": "Live LLM tasks must name mandated local SOTA GGUF models.",
    "prompt_contract_validation": "Run commands and endings prevent conductor drift.",
    "raw_evidence_eligibility_policy": "Raw receipts cannot override artifact-level eligibility.",
    "protected_files_unchanged": "Protected hashes show this run did not rewrite records.",
    "preconditions_checked": "Git state, hashes, roadmap identity, and collision scans are frozen first.",
    "inference_substrate": "This report aggregates checked-in artifacts only.",
    "verifier_is_oracle": "The handoff audits records and is not an answer oracle.",
    "field_provenance": "Every required field cites its source evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands show the verification boundary.",
    "test_exit_codes": "Exit codes are preserved without converting failures into passes.",
    "duration_s": "Wall time records the audit cost without padding.",
    "reproducibility_checksum": "A normalized checksum detects silent payload drift.",
    "honest_verdict": "The verdict states terminal status and preserved caveats plainly.",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def _tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def classify_v541_declared_tasks(
    root: Path,
    capstone_matrix: JsonMap,
    conductor_receipts: JsonMap | None = None,
) -> JsonDict:
    rows: JsonDict = {}
    receipts = conductor_receipts or {}
    for task_id, prior in capstone_matrix.items():
        if not isinstance(prior, Mapping):
            continue
        declared = Path(str(prior.get("declared_deliverable") or prior.get("deliverable") or ""))
        receipt = receipts.get(str(task_id))
        typed_receipt = receipt if isinstance(receipt, Mapping) else None
        classification = classify_artifact_path(
            root / declared, conductor_receipt=typed_receipt
        ).to_dict()
        payload, _meta = read_json_mapping(root / declared)
        rows[str(task_id)] = {
            "task_id": str(task_id),
            "title": str(prior.get("title") or task_id),
            "track": str(prior.get("track") or ""),
            "declared_deliverable": declared.as_posix(),
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"],
            "terminal_class": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "capstone_terminal_class": prior.get("terminal_class"),
            "capstone_terminal": prior.get("terminal"),
            "receipt_status": classification["conductor_receipt_status"],
            "receipt_override_attempted": classification["receipt_override_attempted"],
            "receipt_overrode": classification["receipt_overrode"],
            "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_stamped": bool(payload.get("corrigendum_pending")),
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": same_number_aliases(
                root, str(task_id), declared
            ),
        }
    return rows


def load_v542_roadmap(root: Path) -> tuple[JsonDict, JsonDict]:
    next_path = root / V542_NEXT_ROADMAP_RELATIVE_PATH
    active_path = root / V542_ROADMAP_RELATIVE_PATH
    next_data = read_yaml_mapping(next_path)
    active_data = read_yaml_mapping(active_path)
    if next_data.get("milestone") == MILESTONE_V542:
        chosen_rel = V542_NEXT_ROADMAP_RELATIVE_PATH
        data = next_data
        note = "research-roadmap-next.yaml contains V542"
    elif active_data.get("milestone") == MILESTONE_V542:
        chosen_rel = V542_ROADMAP_RELATIVE_PATH
        data = active_data
        note = "active research-roadmap.yaml already contains V542; no activation performed"
    else:
        chosen_rel = V542_NEXT_ROADMAP_RELATIVE_PATH if next_data else V542_ROADMAP_RELATIVE_PATH
        data = next_data or active_data
        note = "V542 roadmap milestone was not found"
    return data, {
        "path": chosen_rel.as_posix(),
        "sha256": path_sha256(root / chosen_rel),
        "milestone": data.get("milestone"),
        "requested_next_path": V542_NEXT_ROADMAP_RELATIVE_PATH.as_posix(),
        "research_roadmap_next_present": next_path.exists(),
        "active_roadmap_milestone": active_data.get("milestone"),
        "selection_note": note,
    }


def model_specs_named_in_prompt(prompt: str) -> list[str]:
    return sorted(set(GGUF_ID_RE.findall(str(prompt))))


def _validate_agent_route(task_id: str, agent_type: Any, model: Any) -> list[JsonDict]:
    failures: list[JsonDict] = []
    if agent_type == "codex" and model != "gpt-5.5":
        failures.append(
            {
                "task_id": task_id,
                "expected_agent_type": "codex",
                "expected_model": "gpt-5.5",
                "agent_type": agent_type,
                "model": model,
            }
        )
    if agent_type == "gemini":
        failures.append(
            {
                "task_id": task_id,
                "expected_agent_type": "not_gemini",
                "expected_model": None,
                "agent_type": agent_type,
                "model": model,
            }
        )
    if model == "opus" and agent_type not in (None, "claude"):
        failures.append(
            {
                "task_id": task_id,
                "expected_agent_type": "default_or_claude",
                "expected_model": "opus",
                "agent_type": agent_type,
                "model": model,
            }
        )
    return failures


def _validate_model_policy(task: JsonMap) -> list[JsonDict]:
    task_id = str(task.get("id") or "")
    agent_type = task.get("agent_type")
    model = task.get("model")
    prompt = str(task.get("prompt") or "")
    failures: list[JsonDict] = []
    if agent_type == "codex" and model != "gpt-5.5":
        failures.append({"task_id": task_id, "reason": "codex_requires_gpt_5_5"})
    if agent_type == "gemini" or model == "gemini-3.1-pro-preview":
        failures.append({"task_id": task_id, "reason": "gemini_not_allowed"})
    if model not in (None, "sonnet", "opus", "gpt-5.5"):
        failures.append({"task_id": task_id, "reason": "unknown_agent_model", "model": model})
    if task.get("requires_gpu") is True or "MODEL_SPECS" in prompt:
        named = model_specs_named_in_prompt(prompt)
        if task.get("requires_gpu") is True and not named:
            failures.append({"task_id": task_id, "reason": "missing_model_specs_gguf_ids"})
        unknown = [model_id for model_id in named if model_id not in MANDATED_GGUF_IDS]
        if unknown:
            failures.append({"task_id": task_id, "reason": "non_mandated_gguf_id", "ids": unknown})
        if "all three mandated" in prompt.lower() and not MANDATED_GGUF_IDS <= set(named):
            failures.append(
                {
                    "task_id": task_id,
                    "reason": "missing_all_three_mandated_gguf_ids",
                    "expected": sorted(MANDATED_GGUF_IDS),
                    "found": named,
                }
            )
    return failures


def validate_v542_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    id_counts = Counter(ids)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    duplicate_id_count = sum(count - 1 for count in id_counts.values() if count > 1)
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
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            if not ok:
                gate_failures.append({"task_id": task_id, "gate": gate, "reason": reason})

    prior_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        priors = task.get("prior_failures")
        if priors is None:
            continue
        if not isinstance(priors, list) or not priors:
            prior_failures.append({"task_id": task_id, "reason": "empty_prior_failures"})
            continue
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    route_failures: list[JsonDict] = []
    model_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        route_failures.extend(
            _validate_agent_route(task_id, task.get("agent_type"), task.get("model"))
        )
        model_failures.extend(_validate_model_policy(task))

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
        if not has_run or not has_ending:
            prompt_failures.append(
                {
                    "task_id": task_id,
                    "run_command_present": has_run,
                    "protected_conductor_ending": has_ending,
                }
            )

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "task_count": len(tasks),
        "phase_counts": dict(
            sorted(Counter(str(task.get("track") or "unset") for task in tasks).items())
        ),
        "task_id_validation": {
            "task_ids": ids,
            "expected_task_ids": list(EXPECTED_V542_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V542_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "dependency_validation": {"ok": not dependency_failures, "failures": dependency_failures},
        "gated_on_validation": {"ok": not gate_failures, "failures": gate_failures},
        "prior_failure_validation": {"ok": not prior_failures, "failures": prior_failures},
        "agent_routing_validation": {"ok": not route_failures, "failures": route_failures},
        "model_policy_validation": {"ok": not model_failures, "failures": model_failures},
        "prompt_contract_validation": {"ok": not prompt_failures, "failures": prompt_failures},
        "retired_dependency_count": int(retired_dependency_count),
        "id_collision_count": int(duplicate_id_count),
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


def scan_reserved_id_collisions(
    root: Path,
    *,
    allowed_reserved_paths: set[str] | None = None,
    staged_deliverables: set[str] | None = None,
) -> JsonDict:
    allowed = set(allowed_reserved_paths or set())
    deliverables = set(staged_deliverables or set())
    unexpected: dict[str, list[str]] = {str(exp_id): [] for exp_id in RESERVED_EXP_IDS}
    for rel in _experiment_paths(root):
        number = exp_number(rel.name)
        rel_text = rel.as_posix()
        if number in RESERVED_EXP_IDS and rel_text not in allowed:
            unexpected[str(number)].append(rel_text)
    existing_deliverables = sorted(
        rel for rel in deliverables if (root / rel).exists() and rel not in allowed
    )
    for rel in existing_deliverables:
        number = exp_number(Path(rel).name)
        if number in RESERVED_EXP_IDS and rel not in unexpected[str(number)]:
            unexpected[str(number)].append(rel)
    unexpected = {key: sorted(value) for key, value in unexpected.items() if value}
    return {
        "scan_roots": [path.as_posix() for path in EXPERIMENT_SCAN_ROOTS],
        "reserved_exp_ids": list(RESERVED_EXP_IDS),
        "allowed_reserved_paths": sorted(allowed),
        "staged_deliverables": sorted(deliverables),
        "existing_unallowed_deliverables": existing_deliverables,
        "unexpected_reserved_paths_by_exp_id": unexpected,
        "unexpected_reserved_collision_count": sum(len(paths) for paths in unexpected.values()),
        "tracked_and_untracked_basis": "filesystem scan covers tracked and untracked files under experiment roots",
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


def input_hashes(root: Path) -> JsonDict:
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in INPUT_RELATIVE_PATHS
    }


def _allowed_reserved_paths() -> set[str]:
    return set(ALLOWED_LOCAL_RESERVED_PATHS)


def _v541_counts(matrix: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("terminal_class")) for row in matrix.values())
    nonterminal = sum(1 for row in matrix.values() if row.get("terminal") is not True)
    flagged = sum(
        1
        for row in matrix.values()
        if row.get("flagged_adversarial_stamped")
        or row.get("corrigendum_pending_stamped")
        or row.get("terminal_class") == "flagged"
    )
    return {
        "missing": int(counts.get("missing", 0)),
        "nonterminal": int(nonterminal),
        "blocked": int(counts.get("blocked", 0)),
        "skipped": int(counts.get("skipped", 0)),
        "null": int(counts.get("null", 0)),
        "flagged": int(max(counts.get("flagged", 0), flagged)),
        "retired": int(counts.get("retired", 0)),
        "ready": int(counts.get("ready", 0)),
        "complete": int(counts.get("complete", 0)),
        "unknown": int(counts.get("unknown", 0)),
        "classification_counts": dict(sorted((key, int(value)) for key, value in counts.items())),
    }


def _v542_task_deliverables(data: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in _tasks(data):
        rows.append(
            {
                "task_id": str(task.get("id") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "track": str(task.get("track") or ""),
                "requires": list(task.get("requires") or [])
                if isinstance(task.get("requires"), list)
                else [],
                "gated_on": list(task.get("gated_on") or [])
                if isinstance(task.get("gated_on"), list)
                else [],
                "agent_type": task.get("agent_type"),
                "model": task.get("model"),
                "requires_gpu": task.get("requires_gpu") is True,
                "model_specs_named_in_prompt": model_specs_named_in_prompt(
                    str(task.get("prompt") or "")
                ),
            }
        )
    return rows


def _is_broad_command(command: str) -> bool:
    return command.strip() == ".venv/bin/pytest tests/python -q"


def _test_exits(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _artifact_is_flagged(row: JsonMap) -> bool:
    return bool(
        row.get("terminal_class") == "flagged"
        or row.get("flagged_adversarial_stamped")
        or row.get("corrigendum_pending_stamped")
    )


def raw_evidence_eligibility_policy(matrix: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    raw_reusable = {
        "exp6275-flagship-asp-constraint-verification-benchmark",
        "exp6282-arc-mechanic-class-live-router",
    }
    source_reusable = {
        "exp6274-asp-energy-semantic-compiler",
        "exp6280-variable-cardinality-mode-jump-backend",
        "exp6282-arc-mechanic-class-live-router",
    }
    for task_id, row in matrix.items():
        if not isinstance(row, Mapping):
            continue
        terminal = row.get("terminal") is True
        terminal_class = str(row.get("terminal_class") or "unknown")
        flagged = _artifact_is_flagged(row)
        artifact_gate_eligible = (
            terminal
            and not flagged
            and terminal_class
            in {
                "complete",
                "ready",
                "positive",
            }
        )
        reason = "eligible"
        if flagged:
            reason = "flagged_artifact"
        elif not terminal:
            reason = "nonterminal_or_missing_artifact"
        elif terminal_class not in {"complete", "ready", "positive"}:
            reason = f"{terminal_class}_artifact_not_gate_ready"
        rows[str(task_id)] = {
            "declared_deliverable": row.get("declared_deliverable"),
            "terminal_class": terminal_class,
            "terminal": terminal,
            "raw_receipts_may_be_reused": str(task_id) in raw_reusable,
            "source_may_be_reused": str(task_id) in source_reusable,
            "artifact_gate_eligible": artifact_gate_eligible,
            "artifact_level_quarantine_reason": reason,
        }
    return {
        "exact_path_over_raw_receipt_precedence": True,
        "raw_receipts_cannot_override_artifact_level_eligibility": True,
        "rows": rows,
        "principle": FIELD_PRINCIPLES["raw_evidence_eligibility_policy"],
    }


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6284",
        V541_CAPSTONE_RELATIVE_PATH.as_posix(),
        OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        V542_ROADMAP_RELATIVE_PATH.as_posix(),
        V542_NEXT_ROADMAP_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        TERMINAL_ARTIFACTS_RELATIVE_PATH.as_posix(),
        "scripts/roadmap_schema.py",
        "scripts/audit_roadmap_gates.py",
        "scripts/exclusion_manifest_lint.py",
        "python/carnot/inference/sota_models.py",
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sorted(sources)}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    v542_identity: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    collision_receipt_before: JsonMap,
    input_hashes_before: JsonMap,
    git_status_after_tests: Sequence[str] | None = None,
) -> JsonDict:
    return {
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests or []),
        "input_hashes_before": input_hashes_before,
        "staged_roadmap_identity": v542_identity,
        "reserved_id_scan_before_artifact_write": collision_receipt_before,
        "protected_hashes_before_artifact_write": before_hashes,
        "active_roadmap_was_not_edited_by_this_task": True,
        "research_roadmap_next_was_not_activated_by_this_task": True,
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    before_hashes: JsonMap | None = None,
    git_status_before: Sequence[str] | None = None,
    collision_receipt_before: JsonMap | None = None,
    input_hashes_before: JsonMap | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    v542_data, v542_identity = load_v542_roadmap(root)
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    staged_deliverables = {row["deliverable"] for row in _v542_task_deliverables(v542_data)}
    collision_before = dict(
        scan_reserved_id_collisions(
            root,
            allowed_reserved_paths=_allowed_reserved_paths(),
            staged_deliverables=staged_deliverables,
        )
        if collision_receipt_before is None
        else collision_receipt_before
    )
    inputs_before = dict(input_hashes(root) if input_hashes_before is None else input_hashes_before)
    capstone_payload, capstone_meta = read_json_mapping(root / V541_CAPSTONE_RELATIVE_PATH)
    retro_payload, retro_meta = read_json_mapping(root / OPERATIONAL_RETRO_RELATIVE_PATH)

    capstone_matrix = capstone_payload.get("exact_declared_deliverable_matrix")
    conductor_receipts = capstone_payload.get("conductor_receipt_matrix")
    matrix = classify_v541_declared_tasks(
        root,
        capstone_matrix if isinstance(capstone_matrix, Mapping) else {},
        conductor_receipts if isinstance(conductor_receipts, Mapping) else {},
    )
    validations = focused_and_broad_validation_receipts_by_task(root, matrix)
    command_rows = [dict(row) for row in (command_receipts or [])]
    if command_rows:
        buckets = _bucket_command_rows(command_rows)
        validations[EXPERIMENT_ID] = {"task_id": EXPERIMENT_ID, **buckets}

    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    v542_validation = validate_v542_roadmap_data(v542_data, retired_ids)
    file_collision_count = int(collision_before.get("unexpected_reserved_collision_count") or 0)
    duplicate_id_count = int(v542_validation["id_collision_count"])
    id_collision_count = file_collision_count + duplicate_id_count
    capstone_roadmap = capstone_payload.get("milestone_roadmap_path_and_hash")
    if not isinstance(capstone_roadmap, Mapping):
        capstone_roadmap = {}

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete",
        "v541_milestone_roadmap_and_hash": {
            "milestone": MILESTONE_V541,
            "capstone_path": V541_CAPSTONE_RELATIVE_PATH.as_posix(),
            "capstone_sha256": capstone_meta.get("sha256"),
            "recorded_roadmap_path": capstone_roadmap.get("roadmap_path"),
            "recorded_roadmap_sha256": capstone_roadmap.get("roadmap_sha256"),
            "recorded_task_ids": capstone_roadmap.get("task_ids"),
            "expected_task_ids": list(EXPECTED_V541_TASK_IDS),
            "recorded_task_count": capstone_roadmap.get("task_count"),
            "research_complete_path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            "research_complete_sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        },
        "v541_task_terminal_matrix": matrix,
        "v541_capstone_path_hash_and_summary": {
            **capstone_meta,
            "summary": {
                "status": capstone_payload.get("status"),
                "honest_verdict": capstone_payload.get("honest_verdict"),
                "counts": capstone_payload.get(
                    "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts"
                ),
                "promotion_ledger": capstone_payload.get("branch_independent_promotion_ledger"),
            },
        },
        "operational_retro_path_hash_and_summary": {
            **retro_meta,
            "summary": {
                "milestone": retro_payload.get("milestone"),
                "retro_type": retro_payload.get("retro_type"),
                "summary": retro_payload.get("summary"),
                "experiments_completed": retro_payload.get("experiments_completed"),
                "timing_integrity_mismatch": retro_payload.get("timing_integrity_mismatch"),
            },
        },
        "focused_and_broad_validation_receipts_by_task": validations,
        "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": _v541_counts(
            matrix
        ),
        "v542_roadmap_path_and_hash": v542_identity,
        "v542_task_ids_and_deliverables": _v542_task_deliverables(v542_data),
        "task_count": v542_validation["task_count"],
        "phase_counts": v542_validation["phase_counts"],
        "dependency_validation": v542_validation["dependency_validation"],
        "gated_on_validation": v542_validation["gated_on_validation"],
        "prior_failure_validation": v542_validation["prior_failure_validation"],
        "retired_dependency_count": v542_validation["retired_dependency_count"],
        "id_collision_count": id_collision_count,
        "agent_routing_validation": v542_validation["agent_routing_validation"],
        "model_policy_validation": v542_validation["model_policy_validation"],
        "prompt_contract_validation": v542_validation["prompt_contract_validation"],
        "raw_evidence_eligibility_policy": raw_evidence_eligibility_policy(matrix),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            v542_identity,
            before,
            status_before,
            collision_before,
            inputs_before,
            git_status_after_tests,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exits(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": "complete: V541 exact states and V542 roadmap contracts validated; raw receipts remain separate from artifact eligibility",
    }
    blocking_command_failures = [
        row
        for row in command_rows
        if int(row.get("exit_code") or 0) != 0
        and not _is_broad_command(str(row.get("command") or ""))
    ]
    if (
        report["task_count"] != 13
        or report["retired_dependency_count"] != 0
        or report["id_collision_count"] != 0
        or not report["dependency_validation"]["ok"]
        or not report["gated_on_validation"]["ok"]
        or not report["prior_failure_validation"]["ok"]
        or not report["agent_routing_validation"]["ok"]
        or not report["model_policy_validation"]["ok"]
        or not report["prompt_contract_validation"]["ok"]
        or not report["protected_files_unchanged"]["unchanged"]
        or blocking_command_failures
    ):
        report["status"] = "blocked"
        report["honest_verdict"] = (
            "blocked: V542 transition validation found a contract failure or task-owned command failure"
        )
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _bucket_command_rows(commands: Sequence[JsonMap]) -> JsonDict:
    focused: list[JsonDict] = []
    broad: list[JsonDict] = []
    for row in commands:
        command = str(row.get("command") or "")
        exit_code = int(row.get("exit_code") or 0)
        target = broad if _is_broad_command(command) else focused
        target.append(
            {
                "command": command,
                "exit_code": exit_code,
                "classification": "passed" if exit_code == 0 else f"nonzero_exit_{exit_code}",
            }
        )
    return {
        "focused": {
            "command_count": len(focused),
            "failed_count": sum(1 for row in focused if row["exit_code"] != 0),
            "commands": focused,
        },
        "broad": {
            "command_count": len(broad),
            "failed_count": sum(1 for row in broad if row["exit_code"] != 0),
            "commands": broad,
        },
    }


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
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if report.get("task_count") != 13:
        errors.append("task_count must be 13")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not (
        type(report.get("retired_dependency_count")) is int
        and report.get("retired_dependency_count") == 0
    ):
        errors.append("retired_dependency_count must be bare integer 0")
    if not (
        type(report.get("id_collision_count")) is int and report.get("id_collision_count") == 0
    ):
        errors.append("id_collision_count must be bare integer 0")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(
        (
            "complete:",
            "complete_ready:",
            "complete_null:",
            "blocked:",
            "blocked_safety:",
            "skipped:",
        )
    ):
        errors.append("honest_verdict lacks accepted Exp6284 prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def run_command(
    command: str, root: Path, timeout_s: int | None = None
) -> JsonDict:  # pragma: no cover
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover
    rows: list[JsonDict] = []
    for command in DEFAULT_TEST_COMMANDS:
        rows.append(run_command(command, root, COMMAND_TIMEOUTS_S.get(command)))
    return rows


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6284 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(root: Path, date: str, *, run_commands: bool) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    before = protected_hashes(root)
    git_before = git_status_lines(root)
    inputs_before = input_hashes(root)
    v542_data, _identity = load_v542_roadmap(root)
    staged_deliverables = {row["deliverable"] for row in _v542_task_deliverables(v542_data)}
    collision_before = scan_reserved_id_collisions(
        root,
        allowed_reserved_paths=_allowed_reserved_paths(),
        staged_deliverables=staged_deliverables,
    )
    preliminary = build_report(
        root,
        date=date,
        command_receipts=[],
        before_hashes=before,
        git_status_before=git_before,
        collision_receipt_before=collision_before,
        input_hashes_before=inputs_before,
        started_at=started,
    )
    write_report(preliminary, root)
    command_rows = run_default_commands(root) if run_commands else []
    final = build_report(
        root,
        date=date,
        command_receipts=command_rows,
        before_hashes=before,
        git_status_before=git_before,
        collision_receipt_before=collision_before,
        input_hashes_before=inputs_before,
        git_status_after_tests=git_status_lines(root),
        started_at=started,
    )
    write_report(final, root)
    return final


def check_roadmap_only(root: Path = REPO_ROOT) -> JsonDict:
    data, identity = load_v542_roadmap(root)
    validation = validate_v542_roadmap_data(
        data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    staged_deliverables = {row["deliverable"] for row in _v542_task_deliverables(data)}
    collision_receipt = scan_reserved_id_collisions(
        root,
        allowed_reserved_paths=_allowed_reserved_paths(),
        staged_deliverables=staged_deliverables,
    )
    file_collisions = int(collision_receipt.get("unexpected_reserved_collision_count") or 0)
    ok = (
        identity.get("milestone") == MILESTONE_V542
        and validation["task_id_validation"]["expected_order"]
        and validation["task_count"] == 13
        and validation["id_collision_count"] == 0
        and validation["retired_dependency_count"] == 0
        and file_collisions == 0
        and validation["dependency_validation"]["ok"]
        and validation["gated_on_validation"]["ok"]
        and validation["prior_failure_validation"]["ok"]
        and validation["agent_routing_validation"]["ok"]
        and validation["model_policy_validation"]["ok"]
        and validation["prompt_contract_validation"]["ok"]
    )
    return {
        "ok": ok,
        "roadmap_identity": identity,
        "collision_validation": collision_receipt,
        **validation,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--check-roadmap-only", action="store_true")
    parser.add_argument("--no-run-commands", action="store_true")
    args = parser.parse_args(argv)
    if args.check_roadmap_only:
        result = check_roadmap_only(REPO_ROOT)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    report = run_experiment(REPO_ROOT, args.date, run_commands=not args.no_run_commands)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
