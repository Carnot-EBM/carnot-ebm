"""Exp6272 V540-to-V541 exact terminal transition.

Spec refs: REQ-INFRA-6272, SCENARIO-INFRA-6272-1,
SCENARIO-INFRA-6272-2, SCENARIO-INFRA-6272-3,
SCENARIO-INFRA-6272-4, SCENARIO-INFRA-6272-5,
SCENARIO-INFRA-6272-6.
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

import yaml

from carnot.experiment_artifacts import atomic_write_json
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


MILESTONE_V540 = "2026.08.540"
MILESTONE_V541 = "2026.08.541"
EXPERIMENT_ID = "exp6272-v541-terminal-transition"
SCHEMA = "carnot.experiment_6272.v541_terminal_transition.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6272_v541_terminal_transition.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

V540_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6271_v540_adversarial_capstone.json")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_540.json")
V541_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V541_NEXT_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

EXPECTED_V540_TASK_IDS = (
    "exp6260-v540-terminal-transition",
    "exp6261-v540-post-marker-source-scope-freeze",
    "exp6262-terminal-artifact-readiness-contract",
    "exp6263-clean-sota-event-replay-bridge",
    "exp6264-energy-familiarity-memory-gate",
    "exp6265-chronological-two-timescale-csl-ab",
    "exp6266-family-task-holdout-csl-audit",
    "exp6267-constraint-memory-shadow-consumer-v2",
    "exp6268-multimodal-sampler-fixture-suite",
    "exp6269-mode-jump-multifamily-ab",
    "exp6270-mode-jump-descriptor-router",
    "exp6271-v540-adversarial-capstone",
)
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
RESERVED_EXP_IDS = tuple(range(6272, 6284))
GATE_OPS = frozenset({"==", "!=", ">", "<", ">=", "<=", "exists", "in"})

CODEX_TASK_IDS = frozenset(
    {
        "exp6272-v541-terminal-transition",
        "exp6274-asp-energy-semantic-compiler",
        "exp6280-variable-cardinality-mode-jump-backend",
        "exp6281-mode-jump-multifamily-rerun",
        "exp6282-arc-mechanic-class-live-router",
    }
)
OPUS_TASK_IDS = frozenset(
    {
        "exp6275-flagship-asp-constraint-verification-benchmark",
        "exp6279-certified-memory-shadow-consumer",
    }
)

PROTECTED_RELATIVE_PATHS = (
    V541_ROADMAP_RELATIVE_PATH,
    V541_NEXT_ROADMAP_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    V540_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    Path("python/carnot/terminal_artifacts.py"),
)
INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    V541_ROADMAP_RELATIVE_PATH,
    V541_NEXT_ROADMAP_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    V540_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    Path("results/experiment_6263_clean_sota_event_replay_bridge.json"),
    Path("results/experiment_6264_energy_familiarity_memory_gate.json"),
    Path("results/experiment_6268_multimodal_sampler_fixture_suite.json"),
    Path("results/experiment_6269_mode_jump_multifamily_ab.json"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("python/carnot/terminal_artifacts.py"),
)
EXPERIMENT_SCAN_ROOTS = (
    Path("python/carnot"),
    Path("scripts/experiments"),
    Path("results"),
    Path("tests/python"),
)
ALLOWED_LOCAL_RESERVED_PATHS = {
    "python/carnot/experiment_6272_v541_terminal_transition.py",
    "tests/python/test_experiment_6272_v541_terminal_transition.py",
    RESULT_RELATIVE_PATH.as_posix(),
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6272_v541_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6272_v541_terminal_transition.py -m pytest tests/python/test_experiment_6272_v541_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6272_v541_terminal_transition.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6272_v541_terminal_transition.py tests/python/test_experiment_6272_v541_terminal_transition.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6272_v541_terminal_transition.py tests/python/test_experiment_6272_v541_terminal_transition.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6272_v541_terminal_transition.py",
    ".venv/bin/python -m carnot.experiment_6272_v541_terminal_transition --check-roadmap-only",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6272_v541_terminal_transition.json",
)
COMMAND_TIMEOUTS_S = {".venv/bin/pytest tests/python -q": 3600}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v540_milestone_roadmap_and_hash",
    "v540_task_terminal_matrix",
    "v540_capstone_path_hash_and_summary",
    "operational_retro_path_hash_and_summary",
    "focused_and_broad_validation_receipts_by_task",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts",
    "v541_roadmap_path_and_hash",
    "v541_task_ids_and_deliverables",
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
    "status": "The handoff closes only when exact evidence and roadmap checks are complete.",
    "v540_milestone_roadmap_and_hash": "The V540 denominator must come from a content-addressed record.",
    "v540_task_terminal_matrix": "Each V540 task keeps the class of its exact deliverable.",
    "v540_capstone_path_hash_and_summary": "The capstone is input evidence, not a replacement classifier.",
    "operational_retro_path_hash_and_summary": "Runtime facts stay separate from scientific readiness.",
    "focused_and_broad_validation_receipts_by_task": "Focused passes and broad failures must not overwrite each other.",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": "Mixed terminal states remain visible in counts.",
    "v541_roadmap_path_and_hash": "The V541 roadmap identity must be explicit and hashed.",
    "v541_task_ids_and_deliverables": "The staged task denominator is exactly Exp6272 through Exp6283.",
    "task_count": "Exactly twelve tasks are reserved for V541.",
    "phase_counts": "Phase counts keep independent work tracks auditable.",
    "dependency_validation": "Dependencies must point to live V541 tasks.",
    "gated_on_validation": "Gates may read only fields promised by upstream artifacts.",
    "prior_failure_validation": "Reruns need a stated prior, difference, and retirement rule.",
    "retired_dependency_count": "Bare zero proves no dependency chain points at a retired id.",
    "id_collision_count": "Bare zero proves the reserved id range has no unexpected file collision.",
    "agent_routing_validation": "Formulaic work stays on Codex and judgment work stays default or Opus.",
    "model_policy_validation": "Model choices must match the declared routing policy.",
    "prompt_contract_validation": "Prompt run commands and endings prevent conductor drift.",
    "protected_files_unchanged": "Protected files must be byte-identical across the write.",
    "preconditions_checked": "Git state, hashes, roadmap identity, and collision scans are recorded.",
    "inference_substrate": "This report aggregates checked-in files, not live inference.",
    "verifier_is_oracle": "The handoff audits records and is not an answer oracle.",
    "field_provenance": "Every field cites the files or checks that produced it.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "The artifact records focused, broad, and bounded validation commands.",
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


def read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


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


def exp_number(text: str) -> int | None:
    match = re.search(r"exp(?:eriment_)?(\d+)|\b(\d{4,5})\b", str(text), re.IGNORECASE)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def same_number_aliases(root: Path, task_id: str, declared_rel: Path) -> list[str]:
    number = exp_number(task_id)
    results_dir = root / "results"
    if number is None or not results_dir.exists():
        return []
    declared = (root / declared_rel).resolve()
    aliases: list[str] = []
    for candidate in sorted(results_dir.glob(f"experiment_{number}*.json")):
        if candidate.resolve() != declared:
            aliases.append(candidate.relative_to(root).as_posix())
    return aliases


def classify_v540_declared_tasks(
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
        if not isinstance(receipt, Mapping):
            receipt = None
        classification = classify_artifact_path(
            root / declared, conductor_receipt=receipt
        ).to_dict()
        aliases = same_number_aliases(root, str(task_id), declared)
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
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": aliases,
        }
    return rows


def _tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    return (
        [dict(task) for task in tasks if isinstance(task, Mapping)]
        if isinstance(tasks, list)
        else []
    )


def load_v541_roadmap(root: Path) -> tuple[JsonDict, JsonDict]:
    next_path = root / V541_NEXT_ROADMAP_RELATIVE_PATH
    active_path = root / V541_ROADMAP_RELATIVE_PATH
    next_data = read_yaml_mapping(next_path)
    active_data = read_yaml_mapping(active_path)
    if next_data.get("milestone") == MILESTONE_V541:
        chosen_rel = V541_NEXT_ROADMAP_RELATIVE_PATH
        data = next_data
        note = "research-roadmap-next.yaml contains V541"
    elif active_data.get("milestone") == MILESTONE_V541:
        chosen_rel = V541_ROADMAP_RELATIVE_PATH
        data = active_data
        note = "active research-roadmap.yaml already contains V541; no activation performed"
    else:
        chosen_rel = V541_NEXT_ROADMAP_RELATIVE_PATH if next_data else V541_ROADMAP_RELATIVE_PATH
        data = next_data or active_data
        note = "V541 roadmap milestone was not found"
    return data, {
        "path": chosen_rel.as_posix(),
        "sha256": path_sha256(root / chosen_rel),
        "milestone": data.get("milestone"),
        "requested_next_path": V541_NEXT_ROADMAP_RELATIVE_PATH.as_posix(),
        "research_roadmap_next_present": next_path.exists(),
        "active_roadmap_milestone": active_data.get("milestone"),
        "selection_note": note,
    }


def load_retired_exp_ids(path: Path) -> set[int]:
    data = read_yaml_mapping(path)
    retired: set[int] = set()
    for key in ("retired", "retired_experiments", "retired_extras"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for value in (row.get("experiment_id"), row.get("retired_by_experiment_id")):
                number = exp_number(str(value))
                if number is not None:
                    retired.add(number)
            ids = row.get("experiment_ids")
            if isinstance(ids, list):
                for value in ids:
                    number = exp_number(str(value))
                    if number is not None:
                        retired.add(number)
    return retired


def required_artifact_fields_from_prompt(prompt: str) -> set[str]:
    lines = str(prompt).splitlines()
    block: list[str] = []
    for index, line in enumerate(lines):
        if "REQUIRED ARTIFACT FIELDS:" not in line.upper():
            continue
        block.append(line.split(":", 1)[1] if ":" in line else line)
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped or stripped.endswith(":") or stripped.startswith("CONCRETE STEPS"):
                break
            block.append(stripped)
        break
    if not block:
        return set()
    words = set(re.findall(r"\b[a-z][a-z0-9_]*\b", " ".join(block)))
    stopwords = {"required", "artifact", "fields", "and", "or", "must", "be", "bare"}
    return words - stopwords


def gate_ok(
    gate: Any,
    tasks_by_id: Mapping[str, JsonMap],
    required_fields_by_id: Mapping[str, set[str]] | set[str],
) -> tuple[bool, str | None]:
    if not isinstance(gate, Mapping):
        return False, "gate_not_mapping"
    for field in ("upstream", "artifact_field", "op", "value"):
        if field not in gate:
            return False, f"missing_{field}"
    if str(gate.get("op")) not in GATE_OPS:
        return False, "bad_op"
    upstream = str(gate.get("upstream"))
    if upstream not in tasks_by_id:
        return False, "unknown_upstream"
    fields = (
        required_fields_by_id.get(upstream, set())
        if isinstance(required_fields_by_id, Mapping)
        else required_fields_by_id
    )
    if str(gate.get("artifact_field")) not in fields:
        return False, "artifact_field_not_in_required_block"
    return True, None


def prior_ok(prior: Any) -> tuple[bool, str | None]:
    if not isinstance(prior, Mapping):
        return False, "prior_not_mapping"
    for field in ("experiment_id", "verdict", "addressed_by"):
        if not str(prior.get(field) or "").strip():
            return False, f"missing_{field}"
    if prior.get("retire_if_same_verdict") is not True:
        return False, "retire_if_same_verdict_not_true"
    return True, None


def module_name_for_task(task: JsonMap) -> str:
    stem = Path(str(task.get("deliverable") or "")).stem
    return stem.replace("-", "_")


def _expected_route(task_id: str) -> tuple[str, str | None]:
    if task_id in CODEX_TASK_IDS:
        return "codex", "gpt-5.5"
    if task_id in OPUS_TASK_IDS:
        return "default_or_claude", "opus"
    return "default", None


def validate_v541_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
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
    except Exception as exc:  # noqa: BLE001 - validation details are serialized.
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
        agent_type = task.get("agent_type")
        model = task.get("model")
        route_kind, expected_model = _expected_route(task_id)
        if route_kind == "codex":
            if agent_type != "codex" or model != expected_model:
                route_failures.append(
                    {
                        "task_id": task_id,
                        "expected_agent_type": "codex",
                        "expected_model": expected_model,
                        "agent_type": agent_type,
                        "model": model,
                    }
                )
        elif route_kind == "default_or_claude":
            if agent_type not in (None, "claude") or model != "opus":
                route_failures.append(
                    {
                        "task_id": task_id,
                        "expected_agent_type": "default_or_claude",
                        "expected_model": "opus",
                        "agent_type": agent_type,
                        "model": model,
                    }
                )
        elif agent_type is not None or model is not None:
            route_failures.append(
                {
                    "task_id": task_id,
                    "expected_agent_type": "default",
                    "expected_model": None,
                    "agent_type": agent_type,
                    "model": model,
                }
            )

        if agent_type == "gemini" or model == "gemini-3.1-pro-preview":
            model_failures.append({"task_id": task_id, "reason": "gemini_not_allowed"})
        if agent_type == "codex" and model != "gpt-5.5":
            model_failures.append({"task_id": task_id, "reason": "codex_requires_gpt_5_5"})
        if model == "opus" and task_id not in OPUS_TASK_IDS:
            model_failures.append({"task_id": task_id, "reason": "opus_route_not_declared"})
        if task_id in CODEX_TASK_IDS and (agent_type != "codex" or model != "gpt-5.5"):
            model_failures.append({"task_id": task_id, "reason": "missing_codex_gpt_5_5"})

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
            "expected_task_ids": list(EXPECTED_V541_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V541_TASK_IDS),
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
) -> JsonDict:
    allowed = set(allowed_reserved_paths or set())
    unexpected: dict[str, list[str]] = {str(exp_id): [] for exp_id in RESERVED_EXP_IDS}
    for rel in _experiment_paths(root):
        number = exp_number(rel.name)
        rel_text = rel.as_posix()
        if number in RESERVED_EXP_IDS and rel_text not in allowed:
            unexpected[str(number)].append(rel_text)
    unexpected = {key: value for key, value in unexpected.items() if value}
    return {
        "scan_roots": [path.as_posix() for path in EXPERIMENT_SCAN_ROOTS],
        "reserved_exp_ids": list(RESERVED_EXP_IDS),
        "allowed_reserved_paths": sorted(allowed),
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


def git_status_lines(root: Path) -> list[str]:  # pragma: no cover - shell edge.
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git_status_failed:{proc.returncode}:{proc.stderr.strip()}"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _allowed_reserved_paths() -> set[str]:
    return set(ALLOWED_LOCAL_RESERVED_PATHS)


def _v540_counts(matrix: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("terminal_class")) for row in matrix.values())
    nonterminal = sum(1 for row in matrix.values() if row.get("terminal") is not True)
    return {
        "missing": int(counts.get("missing", 0)),
        "nonterminal": int(nonterminal),
        "blocked": int(counts.get("blocked", 0)),
        "skipped": int(counts.get("skipped", 0)),
        "null": int(counts.get("null", 0)),
        "flagged": int(counts.get("flagged", 0)),
        "retired": int(counts.get("retired", 0)),
        "ready": int(counts.get("ready", 0)),
        "complete": int(counts.get("complete", 0)),
        "unknown": int(counts.get("unknown", 0)),
        "classification_counts": dict(sorted((key, int(value)) for key, value in counts.items())),
    }


def _v541_task_deliverables(data: JsonMap) -> list[JsonDict]:
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
            }
        )
    return rows


def _is_broad_command(command: str) -> bool:
    return command.strip() == ".venv/bin/pytest tests/python -q"


def _bucket_commands(commands: Sequence[JsonMap]) -> JsonDict:
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


def _artifact_command_rows(payload: JsonMap) -> list[JsonDict]:
    commands = payload.get("test_commands")
    exits = payload.get("test_exit_codes")
    if not isinstance(commands, list) or not isinstance(exits, Mapping):
        return []
    rows: list[JsonDict] = []
    normalized_exits = {str(key): value for key, value in exits.items()}
    for command in commands:
        command_text = str(command)
        exit_value = normalized_exits.get(command_text, 0)
        try:
            exit_code = int(exit_value)
        except (TypeError, ValueError):
            exit_code = 1
        rows.append({"command": command_text, "exit_code": exit_code})
    return rows


def focused_and_broad_validation_receipts_by_task(root: Path, v540_matrix: JsonMap) -> JsonDict:
    receipts: JsonDict = {}
    for task_id, row in v540_matrix.items():
        if not isinstance(row, Mapping):
            continue
        rel = Path(str(row.get("declared_deliverable") or ""))
        payload, meta = read_json_mapping(root / rel)
        buckets = _bucket_commands(_artifact_command_rows(payload))
        state = (
            "commands_recorded"
            if buckets["focused"]["command_count"] or buckets["broad"]["command_count"]
            else "no_command_receipts"
        )
        if not meta["present"]:
            state = "missing_artifact"
        elif not meta["loadable"]:
            state = "unloadable_artifact"
        artifact_classification = row.get("terminal_class")
        if artifact_classification is None:
            artifact_classification = classify_artifact_path(root / rel).classification
        receipts[str(task_id)] = {
            "task_id": str(task_id),
            "artifact_path": rel.as_posix(),
            "artifact_sha256": meta.get("sha256"),
            "artifact_classification": artifact_classification,
            "receipt_state": state,
            "focused": buckets["focused"],
            "broad": buckets["broad"],
        }
    return receipts


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6272",
        V540_CAPSTONE_RELATIVE_PATH.as_posix(),
        OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        V541_ROADMAP_RELATIVE_PATH.as_posix(),
        V541_NEXT_ROADMAP_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "python/carnot/terminal_artifacts.py",
        "scripts/roadmap_schema.py",
        "scripts/audit_roadmap_gates.py",
        "scripts/exclusion_manifest_lint.py",
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sorted(sources)}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def preconditions_checked(
    v541_identity: JsonMap,
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
        "staged_roadmap_identity": v541_identity,
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
    v541_data, v541_identity = load_v541_roadmap(root)
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    collision_before = dict(
        scan_reserved_id_collisions(root, allowed_reserved_paths=_allowed_reserved_paths())
        if collision_receipt_before is None
        else collision_receipt_before
    )
    inputs_before = dict(input_hashes(root) if input_hashes_before is None else input_hashes_before)
    capstone_payload, capstone_meta = read_json_mapping(root / V540_CAPSTONE_RELATIVE_PATH)
    retro_payload, retro_meta = read_json_mapping(root / OPERATIONAL_RETRO_RELATIVE_PATH)

    capstone_matrix = capstone_payload.get("exact_declared_deliverable_matrix")
    conductor_receipts = capstone_payload.get("conductor_receipt_matrix")
    matrix = classify_v540_declared_tasks(
        root,
        capstone_matrix if isinstance(capstone_matrix, Mapping) else {},
        conductor_receipts if isinstance(conductor_receipts, Mapping) else {},
    )
    validations = focused_and_broad_validation_receipts_by_task(root, matrix)
    command_rows = [dict(row) for row in (command_receipts or [])]
    if command_rows:
        validations[EXPERIMENT_ID] = {"task_id": EXPERIMENT_ID, **_bucket_commands(command_rows)}

    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    v541_validation = validate_v541_roadmap_data(v541_data, retired_ids)
    file_collision_count = int(collision_before.get("unexpected_reserved_collision_count") or 0)
    duplicate_id_count = int(v541_validation["id_collision_count"])
    id_collision_count = file_collision_count + duplicate_id_count
    capstone_roadmap = capstone_payload.get("milestone_roadmap_path_and_hash")
    if not isinstance(capstone_roadmap, Mapping):
        capstone_roadmap = {}

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete",
        "v540_milestone_roadmap_and_hash": {
            "milestone": MILESTONE_V540,
            "capstone_path": V540_CAPSTONE_RELATIVE_PATH.as_posix(),
            "capstone_sha256": capstone_meta.get("sha256"),
            "recorded_roadmap_path": capstone_roadmap.get("roadmap_path"),
            "recorded_roadmap_sha256": capstone_roadmap.get("roadmap_sha256"),
            "recorded_task_ids": capstone_roadmap.get("task_ids"),
            "expected_task_ids": list(EXPECTED_V540_TASK_IDS),
            "recorded_task_count": capstone_roadmap.get("task_count"),
            "research_complete_path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            "research_complete_sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        },
        "v540_task_terminal_matrix": matrix,
        "v540_capstone_path_hash_and_summary": {
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
        "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": _v540_counts(
            matrix
        ),
        "v541_roadmap_path_and_hash": v541_identity,
        "v541_task_ids_and_deliverables": _v541_task_deliverables(v541_data),
        "task_count": v541_validation["task_count"],
        "phase_counts": v541_validation["phase_counts"],
        "dependency_validation": v541_validation["dependency_validation"],
        "gated_on_validation": v541_validation["gated_on_validation"],
        "prior_failure_validation": v541_validation["prior_failure_validation"],
        "retired_dependency_count": v541_validation["retired_dependency_count"],
        "id_collision_count": id_collision_count,
        "agent_routing_validation": v541_validation["agent_routing_validation"],
        "model_policy_validation": v541_validation["model_policy_validation"],
        "prompt_contract_validation": v541_validation["prompt_contract_validation"],
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            v541_identity,
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
        "honest_verdict": "complete: V540 exact states and V541 roadmap contracts validated; broad-suite outcomes are recorded separately from focused checks",
    }
    blocking_command_failures = [
        row
        for row in command_rows
        if int(row.get("exit_code") or 0) != 0
        and not _is_broad_command(str(row.get("command") or ""))
    ]
    if (
        report["task_count"] != 12
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
            "blocked: V541 transition validation found a contract failure or task-owned command failure"
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
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if report.get("task_count") != 12:
        errors.append("task_count must be 12")
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
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
            "blocked:",
        )
    ):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def run_command(
    command: str, root: Path, timeout_s: int | None = None
) -> JsonDict:  # pragma: no cover - shell edge.
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


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover - shell edge.
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
        raise ValueError("invalid Exp6272 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(
    root: Path, date: str, *, run_commands: bool
) -> JsonDict:  # pragma: no cover - shell edge.
    started = time.perf_counter()
    before = protected_hashes(root)
    git_before = git_status_lines(root)
    inputs_before = input_hashes(root)
    collision_before = scan_reserved_id_collisions(
        root, allowed_reserved_paths=_allowed_reserved_paths()
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
    data, identity = load_v541_roadmap(root)
    validation = validate_v541_roadmap_data(
        data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    collision_receipt = scan_reserved_id_collisions(
        root, allowed_reserved_paths=_allowed_reserved_paths()
    )
    file_collisions = int(collision_receipt.get("unexpected_reserved_collision_count") or 0)
    ok = (
        identity.get("milestone") == MILESTONE_V541
        and validation["task_id_validation"]["expected_order"]
        and validation["task_count"] == 12
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - shell edge.
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
