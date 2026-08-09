"""Exp6260 V539-to-V540 terminal transition.

Spec refs: REQ-INFRA-6260, SCENARIO-INFRA-6260-1,
SCENARIO-INFRA-6260-2, SCENARIO-INFRA-6260-3,
SCENARIO-INFRA-6260-4, SCENARIO-INFRA-6260-5,
SCENARIO-INFRA-6260-6.
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


MILESTONE_V539 = "2026.08.539"
MILESTONE_V540 = "2026.08.540"
EXPERIMENT_ID = "exp6260-v540-terminal-transition"
SCHEMA = "carnot.experiment_6260.v540_terminal_transition.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6260_v540_terminal_transition.json")
INFERENCE_SUBSTRATE = "deterministic_v539_v540_terminal_transition_audit"

V539_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6238_v539_adversarial_capstone.json")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_539.json")
V540_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V540_NEXT_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
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
RESERVED_EXP_IDS = tuple(range(6260, 6272))
CONCURRENT_EXP_IDS = (6240, 6244, 6245, 6246)
GATE_OPS = frozenset({"==", "!=", ">", "<", ">=", "<=", "exists", "in"})

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    V539_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    Path("python/carnot/terminal_artifacts.py"),
)
INPUT_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    V540_ROADMAP_RELATIVE_PATH,
    V540_NEXT_ROADMAP_RELATIVE_PATH,
    V539_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    Path("results/experiment_6227_llama_server_signal_sender_diagnostic.json"),
    Path("results/experiment_6228_supervised_three_family_runtime_endurance.json"),
    Path("results/experiment_6237_activated_mode_jump_sampler_ab.json"),
    Path("python/carnot/terminal_artifacts.py"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
)
EXPERIMENT_SCAN_ROOTS = (
    Path("python/carnot"),
    Path("scripts/experiments"),
    Path("results"),
    Path("tests/python"),
)
ALLOWED_LOCAL_RESERVED_PATHS = {
    "python/carnot/experiment_6260_v540_terminal_transition.py",
    "tests/python/test_experiment_6260_v540_terminal_transition.py",
    RESULT_RELATIVE_PATH.as_posix(),
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6260_v540_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6260_v540_terminal_transition.py -m pytest tests/python/test_experiment_6260_v540_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6260_v540_terminal_transition.py --fail-under=100",
    ".venv/bin/python -m carnot.experiment_6260_v540_terminal_transition --check-roadmap-only",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6260_v540_terminal_transition.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6260_v540_terminal_transition.json",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v539_milestone_roadmap_and_hash",
    "v539_task_terminal_matrix",
    "exp6228_nonterminal_classification",
    "v539_capstone_path_hash_and_summary",
    "operational_retro_path_hash_and_summary",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts",
    "concurrent_exp6240_6244_6245_6246_collision_receipts",
    "v540_roadmap_path_and_hash",
    "v540_task_ids_and_deliverables",
    "task_count",
    "phase_counts",
    "dependency_validation",
    "gated_on_validation",
    "prior_failure_validation",
    "retired_dependency_count",
    "id_collision_count",
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
    "status": "The handoff is complete only when V539 evidence and V540 contracts are checked.",
    "v539_milestone_roadmap_and_hash": "The V539 denominator comes from the capstone roadmap record.",
    "v539_task_terminal_matrix": "Each V539 task is classified from its exact declared deliverable path.",
    "exp6228_nonterminal_classification": "The preconditions-only runtime artifact must stay nonterminal.",
    "v539_capstone_path_hash_and_summary": "The capstone is an input record, not a promotion source.",
    "operational_retro_path_hash_and_summary": "Operational timing is kept separate from scientific readiness.",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": "Mixed states stay visible in counts.",
    "concurrent_exp6240_6244_6245_6246_collision_receipts": "Concurrent outer-loop ids are recorded separately from the V540 reserved range.",
    "v540_roadmap_path_and_hash": "The audited V540 roadmap identity is content-addressed.",
    "v540_task_ids_and_deliverables": "The V540 task denominator must be exactly Exp6260 through Exp6271.",
    "task_count": "The transition accepts exactly twelve V540 tasks.",
    "phase_counts": "Track counts keep infrastructure, learning, sampler, and synthesis work distinct.",
    "dependency_validation": "Requires chains must reference current non-retired V540 tasks.",
    "gated_on_validation": "Structured gates must name valid upstream fields and operators.",
    "prior_failure_validation": "Prior-failure blocks must state the prior, verdict, difference, and retirement rule.",
    "retired_dependency_count": "Bare zero is the activation-safe retired dependency count.",
    "id_collision_count": "Bare zero is the activation-safe duplicate id count.",
    "model_policy_validation": "All staged tasks must use Codex GPT-5.5 and avoid fresh compute execution.",
    "prompt_contract_validation": "Prompt endings and protected-file clauses are checked mechanically.",
    "protected_files_unchanged": "Protected files are compared by hash before and after the result write.",
    "preconditions_checked": "The report records git state, input hashes, roadmap identity, and reserved-id scans.",
    "inference_substrate": "Declares deterministic file and roadmap audit, not live model inference.",
    "verifier_is_oracle": "False because this verifies records, not benchmark answers.",
    "field_provenance": "Every required field cites concrete files or checks.",
    "field_principles": "Every required field carries the reason it exists.",
    "test_commands": "Commands record focused tests, coverage, schema, gate, exclusion, clutter, suite, and adversarial checks.",
    "test_exit_codes": "Exit codes are reported without changing nonzero results.",
    "duration_s": "Wall time for deterministic aggregation is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed for drift detection.",
    "honest_verdict": "The terminal verdict states the caveat without strengthening prior evidence.",
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


def classify_v539_declared_tasks(root: Path, capstone_matrix: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    for task_id, prior in capstone_matrix.items():
        if not isinstance(prior, Mapping):
            continue
        declared = Path(str(prior.get("declared_deliverable") or prior.get("deliverable") or ""))
        classification = classify_artifact_path(root / declared).to_dict()
        aliases = same_number_aliases(root, str(task_id), declared)
        rows[str(task_id)] = {
            "task_id": str(task_id),
            "title": str(prior.get("title") or task_id),
            "track": str(prior.get("track") or ""),
            "declared_deliverable": declared.as_posix(),
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"],
            "classification": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "capstone_terminal_class": prior.get("terminal_class"),
            "capstone_terminal": prior.get("terminal"),
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": aliases,
        }
    return rows


def _tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    return (
        [dict(task) for task in tasks if isinstance(task, Mapping)] if isinstance(tasks, list) else []
    )


def load_v540_roadmap(root: Path) -> tuple[JsonDict, JsonDict]:
    next_path = root / V540_NEXT_ROADMAP_RELATIVE_PATH
    active_path = root / V540_ROADMAP_RELATIVE_PATH
    next_data = read_yaml_mapping(next_path)
    active_data = read_yaml_mapping(active_path)
    if next_data.get("milestone") == MILESTONE_V540:
        chosen_rel = V540_NEXT_ROADMAP_RELATIVE_PATH
        data = next_data
        note = "research-roadmap-next.yaml contains V540"
    elif active_data.get("milestone") == MILESTONE_V540:
        chosen_rel = V540_ROADMAP_RELATIVE_PATH
        data = active_data
        note = "active research-roadmap.yaml already contains V540; next roadmap is not activated here"
    else:
        chosen_rel = V540_NEXT_ROADMAP_RELATIVE_PATH if next_data else V540_ROADMAP_RELATIVE_PATH
        data = next_data or active_data
        note = "V540 roadmap milestone was not found in next or active roadmap"
    return data, {
        "path": chosen_rel.as_posix(),
        "sha256": path_sha256(root / chosen_rel),
        "milestone": data.get("milestone"),
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
    words = set(re.findall(r"\b[a-z][a-z0-9_]*\b", " ".join(block)))
    return words - {"required", "artifact", "fields", "and", "or", "must", "be", "bare"}


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
    if isinstance(required_fields_by_id, Mapping):
        fields = required_fields_by_id.get(upstream, set())
    else:
        fields = required_fields_by_id
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


def _prompt_schedules_forbidden_execution(prompt: str) -> list[str]:
    text = str(prompt).lower()
    hits: list[str] = []
    for phrase in ("fresh llm", "load an llm", "load a model", "arc solve", "hardware execution"):
        for match in re.finditer(re.escape(phrase), text):
            window = text[max(0, match.start() - 40) : match.end() + 40]
            negated = any(
                marker in window
                for marker in (
                    "do not",
                    "no ",
                    "not ",
                    "without",
                    "make no",
                    "avoid",
                    "provenance only",
                    "schedules",
                )
            )
            if not negated:
                hits.append(phrase)
                break
    return sorted(set(hits))


def validate_v540_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    id_counts = Counter(ids)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    id_collision_count = sum(count - 1 for count in id_counts.values() if count > 1)
    id_set = set(ids)
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        for task_id, task in tasks_by_id.items()
    }

    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - schema diagnostics are serialized.
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
        if not isinstance(priors, list) or not priors:
            prior_failures.append({"task_id": task_id, "reason": "missing_prior_failures"})
            continue
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    model_failures: list[JsonDict] = []
    forbidden_execution_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if task.get("agent_type") != "codex" or task.get("model") != "gpt-5.5":
            model_failures.append(
                {
                    "task_id": task_id,
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                }
            )
        if task.get("requires_gpu") is True:
            forbidden_execution_failures.append({"task_id": task_id, "reason": "requires_gpu_true"})
        if str(task.get("track") or "").lower() == "arc":
            forbidden_execution_failures.append({"task_id": task_id, "reason": "arc_track_task"})
        hits = _prompt_schedules_forbidden_execution(str(task.get("prompt") or ""))
        if hits:
            forbidden_execution_failures.append(
                {"task_id": task_id, "reason": "prompt_schedules_forbidden_execution", "hits": hits}
            )

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
            "expected_task_ids": list(EXPECTED_V540_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V540_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "id_collision_count": int(id_collision_count),
        "dependency_validation": {"ok": not dependency_failures, "failures": dependency_failures},
        "gated_on_validation": {"ok": not gate_failures, "failures": gate_failures},
        "prior_failure_validation": {"ok": not prior_failures, "failures": prior_failures},
        "retired_dependency_count": int(retired_dependency_count),
        "model_policy_validation": {
            "ok": not model_failures and not forbidden_execution_failures,
            "agent_model_failures": model_failures,
            "forbidden_execution_failures": forbidden_execution_failures,
        },
        "prompt_contract_validation": {"ok": not prompt_failures, "failures": prompt_failures},
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
    concurrent: dict[str, list[str]] = {str(exp_id): [] for exp_id in CONCURRENT_EXP_IDS}
    unexpected: dict[str, list[str]] = {str(exp_id): [] for exp_id in RESERVED_EXP_IDS}
    for rel in _experiment_paths(root):
        number = exp_number(rel.name)
        rel_text = rel.as_posix()
        if number in CONCURRENT_EXP_IDS:
            concurrent[str(number)].append(rel_text)
        if number in RESERVED_EXP_IDS and rel_text not in allowed:
            unexpected[str(number)].append(rel_text)
    concurrent = {key: value for key, value in concurrent.items() if value}
    unexpected = {key: value for key, value in unexpected.items() if value}
    return {
        "scan_roots": [path.as_posix() for path in EXPERIMENT_SCAN_ROOTS],
        "concurrent_exp_ids": list(CONCURRENT_EXP_IDS),
        "reserved_exp_ids": list(RESERVED_EXP_IDS),
        "allowed_reserved_paths": sorted(allowed),
        "concurrent_paths_by_exp_id": concurrent,
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


def _allowed_reserved_paths(v540_data: JsonMap) -> set[str]:
    return ALLOWED_LOCAL_RESERVED_PATHS | {
        str(task.get("deliverable") or "") for task in _tasks(v540_data)
    }


def _v539_counts(matrix: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("classification")) for row in matrix.values())
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


def _v540_task_deliverables(data: JsonMap) -> list[JsonDict]:
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


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6260",
        V539_CAPSTONE_RELATIVE_PATH.as_posix(),
        OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        V540_ROADMAP_RELATIVE_PATH.as_posix(),
        V540_NEXT_ROADMAP_RELATIVE_PATH.as_posix(),
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
    root: Path,
    v540_identity: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    collision_receipt_before: JsonMap,
    input_hashes_before: JsonMap,
) -> JsonDict:
    return {
        "git_status_before": list(git_status_before),
        "input_hashes_before": input_hashes_before,
        "staged_roadmap_identity": v540_identity,
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
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    v540_data, v540_identity = load_v540_roadmap(root)
    before = dict(before_hashes or protected_hashes(root))
    status_before = list(git_status_before or git_status_lines(root))
    allowed = _allowed_reserved_paths(v540_data)
    collision_before = dict(
        collision_receipt_before
        or scan_reserved_id_collisions(root, allowed_reserved_paths=allowed)
    )
    inputs_before = dict(input_hashes_before or input_hashes(root))
    capstone_payload, capstone_meta = read_json_mapping(root / V539_CAPSTONE_RELATIVE_PATH)
    retro_payload, retro_meta = read_json_mapping(root / OPERATIONAL_RETRO_RELATIVE_PATH)
    capstone_matrix = capstone_payload.get("exact_task_artifact_matrix")
    matrix = classify_v539_declared_tasks(
        root, capstone_matrix if isinstance(capstone_matrix, Mapping) else {}
    )
    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    v540_validation = validate_v540_roadmap_data(v540_data, retired_ids)
    command_rows = [dict(row) for row in (command_receipts or [])]
    exp6228 = dict(matrix.get("exp6228-supervised-three-family-runtime-endurance") or {})

    capstone_roadmap = capstone_payload.get("roadmap_path_hash_and_task_ids")
    if not isinstance(capstone_roadmap, Mapping):
        capstone_roadmap = {}

    report: JsonDict = {
        "status": "complete",
        "v539_milestone_roadmap_and_hash": {
            "milestone": MILESTONE_V539,
            "capstone_path": V539_CAPSTONE_RELATIVE_PATH.as_posix(),
            "capstone_sha256": capstone_meta.get("sha256"),
            "roadmap_path": capstone_roadmap.get("roadmap_path"),
            "roadmap_sha256": capstone_roadmap.get("roadmap_sha256"),
            "task_ids": capstone_roadmap.get("task_ids"),
            "task_count": capstone_roadmap.get("task_count"),
        },
        "v539_task_terminal_matrix": matrix,
        "exp6228_nonterminal_classification": {
            **exp6228,
            "preserved_nonterminal": exp6228.get("terminal") is False,
            "preconditions_only_status": exp6228.get("status_raw") == "preconditions_recorded",
        },
        "v539_capstone_path_hash_and_summary": {
            **capstone_meta,
            "summary": {
                "status": capstone_payload.get("status"),
                "honest_verdict": capstone_payload.get("honest_verdict"),
                "counts": capstone_payload.get(
                    "missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts"
                ),
                "hardware_boundary": capstone_payload.get("hardware_boundary_and_claim_count"),
            },
        },
        "operational_retro_path_hash_and_summary": {
            **retro_meta,
            "summary": {
                "milestone": retro_payload.get("milestone"),
                "retro_type": retro_payload.get("retro_type"),
                "summary": retro_payload.get("summary"),
            },
        },
        "missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": _v539_counts(
            matrix
        ),
        "concurrent_exp6240_6244_6245_6246_collision_receipts": collision_before,
        "v540_roadmap_path_and_hash": v540_identity,
        "v540_task_ids_and_deliverables": _v540_task_deliverables(v540_data),
        "task_count": v540_validation["task_count"],
        "phase_counts": v540_validation["phase_counts"],
        "dependency_validation": v540_validation["dependency_validation"],
        "gated_on_validation": v540_validation["gated_on_validation"],
        "prior_failure_validation": v540_validation["prior_failure_validation"],
        "retired_dependency_count": v540_validation["retired_dependency_count"],
        "id_collision_count": v540_validation["id_collision_count"],
        "model_policy_validation": v540_validation["model_policy_validation"],
        "prompt_contract_validation": v540_validation["prompt_contract_validation"],
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root, v540_identity, before, status_before, collision_before, inputs_before
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
        "honest_verdict": "complete: V539 exact-path states preserved, Exp6228 remains nonterminal, and V540 roadmap contracts validated without activating a staged roadmap",
    }
    nonzero_test_exit_count = sum(
        1 for row in command_rows if int(row.get("exit_code") or 0) != 0
    )
    if (
        report["task_count"] != 12
        or report["retired_dependency_count"] != 0
        or report["id_collision_count"] != 0
        or collision_before.get("unexpected_reserved_collision_count") != 0
        or not report["dependency_validation"]["ok"]
        or not report["gated_on_validation"]["ok"]
        or not report["prior_failure_validation"]["ok"]
        or not report["model_policy_validation"]["ok"]
        or not report["prompt_contract_validation"]["ok"]
        or not report["protected_files_unchanged"]["unchanged"]
        or report["exp6228_nonterminal_classification"].get("terminal") is not False
        or nonzero_test_exit_count
    ):
        report["status"] = "blocked"
        if nonzero_test_exit_count:
            report["honest_verdict"] = (
                "blocked: one or more recorded validation commands failed or timed out"
            )
        else:
            report["honest_verdict"] = (
                "blocked: V540 transition validation found a contract failure or protected-file mutation"
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
    if report.get("retired_dependency_count") != 0:
        errors.append("retired_dependency_count must be bare 0")
    if report.get("id_collision_count") != 0:
        errors.append("id_collision_count must be bare 0")
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


def write_report(report: JsonMap, root: Path = REPO_ROOT) -> Path:  # pragma: no cover - shell edge.
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6260 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, sort_keys=False)


def run_experiment(
    root: Path, date: str, *, run_commands: bool
) -> JsonDict:  # pragma: no cover - shell edge.
    started = time.perf_counter()
    v540_data, _identity = load_v540_roadmap(root)
    before = protected_hashes(root)
    git_before = git_status_lines(root)
    inputs_before = input_hashes(root)
    collision_before = scan_reserved_id_collisions(
        root, allowed_reserved_paths=_allowed_reserved_paths(v540_data)
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
        started_at=started,
    )
    write_report(final, root)
    return final


def check_roadmap_only(root: Path = REPO_ROOT) -> JsonDict:
    data, identity = load_v540_roadmap(root)
    validation = validate_v540_roadmap_data(
        data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    ok = (
        identity.get("milestone") == MILESTONE_V540
        and validation["task_id_validation"]["expected_order"]
        and validation["task_count"] == 12
        and validation["id_collision_count"] == 0
        and validation["retired_dependency_count"] == 0
        and validation["dependency_validation"]["ok"]
        and validation["gated_on_validation"]["ok"]
        and validation["prior_failure_validation"]["ok"]
        and validation["model_policy_validation"]["ok"]
        and validation["prompt_contract_validation"]["ok"]
    )
    return {"ok": ok, "roadmap_identity": identity, **validation}


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
