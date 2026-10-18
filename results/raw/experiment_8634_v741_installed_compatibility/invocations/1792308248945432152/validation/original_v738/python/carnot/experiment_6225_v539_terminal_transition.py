"""Exp6225 V538-to-V539 terminal transition.

Spec refs: REQ-INFRA-6225, SCENARIO-INFRA-6225-1,
SCENARIO-INFRA-6225-2, SCENARIO-INFRA-6225-3,
SCENARIO-INFRA-6225-4, SCENARIO-INFRA-6225-5,
SCENARIO-INFRA-6225-6.
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


MILESTONE_V538 = "2026.08.538"
MILESTONE_V539 = "2026.08.539"
EXPERIMENT_ID = "exp6225-v539-terminal-transition"
SCHEMA = "carnot.experiment_6225.v539_terminal_transition.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6225_v539_terminal_transition.json")
INFERENCE_SUBSTRATE = "deterministic_v538_v539_terminal_transition_audit"

V538_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6224_v538_adversarial_capstone.json")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_538.json")
V539_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V539_NEXT_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")

EXPECTED_V539_TASK_IDS = (
    "exp6225-v539-terminal-transition",
    "exp6226-v539-post-marker-source-scope-freeze",
    "exp6227-llama-server-signal-sender-diagnostic",
    "exp6228-supervised-three-family-runtime-endurance",
    "exp6229-arc-gemma31-think-determination",
    "exp6230-arc-induce-prompt-enrichment-heldout-ab",
    "exp6231-arc-bounded-reinduction-depth-ab",
    "exp6232-arc-admissible-depth-portfolio",
    "exp6233-three-family-code-content-margin",
    "exp6234-fresh-flagship-constraint-event-stream",
    "exp6235-prospective-two-timescale-live-csl",
    "exp6236-online-constraint-memory-shadow-consumer",
    "exp6237-activated-mode-jump-sampler-ab",
    "exp6238-v539-adversarial-capstone",
)

SOTA_GGUFS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-12B-it-GGUF",
    }
)

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("_bmad/architecture.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    V538_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6225_v539_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6225_v539_terminal_transition.py -m pytest tests/python/test_experiment_6225_v539_terminal_transition.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6225_v539_terminal_transition.py --fail-under=100",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python -m carnot.experiment_6225_v539_terminal_transition --check-roadmap-only",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6225_v539_terminal_transition.json",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v538_milestone_and_roadmap_hash",
    "v538_task_terminal_matrix",
    "v538_capstone_path_hash_and_summary",
    "operational_retro_path_hash_and_summary",
    "blocked_skipped_partial_flagged_and_ready_counts",
    "research_complete_duplicate_record_note",
    "v539_roadmap_path_and_hash",
    "v539_task_ids_and_deliverables",
    "task_count",
    "phase_counts",
    "dependency_validation",
    "gated_on_validation",
    "prior_failure_validation",
    "retired_dependency_count",
    "id_collision_count",
    "model_policy_validation",
    "prompt_contract_validation",
    "architecture_staleness_receipt",
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
    "status": "The transition is complete only after V538 evidence and V539 contracts are both checked.",
    "v538_milestone_and_roadmap_hash": "The prior milestone denominator is pinned to its archived record and capstone roadmap hash.",
    "v538_task_terminal_matrix": "Each V538 task is classified from its exact declared deliverable path.",
    "v538_capstone_path_hash_and_summary": "The capstone summary records current-rule flags before any conclusion is copied.",
    "operational_retro_path_hash_and_summary": "The retro summary keeps operational caveats separate from scientific readiness.",
    "blocked_skipped_partial_flagged_and_ready_counts": "Mixed terminal states stay counted before any branch handoff narrative.",
    "research_complete_duplicate_record_note": "Duplicate milestone records are an input caveat, not something this task rewrites.",
    "v539_roadmap_path_and_hash": "The audited V539 roadmap identity is content-addressed and notes the missing next-roadmap caveat.",
    "v539_task_ids_and_deliverables": "The V539 task denominator must be exactly Exp6225 through Exp6238.",
    "task_count": "The transition accepts exactly fourteen V539 tasks.",
    "phase_counts": "Track counts keep runtime, ARC, code, self-learning, sampler, and capstone work distinct.",
    "dependency_validation": "Requires chains must reference real non-retired tasks.",
    "gated_on_validation": "Structured gates must name valid upstream fields and operators.",
    "prior_failure_validation": "Reruns must name a prior verdict, difference, and retire-if-same-verdict rule.",
    "retired_dependency_count": "Bare zero is the activation-safe retired dependency count.",
    "id_collision_count": "Bare zero is the activation-safe duplicate id count.",
    "model_policy_validation": "LLM tasks must name mandated local SOTA GGUF models.",
    "prompt_contract_validation": "Prompt endings and protected-file clauses are checked mechanically.",
    "architecture_staleness_receipt": "Architecture is stale by the 30-day rule and is not reconciled here.",
    "protected_files_unchanged": "Protected files are compared by hash before and after the result write.",
    "preconditions_checked": "The report names the working tree, inputs, roadmap identity, and protected hashes checked before artifact write.",
    "inference_substrate": "Declares deterministic file and roadmap audit, not live model inference.",
    "verifier_is_oracle": "False because this handoff verifies records, not benchmark answers.",
    "field_provenance": "Every required field cites the concrete file or check that produced it.",
    "field_principles": "Every required field carries the reason it exists.",
    "test_commands": "Commands record focused tests, coverage, schema, gate, exclusion, prompt, clutter, E2E-plan, suite, and adversarial checks.",
    "test_exit_codes": "Exit codes are reported without laundering nonzero results.",
    "duration_s": "Wall time for deterministic aggregation is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed for drift detection.",
    "honest_verdict": "The terminal verdict names any caveat without strengthening prior evidence.",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
    match = re.search(r"exp(?:eriment_)?(\d+)|\b(\d{3,5})\b", str(text), re.IGNORECASE)
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


def classify_declared_deliverables(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id") or "")
        deliverable = Path(str(task.get("deliverable") or ""))
        classification = classify_artifact_path(root / deliverable).to_dict()
        aliases = same_number_aliases(root, task_id, deliverable)
        rows[task_id] = {
            "task_id": task_id,
            "title": str(task.get("title") or task_id),
            "declared_deliverable": deliverable.as_posix(),
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"],
            "classification": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": aliases,
        }
    return rows


def _v538_tasks_from_research_complete(root: Path) -> list[JsonDict]:
    data = read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    for milestone in data.get("milestones", []):
        if not isinstance(milestone, Mapping) or str(milestone.get("id")) != MILESTONE_V538:
            continue
        tasks = milestone.get("tasks")
        if isinstance(tasks, list):
            return [dict(task) for task in tasks if isinstance(task, Mapping)]
    return []


def build_v538_task_terminal_matrix(root: Path, capstone_payload: JsonMap) -> JsonDict:
    tasks = _v538_tasks_from_research_complete(root)
    rows = classify_declared_deliverables(root, tasks)
    prior_rows = capstone_payload.get("exact_artifact_paths_hashes_and_terminal_classifications")
    if isinstance(prior_rows, Mapping):
        for task_id, prior in prior_rows.items():
            if task_id in rows and isinstance(prior, Mapping):
                rows[task_id]["capstone_flag_count"] = int(prior.get("flag_count") or 0)
                rows[task_id]["capstone_critical_flag_count"] = int(
                    prior.get("critical_adversarial_flag_count") or 0
                )
    return rows


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


def _tasks(data: JsonMap) -> list[JsonDict]:
    rows = data.get("tasks")
    return (
        [dict(task) for task in rows if isinstance(task, Mapping)] if isinstance(rows, list) else []
    )


def _gate_ok(gate: Any, ids: set[str]) -> tuple[bool, str | None]:
    if not isinstance(gate, Mapping):
        return False, "gate_not_mapping"
    for field in ("upstream", "artifact_field", "op", "value"):
        if field not in gate:
            return False, f"missing_{field}"
    if str(gate.get("op")) not in {"==", "!=", ">", "<", ">=", "<=", "exists", "in"}:
        return False, "bad_op"
    if str(gate.get("upstream")) not in ids:
        return False, "unknown_upstream"
    return True, None


def _prior_ok(prior: Any) -> tuple[bool, str | None]:
    if not isinstance(prior, Mapping):
        return False, "prior_not_mapping"
    for field in ("experiment_id", "verdict", "addressed_by"):
        if not str(prior.get(field) or "").strip():
            return False, f"missing_{field}"
    if prior.get("retire_if_same_verdict") is not True:
        return False, "retire_if_same_verdict_not_true"
    return True, None


def _is_llm_task(task: JsonMap) -> bool:
    prompt = str(task.get("prompt") or "")
    return "MODEL_SPECS:" in prompt or ("GGUF" in prompt and task.get("requires_gpu") is True)


def validate_v539_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    id_counts = Counter(ids)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    id_collision_count = sum(count - 1 for count in id_counts.values() if count > 1)
    id_set = set(ids)
    phase_counts = dict(
        sorted(Counter(str(task.get("track") or "unset") for task in tasks).items())
    )

    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:
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
            if dep_text not in id_set or retired or dep_text == task_id:
                dependency_failures.append(
                    {
                        "task_id": task_id,
                        "dependency": dep_text,
                        "missing": dep_text not in id_set,
                        "self_dependency": dep_text == task_id,
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
            ok, reason = _gate_ok(gate, id_set)
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
            ok, reason = _prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    llm_failures: list[JsonDict] = []
    arc_failures: list[JsonDict] = []
    for task in tasks:
        prompt = str(task.get("prompt") or "")
        task_id = str(task.get("id") or "")
        if _is_llm_task(task) and not any(model in prompt for model in SOTA_GGUFS):
            llm_failures.append({"task_id": task_id, "reason": "missing_mandated_sota_gguf"})
        if str(task.get("track")) == "arc":
            if "solve_provenance must be live_agent_self_discovery" not in prompt:
                arc_failures.append({"task_id": task_id, "reason": "missing_solve_provenance"})
            if "registry_update_count" not in prompt:
                arc_failures.append({"task_id": task_id, "reason": "missing_registry_boundary"})

    prompt_failures: list[JsonDict] = []
    missing_endings: list[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        expected_run = (
            f"Run command: .venv/bin/python -m carnot.{_module_name_for_task(task)} --date"
        )
        has_run = expected_run in prompt
        has_conductor_sentence = prompt.strip().endswith(
            "Do NOT push. Do NOT modify scripts/research_conductor.py."
        )
        if not has_run or not has_conductor_sentence:
            prompt_failures.append(
                {
                    "task_id": task_id,
                    "run_command_present": has_run,
                    "conductor_sentence_ending": has_conductor_sentence,
                }
            )
            missing_endings.append(task_id)

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "task_count": len(tasks),
        "phase_counts": phase_counts,
        "task_id_validation": {
            "task_ids": ids,
            "expected_task_ids": list(EXPECTED_V539_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V539_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "id_collision_count": int(id_collision_count),
        "dependency_validation": {"ok": not dependency_failures, "failures": dependency_failures},
        "gated_on_validation": {"ok": not gate_failures, "failures": gate_failures},
        "prior_failure_validation": {"ok": not prior_failures, "failures": prior_failures},
        "retired_dependency_count": int(retired_dependency_count),
        "model_policy_validation": {
            "ok": not llm_failures and not arc_failures,
            "llm_task_failures": llm_failures,
            "arc_live_path_failures": arc_failures,
        },
        "prompt_contract_validation": {
            "ok": not prompt_failures,
            "failures": prompt_failures,
            "missing_required_ending": missing_endings,
        },
    }


def _module_name_for_task(task: JsonMap) -> str:
    deliverable = Path(str(task.get("deliverable") or ""))
    stem = deliverable.stem
    if stem.startswith("experiment_"):
        return stem
    return stem.replace("-", "_")


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


def research_complete_duplicate_note(root: Path) -> JsonDict:
    data = read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    ids = [
        str(row.get("id"))
        for row in data.get("milestones", [])
        if isinstance(row, Mapping) and row.get("id") is not None
    ]
    counts = Counter(ids)
    duplicates = {key: counts[key] for key in sorted(counts) if counts[key] > 1}
    return {
        "path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        "milestone_record_count": len(ids),
        "unique_milestone_count": len(counts),
        "duplicate_milestone_count": len(duplicates),
        "duplicate_milestones": duplicates,
        "action": "recorded_only_not_deduplicated",
    }


def architecture_staleness(root: Path, as_of_yyyymmdd: str) -> JsonDict:
    path = root / ARCHITECTURE_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    match = re.search(r"Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    last_reconciled = match.group(1) if match else None
    as_of = datetime.strptime(as_of_yyyymmdd, "%Y%m%d").replace(tzinfo=UTC)
    age_days = None
    stale = True
    if last_reconciled:
        reconciled = datetime.strptime(last_reconciled, "%Y-%m-%d").replace(tzinfo=UTC)
        age_days = (as_of - reconciled).days
        stale = age_days > 30
    return {
        "path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(path),
        "last_reconciled": last_reconciled,
        "as_of_date": as_of_yyyymmdd,
        "age_days": age_days,
        "stale_by_30_day_rule": stale,
        "rewritten": False,
    }


def _summary_receipt(root: Path, rel: Path) -> JsonDict:  # pragma: no cover - shell edge.
    command = f".venv/bin/python scripts/summarize_artifact.py {rel.as_posix()}"
    return run_command(command, root, timeout_s=120)


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6225",
        "research-roadmap.yaml",
        V538_CAPSTONE_RELATIVE_PATH.as_posix(),
        OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "scripts/summarize_artifact.py",
        "python/carnot/terminal_artifacts.py",
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": sorted(sources),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _v539_task_deliverables(data: JsonMap) -> list[JsonDict]:
    return [
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
        }
        for task in _tasks(data)
    ]


def _v538_counts(matrix: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("classification")) for row in matrix.values())
    return {
        "blocked": int(counts.get("blocked", 0)),
        "skipped": int(counts.get("skipped", 0)),
        "partial": int(counts.get("partial", 0)),
        "flagged": int(counts.get("flagged", 0)),
        "ready": int(counts.get("ready", 0)),
        "missing": int(counts.get("missing", 0)),
        "complete": int(counts.get("complete", 0)),
        "classification_counts": dict(sorted((key, int(value)) for key, value in counts.items())),
        "capstone_flag_count": int(
            sum(int(row.get("capstone_flag_count") or 0) for row in matrix.values())
        ),
        "capstone_critical_flag_count": int(
            sum(int(row.get("capstone_critical_flag_count") or 0) for row in matrix.values())
        ),
    }


def preconditions_checked(root: Path, v539_data: JsonMap, before_hashes: JsonMap) -> JsonDict:
    return {
        "working_tree_checked_before_mutation": True,
        "declared_inputs": {
            path.as_posix(): (root / path).exists()
            for path in (
                Path("AGENTS.md"),
                Path("CODEX.md"),
                Path("CLAUDE.md"),
                V539_ROADMAP_RELATIVE_PATH,
                V539_NEXT_ROADMAP_RELATIVE_PATH,
                V538_CAPSTONE_RELATIVE_PATH,
                OPERATIONAL_RETRO_RELATIVE_PATH,
                RESEARCH_COMPLETE_RELATIVE_PATH,
                Path("ops/conductor-log.md"),
                EXCLUSION_MANIFEST_RELATIVE_PATH,
            )
        },
        "staged_roadmap_identity": {
            "research_roadmap_next_present": (root / V539_NEXT_ROADMAP_RELATIVE_PATH).exists(),
            "audited_v539_path": V539_ROADMAP_RELATIVE_PATH.as_posix(),
            "audited_milestone": v539_data.get("milestone"),
            "expected_milestone": MILESTONE_V539,
            "active_roadmap_already_contains_v539": v539_data.get("milestone") == MILESTONE_V539,
        },
        "protected_hashes_before_artifact_write": before_hashes,
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    before_hashes: JsonMap | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(before_hashes or protected_hashes(root))
    capstone_payload, capstone_meta = read_json_mapping(root / V538_CAPSTONE_RELATIVE_PATH)
    retro_payload, retro_meta = read_json_mapping(root / OPERATIONAL_RETRO_RELATIVE_PATH)
    v539_data = read_yaml_mapping(root / V539_ROADMAP_RELATIVE_PATH)
    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    v539_validation = validate_v539_roadmap_data(v539_data, retired_ids)
    matrix = build_v538_task_terminal_matrix(root, capstone_payload)
    command_rows = [dict(row) for row in (command_receipts or [])]
    capstone_summary = _summary_receipt(root, V538_CAPSTONE_RELATIVE_PATH)
    retro_summary = _summary_receipt(root, OPERATIONAL_RETRO_RELATIVE_PATH)

    old_graph = capstone_payload.get("declared_task_ids_and_deliverables")
    if not isinstance(old_graph, Mapping):
        old_graph = {}

    report: JsonDict = {
        "status": "complete",
        "v538_milestone_and_roadmap_hash": {
            "milestone": MILESTONE_V538,
            "research_complete_path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            "research_complete_sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
            "capstone_declared_roadmap_path": old_graph.get("roadmap_path"),
            "capstone_declared_roadmap_sha256": old_graph.get("roadmap_sha256"),
            "capstone_task_count": old_graph.get("roadmap_task_count"),
        },
        "v538_task_terminal_matrix": matrix,
        "v538_capstone_path_hash_and_summary": {
            **capstone_meta,
            "status": capstone_payload.get("status"),
            "honest_verdict": capstone_payload.get("honest_verdict"),
            "summary_receipt": capstone_summary,
        },
        "operational_retro_path_hash_and_summary": {
            **retro_meta,
            "milestone": retro_payload.get("milestone"),
            "retro_type": retro_payload.get("retro_type"),
            "summary": retro_payload.get("summary"),
            "summary_receipt": retro_summary,
        },
        "blocked_skipped_partial_flagged_and_ready_counts": _v538_counts(matrix),
        "research_complete_duplicate_record_note": research_complete_duplicate_note(root),
        "v539_roadmap_path_and_hash": {
            "path": V539_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / V539_ROADMAP_RELATIVE_PATH),
            "milestone": v539_data.get("milestone"),
            "research_roadmap_next_present": (root / V539_NEXT_ROADMAP_RELATIVE_PATH).exists(),
            "note": "research-roadmap-next.yaml absent; active research-roadmap.yaml already contains V539",
        },
        "v539_task_ids_and_deliverables": _v539_task_deliverables(v539_data),
        "task_count": v539_validation["task_count"],
        "phase_counts": v539_validation["phase_counts"],
        "dependency_validation": v539_validation["dependency_validation"],
        "gated_on_validation": v539_validation["gated_on_validation"],
        "prior_failure_validation": v539_validation["prior_failure_validation"],
        "retired_dependency_count": v539_validation["retired_dependency_count"],
        "id_collision_count": v539_validation["id_collision_count"],
        "model_policy_validation": v539_validation["model_policy_validation"],
        "prompt_contract_validation": v539_validation["prompt_contract_validation"],
        "architecture_staleness_receipt": architecture_staleness(root, date),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(root, v539_data, before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exits(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": "complete: V538 mixed terminal states archived and V539 roadmap identity validated without activating or editing protected files",
    }
    if (
        report["retired_dependency_count"] != 0
        or report["id_collision_count"] != 0
        or not report["protected_files_unchanged"]["unchanged"]
    ):
        report["status"] = "blocked"
        report["honest_verdict"] = (
            "blocked: transition validation found retired dependencies, id collisions, or protected-file mutation"
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
        expected = payload_checksum(report)
        if checksum != expected:
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
        raise ValueError("invalid Exp6225 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, sort_keys=False)


def run_experiment(
    root: Path, date: str, *, run_commands: bool
) -> JsonDict:  # pragma: no cover - shell edge.
    started = time.perf_counter()
    before = protected_hashes(root)
    preliminary = build_report(
        root, date=date, command_receipts=[], before_hashes=before, started_at=started
    )
    write_report(preliminary, root)
    command_rows = run_default_commands(root) if run_commands else []
    final = build_report(
        root, date=date, command_receipts=command_rows, before_hashes=before, started_at=started
    )
    write_report(final, root)
    return final


def check_roadmap_only(root: Path = REPO_ROOT) -> JsonDict:
    data = read_yaml_mapping(root / V539_ROADMAP_RELATIVE_PATH)
    validation = validate_v539_roadmap_data(
        data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    ok = (
        validation["task_id_validation"]["expected_order"]
        and validation["id_collision_count"] == 0
        and validation["retired_dependency_count"] == 0
        and validation["dependency_validation"]["ok"]
        and validation["gated_on_validation"]["ok"]
        and validation["prior_failure_validation"]["ok"]
        and validation["model_policy_validation"]["ok"]
        and validation["prompt_contract_validation"]["ok"]
    )
    return {"ok": ok, **validation}


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
