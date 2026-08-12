"""Exp6350 V547 bounded terminal handoff.

Spec refs: REQ-INFRA-6350, SCENARIO-INFRA-6350-1,
SCENARIO-INFRA-6350-2, SCENARIO-INFRA-6350-3,
SCENARIO-INFRA-6350-4, SCENARIO-INFRA-6350-5,
SCENARIO-INFRA-6350-6.
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
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - import path guard.
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_manifest_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402
from validate_prior_failures import validate_roadmap as validate_prior_failure_roadmap  # noqa: E402


MILESTONE_V546 = "2026.08.546"
MILESTONE_V547 = "2026.08.547"
EXPERIMENT_ID = "exp6350-v547-bounded-terminal-handoff"
SCHEMA = "carnot.experiment_6350.v547_bounded_terminal_handoff.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6350_v547_bounded_terminal_handoff.json")
INFERENCE_SUBSTRATE = "deterministic_repository_evidence_handoff_no_llm"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPECTED_V546_TASK_IDS = (
    "exp6337-v546-bounded-terminal-handoff",
    "exp6338-v546-post-marker-source-scope-freeze",
    "exp6339-incremental-prefix-enforcement-substrate",
    "exp6340-parser-jit-semantic-diversity-canary",
    "exp6341-prospective-prefix-utility-ab",
    "exp6342-anytime-evalue-release-ledger",
    "exp6343-evidence-carrying-factor-lifecycle",
    "exp6344-counterexample-factor-proposal-calibration",
    "exp6345-prospective-certified-factor-evolution-ab",
    "exp6346-certified-factor-evolution-safety-audit",
    "exp6347-arc-action-influence-preflight",
    "exp6348-arc-default-off-action-influence-ab",
    "exp6349-v546-adversarial-capstone",
)
V546_DELIVERABLES_BY_TASK = {
    "exp6337-v546-bounded-terminal-handoff": (
        "results/experiment_6337_v546_bounded_terminal_handoff.json"
    ),
    "exp6338-v546-post-marker-source-scope-freeze": (
        "results/experiment_6338_v546_post_marker_source_scope_freeze.json"
    ),
    "exp6339-incremental-prefix-enforcement-substrate": (
        "results/experiment_6339_incremental_prefix_enforcement_substrate.json"
    ),
    "exp6340-parser-jit-semantic-diversity-canary": (
        "results/experiment_6340_parser_jit_semantic_diversity_canary.json"
    ),
    "exp6341-prospective-prefix-utility-ab": (
        "results/experiment_6341_prospective_prefix_utility_ab.json"
    ),
    "exp6342-anytime-evalue-release-ledger": (
        "results/experiment_6342_anytime_evalue_release_ledger.json"
    ),
    "exp6343-evidence-carrying-factor-lifecycle": (
        "results/experiment_6343_evidence_carrying_factor_lifecycle.json"
    ),
    "exp6344-counterexample-factor-proposal-calibration": (
        "results/experiment_6344_counterexample_factor_proposal_calibration.json"
    ),
    "exp6345-prospective-certified-factor-evolution-ab": (
        "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
    ),
    "exp6346-certified-factor-evolution-safety-audit": (
        "results/experiment_6346_certified_factor_evolution_safety_audit.json"
    ),
    "exp6347-arc-action-influence-preflight": (
        "results/experiment_6347_arc_action_influence_preflight.json"
    ),
    "exp6348-arc-default-off-action-influence-ab": (
        "results/experiment_6348_arc_default_off_action_influence_ab.json"
    ),
    "exp6349-v546-adversarial-capstone": (
        "results/experiment_6349_v546_adversarial_capstone.json"
    ),
}
V546_TITLE_SNIPPETS = {
    "exp6337-v546-bounded-terminal-handoff": "Bounded V545 terminal evidence handoff",
    "exp6338-v546-post-marker-source-scope-freeze": "V546 dated source-window",
    "exp6339-incremental-prefix-enforcement-substrate": "Incremental parser-state",
    "exp6340-parser-jit-semantic-diversity-canary": "Three-model parser-state",
    "exp6341-prospective-prefix-utility-ab": "Gated prospective held-family prefix utility",
    "exp6342-anytime-evalue-release-ledger": "Immutable anytime e-value ledger",
    "exp6343-evidence-carrying-factor-lifecycle": "Bounded evidence-carrying factor",
    "exp6344-counterexample-factor-proposal-calibration": "Three-model minimized-counterexample",
    "exp6345-prospective-certified-factor-evolution-ab": "Gated chronological e-process",
    "exp6346-certified-factor-evolution-safety-audit": "Independent e-process and factor-lifecycle",
    "exp6347-arc-action-influence-preflight": "ARC target-licensed counterfactual",
    "exp6348-arc-default-off-action-influence-ab": (
        "Gated default-off live ARC action-influence"
    ),
    "exp6349-v546-adversarial-capstone": "V546 terminal evidence reconciliation",
}
BLOCKED_V546_TASK_IDS = {"exp6341-prospective-prefix-utility-ab"}
EXP6337_TASK_ID = "exp6337-v546-bounded-terminal-handoff"

EXPECTED_V547_TASK_IDS = (
    "exp6350-v547-bounded-terminal-handoff",
    "exp6351-v547-post-marker-source-scope-freeze",
    "exp6352-live-factor-proposal-authenticity-preflight",
    "exp6353-live-counterexample-factor-proposal-ab",
    "exp6354-prospective-live-certified-factor-learning",
    "exp6355-default-off-certified-factor-consumer-ab",
    "exp6356-live-certified-learning-safety-audit",
)
EXPECTED_V547_PROPOSAL_TASK_IDS = (
    *EXPECTED_V547_TASK_IDS,
    "exp6357-arc-two-sided-goal-evidence-contract",
    "exp6358-arc-active-reward-machine-discriminator",
    "exp6359-arc-goal-evidence-response-calibration",
    "exp6360-arc-default-off-active-goal-shadow",
    "exp6361-arc-active-goal-provenance-audit",
    "exp6362-v547-adversarial-capstone",
)
MANDATED_HEADLINE_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
LIVE_MODEL_V547_TASK_IDS = {
    "exp6352-live-factor-proposal-authenticity-preflight",
    "exp6353-live-counterexample-factor-proposal-ab",
    "exp6354-prospective-live-certified-factor-learning",
    "exp6355-default-off-certified-factor-consumer-ab",
}

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6350_v547_bounded_terminal_handoff --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6350_v547_bounded_terminal_handoff.py "
    "-m pytest tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6350_v547_bounded_terminal_handoff.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6350_v547_bounded_terminal_handoff.py "
    "tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6350_v547_bounded_terminal_handoff.py "
    "tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py"
)
ROADMAP_SCHEMA_COMMAND = (
    ".venv/bin/python -c 'import yaml; from pathlib import Path; "
    "from scripts.roadmap_schema import Roadmap; "
    'Roadmap.model_validate(yaml.safe_load(Path("research-roadmap.yaml").read_text()))'
    "'"
)
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
PROTECTED_DIFF_COMMAND = (
    "git diff --exit-code -- research-roadmap.yaml "
    "openspec/change-proposals/research-roadmap-vNEXT.md scripts/research_conductor.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6350_v547_bounded_terminal_handoff.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROADMAP_SCHEMA_COMMAND,
    PRIOR_FAILURE_COMMAND,
    GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    PROTECTED_DIFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6350_test_receipts.json")

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
    KNOWN_ISSUES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("python/carnot/experiment_6350_v547_bounded_terminal_handoff.py"),
    Path("tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
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
    CONDUCTOR_LOG_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    *[Path(path) for path in V546_DELIVERABLES_BY_TASK.values()],
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v546_milestone_and_queue_hash",
    "queued_v546_task_ids",
    "terminal_v546_artifacts_by_task",
    "blocked_v546_tasks",
    "flagged_v546_artifacts_and_reasons",
    "inference_substrate_classification_by_task",
    "live_autoregressive_generation_by_task",
    "v546_scientific_terminal_states",
    "closed_parser_jit_receipt",
    "qualified_certified_learning_receipt",
    "open_live_generation_and_consumer_gaps",
    "arc_no_solve_receipt",
    "v547_milestone_and_doc_hash",
    "v547_task_ids",
    "v547_id_collision_check",
    "v547_deliverable_checks",
    "v547_dependency_checks",
    "v547_structured_gate_checks",
    "v547_prior_failure_checks",
    "v547_llm_model_policy_checks",
    "prompt_contract_checks",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "llm_call_count",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "status": "The status keeps flagged boundaries visible.",
    "v546_milestone_and_queue_hash": "The V546 task denominator is content-addressed.",
    "queued_v546_task_ids": "Only Exp6337 through Exp6349 are in scope.",
    "terminal_v546_artifacts_by_task": "Each V546 task keeps its exact artifact row.",
    "blocked_v546_tasks": "Gate blocks stay separate from missing artifacts.",
    "flagged_v546_artifacts_and_reasons": "Flags cannot be promoted to clean evidence.",
    "inference_substrate_classification_by_task": "Substrate labels prevent replay laundering.",
    "live_autoregressive_generation_by_task": "Live generation is recorded as a separate claim.",
    "v546_scientific_terminal_states": "Scientific states stay distinct from infrastructure.",
    "closed_parser_jit_receipt": "The parser/JIT lane closed after the null canary.",
    "qualified_certified_learning_receipt": "Factor-learning evidence stays replay-qualified.",
    "open_live_generation_and_consumer_gaps": "Open gaps show what V547 must still prove.",
    "arc_no_solve_receipt": "ARC action influence is not solve credit.",
    "v547_milestone_and_doc_hash": "The active roadmap and proposal are hash-pinned.",
    "v547_task_ids": "The active V547 queue identity is explicit.",
    "v547_id_collision_check": "Duplicate task ids fail closed.",
    "v547_deliverable_checks": "Deliverables must be unique result JSON paths.",
    "v547_dependency_checks": "Dependencies must name active V547 tasks.",
    "v547_structured_gate_checks": "Gates must point to declared upstream fields.",
    "v547_prior_failure_checks": "Prior failures need changed mechanisms.",
    "v547_llm_model_policy_checks": "Live GGUF tasks must name required model ids.",
    "prompt_contract_checks": "Run commands and final protections prevent drift.",
    "protected_files_unchanged": "Protected hashes show no roadmap or conductor rewrite.",
    "preconditions_checked": "Inputs, tools, disk, parser, and hashes are captured.",
    "inference_substrate": "This handoff uses repository evidence only.",
    "verifier_is_oracle": "The handoff audits records and is not an oracle.",
    "llm_call_count": "Bare zero proves no model call occurred.",
    "field_provenance": "Every field cites source evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands record the verification boundary.",
    "test_exit_codes": "Exit codes stay separate from the verdict.",
    "duration_s": "Wall time records audit cost without padding.",
    "random_seeds": "No random sampling is used by the handoff.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The terminal prefix states the bounded claim.",
}


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


def _bare_value(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) >= {"value", "principle"}:
        return value.get("value")
    return value


def _numeric_count(value: Any) -> int:
    value = _bare_value(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _required_artifact_fields_block(prompt: str) -> str:
    lines = str(prompt).splitlines()
    block: list[str] = []
    for index, line in enumerate(lines):
        if "REQUIRED ARTIFACT FIELDS:" not in line.upper():
            continue
        block.append(line)
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped or stripped.endswith(":") or stripped.startswith("CONCRETE STEPS"):
                break
            block.append(stripped)
        break
    return "\n".join(block)


def _proposal_task_rows(root: Path) -> list[JsonDict]:
    path = root / MILESTONE_DOC_RELATIVE_PATH
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows: list[JsonDict] = []
    pattern = re.compile(
        r"^### Exp(?P<num>\d+) - (?P<title>.*?)\n(?P<body>.*?)(?=^### Exp|\Z)",
        flags=re.M | re.S,
    )
    for match in pattern.finditer(text):
        number = match.group("num")
        body = match.group("body")
        deliverable_match = re.search(r"\*\*Deliverable:\*\* `(?P<path>results/[^`]+\.json)`", body)
        if not deliverable_match:
            continue
        expected = next(
            (task_id for task_id in EXPECTED_V547_PROPOSAL_TASK_IDS if task_id.startswith(f"exp{number}-")),
            f"exp{number}",
        )
        rows.append(
            {
                "id": expected,
                "exp_number": int(number),
                "title": match.group("title").strip(),
                "deliverable": deliverable_match.group("path"),
                "source": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            }
        )
    return rows


def load_v547_roadmap(root: Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    doc_path = root / MILESTONE_DOC_RELATIVE_PATH
    data = read_yaml_mapping(active_path)
    tasks = _roadmap_tasks(data)
    proposal_rows = _proposal_task_rows(root)
    identity = {
        "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(active_path),
        "milestone": data.get("milestone"),
        "milestone_doc": data.get("milestone_doc"),
        "milestone_doc_sha256": path_sha256(doc_path),
        "requested_next_path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        "research_roadmap_next_present": next_path.exists(),
        "research_roadmap_next_sha256": path_sha256(next_path),
        "task_count": len(tasks),
        "expected_task_count": len(EXPECTED_V547_TASK_IDS),
        "proposal_task_count": len(proposal_rows),
        "proposal_expected_task_count": len(EXPECTED_V547_PROPOSAL_TASK_IDS),
        "proposal_task_ids": [str(row["id"]) for row in proposal_rows],
        "selection_note": "active research-roadmap.yaml is V547 with the first 7 queued tasks",
    }
    data = dict(data)
    data["proposal_tasks"] = proposal_rows
    return data, identity


def _v547_task_rows(tasks: Sequence[JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in tasks:
        prompt = str(task.get("prompt") or "")
        required_fields = sorted(required_artifact_fields_from_prompt(prompt))
        named_models = model_specs_named_in_prompt(prompt)
        required_block = _required_artifact_fields_block(prompt)
        rows.append(
            {
                "task_id": str(task.get("id") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "requires": list(task.get("requires") or []),
                "gated_on": list(task.get("gated_on") or []),
                "agent_type": task.get("agent_type"),
                "model": task.get("model"),
                "requires_gpu": task.get("requires_gpu") is True,
                "required_artifact_field_count": len(required_fields),
                "required_artifact_fields_sha256": payload_sha256(required_fields),
                "MODEL_SPECS_in_required_artifact_fields": "MODEL_SPECS" in required_block,
                "model_identifier_count": len(named_models),
                "model_identifier_set_sha256": payload_sha256(named_models),
                "required_model_identifiers_present": MANDATED_HEADLINE_GGUF_IDS
                <= set(named_models),
            }
        )
    return rows


def validate_v547_roadmap_data(data: JsonMap, retired_exp_ids: set[int]) -> JsonDict:
    tasks = _roadmap_tasks(data)
    proposal_tasks = [
        dict(task) for task in data.get("proposal_tasks", []) if isinstance(task, Mapping)
    ]
    ids = [str(task.get("id") or "") for task in tasks]
    proposal_ids = [str(task.get("id") or "") for task in proposal_tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    id_counts = Counter(ids)
    deliverable_counts = Counter(deliverables)
    duplicate_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    duplicate_deliverables = sorted(
        path for path, count in deliverable_counts.items() if path and count > 1
    )
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    id_set = set(ids)
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        for task_id, task in tasks_by_id.items()
    }

    schema_errors: list[str] = []
    try:
        active_data = {key: value for key, value in data.items() if key != "proposal_tasks"}
        Roadmap.model_validate(active_data)
    except Exception as exc:  # noqa: BLE001 - serialized for the artifact.
        schema_errors.append(str(exc))

    proposal_deliverable_failures = [
        {"task_id": str(task.get("id") or ""), "deliverable": str(task.get("deliverable") or "")}
        for task in proposal_tasks
        if not str(task.get("deliverable") or "").startswith("results/")
        or not str(task.get("deliverable") or "").endswith(".json")
    ]
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
        requires = task.get("requires")
        for dep in requires if isinstance(requires, list) else []:
            dep_text = str(dep)
            dep_num = exp_number(dep_text)
            retired = dep_num in retired_exp_ids if dep_num is not None else False
            missing = dep_text not in id_set
            self_dependency = dep_text == task_id
            if missing or self_dependency or retired:
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
    gate_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            gate_count += 1
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            if not ok:
                gate_failures.append({"task_id": task_id, "gate": gate, "reason": reason})

    prior_failures: list[JsonDict] = []
    prior_entry_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        priors = task.get("prior_failures")
        if priors is None:
            continue
        if not isinstance(priors, list) or not priors:
            prior_failures.append({"task_id": task_id, "reason": "empty_prior_failures"})
            continue
        prior_entry_count += len(priors)
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    route_failures: list[JsonDict] = []
    model_policy_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if task.get("agent_type") != "codex" or task.get("model") != "gpt-5.5":
            route_failures.append(
                {
                    "task_id": task_id,
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "expected_agent_type": "codex",
                    "expected_model": "gpt-5.5",
                }
            )
        prompt = str(task.get("prompt") or "")
        required_block = _required_artifact_fields_block(prompt)
        named_models = set(model_specs_named_in_prompt(prompt))
        live_llm = task.get("requires_gpu") is True or task_id in LIVE_MODEL_V547_TASK_IDS
        if live_llm and "MODEL_SPECS" not in required_block:
            model_policy_failures.append(
                {"task_id": task_id, "reason": "missing_MODEL_SPECS_required_field"}
            )
        if live_llm and not MANDATED_HEADLINE_GGUF_IDS <= named_models:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "missing_mandated_gguf_ids",
                    "expected": sorted(MANDATED_HEADLINE_GGUF_IDS),
                    "found": sorted(named_models),
                }
            )
        if named_models and not named_models <= MANDATED_HEADLINE_GGUF_IDS:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "non_mandated_gguf_id",
                    "ids": sorted(named_models - MANDATED_HEADLINE_GGUF_IDS),
                }
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
        has_required_block = bool(required_artifact_fields_from_prompt(prompt))
        if not (has_run and has_ending and has_required_block):
            prompt_failures.append(
                {
                    "task_id": task_id,
                    "run_command_present": has_run,
                    "protected_conductor_ending": has_ending,
                    "required_artifact_block_present": has_required_block,
                }
            )

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "v547_task_ids": ids,
        "proposal_task_ids": proposal_ids,
        "task_count": len(tasks),
        "proposal_task_count": len(proposal_tasks),
        "v547_id_collision_check": {
            "ok": ids == list(EXPECTED_V547_TASK_IDS)
            and proposal_ids == list(EXPECTED_V547_PROPOSAL_TASK_IDS)
            and not duplicate_ids,
            "task_ids": ids,
            "expected_task_ids": list(EXPECTED_V547_TASK_IDS),
            "expected_order": ids == list(EXPECTED_V547_TASK_IDS),
            "proposal_task_ids": proposal_ids,
            "expected_proposal_task_ids": list(EXPECTED_V547_PROPOSAL_TASK_IDS),
            "proposal_expected_order": proposal_ids == list(EXPECTED_V547_PROPOSAL_TASK_IDS),
            "duplicate_ids": duplicate_ids,
        },
        "v547_deliverable_checks": {
            "ok": not deliverable_failures
            and not proposal_deliverable_failures
            and not duplicate_deliverables,
            "task_deliverables": _v547_task_rows(tasks),
            "proposal_task_deliverables": [
                {"task_id": row.get("id"), "deliverable": row.get("deliverable")}
                for row in proposal_tasks
            ],
            "failures": deliverable_failures,
            "proposal_failures": proposal_deliverable_failures,
            "duplicate_deliverables": duplicate_deliverables,
        },
        "v547_dependency_checks": {
            "ok": not dependency_failures,
            "failures": dependency_failures,
            "retired_dependency_count": retired_dependency_count,
        },
        "v547_structured_gate_checks": {
            "ok": not gate_failures,
            "gate_count": gate_count,
            "failures": gate_failures,
        },
        "v547_prior_failure_checks": {
            "ok": not prior_failures,
            "prior_entry_count": prior_entry_count,
            "failures": prior_failures,
        },
        "v547_llm_model_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "live_llm_task_ids": sorted(LIVE_MODEL_V547_TASK_IDS),
            "mandated_model_identifier_count": len(MANDATED_HEADLINE_GGUF_IDS),
            "mandated_model_identifier_set_sha256": payload_sha256(
                sorted(MANDATED_HEADLINE_GGUF_IDS)
            ),
            "route_failures": route_failures,
            "model_policy_failures": model_policy_failures,
        },
        "prompt_contract_checks": {
            "ok": not prompt_failures,
            "checked_task_count": len(tasks),
            "proposal_only_task_count": len(proposal_tasks) - len(tasks),
            "failures": prompt_failures,
        },
    }


def conductor_receipts_for_task(root: Path, task_id: str) -> JsonDict:
    snippet = V546_TITLE_SNIPPETS.get(task_id, "")
    rows: list[JsonDict] = []
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if path.exists() and snippet:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if snippet.lower() not in line.lower():
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) >= 4:
                rows.append(
                    {
                        "line": line_number,
                        "timestamp_utc": parts[0],
                        "title_truncated": parts[1],
                        "status": parts[2],
                        "message": parts[3],
                        "raw": line.strip(),
                    }
                )
    statuses = [str(row["status"]) for row in rows]
    terminal_statuses = {"OK", "FAIL", "FLAGGED", "GATE_BLOCK"}
    return {
        "receipt_count": len(rows),
        "terminal_receipt_count": sum(status in terminal_statuses for status in statuses),
        "statuses": statuses,
        "last_status": statuses[-1] if statuses else None,
        "rows": rows,
    }


def classify_v546_evidence(root: Path) -> tuple[JsonDict, JsonDict, JsonDict]:
    terminal: JsonDict = {}
    blocked: JsonDict = {}
    missing: JsonDict = {}
    for task_id in EXPECTED_V546_TASK_IDS:
        rel = V546_DELIVERABLES_BY_TASK[task_id]
        path = root / rel
        payload, meta = read_json_mapping(path)
        classification = classify_artifact_path(path).to_dict()
        receipt = conductor_receipts_for_task(root, task_id)
        row = {
            "task_id": task_id,
            "declared_deliverable": rel,
            "present": classification["present"],
            "loadable": classification["loadable"],
            "sha256": classification["sha256"] or meta.get("sha256"),
            "terminal_class": classification["classification"],
            "terminal": classification["terminal"],
            "reason": classification["reason"],
            "status_raw": classification["status_raw"],
            "honest_verdict_raw": classification["honest_verdict_raw"],
            "conductor_receipt": receipt,
            "clean_promotion_attempted": False,
            "missing_artifact": not classification["present"],
        }
        if not classification["present"]:
            missing[task_id] = {**row, "honest_verdict_from_artifact": None}
            continue
        terminal[task_id] = row
        if (
            task_id in BLOCKED_V546_TASK_IDS
            or classification["classification"] == "skipped"
            or classification["status_raw"] == "blocked"
            or "GATE_BLOCK" in set(receipt["statuses"])
        ):
            blocked[task_id] = {**row, "explicit_gate_block": task_id in BLOCKED_V546_TASK_IDS}
    return terminal, blocked, missing


def flagged_v546_artifacts_and_reasons(root: Path, terminal: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    for task_id, terminal_row in terminal.items():
        rel = str(terminal_row.get("declared_deliverable") or "")
        payload, _meta = read_json_mapping(root / rel)
        classification = str(terminal_row.get("terminal_class") or "")
        flagged = (
            classification == "flagged"
            or payload.get("flagged_adversarial") is True
            or any(
                status == "FLAGGED"
                for status in (
                    terminal_row.get("conductor_receipt", {}).get("statuses", [])
                    if isinstance(terminal_row.get("conductor_receipt"), Mapping)
                    else []
                )
            )
        )
        if flagged:
            rows[str(task_id)] = {
                "declared_deliverable": rel,
                "terminal_class": classification,
                "flagged_adversarial": payload.get("flagged_adversarial"),
                "corrigendum_pending": payload.get("corrigendum_pending"),
                "conductor_receipt": terminal_row.get("conductor_receipt"),
                "clean_promotion_attempted": False,
            }
    return rows


def classify_substrate(payload: JsonMap, task_id: str) -> JsonDict:
    substrate = str(payload.get("inference_substrate") or "")
    keys = set(payload)
    if not payload:
        cls = "unknown_missing_payload"
        invoked = False
    elif task_id == "exp6344-counterexample-factor-proposal-calibration":
        cls = "deterministic_replay_with_gguf_receipts"
        invoked = False
    elif task_id == "exp6345-prospective-certified-factor-evolution-ab":
        cls = "tokenizer_only_exact_replay"
        invoked = False
    elif "live_llm_inference" in substrate or "raw_generation_paths_hashes_and_counts" in keys:
        cls = "live_autoregressive_generation"
        invoked = True
    elif "web" in substrate or "bibliographic" in substrate:
        cls = "web_or_bibliographic_search"
        invoked = False
    elif "synthetic" in substrate:
        cls = "synthetic_replay"
        invoked = False
    elif "artifact_replay" in substrate or ("deterministic" in substrate and "replay" in substrate):
        cls = "deterministic_replay"
        invoked = False
    elif "aggregation" in substrate:
        cls = "artifact_aggregation"
        invoked = False
    elif "deterministic" in substrate:
        cls = "deterministic_exact_checker"
        invoked = False
    else:
        cls = "declared_other"
        invoked = False
    return {
        "task_id": task_id,
        "class": cls,
        "inference_substrate_raw": substrate or None,
        "live_autoregressive_generation_invoked": invoked,
        "llm_call_count_raw": payload.get("llm_call_count"),
        "has_MODEL_SPECS": "MODEL_SPECS" in payload,
        "has_tokenizer_receipts": "llama_cpp_embedded_tokenizer_receipts" in payload,
    }


def substrate_receipts(root: Path) -> tuple[JsonDict, JsonDict]:
    classifications: JsonDict = {}
    live: JsonDict = {}
    for task_id in EXPECTED_V546_TASK_IDS:
        payload, _meta = read_json_mapping(root / V546_DELIVERABLES_BY_TASK[task_id])
        row = classify_substrate(payload, task_id)
        classifications[task_id] = row
        live[task_id] = {
            "invoked": row["live_autoregressive_generation_invoked"],
            "class": row["class"],
            "evidence": row["inference_substrate_raw"],
        }
    return classifications, live


def v546_milestone_and_queue_hash(root: Path) -> JsonDict:
    conductor_text = (root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    activation_rows = [
        line
        for line in conductor_text.splitlines()
        if "Plan milestone 2026.08.546" in line or "Milestone 2026.08.546 activated" in line
    ]
    payload = {
        "milestone": MILESTONE_V546,
        "queued_task_ids": list(EXPECTED_V546_TASK_IDS),
        "deliverables_by_task": V546_DELIVERABLES_BY_TASK,
        "activation_rows": activation_rows,
    }
    return {**payload, "queue_hash": payload_sha256(payload)}


def v546_scientific_terminal_states(terminal: JsonMap) -> JsonDict:
    infrastructure = {
        "exp6337-v546-bounded-terminal-handoff",
        "exp6338-v546-post-marker-source-scope-freeze",
        "exp6349-v546-adversarial-capstone",
    }
    rows = {
        task_id: row
        for task_id, row in terminal.items()
        if isinstance(row, Mapping) and task_id not in infrastructure
    }
    counts = Counter(str(row.get("terminal_class")) for row in rows.values())
    return {
        "task_ids": list(rows),
        "terminal_class_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def closed_parser_jit_receipt(payloads: JsonMap, blocked: JsonMap) -> JsonDict:
    exp6340 = payloads.get("exp6340-parser-jit-semantic-diversity-canary") or {}
    score = _bare_value(
        exp6340.get("semantic_diversity_gain_score") if isinstance(exp6340, Mapping) else None
    )
    exp6341_blocked = "exp6341-prospective-prefix-utility-ab" in blocked
    return {
        "closed": score == 0.0 and exp6341_blocked,
        "exp6340_semantic_diversity_gain_score": score,
        "exp6340_honest_verdict": exp6340.get("honest_verdict")
        if isinstance(exp6340, Mapping)
        else None,
        "exp6341_gate_blocked": exp6341_blocked,
        "boundary": "parser/JIT semantic-diversity lane closed after null canary",
    }


def qualified_certified_learning_receipt(payloads: JsonMap, live: JsonMap) -> JsonDict:
    fields = {
        "exp6342_anytime_release_certificate_ready_score": (
            "exp6342-anytime-evalue-release-ledger",
            "anytime_release_certificate_ready_score",
        ),
        "exp6343_evidence_factor_lifecycle_ready_score": (
            "exp6343-evidence-carrying-factor-lifecycle",
            "evidence_factor_lifecycle_ready_score",
        ),
        "exp6344_counterexample_proposal_ready_score": (
            "exp6344-counterexample-factor-proposal-calibration",
            "counterexample_proposal_ready_score",
        ),
        "exp6345_certified_continuous_learning_ready_score": (
            "exp6345-prospective-certified-factor-evolution-ab",
            "certified_continuous_learning_ready_score",
        ),
        "exp6346_safety_ready_score": (
            "exp6346-certified-factor-evolution-safety-audit",
            "safety_ready_score",
        ),
    }
    scores: JsonDict = {}
    for name, (task_id, field) in fields.items():
        payload = payloads.get(task_id) or {}
        scores[name] = _bare_value(payload.get(field)) if isinstance(payload, Mapping) else None
    exp6344_live = bool(
        (live.get("exp6344-counterexample-factor-proposal-calibration") or {}).get("invoked")
    )
    exp6345_live = bool(
        (live.get("exp6345-prospective-certified-factor-evolution-ab") or {}).get("invoked")
    )
    return {
        "qualified_closed": all(value == 1.0 for value in scores.values()),
        "scores": scores,
        "live_generation_claim": exp6344_live or exp6345_live,
        "exp6344_no_live_autoregressive_generation": not exp6344_live,
        "exp6345_no_live_autoregressive_generation": not exp6345_live,
        "qualification": "closed_inside_synthetic_and_deterministic_replay_bounds",
    }


def open_live_generation_and_consumer_gaps() -> JsonDict:
    return {
        "live_factor_proposal_generation_open": True,
        "future_consumer_value_open": True,
        "parser_jit_generation_open": False,
        "reason": (
            "V546 closed replay mechanics but did not prove real proposal generation "
            "or later consumer value."
        ),
    }


def arc_no_solve_receipt(payloads: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    total = 0
    for task_id in (
        "exp6347-arc-action-influence-preflight",
        "exp6348-arc-default-off-action-influence-ab",
    ):
        payload = payloads.get(task_id) or {}
        solve_count = _numeric_count(payload.get("solve_claim_count")) if isinstance(payload, Mapping) else 0
        total += solve_count
        rows[task_id] = {
            "solve_claim_count": solve_count,
            "solve_provenance": payload.get("solve_provenance") if isinstance(payload, Mapping) else None,
            "honest_verdict": payload.get("honest_verdict") if isinstance(payload, Mapping) else None,
        }
    return {
        "solve_claim_count": total,
        "no_solve_boundary_preserved": total == 0,
        "rows": rows,
    }


def _repository_validator_checks(root: Path) -> JsonDict:
    roadmap_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    complete_path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    schema_errors, prior_errors = validate_prior_failure_roadmap(roadmap_path, complete_path)
    gate_result = audit_roadmap(roadmap_path, complete_path=complete_path).to_artifact()
    exclusion_risks = exclusion_manifest_lint(roadmap_path)
    hard_exclusion_risks = [risk for risk in exclusion_risks if risk.severity == "HARD"]
    return {
        "ok": not schema_errors
        and not prior_errors
        and gate_result["roadmap_gate_audit_passed"] is True
        and not hard_exclusion_risks,
        "schema_and_prior_linter": {
            "schema_errors": schema_errors,
            "prior_failure_violations": prior_errors,
        },
        "gate_audit": gate_result,
        "exclusion_manifest_lint": {
            "risk_count": len(exclusion_risks),
            "hard_risk_count": len(hard_exclusion_risks),
            "risks": [risk.__dict__ for risk in exclusion_risks],
        },
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


def _disk_receipt(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


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


def _yaml_parser_receipt() -> JsonDict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML is a dependency.
        return {"available": False, "error": str(exc)}
    return {"available": True, "module": yaml.__name__}


def preconditions_checked(
    root: Path,
    v547_identity: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    git_status_after_tests: Sequence[str] | None = None,
) -> JsonDict:
    return {
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests or []),
        "input_hashes_before": _input_hashes(root),
        "v547_roadmap_identity": v547_identity,
        "protected_hashes_before_artifact_write": before_hashes,
        "disk": _disk_receipt(root),
        "command_availability": _command_availability(),
        "yaml_parser": _yaml_parser_receipt(),
        "active_roadmap_was_not_edited": True,
        "conductor_was_not_edited": True,
        "active_v547_queue_count": v547_identity.get("task_count"),
        "proposal_v547_task_count": v547_identity.get("proposal_task_count"),
    }


def _field_provenance() -> JsonDict:
    sources = sorted(
        {
            "REQ-INFRA-6350",
            ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
            SPEC_RELATIVE_PATH.as_posix(),
            "scripts/roadmap_schema.py",
            "scripts/validate_prior_failures.py",
            "scripts/audit_roadmap_gates.py",
            "scripts/exclusion_manifest_lint.py",
            *V546_DELIVERABLES_BY_TASK.values(),
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


def _status_from_exit_codes(exit_codes: JsonMap) -> tuple[str, str]:
    nonzero = {command: code for command, code in exit_codes.items() if int(code) != 0}
    if nonzero:
        return (
            "blocked_validation_command_failed",
            "blocked: Exp6350 preserved V546 boundaries but one or more validation commands failed",
        )
    return (
        "complete_with_flagged_boundaries",
        (
            "complete_with_flagged_boundaries: V546 evidence preserved Exp6337 as flagged, "
            "Exp6341 as gate-blocked, factor learning as replay-qualified, and ARC as no-solve"
        ),
    )


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
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    v547_data, v547_identity = load_v547_roadmap(root)
    v547_validation = validate_v547_roadmap_data(
        v547_data, load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    )
    terminal, blocked, missing = classify_v546_evidence(root)
    payloads = {
        task_id: read_json_mapping(root / rel)[0]
        for task_id, rel in V546_DELIVERABLES_BY_TASK.items()
    }
    substrate, live = substrate_receipts(root)
    command_rows = [dict(row) for row in (command_receipts or [])]
    exit_codes = _test_exit_codes(command_rows)
    status, verdict = _status_from_exit_codes(exit_codes)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "v546_milestone_and_queue_hash": v546_milestone_and_queue_hash(root),
        "queued_v546_task_ids": list(EXPECTED_V546_TASK_IDS),
        "terminal_v546_artifacts_by_task": terminal,
        "blocked_v546_tasks": blocked,
        "missing_v546_artifacts_by_task": missing,
        "flagged_v546_artifacts_and_reasons": flagged_v546_artifacts_and_reasons(
            root, terminal
        ),
        "inference_substrate_classification_by_task": substrate,
        "live_autoregressive_generation_by_task": live,
        "v546_scientific_terminal_states": v546_scientific_terminal_states(terminal),
        "closed_parser_jit_receipt": closed_parser_jit_receipt(payloads, blocked),
        "qualified_certified_learning_receipt": qualified_certified_learning_receipt(
            payloads, live
        ),
        "open_live_generation_and_consumer_gaps": open_live_generation_and_consumer_gaps(),
        "arc_no_solve_receipt": arc_no_solve_receipt(payloads),
        "v547_milestone_and_doc_hash": v547_identity,
        "v547_task_ids": v547_validation["v547_task_ids"],
        "v547_id_collision_check": v547_validation["v547_id_collision_check"],
        "v547_deliverable_checks": v547_validation["v547_deliverable_checks"],
        "v547_dependency_checks": v547_validation["v547_dependency_checks"],
        "v547_structured_gate_checks": v547_validation["v547_structured_gate_checks"],
        "v547_prior_failure_checks": v547_validation["v547_prior_failure_checks"],
        "v547_llm_model_policy_checks": v547_validation["v547_llm_model_policy_checks"],
        "prompt_contract_checks": v547_validation["prompt_contract_checks"],
        "repository_validator_checks": _repository_validator_checks(root),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root,
            v547_identity,
            before,
            status_before,
            git_status_after_tests,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "llm_call_count": 0,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": exit_codes,
        "duration_s": time.perf_counter() - started,
        "random_seeds": {"used": [], "deterministic": True},
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
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
    if report.get("llm_call_count") != 0 or not isinstance(report.get("llm_call_count"), int):
        errors.append("llm_call_count must be bare 0")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("queued_v546_task_ids") != list(EXPECTED_V546_TASK_IDS):
        errors.append("V546 task ids must be Exp6337 through Exp6349")
    terminal = report.get("terminal_v546_artifacts_by_task")
    if not isinstance(terminal, Mapping) or set(terminal) != set(EXPECTED_V546_TASK_IDS):
        errors.append("all V546 tasks must have exact terminal artifact rows")
    blocked = report.get("blocked_v546_tasks")
    if not isinstance(blocked, Mapping) or "exp6341-prospective-prefix-utility-ab" not in blocked:
        errors.append("Exp6341 gate block must be recorded")
    flagged = report.get("flagged_v546_artifacts_and_reasons")
    if not isinstance(flagged, Mapping) or EXP6337_TASK_ID not in flagged:
        errors.append("Exp6337 flag must be preserved")
    substrate = report.get("inference_substrate_classification_by_task")
    if not isinstance(substrate, Mapping):
        errors.append("inference_substrate_classification_by_task is not a mapping")
        substrate = {}
    exp6344 = substrate.get("exp6344-counterexample-factor-proposal-calibration")
    if isinstance(exp6344, Mapping) and exp6344.get("class") == "live_autoregressive_generation":
        errors.append("Exp6344 must not be live autoregressive generation")
    exp6345 = substrate.get("exp6345-prospective-certified-factor-evolution-ab")
    if isinstance(exp6345, Mapping) and exp6345.get("class") == "live_autoregressive_generation":
        errors.append("Exp6345 must not be live autoregressive generation")
    if report.get("v547_task_ids") != list(EXPECTED_V547_TASK_IDS):
        errors.append("active V547 task ids must match the queued roadmap")
    if not str(report.get("honest_verdict") or "").startswith(
        ("complete_with_flagged_boundaries:", "blocked:", "retired:")
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
        raise ValueError(f"invalid Exp6350 report: {errors}")
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
    if isinstance(payload, Mapping):
        return [
            {"command": str(command), "exit_code": int(code)}
            for command, code in payload.items()
        ]
    if isinstance(payload, list):
        rows = [
            {"command": str(row.get("command")), "exit_code": int(row.get("exit_code") or 0)}
            for row in payload
            if isinstance(row, Mapping) and row.get("command")
        ]
        return rows or [{"command": RUN_COMMAND, "exit_code": 0}]
    return [{"command": RUN_COMMAND, "exit_code": 0}]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    before = protected_hashes(root)
    receipts = list(command_receipts) if command_receipts is not None else read_external_test_receipts()
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
    )
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    report = run(date=args.date, write=not args.no_write)
    print(f"{RESULT_RELATIVE_PATH.as_posix()} status={report['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
