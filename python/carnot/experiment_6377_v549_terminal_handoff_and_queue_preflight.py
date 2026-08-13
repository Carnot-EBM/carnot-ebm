"""Exp6377 V549 terminal handoff and queue preflight.

Spec refs: REQ-INFRA-6377, SCENARIO-INFRA-6377-1,
SCENARIO-INFRA-6377-2, SCENARIO-INFRA-6377-3,
SCENARIO-INFRA-6377-4, SCENARIO-INFRA-6377-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_6272_v541_terminal_transition import (
    exp_number,
    gate_ok,
    git_status_lines,
    load_retired_exp_ids,
    prior_ok,
    read_yaml_mapping,
    required_artifact_fields_from_prompt,
)
from carnot.experiment_6284_v542_terminal_transition import model_specs_named_in_prompt
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

from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_manifest_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402
from validate_prior_failures import validate_roadmap as validate_prior_failure_roadmap  # noqa: E402


MILESTONE_V548 = "2026.08.548"
MILESTONE_V549 = "2026.08.549"
RUN_DATE = "20260813"
EXPERIMENT_ID = "exp6377-v549-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6377.v549_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

ACTIVE_V548_TASK_IDS = (
    "exp6363-v548-terminal-handoff-and-queue-preflight",
    "exp6364-v548-post-marker-source-scope-freeze",
    "exp6365-gguf-child-failure-forensics-runtime-contract",
    "exp6366-repaired-live-factor-proposal-authenticity",
)
PROPOSAL_ONLY_V548_IDS = (
    "exp6367-verified-frontier-factor-proposal-ab",
    "exp6368-prospective-verified-frontier-factor-learning",
    "exp6369-dependency-guided-factor-rollback-stress",
    "exp6370-default-off-certified-factor-consumer-ab",
    "exp6371-live-learning-consumer-and-rollback-audit",
    "exp6372-arc-two-sided-goal-evidence-contract",
    "exp6373-arc-active-reward-machine-discriminator",
    "exp6374-arc-goal-evidence-response-calibration",
    "exp6375-arc-default-off-active-goal-shadow",
    "exp6376-v548-adversarial-capstone",
)
V548_DELIVERABLES_BY_TASK = {
    "exp6363-v548-terminal-handoff-and-queue-preflight": (
        "results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json"
    ),
    "exp6364-v548-post-marker-source-scope-freeze": (
        "results/experiment_6364_v548_post_marker_source_scope_freeze.json"
    ),
    "exp6365-gguf-child-failure-forensics-runtime-contract": (
        "results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json"
    ),
    "exp6366-repaired-live-factor-proposal-authenticity": (
        "results/experiment_6366_repaired_live_factor_proposal_authenticity.json"
    ),
}
V548_TITLE_SNIPPETS = {
    "exp6363-v548-terminal-handoff-and-queue-preflight": (
        "V547 terminal evidence handoff and V548 queue pref"
    ),
    "exp6364-v548-post-marker-source-scope-freeze": (
        "V548 dated source delta and three-lane scope freez"
    ),
    "exp6365-gguf-child-failure-forensics-runtime-contract": (
        "Three-model GGUF child failure forensics and obser"
    ),
    "exp6366-repaired-live-factor-proposal-authenticity": (
        "Gated repaired three-model live factor proposal au"
    ),
}

EXPECTED_V549_TASK_IDS = (
    "exp6377-v549-terminal-handoff-and-queue-preflight",
    "exp6378-v549-post-marker-source-scope-freeze",
    "exp6379-canonical-factor-edit-transport-contract",
    "exp6380-three-family-canonical-factor-transport-canary",
    "exp6381-verified-frontier-live-factor-proposal-ab",
    "exp6382-chronological-verified-factor-self-learning",
    "exp6383-dependency-guided-factor-rollback-stress",
    "exp6384-default-off-certified-factor-consumer-ab",
    "exp6385-live-factor-learning-and-rollback-safety-audit",
    "exp6386-arc-two-sided-goal-evidence-contract",
    "exp6387-arc-active-reward-machine-discriminator",
    "exp6388-arc-goal-evidence-response-calibration",
    "exp6389-arc-default-off-active-goal-shadow",
    "exp6390-v549-adversarial-capstone",
)
MANDATED_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
GGUF_ID_RE = re.compile(r"[\w.-]+/[\w.-]+-GGUF")
FINAL_PROHIBITION_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6377_v549_terminal_handoff_and_queue_preflight "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6377_v549_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6377_v549_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6377_v549_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6377_v549_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6377_v549_terminal_handoff_and_queue_preflight.py"
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
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROADMAP_SCHEMA_COMMAND,
    PRIOR_FAILURE_COMMAND,
    GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    FULL_PYTEST_COMMAND,
    RUN_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6377_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *[Path(path) for path in V548_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v548_active_roadmap_path_and_hash",
    "v548_active_task_ids",
    "v548_terminal_artifacts_by_task",
    "v548_conductor_outcomes",
    "v548_blocked_null_clean_and_proposal_only_states",
    "exp6365_runtime_boundary",
    "exp6366_transport_failure_boundary",
    "v549_milestone_doc_and_queue_hashes",
    "v549_task_ids",
    "v549_id_and_deliverable_checks",
    "v549_dependency_and_gate_checks",
    "v549_gate_field_cross_reference_checks",
    "v549_prior_failure_checks",
    "v549_exclusion_manifest_checks",
    "v549_agent_model_and_llm_policy_checks",
    "prompt_contract_checks",
    "active_roadmap_modified",
    "conductor_modified",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "status": "The status states whether the V549 queue is activation-safe.",
    "v548_active_roadmap_path_and_hash": "The prior V548 active queue hash is pinned from Exp6363 evidence.",
    "v548_active_task_ids": "Only four active V548 tasks define the handoff denominator.",
    "v548_terminal_artifacts_by_task": "Each V548 task keeps its exact artifact state.",
    "v548_conductor_outcomes": "Conductor outcomes stay separate from artifact fields.",
    "v548_blocked_null_clean_and_proposal_only_states": "Blocked, null, clean, and proposal-only states are not mixed.",
    "exp6365_runtime_boundary": "Complete GGUF runtime is not treated as factor transport success.",
    "exp6366_transport_failure_boundary": "Raw nonempty outputs are not treated as parse-valid or exact-checked edits.",
    "v549_milestone_doc_and_queue_hashes": "The V549 proposal and queue sources are hash-pinned.",
    "v549_task_ids": "The audited V549 queue identity is explicit.",
    "v549_id_and_deliverable_checks": "The queue must contain fourteen unique ordered IDs and result JSON deliverables.",
    "v549_dependency_and_gate_checks": "Dependencies and structured gates must be ordered and valid.",
    "v549_gate_field_cross_reference_checks": "Gate fields must appear in upstream required artifact fields.",
    "v549_prior_failure_checks": "Prior failures must name the changed mechanism and retirement rule.",
    "v549_exclusion_manifest_checks": "Retired upstream references must fail before execution.",
    "v549_agent_model_and_llm_policy_checks": "Agent routing and local GGUF policy are checked.",
    "prompt_contract_checks": "Rendered prompts must contain the operational contract the agent receives.",
    "active_roadmap_modified": "The active roadmap must stay byte-identical during this run.",
    "conductor_modified": "The conductor source must stay byte-identical during this run.",
    "protected_files_unchanged": "Protected hashes prove no handoff-side rewrite occurred.",
    "preconditions_checked": "Input hashes and artifact classifications are frozen before field reads.",
    "inference_substrate": "This task uses repository evidence with no model call.",
    "verifier_is_oracle": "The handoff reconciles records and is not a correctness oracle.",
    "field_principles": "Every required field and structured gate expression has a reason.",
    "field_provenance": "Every required field identifies its source kind.",
    "random_seed": "No random sampling is used by this deterministic handoff.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict uses a terminal prefix and names the queue boundary.",
}
FIELD_PROVENANCE = {
    "status": "derived",
    "v548_active_roadmap_path_and_hash": "upstream",
    "v548_active_task_ids": "constant",
    "v548_terminal_artifacts_by_task": "upstream",
    "v548_conductor_outcomes": "measured",
    "v548_blocked_null_clean_and_proposal_only_states": "derived",
    "exp6365_runtime_boundary": "upstream",
    "exp6366_transport_failure_boundary": "upstream",
    "v549_milestone_doc_and_queue_hashes": "measured",
    "v549_task_ids": "upstream",
    "v549_id_and_deliverable_checks": "derived",
    "v549_dependency_and_gate_checks": "derived",
    "v549_gate_field_cross_reference_checks": "derived",
    "v549_prior_failure_checks": "derived",
    "v549_exclusion_manifest_checks": "derived",
    "v549_agent_model_and_llm_policy_checks": "derived",
    "prompt_contract_checks": "derived",
    "active_roadmap_modified": "measured",
    "conductor_modified": "measured",
    "protected_files_unchanged": "measured",
    "preconditions_checked": "measured",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured",
    "tests_run": "measured",
    "reproducibility_checksum": "derived",
    "honest_verdict": "derived",
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


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def protected_files_unchanged(before: JsonMap, after: JsonMap) -> JsonDict:
    rows = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"ok": all(row["unchanged"] for row in rows.values()), "rows": rows}


def _summarize_artifact(root: Path, rel_path: str) -> JsonDict:
    path = root / rel_path
    if not path.exists():
        return {
            "invoked_before_field_import": False,
            "reason": "artifact_absent",
            "exit_code": None,
            "live_adversarial_findings": [],
        }
    command = [sys.executable, "scripts/summarize_artifact.py", rel_path]
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    findings = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("[") or "DURATION_TOO_SHORT" in line
    ]
    return {
        "invoked_before_field_import": True,
        "command": " ".join(command),
        "exit_code": result.returncode,
        "stdout_sha256": payload_sha256(result.stdout),
        "stderr_sha256": payload_sha256(result.stderr),
        "live_adversarial_findings": findings,
    }


def _conductor_rows(root: Path, task_id: str) -> list[JsonDict]:
    snippet = V548_TITLE_SNIPPETS[task_id].lower()
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if snippet not in line.lower():
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
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
    return rows


def v548_conductor_outcomes(root: Path) -> JsonDict:
    outcomes: JsonDict = {}
    for task_id in ACTIVE_V548_TASK_IDS:
        rows = _conductor_rows(root, task_id)
        counts = Counter(str(row["status"]) for row in rows)
        outcomes[task_id] = {
            **dict(sorted(counts.items())),
            "attempt_count": len(rows),
            "rows": rows,
        }
    return outcomes


def classify_v548_artifacts(root: Path) -> tuple[JsonDict, JsonDict]:
    outcomes = v548_conductor_outcomes(root)
    rows: JsonDict = {}
    for task_id in ACTIVE_V548_TASK_IDS:
        rel = V548_DELIVERABLES_BY_TASK[task_id]
        summary = _summarize_artifact(root, rel)
        path = root / rel
        classification = classify_artifact_path(path).to_dict()
        payload, meta = read_json_mapping(path)
        rows[task_id] = {
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
            "summary_receipt": summary,
            "conductor_receipt": outcomes[task_id],
            "stamped_flagged_adversarial": payload.get("flagged_adversarial"),
            "corrigendum_pending": payload.get("corrigendum_pending"),
        }
    return rows, outcomes


def v548_state_receipt(rows: JsonMap) -> JsonDict:
    blocked: list[str] = []
    null: list[str] = []
    clean: list[str] = []
    missing: list[str] = []
    for task_id, row_any in rows.items():
        row = row_any if isinstance(row_any, Mapping) else {}
        terminal_class = str(row.get("terminal_class") or "")
        status_raw = str(row.get("status_raw") or "")
        verdict = str(row.get("honest_verdict_raw") or "")
        if terminal_class == "missing":
            missing.append(str(task_id))
        elif terminal_class == "blocked" or status_raw.startswith("blocked"):
            blocked.append(str(task_id))
        elif terminal_class == "null" or status_raw.startswith("complete_null") or verdict.startswith(
            "complete_null"
        ):
            null.append(str(task_id))
        elif terminal_class in {"complete", "ready", "positive"}:
            clean.append(str(task_id))
    return {
        "blocked": blocked,
        "null": null,
        "clean": clean,
        "missing_active": missing,
        "proposal_only": {
            "task_ids": list(PROPOSAL_ONLY_V548_IDS),
            "executed_count": 0,
            "counted_as_executed": False,
            "counted_as_blocked": False,
            "counted_as_missing": False,
            "reason": "These identities were proposal-only V548 scope and did not enter the active queue.",
        },
        "counts": {
            "blocked": len(blocked),
            "null": len(null),
            "clean": len(clean),
            "missing_active": len(missing),
            "proposal_only": len(PROPOSAL_ONLY_V548_IDS),
        },
    }


def exp6365_runtime_boundary(root: Path) -> JsonDict:
    payload, meta = read_json_mapping(
        root / V548_DELIVERABLES_BY_TASK["exp6365-gguf-child-failure-forensics-runtime-contract"]
    )
    child_rows = payload.get("child_exit_signal_timeout_and_usage_receipts_by_model", {})
    vram_rows = payload.get("vram_rise_and_release_receipts_by_model", {})
    models_used = list(payload.get("models_used") or [])
    return {
        "artifact_path": V548_DELIVERABLES_BY_TASK[
            "exp6365-gguf-child-failure-forensics-runtime-contract"
        ],
        "artifact_sha256": meta["sha256"],
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "models_used": models_used,
        "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
        "all_three_mandated_models_used": set(models_used) == MANDATED_GGUF_IDS,
        "gguf_runtime_observability_ready_score": payload.get(
            "gguf_runtime_observability_ready_score"
        ),
        "all_child_contracts_ok": bool(child_rows)
        and all(
            bool(row.get("contract_ok")) for row in child_rows.values() if isinstance(row, Mapping)
        ),
        "all_vram_rise_and_release_proved": bool(vram_rows)
        and all(
            bool(row.get("proved_rise_and_release"))
            for row in vram_rows.values()
            if isinstance(row, Mapping)
        ),
        "raw_outputs_nonempty": all(
            bool(row.get("raw_bytes_nonempty_before_parse"))
            for row in (payload.get("raw_output_paths_hashes_and_byte_counts", {}) or {}).values()
            if isinstance(row, Mapping)
        ),
        "proposal_quality_claimed": False,
        "boundary": "runtime_observability_only_no_factor_quality_claim",
    }


def exp6366_transport_failure_boundary(root: Path) -> JsonDict:
    payload, meta = read_json_mapping(
        root / V548_DELIVERABLES_BY_TASK["exp6366-repaired-live-factor-proposal-authenticity"]
    )
    raw = payload.get("raw_output_before_parse_paths_hashes_and_counts", {})
    parse = payload.get("parse_valid_invalid_timeout_and_abstain_counts_by_model", {})
    parse_rows = parse.get("by_model", {}) if isinstance(parse, Mapping) else {}
    exact = payload.get("exact_checker_paths_hashes_versions_calls_costs_and_errors", {})
    exact_counts = payload.get("exact_pass_fail_counts_by_model", {})
    valid_count = sum(
        int(row.get("valid") or 0) for row in parse_rows.values() if isinstance(row, Mapping)
    )
    invalid_count = sum(
        int(row.get("invalid") or 0) for row in parse_rows.values() if isinstance(row, Mapping)
    )
    exact_call_count = int(exact.get("exact_checker_calls") or 0) if isinstance(exact, Mapping) else 0
    return {
        "artifact_path": V548_DELIVERABLES_BY_TASK[
            "exp6366-repaired-live-factor-proposal-authenticity"
        ],
        "artifact_sha256": meta["sha256"],
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "models_used": list(payload.get("models_used") or []),
        "repaired_live_factor_proposal_authenticity_ready_score": payload.get(
            "repaired_live_factor_proposal_authenticity_ready_score"
        ),
        "total_raw_output_count": raw.get("total_raw_output_count")
        if isinstance(raw, Mapping)
        else None,
        "total_raw_output_byte_count": raw.get("total_byte_count")
        if isinstance(raw, Mapping)
        else None,
        "all_raw_outputs_nonempty_before_parse": raw.get("all_raw_outputs_nonempty_before_parse")
        if isinstance(raw, Mapping)
        else None,
        "parse_valid_count": valid_count,
        "parse_invalid_count": invalid_count,
        "exact_checker_call_count": exact_call_count,
        "exact_pass_fail_totals": {
            "total_exact_calls": exact_counts.get("total_exact_calls")
            if isinstance(exact_counts, Mapping)
            else None,
            "total_exact_pass": exact_counts.get("total_exact_pass")
            if isinstance(exact_counts, Mapping)
            else None,
            "total_exact_fail": exact_counts.get("total_exact_fail")
            if isinstance(exact_counts, Mapping)
            else None,
        },
        "hidden_state_access_count": payload.get("hidden_state_access_count"),
        "protected_validation_leak_count": payload.get("protected_validation_leak_count"),
        "source_model_weight_mutation_count": payload.get("source_model_weight_mutation_count"),
        "transport_ready": valid_count > 0 and exact_call_count > 0,
        "boundary": "live_generation_nonempty_but_no_parse_valid_factor_edits_or_exact_calls",
    }


def load_v549_queue(root: Path) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_data = read_yaml_mapping(active_path)
    if next_path.exists():
        data = read_yaml_mapping(next_path)
        chosen = ROADMAP_NEXT_RELATIVE_PATH
        note = "research-roadmap-next.yaml exists and was audited"
    else:
        data = active_data
        chosen = ACTIVE_ROADMAP_RELATIVE_PATH
        note = "research-roadmap-next.yaml absent; active research-roadmap.yaml audited"
    identity = {
        "active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "present": active_path.exists(),
            "sha256": path_sha256(active_path),
            "milestone": active_data.get("milestone"),
        },
        "requested_next_roadmap": {
            "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            "present": next_path.exists(),
            "sha256": path_sha256(next_path),
        },
        "audited_queue": {
            "path": chosen.as_posix(),
            "present": (root / chosen).exists(),
            "sha256": path_sha256(root / chosen),
            "milestone": data.get("milestone"),
            "selection_note": note,
        },
        "milestone_doc": {
            "path": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            "present": (root / MILESTONE_DOC_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / MILESTONE_DOC_RELATIVE_PATH),
        },
        "conductor_source": {
            "path": RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix(),
            "present": (root / RESEARCH_CONDUCTOR_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / RESEARCH_CONDUCTOR_RELATIVE_PATH),
        },
        "conductor_log": {
            "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
        },
        "exclusion_manifest": {
            "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            "present": (root / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        },
    }
    return dict(data), identity


def _tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def _gate_expression(task_id: str, gate: JsonMap) -> str:
    return (
        f"gate:{task_id}:{gate.get('upstream')}.{gate.get('artifact_field')}"
        f"{gate.get('op')}{json.dumps(gate.get('value'), sort_keys=True)}"
    )


def render_prompt(raw_prompt: str, root: Path, date: str) -> tuple[str, JsonDict]:
    try:
        rendered = raw_prompt.format(project_root=root.as_posix(), date=date)
        return rendered, {"format_ok": True, "error": None}
    except (KeyError, IndexError, ValueError) as exc:
        rendered = raw_prompt.replace("{project_root}", root.as_posix()).replace("{date}", date)
        return rendered, {"format_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _is_llm_task(task: JsonMap, rendered_prompt: str) -> bool:
    raw_prompt = str(task.get("prompt") or "")
    return (
        task.get("requires_gpu") is True
        or "MODEL_SPECS must include" in raw_prompt
        or "MODEL_SPECS must include" in rendered_prompt
    )


def _risk_rows(risks: Sequence[Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for risk in risks:
        try:
            rows.append(asdict(risk))
        except TypeError:
            rows.append({"repr": repr(risk)})
    return rows


def validate_v549_queue_data(
    data: JsonMap,
    root: Path,
    date: str,
    *,
    retired_exp_ids: set[int] | None = None,
) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    rendered_prompts: dict[str, str] = {}
    render_receipts: dict[str, JsonDict] = {}
    for task_id, task in tasks_by_id.items():
        rendered, receipt = render_prompt(str(task.get("prompt") or ""), root, date)
        rendered_prompts[task_id] = rendered
        render_receipts[task_id] = receipt
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(rendered_prompts.get(task_id, ""))
        for task_id in tasks_by_id
    }

    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        schema_errors.append(str(exc))

    exp_numbers = [exp_number(task_id) for task_id in ids]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    missing_ids = [task_id for task_id in EXPECTED_V549_TASK_IDS if task_id not in ids]
    extra_ids = [task_id for task_id in ids if task_id not in EXPECTED_V549_TASK_IDS]
    duplicate_deliverables = sorted(
        path for path, count in Counter(deliverables).items() if path and count > 1
    )
    deliverable_failures = [
        {"task_id": str(task.get("id") or ""), "deliverable": str(task.get("deliverable") or "")}
        for task in tasks
        if not str(task.get("deliverable") or "").startswith("results/")
        or not str(task.get("deliverable") or "").endswith(".json")
    ]

    retired_ids = retired_exp_ids
    if retired_ids is None:
        retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    id_index = {task_id: index for index, task_id in enumerate(ids)}
    dependency_failures: list[JsonDict] = []
    gate_failures: list[JsonDict] = []
    gate_cross_ref_failures: list[JsonDict] = []
    retired_upstream_references: list[JsonDict] = []
    gate_expressions: list[str] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        requires = task.get("requires")
        for dependency in requires if isinstance(requires, list) else []:
            dep = str(dependency)
            dep_exp_number = exp_number(dep)
            if dep not in id_index or id_index[dep] >= task_index:
                dependency_failures.append({"task_id": task_id, "dependency": dep})
            if dep_exp_number in retired_ids:
                retired_upstream_references.append({"task_id": task_id, "dependency": dep})
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            expression = _gate_expression(task_id, gate) if isinstance(gate, Mapping) else ""
            if expression:
                gate_expressions.append(expression)
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            if not ok:
                gate_failures.append({"task_id": task_id, "gate": gate, "reason": reason})
            if isinstance(gate, Mapping):
                upstream = str(gate.get("upstream") or "")
                field = str(gate.get("artifact_field") or "")
                if field not in required_fields_by_id.get(upstream, set()):
                    gate_cross_ref_failures.append(
                        {"task_id": task_id, "upstream": upstream, "artifact_field": field}
                    )
                upstream_exp_number = exp_number(upstream)
                if upstream_exp_number in retired_ids:
                    retired_upstream_references.append(
                        {"task_id": task_id, "gate_upstream": upstream}
                    )

    prior_failures: list[JsonDict] = []
    prior_entry_count = 0
    for task in tasks:
        task_id = str(task.get("id") or "")
        priors = task.get("prior_failures")
        if not isinstance(priors, list) or not priors:
            prior_failures.append({"task_id": task_id, "reason": "missing_or_empty_prior_failures"})
            continue
        prior_entry_count += len(priors)
        for prior in priors:
            ok, reason = prior_ok(prior)
            if not ok:
                prior_failures.append({"task_id": task_id, "prior": prior, "reason": reason})

    route_failures: list[JsonDict] = []
    model_policy_failures: list[JsonDict] = []
    llm_task_ids: list[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        raw_prompt = str(task.get("prompt") or "")
        rendered_prompt = rendered_prompts.get(task_id, "")
        prompt_for_models = raw_prompt + "\n" + rendered_prompt
        agent_type = str(task.get("agent_type") or "claude")
        model = str(task.get("model") or "")
        if agent_type == "codex" and model != "gpt-5.5":
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        if agent_type == "gemini":
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        if agent_type == "claude" and model not in {"", "sonnet", "opus"}:
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})

        named_models = set(model_specs_named_in_prompt(prompt_for_models)) | set(
            GGUF_ID_RE.findall(prompt_for_models)
        )
        if named_models and not named_models <= MANDATED_GGUF_IDS:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "non_mandated_gguf_id",
                    "ids": sorted(named_models - MANDATED_GGUF_IDS),
                }
            )
        if _is_llm_task(task, rendered_prompt):
            llm_task_ids.append(task_id)
            prompt_lower = rendered_prompt.lower()
            if "MODEL_SPECS" not in rendered_prompt:
                model_policy_failures.append({"task_id": task_id, "reason": "missing_model_specs"})
            if not (MANDATED_GGUF_IDS & named_models):
                model_policy_failures.append(
                    {
                        "task_id": task_id,
                        "reason": "missing_mandated_gguf_id",
                        "expected_any_of": sorted(MANDATED_GGUF_IDS),
                    }
                )
            if "embedded" not in prompt_lower or "tokenizer" not in prompt_lower:
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "missing_embedded_tokenizer_rule"}
                )
            if not (
                "never call autotokenizer" in prompt_lower
                or "do not call autotokenizer" in prompt_lower
                or "no autotokenizer" in prompt_lower
                or re.search(r"\bno\b.{0,80}\bautotokenizer\b", prompt_lower) is not None
            ):
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "missing_no_autotokenizer_rule"}
                )
            if "autotokenizer.from_pretrained" in prompt_lower:
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "forbidden_autotokenizer_from_pretrained"}
                )

    prompt_failures: list[JsonDict] = []
    required_sections = ("CONTEXT", "EXISTING CODE TO READ FIRST", "TASK", "CONCRETE STEPS")
    raw_placeholder_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        raw_prompt = str(task.get("prompt") or "")
        rendered_prompt = rendered_prompts.get(task_id, "")
        if "{project_root}" not in raw_prompt or "{date}" not in raw_prompt:
            raw_placeholder_failures.append({"task_id": task_id, "reason": "missing_placeholder"})
        checks = {
            "format_failed": not render_receipts.get(task_id, {}).get("format_ok", False),
            "missing_context": "CONTEXT" not in rendered_prompt,
            "missing_existing_code": "EXISTING CODE TO READ FIRST" not in rendered_prompt,
            "missing_task": "\n      TASK" not in rendered_prompt and "\nTASK" not in rendered_prompt,
            "missing_concrete_steps": "CONCRETE STEPS" not in rendered_prompt,
            "missing_project_root_literal": root.as_posix() not in rendered_prompt,
            "missing_date_literal": date not in rendered_prompt,
            "missing_run_command": "Run command:" not in rendered_prompt,
            "missing_final_prohibition": not rendered_prompt.strip().endswith(
                FINAL_PROHIBITION_LINE
            ),
            "missing_required_artifact_block": not required_artifact_fields_from_prompt(
                rendered_prompt
            ),
            "missing_required_section": any(
                section not in rendered_prompt for section in required_sections
            ),
        }
        for reason, failed in checks.items():
            if failed:
                prompt_failures.append({"task_id": task_id, "reason": reason})

    schema_errors_linter, prior_errors_linter = validate_prior_failure_roadmap(
        root / ACTIVE_ROADMAP_RELATIVE_PATH,
        root / RESEARCH_COMPLETE_RELATIVE_PATH,
    )
    gate_audit = audit_roadmap(
        root / ACTIVE_ROADMAP_RELATIVE_PATH,
        complete_path=root / RESEARCH_COMPLETE_RELATIVE_PATH,
    ).to_artifact()
    exclusion_risks = exclusion_manifest_lint(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    hard_exclusion_count = sum(1 for risk in exclusion_risks if risk.severity == "HARD")

    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "v549_task_ids": ids,
        "v549_id_and_deliverable_checks": {
            "ok": ids == list(EXPECTED_V549_TASK_IDS)
            and not duplicate_ids
            and not deliverable_failures
            and not duplicate_deliverables
            and exp_numbers == sorted(exp_numbers)
            and None not in exp_numbers,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V549_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V549_TASK_IDS),
            "missing_expected_task_ids": missing_ids,
            "extra_task_ids": extra_ids,
            "duplicate_task_ids": duplicate_ids,
            "unique_deliverables": not duplicate_deliverables,
            "duplicate_deliverables": duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
        },
        "v549_dependency_and_gate_checks": {
            "ok": not dependency_failures and not gate_failures,
            "dependency_failures": dependency_failures,
            "gate_count": len(gate_expressions),
            "gate_failures": gate_failures,
            "structured_gate_expressions": gate_expressions,
        },
        "v549_gate_field_cross_reference_checks": {
            "ok": not gate_cross_ref_failures,
            "failures": gate_cross_ref_failures,
            "checked_gate_count": len(gate_expressions),
        },
        "v549_prior_failure_checks": {
            "ok": not prior_failures and not schema_errors_linter and not prior_errors_linter,
            "prior_entry_count": prior_entry_count,
            "failures": prior_failures,
            "validate_prior_failures": {
                "schema_errors": schema_errors_linter,
                "prior_failure_violations": prior_errors_linter,
            },
            "gate_audit_prior_missing": gate_audit["n_prior_failures_missing"],
            "gate_audit_passed": gate_audit["roadmap_gate_audit_passed"],
        },
        "v549_exclusion_manifest_checks": {
            "ok": hard_exclusion_count == 0 and not retired_upstream_references,
            "hard_risk_count": hard_exclusion_count,
            "risk_count": len(exclusion_risks),
            "risks": _risk_rows(exclusion_risks),
            "retired_upstream_references": retired_upstream_references,
        },
        "v549_agent_model_and_llm_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "route_failures": route_failures,
            "model_policy_failures": model_policy_failures,
            "llm_task_ids": llm_task_ids,
            "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
        },
        "prompt_contract_checks": {
            "ok": not prompt_failures and not raw_placeholder_failures,
            "checked_task_count": len(tasks),
            "rendered_prompt_count": len(rendered_prompts),
            "raw_placeholder_contract_ok": not raw_placeholder_failures,
            "raw_placeholder_failures": raw_placeholder_failures,
            "render_receipts": render_receipts,
            "failures": prompt_failures,
        },
    }


def v548_active_roadmap_receipt(root: Path) -> JsonDict:
    exp6363_payload, _meta = read_json_mapping(
        root / V548_DELIVERABLES_BY_TASK["exp6363-v548-terminal-handoff-and-queue-preflight"]
    )
    upstream = exp6363_payload.get("v548_milestone_doc_and_queue_hashes", {})
    active_v548 = upstream.get("active_roadmap", {}) if isinstance(upstream, Mapping) else {}
    current = read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    return {
        "source": "results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json",
        "path": active_v548.get("path", ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "sha256_at_exp6363": active_v548.get("sha256"),
        "milestone_at_exp6363": active_v548.get("milestone"),
        "current_active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "milestone": current.get("milestone"),
        },
        "same_as_current_active": active_v548.get("sha256")
        == path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "boundary_note": "The V548 active roadmap has been consumed; Exp6363 supplies the V548 active hash.",
    }


def preconditions_checked(root: Path, before_hashes: JsonMap) -> JsonDict:
    return {
        "repo_root": root.as_posix(),
        "date": RUN_DATE,
        "active_roadmap_sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "next_roadmap_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "next_roadmap_sha256": path_sha256(root / ROADMAP_NEXT_RELATIVE_PATH),
        "milestone_doc_sha256": path_sha256(root / MILESTONE_DOC_RELATIVE_PATH),
        "conductor_source_sha256": path_sha256(root / RESEARCH_CONDUCTOR_RELATIVE_PATH),
        "conductor_log_sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "v548_artifact_path_classifications": {
            task_id: classify_artifact_path(root / rel).to_dict()
            for task_id, rel in V548_DELIVERABLES_BY_TASK.items()
        },
        "protected_hashes_before": dict(before_hashes),
        "git_status_before": git_status_lines(root),
    }


def _field_principles(gate_expressions: Sequence[str]) -> JsonDict:
    principles = dict(FIELD_PRINCIPLES)
    for expression in gate_expressions:
        principles[expression] = "This structured gate expression must remain auditable."
    return principles


def _test_rows(command_receipts: Sequence[JsonMap] | None) -> list[JsonDict]:
    if command_receipts:
        return [dict(row) for row in command_receipts]
    return [
        {"command": command, "exit_code": None, "source": "declared"}
        for command in DEFAULT_TEST_COMMANDS
    ]


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None,
    before_hashes: JsonMap,
    duration_s: float,
) -> JsonDict:
    v548_rows, conductor = classify_v548_artifacts(root)
    states = v548_state_receipt(v548_rows)
    queue_data, queue_hashes = load_v549_queue(root)
    queue_checks = validate_v549_queue_data(queue_data, root, date)
    after_hashes = protected_hashes(root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    gate_expressions = queue_checks["v549_dependency_and_gate_checks"][
        "structured_gate_expressions"
    ]
    check_keys = (
        "v549_id_and_deliverable_checks",
        "v549_dependency_and_gate_checks",
        "v549_gate_field_cross_reference_checks",
        "v549_prior_failure_checks",
        "v549_exclusion_manifest_checks",
        "v549_agent_model_and_llm_policy_checks",
        "prompt_contract_checks",
    )
    all_v549_checks_ok = all(bool(queue_checks[key]["ok"]) for key in check_keys)
    status = (
        "complete_v549_queue_preflight_passed"
        if all_v549_checks_ok
        else "blocked_v549_queue_preflight_failed"
    )
    verdict = (
        "complete_v549_queue_preflight_passed: V548 evidence is bounded and the fourteen-task V549 queue validates"
        if all_v549_checks_ok
        else "blocked_v549_queue_preflight_failed: one or more V549 queue checks failed"
    )
    report: JsonDict = {
        "status": status,
        "v548_active_roadmap_path_and_hash": v548_active_roadmap_receipt(root),
        "v548_active_task_ids": list(ACTIVE_V548_TASK_IDS),
        "v548_terminal_artifacts_by_task": v548_rows,
        "v548_conductor_outcomes": conductor,
        "v548_blocked_null_clean_and_proposal_only_states": states,
        "exp6365_runtime_boundary": exp6365_runtime_boundary(root),
        "exp6366_transport_failure_boundary": exp6366_transport_failure_boundary(root),
        "v549_milestone_doc_and_queue_hashes": queue_hashes,
        "v549_task_ids": queue_checks["v549_task_ids"],
        "v549_id_and_deliverable_checks": queue_checks["v549_id_and_deliverable_checks"],
        "v549_dependency_and_gate_checks": queue_checks["v549_dependency_and_gate_checks"],
        "v549_gate_field_cross_reference_checks": queue_checks[
            "v549_gate_field_cross_reference_checks"
        ],
        "v549_prior_failure_checks": queue_checks["v549_prior_failure_checks"],
        "v549_exclusion_manifest_checks": queue_checks["v549_exclusion_manifest_checks"],
        "v549_agent_model_and_llm_policy_checks": queue_checks[
            "v549_agent_model_and_llm_policy_checks"
        ],
        "prompt_contract_checks": queue_checks["prompt_contract_checks"],
        "active_roadmap_modified": before_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix())
        != after_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "conductor_modified": before_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix())
        != after_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix()),
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions_checked(root, before_hashes),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(gate_expressions),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": round(float(duration_s), 12),
        "tests_run": _test_rows(command_receipts),
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
    if isinstance(principles, Mapping):
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field_principles entry: {field}")
        gates = report.get("v549_dependency_and_gate_checks", {})
        expressions = (
            gates.get("structured_gate_expressions", []) if isinstance(gates, Mapping) else []
        )
        for expression in expressions:
            if expression not in principles:
                errors.append(f"missing field_principles entry: {expression}")
    else:
        errors.append("field_principles must be a mapping")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance must be a mapping")
    elif set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("random_seed") is not None:
        errors.append("random_seed must be null")
    states = report.get("v548_blocked_null_clean_and_proposal_only_states", {})
    blocked = states.get("blocked", []) if isinstance(states, Mapping) else []
    null = states.get("null", []) if isinstance(states, Mapping) else []
    proposal = states.get("proposal_only", {}) if isinstance(states, Mapping) else {}
    if "exp6363-v548-terminal-handoff-and-queue-preflight" not in blocked:
        errors.append("Exp6363 blocked state must be preserved")
    if "exp6366-repaired-live-factor-proposal-authenticity" not in null:
        errors.append("Exp6366 null state must be preserved")
    if isinstance(proposal, Mapping) and proposal.get("executed_count") != 0:
        errors.append("proposal-only V548 IDs must not be counted as executed")
    status = str(report.get("status") or "")
    if status.startswith("complete"):
        for key in (
            "v549_id_and_deliverable_checks",
            "v549_dependency_and_gate_checks",
            "v549_gate_field_cross_reference_checks",
            "v549_prior_failure_checks",
            "v549_exclusion_manifest_checks",
            "v549_agent_model_and_llm_policy_checks",
            "prompt_contract_checks",
        ):
            value = report.get(key)
            if not isinstance(value, Mapping) or value.get("ok") is not True:
                errors.append("complete report has failed V549 checks")
                break
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete_", "blocked_", "passed_", "success_", "shipped_")):
        errors.append("honest_verdict lacks terminal prefix")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=True)


def read_external_test_receipts(path: Path = EXTERNAL_TEST_RECEIPT_PATH) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    before = protected_hashes(root)
    receipts = list(command_receipts or read_external_test_receipts())
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
        duration_s=max(time.perf_counter() - started, 0.0001),
    )
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Exp6377 V549 handoff artifact.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    report = run(date=args.date, write=not args.no_write)
    print(
        json.dumps({"path": str(RESULT_RELATIVE_PATH), "status": report["status"]}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
