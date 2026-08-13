"""Exp6363 V548 terminal handoff and queue preflight.

Spec refs: REQ-INFRA-6363, SCENARIO-INFRA-6363-1,
SCENARIO-INFRA-6363-2, SCENARIO-INFRA-6363-3,
SCENARIO-INFRA-6363-4, SCENARIO-INFRA-6363-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6352_live_factor_proposal_authenticity_preflight as exp6352_source
from carnot.experiment_6272_v541_terminal_transition import (
    gate_ok,
    git_status_lines,
    module_name_for_task,
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


MILESTONE_V547 = "2026.08.547"
MILESTONE_V548 = "2026.08.548"
RUN_DATE = "20260813"
EXPERIMENT_ID = "exp6363-v548-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6363.v548_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json"
)
INFERENCE_SUBSTRATE = "deterministic_repository_evidence_handoff_no_llm"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6352_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6352_live_factor_proposal_authenticity_preflight.py"
)

EXPECTED_V547_TASK_IDS = (
    "exp6350-v547-bounded-terminal-handoff",
    "exp6351-v547-post-marker-source-scope-freeze",
    "exp6352-live-factor-proposal-authenticity-preflight",
    "exp6353-live-counterexample-factor-proposal-ab",
    "exp6354-prospective-live-certified-factor-learning",
    "exp6355-default-off-certified-factor-consumer-ab",
    "exp6356-live-certified-learning-safety-audit",
)
PROPOSAL_ONLY_V547_IDS = (
    "exp6357-arc-two-sided-goal-evidence-contract",
    "exp6358-arc-active-reward-machine-discriminator",
    "exp6359-arc-goal-evidence-response-calibration",
    "exp6360-arc-default-off-active-goal-shadow",
    "exp6361-arc-active-goal-provenance-audit",
    "exp6362-v547-adversarial-capstone",
)
V547_DELIVERABLES_BY_TASK = {
    "exp6350-v547-bounded-terminal-handoff": (
        "results/experiment_6350_v547_bounded_terminal_handoff.json"
    ),
    "exp6351-v547-post-marker-source-scope-freeze": (
        "results/experiment_6351_v547_post_marker_source_scope_freeze.json"
    ),
    "exp6352-live-factor-proposal-authenticity-preflight": (
        "results/experiment_6352_live_factor_proposal_authenticity_preflight.json"
    ),
    "exp6353-live-counterexample-factor-proposal-ab": (
        "results/experiment_6353_live_counterexample_factor_proposal_ab.json"
    ),
    "exp6354-prospective-live-certified-factor-learning": (
        "results/experiment_6354_prospective_live_certified_factor_learning.json"
    ),
    "exp6355-default-off-certified-factor-consumer-ab": (
        "results/experiment_6355_default_off_certified_factor_consumer_ab.json"
    ),
    "exp6356-live-certified-learning-safety-audit": (
        "results/experiment_6356_live_certified_learning_safety_audit.json"
    ),
}
V547_TITLE_SNIPPETS = {
    "exp6350-v547-bounded-terminal-handoff": "Bounded V546 terminal evidence handoff into V547",
    "exp6351-v547-post-marker-source-scope-freeze": "V547 dated source window",
    "exp6352-live-factor-proposal-authenticity-preflight": (
        "Three-model live factor proposal authenticity"
    ),
    "exp6353-live-counterexample-factor-proposal-ab": (
        "Gated real counterexample-directed factor proposal"
    ),
    "exp6354-prospective-live-certified-factor-learning": (
        "Gated prospective read-only-then-commit certified"
    ),
    "exp6355-default-off-certified-factor-consumer-ab": (
        "Gated default-off certified factor future consumer"
    ),
    "exp6356-live-certified-learning-safety-audit": (
        "Independent live certified learning and consumer"
    ),
}

EXPECTED_V548_TASK_IDS = (
    "exp6363-v548-terminal-handoff-and-queue-preflight",
    "exp6364-v548-post-marker-source-scope-freeze",
    "exp6365-gguf-child-failure-forensics-runtime-contract",
    "exp6366-repaired-live-factor-proposal-authenticity",
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
ACTIVE_V548_TASK_IDS = EXPECTED_V548_TASK_IDS[:4]
MISSING_ACTIVE_V548_TASK_IDS = EXPECTED_V548_TASK_IDS[4:]
MANDATED_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
GGUF_ID_RE = re.compile(r"[\w.-]+/[\w.-]+-GGUF")

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6363_v548_terminal_handoff_and_queue_preflight --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6363_v548_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6363_v548_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6363_v548_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6363_v548_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6363_v548_terminal_handoff_and_queue_preflight.py"
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
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json"
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
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    FULL_PYTEST_COMMAND,
    RUN_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6363_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    EXP6352_SOURCE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v547_active_roadmap_path_and_hash",
    "v547_active_task_ids",
    "v547_terminal_artifacts_by_task",
    "v547_conductor_outcomes_and_attempt_counts",
    "v547_flagged_blocked_missing_and_null_states",
    "exp6352_generation_failure_receipt",
    "exp6352_source_artifact_drift_receipt",
    "proposal_only_v547_ids_not_executed",
    "v548_milestone_doc_and_queue_hashes",
    "v548_task_ids",
    "v548_id_collision_check",
    "v548_deliverable_checks",
    "v548_dependency_and_structured_gate_checks",
    "v548_gate_field_cross_reference_checks",
    "v548_prior_failure_checks",
    "v548_agent_model_and_llm_policy_checks",
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
    "status": "The status states whether the V548 queue can start cleanly.",
    "v547_active_roadmap_path_and_hash": "The roadmap source and current hash are explicit.",
    "v547_active_task_ids": "Only the seven active V547 tasks define the handoff denominator.",
    "v547_terminal_artifacts_by_task": "Each V547 task keeps its exact artifact state.",
    "v547_conductor_outcomes_and_attempt_counts": "Conductor attempts stay separate from artifact fields.",
    "v547_flagged_blocked_missing_and_null_states": "Failed states are not averaged into clean evidence.",
    "exp6352_generation_failure_receipt": "The live-generation failure is measured without diagnosis.",
    "exp6352_source_artifact_drift_receipt": "Source and artifact sampling drift is visible.",
    "proposal_only_v547_ids_not_executed": "Proposal-only V547 identities are not counted as runs.",
    "v548_milestone_doc_and_queue_hashes": "The V548 proposal, active queue, and optional next queue are hash-pinned.",
    "v548_task_ids": "The audited V548 queue identity is explicit.",
    "v548_id_collision_check": "The queue must contain fourteen unique expected task IDs.",
    "v548_deliverable_checks": "Deliverables must be unique result JSON paths.",
    "v548_dependency_and_structured_gate_checks": "Dependencies and gates must be ordered and structured.",
    "v548_gate_field_cross_reference_checks": "Gate fields must appear in upstream required artifact fields.",
    "v548_prior_failure_checks": "Prior failures must name the changed mechanism.",
    "v548_agent_model_and_llm_policy_checks": "Agent routing and mandatory GGUF policy are checked.",
    "prompt_contract_checks": "Prompt sections, run command, date, root, and final prohibition are checked.",
    "active_roadmap_modified": "The active roadmap must stay byte-identical during the run.",
    "conductor_modified": "The conductor must stay byte-identical during the run.",
    "protected_files_unchanged": "Protected hashes prove no handoff-side rewrite occurred.",
    "preconditions_checked": "Input hashes and artifact classifications are frozen before field reads.",
    "inference_substrate": "This task uses repository evidence with no model call.",
    "verifier_is_oracle": "The handoff reconciles records and is not a scientific oracle.",
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
    "v547_active_roadmap_path_and_hash": "measured",
    "v547_active_task_ids": "constant",
    "v547_terminal_artifacts_by_task": "upstream",
    "v547_conductor_outcomes_and_attempt_counts": "measured",
    "v547_flagged_blocked_missing_and_null_states": "derived",
    "exp6352_generation_failure_receipt": "upstream",
    "exp6352_source_artifact_drift_receipt": "derived",
    "proposal_only_v547_ids_not_executed": "constant",
    "v548_milestone_doc_and_queue_hashes": "measured",
    "v548_task_ids": "upstream",
    "v548_id_collision_check": "derived",
    "v548_deliverable_checks": "derived",
    "v548_dependency_and_structured_gate_checks": "derived",
    "v548_gate_field_cross_reference_checks": "derived",
    "v548_prior_failure_checks": "derived",
    "v548_agent_model_and_llm_policy_checks": "derived",
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
    result = subprocess.run(  # noqa: S603 - fixed local script path and args.
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
    snippet = V547_TITLE_SNIPPETS[task_id].lower()
    rows: list[JsonDict] = []
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if not path.exists():
        return rows
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


def v547_conductor_outcomes(root: Path) -> JsonDict:
    outcomes: JsonDict = {}
    for task_id in EXPECTED_V547_TASK_IDS:
        rows = _conductor_rows(root, task_id)
        counts = Counter(str(row["status"]) for row in rows)
        outcomes[task_id] = {
            **dict(sorted(counts.items())),
            "attempt_count": len(rows),
            "preemptive_skip_count": sum(
                "pre-emptive skip" in str(row["message"]).lower() for row in rows
            ),
            "rows": rows,
        }
    return outcomes


def classify_v547_artifacts(root: Path) -> tuple[JsonDict, JsonDict]:
    outcomes = v547_conductor_outcomes(root)
    rows: JsonDict = {}
    for task_id in EXPECTED_V547_TASK_IDS:
        rel = V547_DELIVERABLES_BY_TASK[task_id]
        summary = _summarize_artifact(root, rel)
        path = root / rel
        classification = classify_artifact_path(path).to_dict()
        payload, meta = read_json_mapping(path)
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
            "summary_receipt": summary,
            "conductor_receipt": outcomes[task_id],
            "stamped_flagged_adversarial": payload.get("flagged_adversarial"),
            "corrigendum_pending": payload.get("corrigendum_pending"),
        }
        rows[task_id] = row
    return rows, outcomes


def v547_state_receipt(rows: JsonMap, outcomes: JsonMap) -> JsonDict:
    flagged: list[str] = []
    blocked: list[str] = []
    missing: list[str] = []
    null: list[str] = []
    retired: list[str] = []
    for task_id, row_any in rows.items():
        row = row_any if isinstance(row_any, Mapping) else {}
        receipt = outcomes.get(task_id, {}) if isinstance(outcomes, Mapping) else {}
        statuses = set(receipt) if isinstance(receipt, Mapping) else set()
        terminal_class = str(row.get("terminal_class") or "")
        status_raw = str(row.get("status_raw") or "")
        verdict = str(row.get("honest_verdict_raw") or "")
        messages = " ".join(str(r.get("message") or "") for r in receipt.get("rows", []))
        if terminal_class == "flagged" or row.get("stamped_flagged_adversarial") is True:
            flagged.append(str(task_id))
        if terminal_class == "missing":
            missing.append(str(task_id))
        if status_raw.startswith("complete_null") or verdict.startswith("complete_null"):
            null.append(str(task_id))
        if "GATE_BLOCK" in statuses and "Pre-emptive skip" in messages:
            retired.append(str(task_id))
        elif "GATE_BLOCK" in statuses or status_raw.startswith("blocked"):
            blocked.append(str(task_id))
    return {
        "flagged": flagged,
        "blocked": blocked,
        "missing": missing,
        "retired_upstream": retired,
        "null": null,
        "counts": {
            "flagged": len(flagged),
            "blocked": len(blocked),
            "missing": len(missing),
            "retired_upstream": len(retired),
            "null": len(null),
        },
    }


def exp6352_generation_failure_receipt(root: Path) -> JsonDict:
    payload, meta = read_json_mapping(
        root / V547_DELIVERABLES_BY_TASK["exp6352-live-factor-proposal-authenticity-preflight"]
    )
    process = payload.get("generation_process_receipts_by_model", {})
    calls = payload.get("generation_call_token_time_and_exit_receipts", {})
    raw = payload.get("raw_model_output_paths_hashes_and_counts", {})
    raw_by_model = raw.get("by_model", {}) if isinstance(raw, Mapping) else {}
    process_rows = process.values() if isinstance(process, Mapping) else []
    call_rows = calls.values() if isinstance(calls, Mapping) else []
    returncodes = [
        row.get("exit_state", {}).get("returncode")
        for row in process_rows
        if isinstance(row, Mapping)
    ]
    token_counts = [row.get("token_counts", {}) for row in call_rows if isinstance(row, Mapping)]
    raw_rows = raw_by_model.values() if isinstance(raw_by_model, Mapping) else []
    stderr_present = "stderr" in payload or any(
        "stderr" in row for row in process_rows if isinstance(row, Mapping)
    )
    return {
        "artifact_path": V547_DELIVERABLES_BY_TASK[
            "exp6352-live-factor-proposal-authenticity-preflight"
        ],
        "artifact_sha256": meta["sha256"],
        "models_used": list(payload.get("models_used") or []),
        "models_used_empty": payload.get("models_used") == [],
        "live_autoregressive_generation_invoked": payload.get(
            "live_autoregressive_generation_invoked"
        ),
        "returncodes_by_model": {
            model: row.get("exit_state", {}).get("returncode")
            for model, row in process.items()
            if isinstance(process, Mapping) and isinstance(row, Mapping)
        },
        "all_generation_children_returned_code_1": bool(returncodes)
        and all(code == 1 for code in returncodes),
        "total_raw_byte_count": sum(
            int(row.get("byte_count") or 0) for row in raw_rows if isinstance(row, Mapping)
        ),
        "total_prompt_tokens": sum(
            int(row.get("prompt_tokens") or 0) for row in token_counts if isinstance(row, Mapping)
        ),
        "total_completion_tokens": sum(
            int(row.get("completion_tokens") or 0)
            for row in token_counts
            if isinstance(row, Mapping)
        ),
        "stderr_preserved_in_artifact": stderr_present,
        "root_cause_inferred": False,
    }


def exp6352_source_artifact_drift_receipt(root: Path) -> JsonDict:
    payload, _meta = read_json_mapping(
        root / V547_DELIVERABLES_BY_TASK["exp6352-live-factor-proposal-authenticity-preflight"]
    )
    process = payload.get("generation_process_receipts_by_model", {})
    rows = process.values() if isinstance(process, Mapping) else []
    artifact_n_ctx = sorted(
        {
            int(row.get("sampling", {}).get("n_ctx"))
            for row in rows
            if isinstance(row, Mapping) and row.get("sampling", {}).get("n_ctx") is not None
        }
    )
    verdict = str(payload.get("honest_verdict") or "").lower()
    live_bool = payload.get("live_autoregressive_generation_invoked")
    return {
        "source_path": EXP6352_SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": path_sha256(root / EXP6352_SOURCE_RELATIVE_PATH),
        "artifact_process_n_ctx_values": artifact_n_ctx,
        "source_sampling_n_ctx": int(exp6352_source.SAMPLING_PARAMETERS["n_ctx"]),
        "n_ctx_mismatch": artifact_n_ctx != [int(exp6352_source.SAMPLING_PARAMETERS["n_ctx"])],
        "top_level_random_seed_present": "random_seed" in payload,
        "random_seeds_field_present": "random_seeds" in payload,
        "prose_vs_boolean_generation_contradiction": "live generation ran" in verdict
        and live_bool is False,
        "root_cause_inferred": False,
    }


def proposal_only_v547_receipt() -> JsonDict:
    return {
        "task_ids": list(PROPOSAL_ONLY_V547_IDS),
        "executed_count": 0,
        "counted_as_executed": False,
        "reason": "These identities were proposal-only V547 scope and did not enter the active queue.",
    }


def load_v548_queue(root: Path) -> tuple[JsonDict, JsonDict]:
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
        "requested_next_roadmap": {
            "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            "present": next_path.exists(),
            "sha256": path_sha256(next_path),
        },
        "active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "present": active_path.exists(),
            "sha256": path_sha256(active_path),
            "milestone": active_data.get("milestone"),
        },
        "audited_queue": {
            "path": chosen.as_posix(),
            "sha256": path_sha256(root / chosen),
            "milestone": data.get("milestone"),
            "selection_note": note,
        },
        "milestone_doc": {
            "path": MILESTONE_DOC_RELATIVE_PATH.as_posix(),
            "present": (root / MILESTONE_DOC_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / MILESTONE_DOC_RELATIVE_PATH),
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


def validate_v548_queue_data(data: JsonMap, root: Path, date: str) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    required_fields_by_id = {
        task_id: required_artifact_fields_from_prompt(str(task.get("prompt") or ""))
        for task_id, task in tasks_by_id.items()
    }
    schema_errors: list[str] = []
    try:
        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - serialized into evidence.
        schema_errors.append(str(exc))

    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    missing_ids = [task_id for task_id in EXPECTED_V548_TASK_IDS if task_id not in ids]
    extra_ids = [task_id for task_id in ids if task_id not in EXPECTED_V548_TASK_IDS]
    duplicate_deliverables = sorted(
        path for path, count in Counter(deliverables).items() if path and count > 1
    )
    deliverable_failures = [
        {"task_id": str(task.get("id") or ""), "deliverable": str(task.get("deliverable") or "")}
        for task in tasks
        if not str(task.get("deliverable") or "").startswith("results/")
        or not str(task.get("deliverable") or "").endswith(".json")
    ]

    dependency_failures: list[JsonDict] = []
    gate_failures: list[JsonDict] = []
    gate_cross_ref_failures: list[JsonDict] = []
    gate_expressions: list[str] = []
    id_index = {task_id: index for index, task_id in enumerate(ids)}
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        requires = task.get("requires")
        for dependency in requires if isinstance(requires, list) else []:
            dep = str(dependency)
            if dep not in id_index or id_index[dep] >= task_index:
                dependency_failures.append({"task_id": task_id, "dependency": dep})
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
    for task in tasks:
        task_id = str(task.get("id") or "")
        agent_type = str(task.get("agent_type") or "claude")
        model = str(task.get("model") or "")
        if agent_type == "codex" and model != "gpt-5.5":
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        if agent_type == "gemini":
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        if agent_type == "claude" and model not in {"sonnet", "opus"}:
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        prompt = str(task.get("prompt") or "")
        named_models = set(model_specs_named_in_prompt(prompt)) | set(GGUF_ID_RE.findall(prompt))
        requires_models = task.get("requires_gpu") is True or "MODEL_SPECS" in prompt
        if requires_models and not MANDATED_GGUF_IDS <= named_models:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "missing_mandated_gguf_ids",
                    "expected": sorted(MANDATED_GGUF_IDS),
                    "found": sorted(named_models),
                }
            )
        if named_models and not named_models <= MANDATED_GGUF_IDS:
            model_policy_failures.append(
                {
                    "task_id": task_id,
                    "reason": "non_mandated_gguf_id",
                    "ids": sorted(named_models - MANDATED_GGUF_IDS),
                }
            )

    prompt_failures: list[JsonDict] = []
    required_sections = ("CONTEXT", "EXISTING CODE TO READ FIRST", "TASK", "CONCRETE STEPS")
    for task in tasks:
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        checks = {
            "missing_context": "CONTEXT" not in prompt,
            "missing_existing_code": "EXISTING CODE TO READ FIRST" not in prompt,
            "missing_task": "\n      TASK" not in prompt and "\nTASK" not in prompt,
            "missing_concrete_steps": "CONCRETE STEPS" not in prompt,
            "missing_project_root_literal": root.as_posix() not in prompt,
            "missing_date_literal": date not in prompt,
            "missing_run_command": "Run command:" not in prompt,
            "missing_final_prohibition": not prompt.strip().endswith(
                "Do NOT push. Do NOT modify scripts/research_conductor.py."
            ),
        }
        checks["missing_required_artifact_block"] = not required_artifact_fields_from_prompt(prompt)
        checks["missing_required_section"] = any(
            section not in prompt for section in required_sections
        )
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
        "v548_task_ids": ids,
        "v548_id_collision_check": {
            "ok": ids == list(EXPECTED_V548_TASK_IDS) and not duplicate_ids,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V548_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V548_TASK_IDS),
            "missing_expected_task_ids": missing_ids,
            "extra_task_ids": extra_ids,
            "duplicate_task_ids": duplicate_ids,
        },
        "v548_deliverable_checks": {
            "ok": not deliverable_failures and not duplicate_deliverables,
            "unique_deliverables": not duplicate_deliverables,
            "deliverable_count": len(deliverables),
            "failures": deliverable_failures,
            "duplicate_deliverables": duplicate_deliverables,
        },
        "v548_dependency_and_structured_gate_checks": {
            "ok": not dependency_failures and not gate_failures,
            "dependency_failures": dependency_failures,
            "gate_count": len(gate_expressions),
            "gate_failures": gate_failures,
            "structured_gate_expressions": gate_expressions,
        },
        "v548_gate_field_cross_reference_checks": {
            "ok": not gate_cross_ref_failures,
            "failures": gate_cross_ref_failures,
            "checked_gate_count": len(gate_expressions),
        },
        "v548_prior_failure_checks": {
            "ok": not prior_failures and not schema_errors_linter and not prior_errors_linter,
            "prior_entry_count": prior_entry_count,
            "failures": prior_failures,
            "validate_prior_failures": {
                "schema_errors": schema_errors_linter,
                "prior_failure_violations": prior_errors_linter,
            },
            "gate_audit_prior_missing": gate_audit["n_prior_failures_missing"],
            "exclusion_manifest_hard_risk_count": hard_exclusion_count,
        },
        "v548_agent_model_and_llm_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "route_failures": route_failures,
            "model_policy_failures": model_policy_failures,
            "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
        },
        "prompt_contract_checks": {
            "ok": not prompt_failures,
            "checked_task_count": len(tasks),
            "failures": prompt_failures,
        },
    }


def v547_active_roadmap_receipt(root: Path) -> JsonDict:
    active = read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    return {
        "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "current_milestone": active.get("milestone"),
        "v547_active_available_in_file": active.get("milestone") == MILESTONE_V547,
        "v547_task_ids_source": "conductor_log_and_declared_v547_artifact_paths",
        "expected_v547_task_count": len(EXPECTED_V547_TASK_IDS),
    }


def preconditions_checked(root: Path, before_hashes: JsonMap) -> JsonDict:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    conductor_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    exclusion_path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    artifact_classes = {
        task_id: classify_artifact_path(root / rel).to_dict()
        for task_id, rel in V547_DELIVERABLES_BY_TASK.items()
    }
    return {
        "repo_root": root.as_posix(),
        "date": RUN_DATE,
        "active_roadmap_sha256": path_sha256(active_path),
        "next_roadmap_present": next_path.exists(),
        "next_roadmap_sha256": path_sha256(next_path),
        "conductor_sha256": path_sha256(conductor_path),
        "exclusion_manifest_loaded": exclusion_path.exists(),
        "exclusion_manifest_sha256": path_sha256(exclusion_path),
        "v547_artifact_path_classifications": artifact_classes,
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
    v547_rows, conductor = classify_v547_artifacts(root)
    states = v547_state_receipt(v547_rows, conductor)
    queue_data, queue_hashes = load_v548_queue(root)
    queue_checks = validate_v548_queue_data(queue_data, root, date)
    after_hashes = protected_hashes(root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    gate_expressions = queue_checks["v548_dependency_and_structured_gate_checks"][
        "structured_gate_expressions"
    ]
    id_ok = queue_checks["v548_id_collision_check"]["ok"]
    status = "complete_v548_queue_preflight_passed" if id_ok else "blocked_v548_queue_incomplete"
    verdict = (
        "complete_v548_queue_preflight_passed: V547 evidence and V548 queue are bounded"
        if id_ok
        else "blocked_v548_queue_incomplete: active V548 queue has 4 of 14 expected tasks"
    )
    report: JsonDict = {
        "status": status,
        "v547_active_roadmap_path_and_hash": v547_active_roadmap_receipt(root),
        "v547_active_task_ids": list(EXPECTED_V547_TASK_IDS),
        "v547_terminal_artifacts_by_task": v547_rows,
        "v547_conductor_outcomes_and_attempt_counts": conductor,
        "v547_flagged_blocked_missing_and_null_states": states,
        "exp6352_generation_failure_receipt": exp6352_generation_failure_receipt(root),
        "exp6352_source_artifact_drift_receipt": exp6352_source_artifact_drift_receipt(root),
        "proposal_only_v547_ids_not_executed": proposal_only_v547_receipt(),
        "v548_milestone_doc_and_queue_hashes": queue_hashes,
        "v548_task_ids": queue_checks["v548_task_ids"],
        "v548_id_collision_check": queue_checks["v548_id_collision_check"],
        "v548_deliverable_checks": queue_checks["v548_deliverable_checks"],
        "v548_dependency_and_structured_gate_checks": queue_checks[
            "v548_dependency_and_structured_gate_checks"
        ],
        "v548_gate_field_cross_reference_checks": queue_checks[
            "v548_gate_field_cross_reference_checks"
        ],
        "v548_prior_failure_checks": queue_checks["v548_prior_failure_checks"],
        "v548_agent_model_and_llm_policy_checks": queue_checks[
            "v548_agent_model_and_llm_policy_checks"
        ],
        "prompt_contract_checks": queue_checks["prompt_contract_checks"],
        "active_roadmap_modified": before_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix())
        != after_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "conductor_modified": before_hashes.get(CONDUCTOR_LOG_RELATIVE_PATH.as_posix())
        != after_hashes.get(CONDUCTOR_LOG_RELATIVE_PATH.as_posix()),
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
        gates = report.get("v548_dependency_and_structured_gate_checks", {})
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
    states = report.get("v547_flagged_blocked_missing_and_null_states", {})
    flagged = states.get("flagged", []) if isinstance(states, Mapping) else []
    if "exp6350-v547-bounded-terminal-handoff" not in flagged:
        errors.append("Exp6350 flagged state must be preserved")
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
    parser = argparse.ArgumentParser(description="Write Exp6363 V548 handoff artifact.")
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
