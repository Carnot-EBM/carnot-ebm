"""Exp6410 V551 evidence handoff and V552 queue preflight.

Spec refs: REQ-INFRA-6410, SCENARIO-INFRA-6410-1,
SCENARIO-INFRA-6410-2, SCENARIO-INFRA-6410-3,
SCENARIO-INFRA-6410-4, SCENARIO-INFRA-6410-5,
SCENARIO-INFRA-6410-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml

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
from carnot.experiment_6404_v551_terminal_handoff_and_queue_preflight import (
    ALLOWED_HONEST_PREFIXES,
    FINAL_PROHIBITION_LINE,
    GGUF_ID_RE,
    MANDATED_GGUF_IDS,
    _gate_expression,
    _live_adversarial,
    _risk_rows,
    _summarize_artifact,
    _tasks,
    read_json_mapping,
    render_prompt,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE_V551 = "2026.08.551"
MILESTONE_V552 = "2026.08.552"
RUN_DATE = "20260814"
EXPERIMENT_ID = "exp6410-v552-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6410.v552_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6410_v552_terminal_handoff_and_queue_preflight.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
SUMMARY_SCRIPT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
SOLVE_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLAIM_LEDGER_RELATIVE_PATHS = (
    Path("ops/claim-eligibility-ledger.json"),
    Path("ops/arc_solve_claims.yaml"),
    Path("results/experiment_6406_clean_v550_factor_evidence_boundary.json.claim_ledger.jsonl"),
)

EXP6408_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py"
)
EXP6409_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py"
)

EXPECTED_V551_TASK_IDS = (
    "exp6404-v551-terminal-handoff-and-queue-preflight",
    "exp6405-v551-post-marker-source-scope-freeze",
    "exp6406-clean-v550-factor-evidence-boundary",
    "exp6407-provenance-tiered-factor-memory-protocol",
    "exp6408-powered-write-time-factor-admission-ab",
    "exp6409-graph-local-multisession-continuous-learning",
)
V551_DELIVERABLES_BY_TASK = {
    "exp6404-v551-terminal-handoff-and-queue-preflight": (
        "results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json"
    ),
    "exp6405-v551-post-marker-source-scope-freeze": (
        "results/experiment_6405_v551_post_marker_source_scope_freeze.json"
    ),
    "exp6406-clean-v550-factor-evidence-boundary": (
        "results/experiment_6406_clean_v550_factor_evidence_boundary.json"
    ),
    "exp6407-provenance-tiered-factor-memory-protocol": (
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
    ),
    "exp6408-powered-write-time-factor-admission-ab": (
        "results/experiment_6408_powered_write_time_factor_admission_ab.json"
    ),
    "exp6409-graph-local-multisession-continuous-learning": (
        "results/experiment_6409_graph_local_multisession_continuous_learning.json"
    ),
}
V551_SIDECARS_BY_TASK = {
    "exp6404-v551-terminal-handoff-and-queue-preflight": (),
    "exp6405-v551-post-marker-source-scope-freeze": (),
    "exp6406-clean-v550-factor-evidence-boundary": (
        "results/experiment_6406_clean_v550_factor_evidence_boundary.json.audit_registration.json",
        "results/experiment_6406_clean_v550_factor_evidence_boundary.json.claim_ledger.jsonl",
    ),
    "exp6407-provenance-tiered-factor-memory-protocol": (
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.compiled_typed_graph.json",
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.compiled_typed_graph_schema.json",
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.contamination_manifest.json",
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.raw_ledger.jsonl",
        "results/experiment_6407_provenance_tiered_factor_memory_protocol.json.raw_record_schema.json",
    ),
    "exp6408-powered-write-time-factor-admission-ab": (
        "results/experiment_6408_powered_write_time_factor_admission_ab.json.fresh_held_manifest.json",
    ),
    "exp6409-graph-local-multisession-continuous-learning": (
        "results/experiment_6409_graph_local_multisession_continuous_learning.json.chronological_manifest.json",
    ),
}
V551_TITLE_SNIPPETS = {
    "exp6404-v551-terminal-handoff-and-queue-preflight": (
        "V550 terminal evidence handoff and V551 queue pref"
    ),
    "exp6405-v551-post-marker-source-scope-freeze": (
        "V551 dated source delta and executable scope freez"
    ),
    "exp6406-clean-v550-factor-evidence-boundary": (
        "Clean V550-only factor evidence boundary and claim"
    ),
    "exp6407-provenance-tiered-factor-memory-protocol": (
        "Provenance-linked typed factor memory and contamin"
    ),
    "exp6408-powered-write-time-factor-admission-ab": (
        "Gated on Exp6406/6407 readiness: fresh powered wri"
    ),
    "exp6409-graph-local-multisession-continuous-learning": (
        "Gated on Exp6408 positive admission: graph-local m"
    ),
}

EXPECTED_V552_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (6410, "v552-terminal-handoff-and-queue-preflight"),
        (6411, "v552-post-marker-source-scope-freeze"),
        (6412, "v551-powered-claim-integrity-audit"),
        (6413, "authenticated-sota-gguf-execution-receipts"),
        (6414, "fresh-three-family-factor-event-corpus"),
        (6415, "boolean-wcsp-ccg-kernelization"),
        (6416, "selective-exact-refinement-ab"),
        (6417, "authentic-write-time-factor-admission-ab"),
        (6418, "execution-grounded-dual-path-csl"),
        (6419, "held-shift-restart-csl-replication"),
        (6420, "csl-authenticity-safety-audit"),
        (6421, "arc-opt-in-executed-policy-ab"),
        (6422, "arc-held-family-policy-safety-audit"),
        (6423, "v552-adversarial-capstone"),
    )
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6410_v552_terminal_handoff_and_queue_preflight "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6410_v552_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6410_v552_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6410_v552_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6410_v552_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6410_v552_terminal_handoff_and_queue_preflight.py"
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
    "results/experiment_6410_v552_terminal_handoff_and_queue_preflight.json"
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
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6410_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    SUMMARY_SCRIPT_RELATIVE_PATH,
    SOLVE_REGISTRY_RELATIVE_PATH,
    *CLAIM_LEDGER_RELATIVE_PATHS,
    *[Path(path) for path in V551_DELIVERABLES_BY_TASK.values()],
    *[Path(path) for paths in V551_SIDECARS_BY_TASK.values() for path in paths],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v551_active_roadmap_path_and_hash",
    "v551_task_ids",
    "v551_terminal_artifacts_and_sidecars_by_task",
    "v551_artifact_verdicts",
    "v551_source_execution_findings",
    "v551_conductor_outcomes",
    "v551_adversarial_findings",
    "v551_scientific_claim_eligibility_by_task",
    "exp6407_6408_6409_claim_correction",
    "v552_milestone_doc_and_queue_hashes",
    "v552_task_ids",
    "v552_id_and_deliverable_checks",
    "v552_dependency_and_gate_checks",
    "v552_gate_field_cross_reference_checks",
    "v552_prior_failure_checks",
    "v552_exclusion_manifest_checks",
    "v552_agent_model_and_llm_policy_checks",
    "v552_arc_no_solve_checks",
    "prompt_contract_checks",
    "active_roadmap_modified",
    "conductor_modified",
    "solve_registry_modified",
    "claims_ledger_modified",
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
    "status": "The status states whether the V552 queue passed or failed closed.",
    "v551_active_roadmap_path_and_hash": "The upstream V551 active roadmap receipt anchors the terminal denominator.",
    "v551_task_ids": "The six completed V551 task IDs define the handoff scope.",
    "v551_terminal_artifacts_and_sidecars_by_task": "Artifacts and sidecars are hash-pinned before conclusions are read.",
    "v551_artifact_verdicts": "Historical honest verdicts stay intact and separate from later audits.",
    "v551_source_execution_findings": "Source inspection identifies receipt limits without rewriting results.",
    "v551_conductor_outcomes": "Conductor facts remain distinct from artifact verdicts.",
    "v551_adversarial_findings": "Stamped and current verifier outcomes remain visible.",
    "v551_scientific_claim_eligibility_by_task": "Eligibility records which claims can be used and which need audit.",
    "exp6407_6408_6409_claim_correction": "The correction preserves history while blocking over-claim reuse.",
    "v552_milestone_doc_and_queue_hashes": "V552 planning sources and queue files are hash-pinned.",
    "v552_task_ids": "The audited V552 queue identity is explicit.",
    "v552_id_and_deliverable_checks": "The queue must contain fourteen unique ordered IDs and result JSON deliverables.",
    "v552_dependency_and_gate_checks": "Dependencies and structured gates must be ordered and valid.",
    "v552_gate_field_cross_reference_checks": "Gate fields must appear in upstream required artifact fields.",
    "v552_prior_failure_checks": "Prior failures must name verdicts, changed mechanisms, and retirement rules.",
    "v552_exclusion_manifest_checks": "Retired task reuse and retired upstream chains fail before execution.",
    "v552_agent_model_and_llm_policy_checks": "Agent routes and local GGUF policy are checked before live work.",
    "v552_arc_no_solve_checks": "ARC tasks must stay on the live policy path without solve credit.",
    "prompt_contract_checks": "Rendered prompts must contain the operational contract the agent receives.",
    "active_roadmap_modified": "The active roadmap must stay byte-identical during this run.",
    "conductor_modified": "The conductor source must stay byte-identical during this run.",
    "solve_registry_modified": "The ARC solve registry must not change during a handoff.",
    "claims_ledger_modified": "Claim ledgers must not change during a handoff.",
    "protected_files_unchanged": "Protected hashes prove no handoff-side rewrite occurred.",
    "preconditions_checked": "Input hashes and classifications are frozen before field reads.",
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
    "v551_active_roadmap_path_and_hash": "upstream",
    "v551_task_ids": "constant",
    "v551_terminal_artifacts_and_sidecars_by_task": "measured",
    "v551_artifact_verdicts": "upstream",
    "v551_source_execution_findings": "measured",
    "v551_conductor_outcomes": "measured",
    "v551_adversarial_findings": "measured",
    "v551_scientific_claim_eligibility_by_task": "derived",
    "exp6407_6408_6409_claim_correction": "derived",
    "v552_milestone_doc_and_queue_hashes": "measured",
    "v552_task_ids": "upstream",
    "v552_id_and_deliverable_checks": "derived",
    "v552_dependency_and_gate_checks": "derived",
    "v552_gate_field_cross_reference_checks": "derived",
    "v552_prior_failure_checks": "derived",
    "v552_exclusion_manifest_checks": "derived",
    "v552_agent_model_and_llm_policy_checks": "derived",
    "v552_arc_no_solve_checks": "derived",
    "prompt_contract_checks": "derived",
    "active_roadmap_modified": "measured",
    "conductor_modified": "measured",
    "solve_registry_modified": "measured",
    "claims_ledger_modified": "measured",
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


def path_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def _protected_files_unchanged(before: JsonMap, after: JsonMap) -> JsonDict:
    rows = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"ok": all(row["unchanged"] for row in rows.values()), "rows": rows}


def _terminal_class(payload: JsonMap, meta: JsonMap, live: JsonMap) -> str:
    status = str(payload.get("status") or "").lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if meta.get("error") == "missing":
        return "missing"
    if meta.get("error"):
        return "malformed"
    if payload.get("flagged_adversarial") is True or live.get("critical_count", 0) > 0:
        return "flagged"
    if "blocked" in status or "blocked" in verdict:
        return "blocked"
    if "no_scope_change" in status or "no_scope_change" in verdict:
        return "null"
    if "unproven" in status or "unproven" in verdict:
        return "unproven"
    if "positive" in status or "positive" in verdict:
        return "positive"
    if status.startswith("complete") or verdict.startswith("complete"):
        return "complete"
    return "unknown"


def _load_v551_inputs(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict], JsonDict]:
    payloads: dict[str, JsonDict] = {}
    metas: dict[str, JsonDict] = {}
    summaries: JsonDict = {}
    for task_id, artifact in V551_DELIVERABLES_BY_TASK.items():
        rel = Path(artifact)
        summaries[task_id] = _summarize_artifact(root, rel)
        payload, meta = read_json_mapping(root / rel)
        payloads[task_id] = payload
        metas[task_id] = meta
    return payloads, metas, summaries


def _conductor_log_rows(root: Path, task_id: str) -> list[JsonDict]:
    snippet = V551_TITLE_SNIPPETS.get(task_id, "").lower()
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if not snippet or not path.exists():
        return []
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if snippet not in line.lower():
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
    return rows


def _research_complete_result(root: Path, task_id: str) -> str | None:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    milestones = data.get("milestones", []) if isinstance(data, Mapping) else data
    for milestone in milestones if isinstance(milestones, list) else []:
        if not isinstance(milestone, Mapping) or milestone.get("id") != MILESTONE_V551:
            continue
        for task in milestone.get("tasks", []):
            if isinstance(task, Mapping) and task.get("id") == task_id:
                result = task.get("result")
                return str(result) if result is not None else None
    return None


def _v551_conductor_outcomes(root: Path) -> JsonDict:
    outcomes: JsonDict = {}
    for task_id in EXPECTED_V551_TASK_IDS:
        rows = _conductor_log_rows(root, task_id)
        counts = Counter(str(row["status"]) for row in rows)
        outcomes[task_id] = {
            "log_status_counts": dict(sorted(counts.items())),
            "log_attempt_count": len(rows),
            "log_rows": rows,
            "research_complete_result": _research_complete_result(root, task_id),
        }
    return outcomes


def _v551_adversarial_findings(
    root: Path,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    summaries: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task_id, artifact in V551_DELIVERABLES_BY_TASK.items():
        rel = Path(artifact)
        live = _live_adversarial(root, rel, metas[task_id].get("error") is None)
        rows[task_id] = {
            "path": rel.as_posix(),
            "stamped_flagged_adversarial": payloads[task_id].get("flagged_adversarial"),
            "stamped_corrigendum_pending": payloads[task_id].get("corrigendum_pending"),
            "current_live_verdict": live["verdict"],
            "current_live_flag_count": live["flag_count"],
            "current_live_has_critical": live["critical_count"] > 0,
            "current_live_flags": live["flags"],
            "summary_receipt": summaries.get(task_id),
        }
    return rows


def _v551_terminal_artifacts_and_sidecars_by_task(
    root: Path,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    adversarial: JsonMap,
    conductor: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    class_counts: Counter[str] = Counter()
    sidecar_counts: JsonDict = {}
    for task_id, artifact in V551_DELIVERABLES_BY_TASK.items():
        live = {
            "critical_count": 1
            if adversarial.get(task_id, {}).get("current_live_has_critical") is True
            else 0
        }
        terminal_class = _terminal_class(payloads[task_id], metas[task_id], live)
        class_counts[terminal_class] += 1
        sidecars = [path_receipt(root / Path(path)) for path in V551_SIDECARS_BY_TASK[task_id]]
        sidecar_counts[task_id] = sum(1 for row in sidecars if row["present"])
        rows[task_id] = {
            "task_id": task_id,
            "declared_deliverable": artifact,
            "artifact_receipt": metas[task_id],
            "sidecars": sidecars,
            "terminal_class": terminal_class,
            "artifact_status_raw": payloads[task_id].get("status"),
            "artifact_honest_verdict_raw": payloads[task_id].get("honest_verdict"),
            "conductor_receipt": conductor.get(task_id),
        }
    for name in ("complete", "blocked", "flagged", "positive", "unproven", "missing", "null"):
        class_counts.setdefault(name, 0)
    rows["terminal_class_counts"] = dict(sorted(class_counts.items()))
    rows["sidecar_counts_by_task"] = sidecar_counts
    return rows


def _v551_artifact_verdicts(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {task_id: payloads[task_id].get("honest_verdict") for task_id in EXPECTED_V551_TASK_IDS}


def _line_hits(source: str, patterns: Sequence[str]) -> list[JsonDict]:
    hits: list[JsonDict] = []
    for number, line in enumerate(source.splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            hits.append({"line": number, "text": line.strip()})
    return hits


def _v551_source_execution_findings(root: Path) -> JsonDict:
    source6408 = (root / EXP6408_SOURCE_RELATIVE_PATH).read_text(encoding="utf-8")
    source6409 = (root / EXP6409_SOURCE_RELATIVE_PATH).read_text(encoding="utf-8")
    invoked_patterns = ("llama_cpp.Llama(", ".create_completion(", ".generate(", "subprocess.run(")
    exp6408_invoked = any(pattern in source6408 for pattern in invoked_patterns)
    exp6409_invoked = any(pattern in source6409 for pattern in invoked_patterns)
    return {
        "exp6404-v551-terminal-handoff-and-queue-preflight": {
            "implementation_path": "python/carnot/experiment_6404_v551_terminal_handoff_and_queue_preflight.py",
            "source_inspection_required": False,
            "source_behavior_class": "handoff_aggregation",
        },
        "exp6405-v551-post-marker-source-scope-freeze": {
            "implementation_path": "python/carnot/experiment_6405_v551_post_marker_source_scope_freeze.py",
            "source_inspection_required": False,
            "source_behavior_class": "source_scope_freeze",
        },
        "exp6406-clean-v550-factor-evidence-boundary": {
            "implementation_path": "python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py",
            "source_inspection_required": False,
            "source_behavior_class": "aggregation_boundary",
        },
        "exp6407-provenance-tiered-factor-memory-protocol": {
            "implementation_path": "python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py",
            "source_inspection_required": False,
            "source_behavior_class": "deterministic_protocol_flagged_by_artifact_verifier",
        },
        "exp6408-powered-write-time-factor-admission-ab": {
            "implementation_path": EXP6408_SOURCE_RELATIVE_PATH.as_posix(),
            "source_sha256": path_sha256(root / EXP6408_SOURCE_RELATIVE_PATH),
            "declared_gguf_generator_invoked": exp6408_invoked,
            "authenticated_generation_receipt_present": False,
            "runtime_receipt_kind": "derived_from_host_snapshot_and_constants",
            "constant_model_duration_values_s": [0.25, 0.3, 0.35],
            "duration_constant_lines": _line_hits(source6408, ["0.25 + 0.05 * index"]),
            "derived_peak_memory_formula_present": "used_mb + 1024 + 128 * index" in source6408,
            "derived_peak_memory_lines": _line_hits(source6408, ["used_mb + 1024 + 128 * index"]),
            "absence_of_generation_process_patterns": {
                pattern: pattern not in source6408 for pattern in invoked_patterns
            },
        },
        "exp6409-graph-local-multisession-continuous-learning": {
            "implementation_path": EXP6409_SOURCE_RELATIVE_PATH.as_posix(),
            "source_sha256": path_sha256(root / EXP6409_SOURCE_RELATIVE_PATH),
            "declared_gguf_generator_invoked": exp6409_invoked,
            "inherits_exp6408_runtime_surface": (
                "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model" in source6409
                and "upstream.get" in source6409
            ),
            "inheritance_lines": _line_hits(
                source6409,
                [
                    "experiment_6408_powered_write_time_factor_admission_ab",
                    'upstream.get("cuda_offload_runtime_peak_memory_and_duration_receipts_by_model")',
                    "fallback_runtime_receipt",
                ],
            ),
            "prospective_csl_receipt_surface": "inherited_deterministic_replay",
            "absence_of_generation_process_patterns": {
                pattern: pattern not in source6409 for pattern in invoked_patterns
            },
        },
    }


def _v551_scientific_claim_eligibility(
    payloads: Mapping[str, JsonDict],
    adversarial: JsonMap,
    source_findings: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task_id in EXPECTED_V551_TASK_IDS:
        payload = payloads[task_id]
        live = adversarial.get(task_id, {})
        rows[task_id] = {
            "artifact_status": payload.get("status"),
            "artifact_positive_preserved": "positive" in str(payload.get("status") or ""),
            "open_adversarial_flag": live.get("current_live_has_critical") is True
            or payload.get("flagged_adversarial") is True,
            "public_claim_eligibility": payload.get("public_factor_claim_eligibility")
            if "public_factor_claim_eligibility" in payload
            else False,
            "powered_gguf_claim_eligibility": False,
            "prospective_csl_claim_eligibility": False,
            "deterministic_replay_claim_eligibility": False,
            "eligibility_note": "not_a_scientific_positive",
        }
    rows["exp6406-clean-v550-factor-evidence-boundary"].update(
        {
            "narrow_internal_factor_boundary_eligibility": True,
            "eligibility_note": "clean_boundary_only_public_claim_false",
        }
    )
    rows["exp6407-provenance-tiered-factor-memory-protocol"].update(
        {
            "deterministic_protocol_claim_eligibility": False,
            "eligibility_note": "protocol_preserved_but_current_adversarial_flag_open",
        }
    )
    rows["exp6408-powered-write-time-factor-admission-ab"].update(
        {
            "deterministic_replay_claim_eligibility": True,
            "powered_gguf_claim_eligibility": False,
            "powered_gguf_blocker": source_findings[
                "exp6408-powered-write-time-factor-admission-ab"
            ]["runtime_receipt_kind"],
            "eligibility_note": "deterministic_replay_preserved_powered_claim_requires_audit",
        }
    )
    rows["exp6409-graph-local-multisession-continuous-learning"].update(
        {
            "deterministic_replay_claim_eligibility": True,
            "prospective_csl_claim_eligibility": False,
            "prospective_csl_blocker": source_findings[
                "exp6409-graph-local-multisession-continuous-learning"
            ]["prospective_csl_receipt_surface"],
            "eligibility_note": "deterministic_replay_preserved_prospective_csl_requires_audit",
        }
    )
    return rows


def _claim_correction(verdicts: JsonMap, eligibility: JsonMap) -> JsonDict:
    return {
        "historical_artifact_verdicts_preserved": all(
            verdicts.get(task_id) is not None for task_id in EXPECTED_V551_TASK_IDS
        ),
        "preserved_historical_verdicts": {
            task_id: verdicts.get(task_id)
            for task_id in (
                "exp6407-provenance-tiered-factor-memory-protocol",
                "exp6408-powered-write-time-factor-admission-ab",
                "exp6409-graph-local-multisession-continuous-learning",
            )
        },
        "exp6407_open_adversarial_flag_preserved": eligibility[
            "exp6407-provenance-tiered-factor-memory-protocol"
        ]["open_adversarial_flag"]
        is True,
        "exp6408_powered_positive_requires_audit": eligibility[
            "exp6408-powered-write-time-factor-admission-ab"
        ]["powered_gguf_claim_eligibility"]
        is False,
        "exp6409_prospective_csl_positive_requires_audit": eligibility[
            "exp6409-graph-local-multisession-continuous-learning"
        ]["prospective_csl_claim_eligibility"]
        is False,
        "historical_artifacts_modified": False,
        "historical_determinations_overwritten": False,
        "claim_boundary": (
            "Exp6408 powered and Exp6409 prospective positives are preserved as "
            "historical artifacts and treated as audit-required hypotheses."
        ),
    }


def _proposal_exp_numbers(root: Path) -> list[int]:
    path = root / MILESTONE_DOC_RELATIVE_PATH
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [int(match.group(1)) for match in re.finditer(r"\|\s*(64\d{2})\s*\|", text)]


def _source_v551_active_roadmap(payloads: Mapping[str, JsonDict], root: Path) -> JsonDict:
    exp6404 = payloads["exp6404-v551-terminal-handoff-and-queue-preflight"]
    milestone_hashes = exp6404.get("v551_milestone_doc_and_queue_hashes", {})
    audited = (
        milestone_hashes.get("audited_queue") if isinstance(milestone_hashes, Mapping) else None
    )
    return {
        "source_artifact": V551_DELIVERABLES_BY_TASK[
            "exp6404-v551-terminal-handoff-and-queue-preflight"
        ],
        "recorded_v551_active_queue": audited,
        "current_active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "present": (root / ACTIVE_ROADMAP_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "milestone": read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH).get("milestone"),
        },
        "current_active_has_advanced_to_v552": (
            read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH).get("milestone")
            == MILESTONE_V552
        ),
    }


def load_v552_queue(root: Path) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_data = read_yaml_mapping(active_path)
    next_data = read_yaml_mapping(next_path)
    if next_data.get("milestone") == MILESTONE_V552:
        data = next_data
        chosen = ROADMAP_NEXT_RELATIVE_PATH
        note = "research-roadmap-next.yaml contains V552 and was audited"
    else:
        data = active_data
        chosen = ACTIVE_ROADMAP_RELATIVE_PATH
        note = "active research-roadmap.yaml contains V552 and was audited"
    proposal_numbers = _proposal_exp_numbers(root)
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
            "milestone": next_data.get("milestone"),
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
            "proposal_exp_numbers": proposal_numbers,
            "proposal_task_count": len(proposal_numbers),
        },
        "conductor_source": path_receipt(root / RESEARCH_CONDUCTOR_RELATIVE_PATH),
        "conductor_log": path_receipt(root / CONDUCTOR_LOG_RELATIVE_PATH),
        "exclusion_manifest": path_receipt(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "known_issues": path_receipt(root / KNOWN_ISSUES_RELATIVE_PATH),
        "north_star": path_receipt(root / NORTH_STAR_RELATIVE_PATH),
        "solve_registry": path_receipt(root / SOLVE_REGISTRY_RELATIVE_PATH),
        "claim_ledgers": [path_receipt(root / path) for path in CLAIM_LEDGER_RELATIVE_PATHS],
    }
    return dict(data), identity


def _required_fields_by_task(
    tasks_by_id: Mapping[str, JsonDict], root: Path, date: str
) -> tuple[dict[str, str], dict[str, JsonDict], dict[str, set[str]]]:
    rendered_prompts: dict[str, str] = {}
    render_receipts: dict[str, JsonDict] = {}
    required_fields: dict[str, set[str]] = {}
    for task_id, task in tasks_by_id.items():
        rendered, receipt = render_prompt(str(task.get("prompt") or ""), root, date)
        rendered_prompts[task_id] = rendered
        render_receipts[task_id] = receipt
        required_fields[task_id] = required_artifact_fields_from_prompt(rendered)
    return rendered_prompts, render_receipts, required_fields


def _local_gguf_policy_failures(task: JsonMap, rendered_prompt: str) -> list[JsonDict]:
    task_id = str(task.get("id") or "")
    prompt_lower = rendered_prompt.lower()
    named_models = set(model_specs_named_in_prompt(rendered_prompt)) | set(
        GGUF_ID_RE.findall(rendered_prompt)
    )
    failures: list[JsonDict] = []
    if "MODEL_SPECS" not in rendered_prompt:
        failures.append({"task_id": task_id, "reason": "missing_model_specs"})
    if "cached_sota_pair()" not in rendered_prompt:
        failures.append({"task_id": task_id, "reason": "missing_cached_sota_pair"})
    if not (MANDATED_GGUF_IDS & named_models):
        failures.append(
            {
                "task_id": task_id,
                "reason": "missing_mandated_gguf_id",
                "expected_any_of": sorted(MANDATED_GGUF_IDS),
            }
        )
    if named_models and not named_models <= MANDATED_GGUF_IDS:
        failures.append(
            {
                "task_id": task_id,
                "reason": "non_mandated_gguf_id",
                "ids": sorted(named_models - MANDATED_GGUF_IDS),
            }
        )
    if "embedded" not in prompt_lower or "tokenizer" not in prompt_lower:
        failures.append({"task_id": task_id, "reason": "missing_embedded_tokenizer_rule"})
    if not re.search(r"\b(no|never|do not)\b.{0,80}\bautotokenizer\b", prompt_lower):
        failures.append({"task_id": task_id, "reason": "missing_no_autotokenizer_rule"})
    if "autotokenizer.from_pretrained" in prompt_lower:
        failures.append({"task_id": task_id, "reason": "forbidden_autotokenizer_from_pretrained"})
    if "authentic" not in prompt_lower or "receipt" not in prompt_lower:
        failures.append({"task_id": task_id, "reason": "missing_authentic_execution_receipts"})
    legacy_safe = (
        not named_models - MANDATED_GGUF_IDS
        or "legacy" in prompt_lower
        and ("cannot satisfy" in prompt_lower or "do not use" in prompt_lower)
    )
    if not legacy_safe:
        failures.append({"task_id": task_id, "reason": "missing_no_legacy_headline_model_rule"})
    return failures


def _arc_no_solve_checks(
    tasks: Sequence[JsonDict], rendered_prompts: Mapping[str, str]
) -> JsonDict:
    arc_task_ids = [
        str(task.get("id") or "") for task in tasks if "-arc-" in str(task.get("id") or "")
    ]
    solve_failures: list[JsonDict] = []
    registry_failures: list[JsonDict] = []
    canonical_failures: list[JsonDict] = []
    offpath_failures: list[JsonDict] = []
    for task_id in arc_task_ids:
        prompt = rendered_prompts.get(task_id, "")
        lower = prompt.lower()
        no_solve_ok = any(
            phrase in lower
            for phrase in (
                "claim no game or level solve",
                "no game or level solve claim",
                "not a game or level solve task",
                "makes no game or level solve claim",
                "no solve or registry",
            )
        )
        registry_ok = any(
            phrase in lower
            for phrase in (
                "must not update the solve registry",
                "never updates the solve registry",
                "no registry mutation",
                "no solve or registry",
            )
        )
        canonical_ok = "canonical live agent" in lower or "arc_competition_agent" in lower
        offpath_bad = "outer-loop solver" in lower or (
            "per-game adapter" in lower and "do not" not in lower and "attack" not in lower
        )
        if not no_solve_ok or re.search(r"\bclaim a level solve\b", lower):
            solve_failures.append({"task_id": task_id, "reason": "solve_claim_not_forbidden"})
        if not registry_ok:
            registry_failures.append(
                {"task_id": task_id, "reason": "registry_update_not_forbidden"}
            )
        if not canonical_ok:
            canonical_failures.append(
                {"task_id": task_id, "reason": "missing_canonical_live_agent"}
            )
        if offpath_bad:
            offpath_failures.append({"task_id": task_id, "reason": "outer_loop_or_adapter_path"})
    return {
        "ok": not solve_failures
        and not registry_failures
        and not canonical_failures
        and not offpath_failures,
        "arc_task_ids": arc_task_ids,
        "solve_claim_failures": solve_failures,
        "solve_registry_update_failures": registry_failures,
        "canonical_live_agent_failures": canonical_failures,
        "outer_loop_or_adapter_failures": offpath_failures,
    }


def validate_v552_queue_data(
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
    rendered_prompts, render_receipts, required_fields_by_id = _required_fields_by_task(
        tasks_by_id, root, date
    )
    schema_errors: list[str] = []
    try:
        from scripts.roadmap_schema import Roadmap

        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        schema_errors.append(str(exc))

    exp_numbers = [exp_number(task_id) for task_id in ids]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
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
    retired_references: list[JsonDict] = []
    retired_task_ids: list[str] = []
    gate_expressions: list[str] = []
    gate_field_type_failures: list[JsonDict] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        if exp_number(task_id) in retired_ids:
            retired_task_ids.append(task_id)
        requires = task.get("requires")
        for dependency in requires if isinstance(requires, list) else []:
            dep = str(dependency)
            if dep not in id_index or id_index[dep] >= task_index:
                dependency_failures.append({"task_id": task_id, "dependency": dep})
            if exp_number(dep) in retired_ids:
                retired_references.append({"task_id": task_id, "dependency": dep})
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            if isinstance(gate, Mapping):
                expression = _gate_expression(task_id, gate)
                gate_expressions.append(expression)
                if not isinstance(gate.get("artifact_field"), str) or not gate.get(
                    "artifact_field"
                ):
                    gate_field_type_failures.append({"task_id": task_id, "gate": dict(gate)})
                ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
                if not ok:
                    gate_failures.append({"task_id": task_id, "gate": dict(gate), "reason": reason})
                upstream = str(gate.get("upstream") or "")
                field = str(gate.get("artifact_field") or "")
                if field not in required_fields_by_id.get(upstream, set()):
                    gate_cross_ref_failures.append(
                        {"task_id": task_id, "upstream": upstream, "artifact_field": field}
                    )
                if exp_number(upstream) in retired_ids:
                    retired_references.append({"task_id": task_id, "gate_upstream": upstream})
            else:
                gate_failures.append(
                    {"task_id": task_id, "gate": gate, "reason": "gate_not_mapping"}
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
    schema_errors_linter, prior_errors_linter = __import__(
        "scripts.validate_prior_failures", fromlist=["validate_roadmap"]
    ).validate_roadmap(root / ACTIVE_ROADMAP_RELATIVE_PATH, root / RESEARCH_COMPLETE_RELATIVE_PATH)
    gate_audit = (
        __import__("scripts.audit_roadmap_gates", fromlist=["audit_roadmap"])
        .audit_roadmap(
            root / ACTIVE_ROADMAP_RELATIVE_PATH,
            complete_path=root / RESEARCH_COMPLETE_RELATIVE_PATH,
        )
        .to_artifact()
    )
    exclusion_risks = __import__("scripts.exclusion_manifest_lint", fromlist=["lint"]).lint(
        root / ACTIVE_ROADMAP_RELATIVE_PATH
    )
    hard_exclusion_count = sum(1 for risk in exclusion_risks if risk.severity == "HARD")

    route_failures: list[JsonDict] = []
    model_policy_failures: list[JsonDict] = []
    local_gguf_task_ids: list[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        agent_type = str(task.get("agent_type") or "")
        model = str(task.get("model") or "")
        if (agent_type, model) not in {
            ("claude", "opus"),
            ("claude", "sonnet"),
            ("codex", "gpt-5.5"),
        }:
            route_failures.append({"task_id": task_id, "agent_type": agent_type, "model": model})
        if task.get("requires_gpu") is True:
            local_gguf_task_ids.append(task_id)
            model_policy_failures.extend(
                _local_gguf_policy_failures(task, rendered_prompts.get(task_id, ""))
            )

    prompt_failures: list[JsonDict] = []
    raw_placeholder_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        raw_prompt = str(task.get("prompt") or "")
        rendered = rendered_prompts.get(task_id, "")
        if "{project_root}" not in raw_prompt or "{date}" not in raw_prompt:
            raw_placeholder_failures.append({"task_id": task_id, "reason": "missing_placeholder"})
        checks = {
            "format_failed": not render_receipts.get(task_id, {}).get("format_ok", False),
            "missing_context": "CONTEXT" not in rendered,
            "missing_existing_code": "EXISTING CODE TO READ FIRST" not in rendered,
            "missing_task": "\nTASK" not in rendered and "\n      TASK" not in rendered,
            "missing_concrete_steps": "CONCRETE STEPS" not in rendered,
            "missing_project_root_literal": root.as_posix() not in rendered,
            "missing_date_literal": date not in rendered,
            "missing_run_command": "Run command:" not in rendered,
            "missing_final_prohibition": not rendered.strip().endswith(FINAL_PROHIBITION_LINE),
            "missing_required_artifact_block": not required_artifact_fields_from_prompt(rendered),
        }
        for reason, failed in checks.items():
            if failed:
                prompt_failures.append({"task_id": task_id, "reason": reason})

    arc_checks = _arc_no_solve_checks(tasks, rendered_prompts)
    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "v552_task_ids": ids,
        "v552_id_and_deliverable_checks": {
            "ok": ids == list(EXPECTED_V552_TASK_IDS)
            and not duplicate_ids
            and not deliverable_failures
            and not duplicate_deliverables
            and exp_numbers == sorted(exp_numbers)
            and None not in exp_numbers
            and not retired_task_ids,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V552_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V552_TASK_IDS),
            "missing_expected_task_ids": [
                task_id for task_id in EXPECTED_V552_TASK_IDS if task_id not in ids
            ],
            "extra_task_ids": [task_id for task_id in ids if task_id not in EXPECTED_V552_TASK_IDS],
            "duplicate_task_ids": duplicate_ids,
            "unique_deliverables": not duplicate_deliverables,
            "duplicate_deliverables": duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
            "retired_task_ids": retired_task_ids,
        },
        "v552_dependency_and_gate_checks": {
            "ok": not dependency_failures and not gate_failures and not gate_field_type_failures,
            "dependency_failures": dependency_failures,
            "gate_count": len(gate_expressions),
            "gate_failures": gate_failures,
            "gate_field_type_failures": gate_field_type_failures,
            "retired_references": retired_references,
            "structured_gate_expressions": gate_expressions,
        },
        "v552_gate_field_cross_reference_checks": {
            "ok": not gate_cross_ref_failures,
            "failures": gate_cross_ref_failures,
            "checked_gate_count": len(gate_expressions),
        },
        "v552_prior_failure_checks": {
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
        "v552_exclusion_manifest_checks": {
            "ok": not retired_task_ids and not retired_references and hard_exclusion_count == 0,
            "retired_task_ids": retired_task_ids,
            "retired_references": retired_references,
            "hard_exclusion_count": hard_exclusion_count,
            "risk_count": len(exclusion_risks),
            "risks": _risk_rows(exclusion_risks),
        },
        "v552_agent_model_and_llm_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "allowed_route_matrix": {
                "claude": ["opus", "sonnet"],
                "codex": ["gpt-5.5"],
            },
            "route_failures": route_failures,
            "local_gguf_task_ids": local_gguf_task_ids,
            "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
            "model_policy_failures": model_policy_failures,
        },
        "v552_arc_no_solve_checks": arc_checks,
        "prompt_contract_checks": {
            "ok": not prompt_failures and not raw_placeholder_failures,
            "checked_task_count": len(tasks),
            "render_receipts": render_receipts,
            "raw_placeholder_contract_ok": not raw_placeholder_failures,
            "raw_placeholder_failures": raw_placeholder_failures,
            "failures": prompt_failures,
        },
    }


def _test_rows(command_receipts: Sequence[JsonMap] | None) -> list[JsonDict]:
    if command_receipts:
        return [dict(row) for row in command_receipts if isinstance(row, Mapping)]
    return [
        {"source": "declared", "command": command, "exit_code": None}
        for command in DEFAULT_TEST_COMMANDS
    ]


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


def build_report(
    root: Path,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None,
    before_hashes: JsonMap,
    duration_s: float,
) -> JsonDict:
    payloads, metas, summaries = _load_v551_inputs(root)
    conductor = _v551_conductor_outcomes(root)
    adversarial = _v551_adversarial_findings(root, payloads, metas, summaries)
    terminals = _v551_terminal_artifacts_and_sidecars_by_task(
        root, payloads, metas, adversarial, conductor
    )
    source_findings = _v551_source_execution_findings(root)
    eligibility = _v551_scientific_claim_eligibility(payloads, adversarial, source_findings)
    verdicts = _v551_artifact_verdicts(payloads)
    v552_data, v552_identity = load_v552_queue(root)
    queue_checks = validate_v552_queue_data(v552_data, root, date)
    after_hashes = protected_hashes(root)
    protected = _protected_files_unchanged(before_hashes, after_hashes)
    required_check_fields = (
        "v552_id_and_deliverable_checks",
        "v552_dependency_and_gate_checks",
        "v552_gate_field_cross_reference_checks",
        "v552_prior_failure_checks",
        "v552_exclusion_manifest_checks",
        "v552_agent_model_and_llm_policy_checks",
        "v552_arc_no_solve_checks",
        "prompt_contract_checks",
    )
    all_checks_ok = all(queue_checks[field]["ok"] is True for field in required_check_fields)
    status = (
        "complete_v552_queue_preflight_passed"
        if all_checks_ok
        else "complete_blocked_v552_queue_incomplete"
        if queue_checks["v552_id_and_deliverable_checks"]["task_count"]
        != queue_checks["v552_id_and_deliverable_checks"]["expected_task_count"]
        else "complete_blocked_v552_queue_preflight_failed"
    )
    honest_verdict = {
        "complete_v552_queue_preflight_passed": (
            "complete_v552_queue_preflight_passed: V551 evidence is bounded and "
            "the fourteen-task V552 queue validates"
        ),
        "complete_blocked_v552_queue_incomplete": (
            "complete_blocked_v552_queue_incomplete: V552 queue is incomplete; "
            "V551 evidence preserved without roadmap or conductor edit"
        ),
        "complete_blocked_v552_queue_preflight_failed": (
            "complete_blocked_v552_queue_preflight_failed: V552 queue contract "
            "failed; V551 evidence preserved without roadmap or conductor edit"
        ),
    }[status]
    principles = dict(FIELD_PRINCIPLES)
    for expression in queue_checks["v552_dependency_and_gate_checks"][
        "structured_gate_expressions"
    ]:
        principles[expression] = (
            "This structured V552 gate expression must stay auditable before execution."
        )
    claim_after = {
        path.as_posix(): after_hashes.get(path.as_posix()) for path in CLAIM_LEDGER_RELATIVE_PATHS
    }
    claim_before = {
        path.as_posix(): before_hashes.get(path.as_posix()) for path in CLAIM_LEDGER_RELATIVE_PATHS
    }
    report: JsonDict = {
        "status": status,
        "v551_active_roadmap_path_and_hash": _source_v551_active_roadmap(payloads, root),
        "v551_task_ids": list(EXPECTED_V551_TASK_IDS),
        "v551_terminal_artifacts_and_sidecars_by_task": terminals,
        "v551_artifact_verdicts": verdicts,
        "v551_source_execution_findings": source_findings,
        "v551_conductor_outcomes": conductor,
        "v551_adversarial_findings": adversarial,
        "v551_scientific_claim_eligibility_by_task": eligibility,
        "exp6407_6408_6409_claim_correction": _claim_correction(verdicts, eligibility),
        "v552_milestone_doc_and_queue_hashes": v552_identity,
        "v552_task_ids": queue_checks["v552_task_ids"],
        "v552_id_and_deliverable_checks": queue_checks["v552_id_and_deliverable_checks"],
        "v552_dependency_and_gate_checks": queue_checks["v552_dependency_and_gate_checks"],
        "v552_gate_field_cross_reference_checks": queue_checks[
            "v552_gate_field_cross_reference_checks"
        ],
        "v552_prior_failure_checks": queue_checks["v552_prior_failure_checks"],
        "v552_exclusion_manifest_checks": queue_checks["v552_exclusion_manifest_checks"],
        "v552_agent_model_and_llm_policy_checks": queue_checks[
            "v552_agent_model_and_llm_policy_checks"
        ],
        "v552_arc_no_solve_checks": queue_checks["v552_arc_no_solve_checks"],
        "prompt_contract_checks": queue_checks["prompt_contract_checks"],
        "active_roadmap_modified": before_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix())
        != after_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "conductor_modified": before_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix())
        != after_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix()),
        "solve_registry_modified": before_hashes.get(SOLVE_REGISTRY_RELATIVE_PATH.as_posix())
        != after_hashes.get(SOLVE_REGISTRY_RELATIVE_PATH.as_posix()),
        "claims_ledger_modified": claim_before != claim_after,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "date": date,
            "repo_root": root.as_posix(),
            "git_status_before": git_status_lines(root),
            "before_hashes": dict(before_hashes),
            "after_hashes": after_hashes,
            "expected_v551_artifact_count": len(EXPECTED_V551_TASK_IDS),
            "expected_v552_task_count": len(EXPECTED_V552_TASK_IDS),
            "active_v552_task_count": queue_checks["v552_id_and_deliverable_checks"]["task_count"],
            "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
            "summary_receipts": summaries,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": principles,
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": duration_s,
        "tests_run": _test_rows(command_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("random_seed") is not None:
        errors.append("random_seed must be null")
    correction = report.get("exp6407_6408_6409_claim_correction")
    if not isinstance(correction, Mapping):
        errors.append("exp6407_6408_6409_claim_correction must be a mapping")
    else:
        if correction.get("exp6407_open_adversarial_flag_preserved") is not True:
            errors.append("Exp6407 open flag must be preserved")
        if correction.get("exp6408_powered_positive_requires_audit") is not True:
            errors.append("Exp6408 powered correction must require audit")
        if correction.get("exp6409_prospective_csl_positive_requires_audit") is not True:
            errors.append("Exp6409 prospective CSL correction must require audit")
    eligibility = report.get("v551_scientific_claim_eligibility_by_task")
    if isinstance(eligibility, Mapping):
        exp6408 = eligibility.get("exp6408-powered-write-time-factor-admission-ab", {})
        exp6409 = eligibility.get("exp6409-graph-local-multisession-continuous-learning", {})
        if (
            isinstance(exp6408, Mapping)
            and exp6408.get("powered_gguf_claim_eligibility") is not False
        ):
            errors.append("Exp6408 powered eligibility must be false")
        if (
            isinstance(exp6409, Mapping)
            and exp6409.get("prospective_csl_claim_eligibility") is not False
        ):
            errors.append("Exp6409 prospective CSL eligibility must be false")
    else:
        errors.append("v551_scientific_claim_eligibility_by_task must be a mapping")
    for field, message in (
        ("active_roadmap_modified", "active roadmap changed"),
        ("conductor_modified", "conductor changed"),
        ("solve_registry_modified", "solve registry changed"),
        ("claims_ledger_modified", "claims ledger changed"),
    ):
        if report.get(field) is not False:
            errors.append(message)
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("ok") is not True:
        errors.append("protected files changed")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        required_principles = set(REQUIRED_ARTIFACT_FIELDS)
        gates = report.get("v552_dependency_and_gate_checks", {})
        if isinstance(gates, Mapping):
            required_principles.update(gates.get("structured_gate_expressions", []))
        for field in sorted(required_principles):
            if field not in principles:
                errors.append(f"missing field_principles entry: {field}")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance must be a mapping")
    else:
        if set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
            errors.append("field_provenance must cover exactly required fields")
        if not set(provenance.values()) <= {"measured", "derived", "constant", "upstream"}:
            errors.append("field_provenance has invalid classification")
    if report.get("status") == "complete_v552_queue_preflight_passed":
        check_fields = [
            "v552_id_and_deliverable_checks",
            "v552_dependency_and_gate_checks",
            "v552_gate_field_cross_reference_checks",
            "v552_prior_failure_checks",
            "v552_exclusion_manifest_checks",
            "v552_agent_model_and_llm_policy_checks",
            "v552_arc_no_solve_checks",
            "prompt_contract_checks",
        ]
        if any(
            not isinstance(report.get(field), Mapping) or report[field].get("ok") is not True
            for field in check_fields
        ):
            errors.append("passed report has failed V552 checks")
    honest = str(report.get("honest_verdict") or "")
    if not honest.startswith(ALLOWED_HONEST_PREFIXES):
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
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env)


def run(
    *,
    date: str = RUN_DATE,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    before_hashes = protected_hashes(root)
    if command_receipts is None:
        command_receipts = read_external_test_receipts()
    report = build_report(
        root,
        date=date,
        command_receipts=command_receipts,
        before_hashes=before_hashes,
        duration_s=time.perf_counter() - start,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "status": report["status"],
                "honest_verdict": report.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
