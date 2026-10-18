"""Exp6404 V551 terminal handoff and queue preflight.

Spec refs: REQ-INFRA-6404, SCENARIO-INFRA-6404-1,
SCENARIO-INFRA-6404-2, SCENARIO-INFRA-6404-3,
SCENARIO-INFRA-6404-4, SCENARIO-INFRA-6404-5,
SCENARIO-INFRA-6404-6.
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
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adversarial_verify import verify_artifact  # noqa: E402
from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_manifest_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402
from validate_prior_failures import validate_roadmap as validate_prior_failure_roadmap  # noqa: E402


MILESTONE_V551 = "2026.08.551"
RUN_DATE = "20260813"
EXPERIMENT_ID = "exp6404-v551-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6404.v551_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
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
SUMMARY_SCRIPT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
SOLVE_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLAIMS_LEDGER_RELATIVE_PATH = Path("ops/arc_solve_claims.yaml")

EXPECTED_V550_TASK_IDS = (
    "exp6391-v550-terminal-handoff-and-queue-preflight",
    "exp6392-v550-post-marker-source-scope-freeze",
    "exp6393-arc-scalar-gate-metric-contract",
    "exp6394-model-family-factor-harness-freeze",
    "exp6395-held-factor-transport-license-matrix",
    "exp6396-capability-qualified-verified-frontier-ab",
    "exp6397-transactional-continuous-factor-learning",
    "exp6398-default-off-transactional-factor-consumer",
    "exp6399-capability-learning-safety-audit",
    "exp6400-arc-default-off-active-goal-shadow",
    "exp6401-arc-active-goal-causal-holdout",
    "exp6402-arc-active-goal-safety-audit",
    "exp6403-v550-adversarial-capstone",
)
V550_DELIVERABLES_BY_TASK = {
    "exp6391-v550-terminal-handoff-and-queue-preflight": (
        "results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json"
    ),
    "exp6392-v550-post-marker-source-scope-freeze": (
        "results/experiment_6392_v550_post_marker_source_scope_freeze.json"
    ),
    "exp6393-arc-scalar-gate-metric-contract": (
        "results/experiment_6393_arc_scalar_gate_metric_contract.json"
    ),
    "exp6394-model-family-factor-harness-freeze": (
        "results/experiment_6394_model_family_factor_harness_freeze.json"
    ),
    "exp6395-held-factor-transport-license-matrix": (
        "results/experiment_6395_held_factor_transport_license_matrix.json"
    ),
    "exp6396-capability-qualified-verified-frontier-ab": (
        "results/experiment_6396_capability_qualified_verified_frontier_ab.json"
    ),
    "exp6397-transactional-continuous-factor-learning": (
        "results/experiment_6397_transactional_continuous_factor_learning.json"
    ),
    "exp6398-default-off-transactional-factor-consumer": (
        "results/experiment_6398_default_off_transactional_factor_consumer.json"
    ),
    "exp6399-capability-learning-safety-audit": (
        "results/experiment_6399_capability_learning_safety_audit.json"
    ),
    "exp6400-arc-default-off-active-goal-shadow": (
        "results/experiment_6400_arc_default_off_active_goal_shadow.json"
    ),
    "exp6401-arc-active-goal-causal-holdout": (
        "results/experiment_6401_arc_active_goal_causal_holdout.json"
    ),
    "exp6402-arc-active-goal-safety-audit": (
        "results/experiment_6402_arc_active_goal_safety_audit.json"
    ),
    "exp6403-v550-adversarial-capstone": (
        "results/experiment_6403_v550_adversarial_capstone.json"
    ),
}
V550_TITLE_SNIPPETS = {
    "exp6391-v550-terminal-handoff-and-queue-preflight": (
        "V549 terminal evidence handoff and V550 queue pref"
    ),
    "exp6392-v550-post-marker-source-scope-freeze": (
        "V550 dated source delta and executable scope freez"
    ),
    "exp6393-arc-scalar-gate-metric-contract": (
        "ARC scalar gate-metric producer contract and V549"
    ),
    "exp6394-model-family-factor-harness-freeze": (
        "Model-family factor harness development and held-s"
    ),
    "exp6395-held-factor-transport-license-matrix": (
        "Gated on Exp6394 freeze: held factor-transport cap"
    ),
    "exp6396-capability-qualified-verified-frontier-ab": (
        "Gated on Exp6395 licenses: capability-qualified ve"
    ),
    "exp6397-transactional-continuous-factor-learning": (
        "Gated on Exp6396 positive delta: transactional con"
    ),
    "exp6398-default-off-transactional-factor-consumer": (
        "Gated on Exp6397 readiness: default-off transactio"
    ),
    "exp6399-capability-learning-safety-audit": (
        "Independent capability-license transaction and con"
    ),
    "exp6400-arc-default-off-active-goal-shadow": (
        "Gated on Exp6393 scalar improvement: default-off A"
    ),
    "exp6401-arc-active-goal-causal-holdout": (
        "Gated on Exp6400 reachability: held ARC active-goa"
    ),
    "exp6402-arc-active-goal-safety-audit": (
        "Independent ARC active-goal provenance and causal"
    ),
    "exp6403-v550-adversarial-capstone": (
        "V550 adversarial capstone and PRD-gap reconciliati"
    ),
}

EXPECTED_V551_TASK_IDS = (
    "exp6404-v551-terminal-handoff-and-queue-preflight",
    "exp6405-v551-post-marker-source-scope-freeze",
    "exp6406-clean-v550-factor-evidence-boundary",
    "exp6407-provenance-tiered-factor-memory-protocol",
    "exp6408-powered-write-time-factor-admission-ab",
    "exp6409-graph-local-multisession-continuous-learning",
    "exp6410-default-off-tiered-consumer-across-restarts",
    "exp6411-independent-factor-memory-safety-audit",
    "exp6412-fresh-opt-in-active-goal-executed-policy-ab",
    "exp6413-held-game-family-policy-replication",
    "exp6414-independent-arc-executed-policy-safety-audit",
    "exp6415-v551-adversarial-capstone-and-prd-gap-reconciliation",
)
MANDATED_GGUF_IDS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
)
QWEN_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_MODEL_IDS = (
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
GGUF_ID_RE = re.compile(r"[\w.-]+/[\w.-]+-GGUF")
FINAL_PROHIBITION_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
ALLOWED_HONEST_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6404_v551_terminal_handoff_and_queue_preflight "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6404_v551_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6404_v551_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6404_v551_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6404_v551_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6404_v551_terminal_handoff_and_queue_preflight.py"
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
    "results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json"
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
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6404_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    SUMMARY_SCRIPT_RELATIVE_PATH,
    SOLVE_REGISTRY_RELATIVE_PATH,
    CLAIMS_LEDGER_RELATIVE_PATH,
    *[Path(path) for path in V550_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v550_active_roadmap_path_and_hash",
    "v550_task_ids",
    "v550_terminal_artifacts_by_task",
    "v550_artifact_verdicts",
    "v550_conductor_outcomes",
    "v550_adversarial_findings",
    "v550_duration_receipts_by_task",
    "v550_factor_boundary",
    "v550_arc_boundary",
    "v551_milestone_doc_and_queue_hashes",
    "v551_task_ids",
    "v551_id_and_deliverable_checks",
    "v551_dependency_and_gate_checks",
    "v551_gate_field_cross_reference_checks",
    "v551_prior_failure_checks",
    "v551_exclusion_manifest_checks",
    "v551_agent_model_and_llm_policy_checks",
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
    "status": "The status states whether V551 preflight passed or failed closed.",
    "v550_active_roadmap_path_and_hash": "The V550 active roadmap hash anchors the prior milestone denominator.",
    "v550_task_ids": "The fixed V550 task denominator prevents missing terminal evidence.",
    "v550_terminal_artifacts_by_task": "Each V550 task keeps its exact terminal class.",
    "v550_artifact_verdicts": "Artifact honest verdicts stay separate from conductor and verifier facts.",
    "v550_conductor_outcomes": "Conductor outcomes stay separate from artifact verdicts.",
    "v550_adversarial_findings": "Live and summarized adversarial findings stay visible.",
    "v550_duration_receipts_by_task": "Each task duration comes from its own artifact receipt.",
    "v550_factor_boundary": "The factor boundary preserves narrow licenses and public ineligibility.",
    "v550_arc_boundary": "The ARC boundary preserves internal progress without solve or promotion.",
    "v551_milestone_doc_and_queue_hashes": "V551 planning and queue sources are hash-pinned.",
    "v551_task_ids": "The audited V551 queue identity is explicit.",
    "v551_id_and_deliverable_checks": "The queue must contain twelve unique ordered IDs and result JSON deliverables.",
    "v551_dependency_and_gate_checks": "Dependencies and structured gates must be ordered and valid.",
    "v551_gate_field_cross_reference_checks": "Gate fields must appear in upstream required artifact fields.",
    "v551_prior_failure_checks": "Prior failures must name the old verdict, changed mechanism, and retirement rule.",
    "v551_exclusion_manifest_checks": "Retired task reuse and retired upstream chains fail before execution.",
    "v551_agent_model_and_llm_policy_checks": "Agent routing and local GGUF policy are checked before live work.",
    "prompt_contract_checks": "Rendered prompts must contain the operational contract the agent receives.",
    "active_roadmap_modified": "The active roadmap must stay byte-identical during this run.",
    "conductor_modified": "The conductor source must stay byte-identical during this run.",
    "solve_registry_modified": "The ARC solve registry must not change during a handoff.",
    "claims_ledger_modified": "The ARC claims ledger must not change during a handoff.",
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
    "v550_active_roadmap_path_and_hash": "upstream",
    "v550_task_ids": "constant",
    "v550_terminal_artifacts_by_task": "derived",
    "v550_artifact_verdicts": "upstream",
    "v550_conductor_outcomes": "measured",
    "v550_adversarial_findings": "measured",
    "v550_duration_receipts_by_task": "upstream",
    "v550_factor_boundary": "derived",
    "v550_arc_boundary": "derived",
    "v551_milestone_doc_and_queue_hashes": "measured",
    "v551_task_ids": "upstream",
    "v551_id_and_deliverable_checks": "derived",
    "v551_dependency_and_gate_checks": "derived",
    "v551_gate_field_cross_reference_checks": "derived",
    "v551_prior_failure_checks": "derived",
    "v551_exclusion_manifest_checks": "derived",
    "v551_agent_model_and_llm_policy_checks": "derived",
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


def _summarize_artifact(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    if not path.exists():
        return {
            "invoked_before_field_import": False,
            "reason": "artifact_absent",
            "exit_code": None,
            "live_adversarial_findings": [],
        }
    command = [sys.executable, SUMMARY_SCRIPT_RELATIVE_PATH.as_posix(), rel_path.as_posix()]
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


def _live_adversarial(root: Path, rel_path: Path, present: bool) -> JsonDict:
    if not present:
        return {"flag_count": 0, "critical_count": 0, "flags": [], "verdict": "absent"}
    report = verify_artifact(root / rel_path)
    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    severities = Counter(str(flag.get("severity") or "") for flag in flags)
    return {
        "flag_count": len(flags),
        "critical_count": severities.get("critical", 0),
        "flags": flags,
        "verdict": "critical" if severities.get("critical", 0) else ("warn" if flags else "clean"),
    }


def _terminal_class(payload: JsonMap, meta: JsonMap, live: JsonMap) -> str:
    status = str(payload.get("status") or "").lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if meta.get("error") == "missing":
        return "absent"
    if meta.get("error"):
        return "malformed"
    if live.get("critical_count", 0) > 0:
        return "flagged"
    if "retired" in status or "retired" in verdict:
        return "retired"
    if status.startswith("skipped") or verdict.startswith("skipped"):
        return "skipped"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if status.startswith("complete_partial") or verdict.startswith("complete_partial"):
        return "partial"
    if status.startswith("complete_null") or verdict.startswith("complete_null"):
        return "null"
    if status.startswith("complete_positive") or verdict.startswith("complete_positive"):
        return "positive"
    if status.startswith("complete") or verdict.startswith("complete"):
        return "clean"
    return "unknown"


def _task_number(task_id: str) -> int | None:
    match = re.match(r"exp(\d+)-", task_id)
    return int(match.group(1)) if match else None


def _proposal_exp_numbers(root: Path) -> list[int]:
    path = root / MILESTONE_DOC_RELATIVE_PATH
    if not path.exists():
        return []
    return [int(match.group(1)) for match in re.finditer(r"^### Exp(\d+)\b", path.read_text(encoding="utf-8"), re.MULTILINE)]


def _load_v550_inputs(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict], JsonDict]:
    payloads: dict[str, JsonDict] = {}
    metas: dict[str, JsonDict] = {}
    summaries: JsonDict = {}
    for task_id in EXPECTED_V550_TASK_IDS:
        rel = Path(V550_DELIVERABLES_BY_TASK[task_id])
        summaries[task_id] = _summarize_artifact(root, rel)
        payload, meta = read_json_mapping(root / rel)
        payloads[task_id] = payload
        metas[task_id] = meta
    return payloads, metas, summaries


def _conductor_rows(root: Path, task_id: str) -> list[JsonDict]:
    snippet = V550_TITLE_SNIPPETS.get(task_id, "").lower()
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if not snippet or not path.exists():
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


def _v550_conductor_outcomes(root: Path) -> JsonDict:
    outcomes: JsonDict = {}
    for task_id in EXPECTED_V550_TASK_IDS:
        rows = _conductor_rows(root, task_id)
        counts = Counter(str(row["status"]) for row in rows)
        outcomes[task_id] = {**dict(sorted(counts.items())), "attempt_count": len(rows), "rows": rows}
    return outcomes


def _v550_adversarial_findings(
    root: Path,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    summaries: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task_id in EXPECTED_V550_TASK_IDS:
        rel = Path(V550_DELIVERABLES_BY_TASK[task_id])
        payload = payloads[task_id]
        live = _live_adversarial(root, rel, metas[task_id].get("error") is None)
        rows[task_id] = {
            "path": rel.as_posix(),
            "present": metas[task_id].get("present"),
            "live_verdict": live["verdict"],
            "live_flag_count": live["flag_count"],
            "live_has_critical": live["critical_count"] > 0,
            "live_flags": live["flags"],
            "summary_receipt": summaries.get(task_id),
            "public_claim_eligible": (
                payload.get("public_factor_claim_eligibility")
                if "public_factor_claim_eligibility" in payload
                else payload.get("public_claim_eligibility")
            ),
            "public_arc_claim_eligible": payload.get("public_arc_claim_eligibility"),
            "flagged_adversarial": payload.get("flagged_adversarial"),
            "corrigendum_pending": payload.get("corrigendum_pending"),
        }
    return rows


def _terminal_artifacts_by_task(
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    findings: JsonMap,
    conductor: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    class_counts: Counter[str] = Counter()
    for task_id in EXPECTED_V550_TASK_IDS:
        live = {
            "critical_count": 1
            if findings.get(task_id, {}).get("live_has_critical") is True
            else 0
        }
        rel = Path(V550_DELIVERABLES_BY_TASK[task_id])
        terminal_class = _terminal_class(payloads[task_id], metas[task_id], live)
        class_counts[terminal_class] += 1
        rows[task_id] = {
            "task_id": task_id,
            "declared_deliverable": rel.as_posix(),
            "present": metas[task_id].get("present"),
            "loadable": metas[task_id].get("loadable"),
            "sha256": metas[task_id].get("sha256"),
            "terminal_class": terminal_class,
            "status_raw": payloads[task_id].get("status"),
            "honest_verdict_raw": payloads[task_id].get("honest_verdict"),
            "flagged_adversarial": payloads[task_id].get("flagged_adversarial"),
            "corrigendum_pending": payloads[task_id].get("corrigendum_pending"),
            "conductor_receipt": conductor.get(task_id),
        }
    for name in ("clean", "partial", "null", "blocked", "skipped", "absent", "flagged", "retired"):
        class_counts.setdefault(name, 0)
    rows["terminal_class_counts"] = dict(sorted(class_counts.items()))
    return rows


def _duration_receipt(task_id: str, artifact_path: str, payload: JsonMap, meta: JsonMap) -> JsonDict:
    value = payload.get("duration_s")
    source = "artifact.duration_s"
    if meta.get("error") == "missing":
        source = "artifact_absent"
        value = None
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        source = "duration_missing_or_non_numeric"
        value = None
    return {"task_id": task_id, "duration_s": value, "source": source, "artifact_path": artifact_path}


def _duration_receipts(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    return {
        task_id: _duration_receipt(task_id, V550_DELIVERABLES_BY_TASK[task_id], payloads[task_id], metas[task_id])
        for task_id in EXPECTED_V550_TASK_IDS
    }


def _artifact_verdicts(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {task_id: payloads[task_id].get("honest_verdict") for task_id in EXPECTED_V550_TASK_IDS}


def _v550_factor_boundary(payloads: Mapping[str, JsonDict]) -> JsonDict:
    exp6395 = payloads["exp6395-held-factor-transport-license-matrix"]
    exp6396 = payloads["exp6396-capability-qualified-verified-frontier-ab"]
    exp6397 = payloads["exp6397-transactional-continuous-factor-learning"]
    exp6398 = payloads["exp6398-default-off-transactional-factor-consumer"]
    exp6399 = payloads["exp6399-capability-learning-safety-audit"]
    licenses = [
        {
            "model_id": row.get("model_hf_id"),
            "constraint_family": row.get("constraint_family"),
        }
        for row in exp6395.get("capability_license_records", [])
        if isinstance(row, Mapping)
    ]
    rejected_abstained = [
        row
        for row in exp6395.get("rejected_and_abstained_cell_records", [])
        if isinstance(row, Mapping)
    ]
    qwen_abstentions = [
        row
        for row in rejected_abstained
        if row.get("model_hf_id") == QWEN_MODEL_ID and row.get("terminal_disposition") == "abstained"
    ]
    rejected_gemma = [
        row
        for row in rejected_abstained
        if row.get("model_hf_id") in GEMMA_MODEL_IDS and row.get("terminal_disposition") == "rejected"
    ]
    return {
        "exp6395_licensed_cell_count": len(licenses),
        "licensed_cells": licenses,
        "exp6395_qwen_abstention_count": len(qwen_abstentions),
        "qwen_abstention_cells": [
            {"model_id": row.get("model_hf_id"), "constraint_family": row.get("constraint_family")}
            for row in qwen_abstentions
        ],
        "exp6395_rejected_gemma_cell_count": len(rejected_gemma),
        "rejected_gemma_cells": [
            {"model_id": row.get("model_hf_id"), "constraint_family": row.get("constraint_family")}
            for row in rejected_gemma
        ],
        "universal_support_claimed": bool(exp6395.get("universal_support_claimed")),
        "positive_internal_factor_results": {
            "exp6396_status": exp6396.get("status"),
            "exp6396_delta_verified_future_exact_yield": exp6396.get(
                "delta_verified_future_exact_yield"
            ),
            "exp6396_ready_score": exp6396.get("capability_qualified_frontier_ready_score"),
            "exp6397_status": exp6397.get("status"),
            "exp6397_delta_future_exact_yield": exp6397.get(
                "delta_future_exact_yield_over_frozen"
            ),
            "exp6397_ready_score": exp6397.get(
                "transactional_continuous_self_learning_ready_score"
            ),
            "exp6398_status": exp6398.get("status"),
            "exp6398_delta_exact_yield": exp6398.get("delta_exact_yield_over_frozen"),
            "exp6398_ready_score": exp6398.get("default_off_transactional_consumer_ready_score"),
        },
        "exp6399_public_block": {
            "status": exp6399.get("status"),
            "honest_verdict": exp6399.get("honest_verdict"),
            "public_factor_claim_eligibility": exp6399.get("public_factor_claim_eligibility"),
            "claim_boundary_includes_flagged_exp6385": True,
        },
        "public_factor_utility_promotion_count": exp6399.get("utility_promotion_count"),
        "no_universal_support": exp6395.get("universal_support_claimed") is False,
    }


def _v550_arc_boundary(payloads: Mapping[str, JsonDict]) -> JsonDict:
    exp6400 = payloads["exp6400-arc-default-off-active-goal-shadow"]
    exp6401 = payloads["exp6401-arc-active-goal-causal-holdout"]
    exp6402 = payloads["exp6402-arc-active-goal-safety-audit"]
    return {
        "exp6400_shadow_readiness": {
            "ready_score": exp6400.get("arc_active_goal_shadow_ready_score"),
            "active_shadow_treatment_fired_count": exp6400.get("active_shadow_treatment_fired_count"),
            "solve_claim_count": exp6400.get("solve_claim_count"),
        },
        "exp6401_causal_progress": {
            "ready_score": exp6401.get("arc_active_goal_causal_ready_score"),
            "delta_exact_progress_proxy": exp6401.get("delta_exact_progress_proxy"),
            "delta_false_accept_count": exp6401.get("delta_false_accept_count"),
            "treatment_fired_count": (
                exp6401.get("treatment_fired_counts", {}) or {}
            ).get("active_disagreement"),
        },
        "exp6401_internal_route_eligible": bool(exp6401.get("route_promotion_eligible")),
        "exp6402_clean_provenance_audit": exp6402.get("status") == "complete",
        "exp6402_public_arc_eligibility": exp6402.get("public_arc_claim_eligibility"),
        "actual_route_promotion_count": exp6402.get("route_promotion_count"),
        "solve_claim_count": sum(
            int(payloads[task_id].get("solve_claim_count") or 0)
            for task_id in (
                "exp6400-arc-default-off-active-goal-shadow",
                "exp6401-arc-active-goal-causal-holdout",
                "exp6402-arc-active-goal-safety-audit",
            )
        ),
        "solve_registry_modified": any(
            bool(payloads[task_id].get("solve_registry_modified"))
            for task_id in (
                "exp6400-arc-default-off-active-goal-shadow",
                "exp6401-arc-active-goal-causal-holdout",
                "exp6402-arc-active-goal-safety-audit",
            )
        ),
        "no_solve": True,
        "zero_route_promotion": exp6402.get("route_promotion_count") == 0,
    }


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


def _risk_rows(risks: Sequence[Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for risk in risks:
        try:
            rows.append(asdict(risk))
        except TypeError:
            rows.append({"repr": repr(risk)})
    return rows


def render_prompt(raw_prompt: str, root: Path, date: str) -> tuple[str, JsonDict]:
    try:
        rendered = raw_prompt.format(project_root=root.as_posix(), date=date)
        return rendered, {"format_ok": True, "error": None}
    except (KeyError, IndexError, ValueError) as exc:
        rendered = raw_prompt.replace("{project_root}", root.as_posix()).replace("{date}", date)
        return rendered, {"format_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def load_v551_queue(root: Path) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_data = read_yaml_mapping(active_path)
    next_data = read_yaml_mapping(next_path)
    if next_data.get("milestone") == MILESTONE_V551:
        data = next_data
        chosen = ROADMAP_NEXT_RELATIVE_PATH
        note = "research-roadmap-next.yaml contains V551 and was audited"
    else:
        data = active_data
        chosen = ACTIVE_ROADMAP_RELATIVE_PATH
        note = "active research-roadmap.yaml contains V551 and was audited"
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
        "known_issues": {
            "path": KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
            "present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        },
        "solve_registry": {
            "path": SOLVE_REGISTRY_RELATIVE_PATH.as_posix(),
            "present": (root / SOLVE_REGISTRY_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / SOLVE_REGISTRY_RELATIVE_PATH),
        },
        "claims_ledger": {
            "path": CLAIMS_LEDGER_RELATIVE_PATH.as_posix(),
            "present": (root / CLAIMS_LEDGER_RELATIVE_PATH).exists(),
            "sha256": path_sha256(root / CLAIMS_LEDGER_RELATIVE_PATH),
        },
    }
    return dict(data), identity


def validate_v551_queue_data(
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

    exp_numbers = [_task_number(task_id) for task_id in ids]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    missing_ids = [task_id for task_id in EXPECTED_V551_TASK_IDS if task_id not in ids]
    extra_ids = [task_id for task_id in ids if task_id not in EXPECTED_V551_TASK_IDS]
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
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        task_exp_number = exp_number(task_id)
        if task_exp_number in retired_ids:
            retired_task_ids.append(task_id)
        requires = task.get("requires")
        for dependency in requires if isinstance(requires, list) else []:
            dep = str(dependency)
            dep_exp_number = exp_number(dep)
            if dep not in id_index or id_index[dep] >= task_index:
                dependency_failures.append({"task_id": task_id, "dependency": dep})
            if dep_exp_number in retired_ids:
                retired_references.append({"task_id": task_id, "dependency": dep})
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
                    retired_references.append({"task_id": task_id, "gate_upstream": upstream})

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
        agent_type = str(task.get("agent_type") or "")
        model = str(task.get("model") or "")
        if agent_type != "codex" or model != "gpt-5.5":
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
        if task.get("requires_gpu") is True:
            llm_task_ids.append(task_id)
            prompt_lower = rendered_prompt.lower()
            if "MODEL_SPECS" not in rendered_prompt:
                model_policy_failures.append({"task_id": task_id, "reason": "missing_model_specs"})
            if "cached_sota_pair()" not in rendered_prompt:
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "missing_cached_sota_pair"}
                )
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
            if "gpu" not in prompt_lower or (
                "receipt" not in prompt_lower and "cuda" not in prompt_lower
            ):
                model_policy_failures.append({"task_id": task_id, "reason": "missing_real_gpu_receipts"})
            if "legacy headline cell" not in prompt_lower:
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "missing_no_legacy_headline_cell_rule"}
                )
            elif not (
                "no legacy headline cell" in prompt_lower
                or "without legacy headline cell" in prompt_lower
                or "never use legacy headline cell" in prompt_lower
            ):
                model_policy_failures.append(
                    {"task_id": task_id, "reason": "forbidden_legacy_headline_cell"}
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
            "missing_task": "\n      TASK" not in rendered_prompt
            and "\nTASK" not in rendered_prompt,
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
        "v551_task_ids": ids,
        "v551_id_and_deliverable_checks": {
            "ok": ids == list(EXPECTED_V551_TASK_IDS)
            and not duplicate_ids
            and not deliverable_failures
            and not duplicate_deliverables
            and exp_numbers == sorted(exp_numbers)
            and None not in exp_numbers
            and not retired_task_ids,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V551_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V551_TASK_IDS),
            "missing_expected_task_ids": missing_ids,
            "extra_task_ids": extra_ids,
            "duplicate_task_ids": duplicate_ids,
            "unique_deliverables": not duplicate_deliverables,
            "duplicate_deliverables": duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
            "retired_task_ids": retired_task_ids,
        },
        "v551_dependency_and_gate_checks": {
            "ok": not dependency_failures and not gate_failures,
            "dependency_failures": dependency_failures,
            "gate_count": len(gate_expressions),
            "gate_failures": gate_failures,
            "retired_references": retired_references,
            "structured_gate_expressions": gate_expressions,
            "memory_gate_fields": [
                "clean_factor_evidence_boundary_ready_score",
                "provenance_tiered_memory_protocol_ready_score",
                "powered_write_time_admission_ready_score",
                "delta_future_exact_yield",
                "delta_contamination_propagation_rate",
            ],
            "arc_gate_fields": [
                "arc_active_goal_causal_ready_score",
                "delta_exact_progress_proxy",
                "delta_false_accept_count",
            ],
        },
        "v551_gate_field_cross_reference_checks": {
            "ok": not gate_cross_ref_failures,
            "failures": gate_cross_ref_failures,
            "checked_gate_count": len(gate_expressions),
        },
        "v551_prior_failure_checks": {
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
        "v551_exclusion_manifest_checks": {
            "ok": not retired_task_ids and not retired_references and hard_exclusion_count == 0,
            "retired_task_ids": retired_task_ids,
            "retired_references": retired_references,
            "hard_exclusion_count": hard_exclusion_count,
            "risk_count": len(exclusion_risks),
            "risks": _risk_rows(exclusion_risks),
        },
        "v551_agent_model_and_llm_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "all_tasks_codex_gpt55": not route_failures,
            "route_failures": route_failures,
            "llm_task_ids": llm_task_ids,
            "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
            "model_policy_failures": model_policy_failures,
        },
        "prompt_contract_checks": {
            "ok": not prompt_failures and not raw_placeholder_failures,
            "checked_task_count": len(tasks),
            "render_receipts": render_receipts,
            "raw_placeholder_contract_ok": not raw_placeholder_failures,
            "raw_placeholder_failures": raw_placeholder_failures,
            "failures": prompt_failures,
        },
    }


def _source_v550_active_roadmap(payloads: Mapping[str, JsonDict], root: Path) -> JsonDict:
    exp6391 = payloads["exp6391-v550-terminal-handoff-and-queue-preflight"]
    milestone_hashes = exp6391.get("v550_milestone_doc_and_queue_hashes", {})
    recorded = milestone_hashes.get("active_roadmap") if isinstance(milestone_hashes, Mapping) else None
    return {
        "source_artifact": V550_DELIVERABLES_BY_TASK[
            "exp6391-v550-terminal-handoff-and-queue-preflight"
        ],
        "recorded_v550_active_roadmap": recorded,
        "recorded_v550_task_ids": exp6391.get("v550_task_ids"),
        "current_active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "milestone": read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH).get("milestone"),
        },
        "same_as_current_active": (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
        ),
    }


def _test_rows(command_receipts: Sequence[JsonMap] | None) -> list[JsonDict]:
    if command_receipts:
        return [dict(row) for row in command_receipts if isinstance(row, Mapping)]
    return [{"source": "declared", "command": command, "exit_code": None} for command in DEFAULT_TEST_COMMANDS]


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
    payloads, metas, summaries = _load_v550_inputs(root)
    conductor = _v550_conductor_outcomes(root)
    findings = _v550_adversarial_findings(root, payloads, metas, summaries)
    terminal_rows = _terminal_artifacts_by_task(payloads, metas, findings, conductor)
    v551_data, v551_identity = load_v551_queue(root)
    queue_checks = validate_v551_queue_data(v551_data, root, date)
    after_hashes = protected_hashes(root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    id_checks = queue_checks["v551_id_and_deliverable_checks"]
    status = (
        "complete_v551_queue_preflight_passed"
        if id_checks["ok"]
        and queue_checks["v551_dependency_and_gate_checks"]["ok"]
        and queue_checks["v551_gate_field_cross_reference_checks"]["ok"]
        and queue_checks["v551_prior_failure_checks"]["ok"]
        and queue_checks["v551_exclusion_manifest_checks"]["ok"]
        and queue_checks["v551_agent_model_and_llm_policy_checks"]["ok"]
        and queue_checks["prompt_contract_checks"]["ok"]
        else "complete_blocked_v551_queue_incomplete"
        if id_checks["task_count"] != id_checks["expected_task_count"]
        else "complete_blocked_v551_queue_preflight_failed"
    )
    honest_verdict = {
        "complete_v551_queue_preflight_passed": (
            "complete_v551_queue_preflight_passed: V550 evidence is bounded and the "
            "twelve-task V551 queue validates"
        ),
        "complete_blocked_v551_queue_incomplete": (
            "complete_blocked_v551_queue_incomplete: active V551 queue has "
            f"{id_checks['task_count']} of {id_checks['expected_task_count']} expected tasks; "
            "V550 evidence preserved without roadmap or conductor edit"
        ),
        "complete_blocked_v551_queue_preflight_failed": (
            "complete_blocked_v551_queue_preflight_failed: V551 queue preflight failed; "
            "V550 evidence preserved without roadmap or conductor edit"
        ),
    }[status]
    principles = dict(FIELD_PRINCIPLES)
    for expression in queue_checks["v551_dependency_and_gate_checks"]["structured_gate_expressions"]:
        principles[expression] = "This structured V551 gate expression must stay auditable before activation."

    report: JsonDict = {
        "status": status,
        "v550_active_roadmap_path_and_hash": _source_v550_active_roadmap(payloads, root),
        "v550_task_ids": list(EXPECTED_V550_TASK_IDS),
        "v550_terminal_artifacts_by_task": terminal_rows,
        "v550_artifact_verdicts": _artifact_verdicts(payloads),
        "v550_conductor_outcomes": conductor,
        "v550_adversarial_findings": findings,
        "v550_duration_receipts_by_task": _duration_receipts(payloads, metas),
        "v550_factor_boundary": _v550_factor_boundary(payloads),
        "v550_arc_boundary": _v550_arc_boundary(payloads),
        "v551_milestone_doc_and_queue_hashes": v551_identity,
        "v551_task_ids": queue_checks["v551_task_ids"],
        "v551_id_and_deliverable_checks": queue_checks["v551_id_and_deliverable_checks"],
        "v551_dependency_and_gate_checks": queue_checks["v551_dependency_and_gate_checks"],
        "v551_gate_field_cross_reference_checks": queue_checks[
            "v551_gate_field_cross_reference_checks"
        ],
        "v551_prior_failure_checks": queue_checks["v551_prior_failure_checks"],
        "v551_exclusion_manifest_checks": queue_checks["v551_exclusion_manifest_checks"],
        "v551_agent_model_and_llm_policy_checks": queue_checks[
            "v551_agent_model_and_llm_policy_checks"
        ],
        "prompt_contract_checks": queue_checks["prompt_contract_checks"],
        "active_roadmap_modified": before_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix())
        != after_hashes.get(ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "conductor_modified": before_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix())
        != after_hashes.get(RESEARCH_CONDUCTOR_RELATIVE_PATH.as_posix()),
        "solve_registry_modified": before_hashes.get(SOLVE_REGISTRY_RELATIVE_PATH.as_posix())
        != after_hashes.get(SOLVE_REGISTRY_RELATIVE_PATH.as_posix()),
        "claims_ledger_modified": before_hashes.get(CLAIMS_LEDGER_RELATIVE_PATH.as_posix())
        != after_hashes.get(CLAIMS_LEDGER_RELATIVE_PATH.as_posix()),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "date": date,
            "repo_root": root.as_posix(),
            "git_status_before": git_status_lines(root),
            "before_hashes": dict(before_hashes),
            "after_hashes": after_hashes,
            "expected_v550_artifact_count": len(EXPECTED_V550_TASK_IDS),
            "expected_v551_task_count": len(EXPECTED_V551_TASK_IDS),
            "active_v551_task_count": id_checks["task_count"],
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
    factor = report.get("v550_factor_boundary")
    if not isinstance(factor, Mapping):
        errors.append("v550_factor_boundary must be a mapping")
    else:
        if factor.get("exp6395_licensed_cell_count") != 4:
            errors.append("four Exp6395 licenses must be preserved")
        if factor.get("exp6395_qwen_abstention_count") != 3:
            errors.append("Qwen abstention boundary must be preserved")
        if factor.get("exp6395_rejected_gemma_cell_count") != 2:
            errors.append("two rejected Gemma cells must be preserved")
        if factor.get("universal_support_claimed") is not False:
            errors.append("no universal support boundary must be preserved")
        public_block = factor.get("exp6399_public_block")
        if not isinstance(public_block, Mapping) or public_block.get(
            "public_factor_claim_eligibility"
        ) is not False:
            errors.append("public factor block must be preserved")
    arc = report.get("v550_arc_boundary")
    if not isinstance(arc, Mapping):
        errors.append("v550_arc_boundary must be a mapping")
    else:
        if arc.get("actual_route_promotion_count") != 0:
            errors.append("zero route promotion must be preserved")
        if arc.get("solve_claim_count") != 0:
            errors.append("no ARC solve must be preserved")
        if arc.get("exp6402_public_arc_eligibility") is not False:
            errors.append("public ARC ineligibility must be preserved")
    if report.get("active_roadmap_modified") is not False:
        errors.append("active roadmap changed")
    if report.get("conductor_modified") is not False:
        errors.append("conductor changed")
    if report.get("solve_registry_modified") is not False:
        errors.append("solve registry changed")
    if report.get("claims_ledger_modified") is not False:
        errors.append("claims ledger changed")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("ok") is not True:
        errors.append("protected files changed")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        required_principles = set(REQUIRED_ARTIFACT_FIELDS)
        gates = report.get("v551_dependency_and_gate_checks", {})
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
    status = str(report.get("status") or "")
    if status == "complete_v551_queue_preflight_passed":
        check_fields = [
            "v551_id_and_deliverable_checks",
            "v551_dependency_and_gate_checks",
            "v551_gate_field_cross_reference_checks",
            "v551_prior_failure_checks",
            "v551_exclusion_manifest_checks",
            "v551_agent_model_and_llm_policy_checks",
            "prompt_contract_checks",
        ]
        if any(
            not isinstance(report.get(field), Mapping) or report[field].get("ok") is not True
            for field in check_fields
        ):
            errors.append("passed report has failed V551 checks")
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
