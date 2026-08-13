"""Exp6391 V550 terminal handoff and queue preflight.

Spec refs: REQ-INFRA-6391, SCENARIO-INFRA-6391-1,
SCENARIO-INFRA-6391-2, SCENARIO-INFRA-6391-3,
SCENARIO-INFRA-6391-4, SCENARIO-INFRA-6391-5,
SCENARIO-INFRA-6391-6.
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


MILESTONE_V549 = "2026.08.549"
MILESTONE_V550 = "2026.08.550"
RUN_DATE = "20260813"
EXPERIMENT_ID = "exp6391-v550-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6391.v550_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json"
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
V549_DELIVERABLES_BY_TASK = {
    "exp6377-v549-terminal-handoff-and-queue-preflight": (
        "results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json"
    ),
    "exp6378-v549-post-marker-source-scope-freeze": (
        "results/experiment_6378_v549_post_marker_source_scope_freeze.json"
    ),
    "exp6379-canonical-factor-edit-transport-contract": (
        "results/experiment_6379_canonical_factor_edit_transport_contract.json"
    ),
    "exp6380-three-family-canonical-factor-transport-canary": (
        "results/experiment_6380_three_family_canonical_factor_transport_canary.json"
    ),
    "exp6381-verified-frontier-live-factor-proposal-ab": (
        "results/experiment_6381_verified_frontier_live_factor_proposal_ab.json"
    ),
    "exp6382-chronological-verified-factor-self-learning": (
        "results/experiment_6382_chronological_verified_factor_self_learning.json"
    ),
    "exp6383-dependency-guided-factor-rollback-stress": (
        "results/experiment_6383_dependency_guided_factor_rollback_stress.json"
    ),
    "exp6384-default-off-certified-factor-consumer-ab": (
        "results/experiment_6384_default_off_certified_factor_consumer_ab.json"
    ),
    "exp6385-live-factor-learning-and-rollback-safety-audit": (
        "results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"
    ),
    "exp6386-arc-two-sided-goal-evidence-contract": (
        "results/experiment_6386_arc_two_sided_goal_evidence_contract.json"
    ),
    "exp6387-arc-active-reward-machine-discriminator": (
        "results/experiment_6387_arc_active_reward_machine_discriminator.json"
    ),
    "exp6388-arc-goal-evidence-response-calibration": (
        "results/experiment_6388_arc_goal_evidence_response_calibration.json"
    ),
    "exp6389-arc-default-off-active-goal-shadow": (
        "results/experiment_6389_arc_default_off_active_goal_shadow.json"
    ),
    "exp6390-v549-adversarial-capstone": ("results/experiment_6390_v549_adversarial_capstone.json"),
}
V549_TITLE_SNIPPETS = {
    "exp6377-v549-terminal-handoff-and-queue-preflight": (
        "V548 terminal evidence handoff and V549 queue pref"
    ),
    "exp6378-v549-post-marker-source-scope-freeze": (
        "V549 dated source delta and three-lane scope freez"
    ),
    "exp6379-canonical-factor-edit-transport-contract": (
        "Canonical factor-edit instruction and capacity tra"
    ),
    "exp6380-three-family-canonical-factor-transport-canary": (
        "Gated on Exp6379 readiness: three-family canonical"
    ),
    "exp6381-verified-frontier-live-factor-proposal-ab": (
        "Gated on Exp6380 readiness: verified-frontier live"
    ),
    "exp6382-chronological-verified-factor-self-learning": (
        "Gated on Exp6381 positive delta: chronological ver"
    ),
    "exp6383-dependency-guided-factor-rollback-stress": (
        "Dependency-guided certified-factor descendant roll"
    ),
    "exp6384-default-off-certified-factor-consumer-ab": (
        "Gated on Exp6382 and Exp6383 readiness: default-of"
    ),
    "exp6385-live-factor-learning-and-rollback-safety-audit": (
        "Independent canonical-transport learning rollback"
    ),
    "exp6386-arc-two-sided-goal-evidence-contract": (
        "Two-sided live ARC goal-evidence admission contrac"
    ),
    "exp6387-arc-active-reward-machine-discriminator": (
        "Gated on Exp6386 readiness: live-path active rewar"
    ),
    "exp6388-arc-goal-evidence-response-calibration": (
        "Gated on Exp6387 readiness: three-family ARC goal"
    ),
    "exp6389-arc-default-off-active-goal-shadow": (
        "Gated on Exp6388 improvement: default-off live ARC"
    ),
    "exp6390-v549-adversarial-capstone": ("V549 adversarial capstone and PRD-gap reconciliati"),
}

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
EXPECTED_V550_LLM_TASK_IDS = (
    "exp6394-model-family-factor-harness-freeze",
    "exp6395-held-factor-transport-license-matrix",
    "exp6396-capability-qualified-verified-frontier-ab",
    "exp6397-transactional-continuous-factor-learning",
    "exp6398-default-off-transactional-factor-consumer",
    "exp6400-arc-default-off-active-goal-shadow",
    "exp6401-arc-active-goal-causal-holdout",
)
MANDATED_GGUF_IDS = frozenset(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)
QWEN_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_MODEL_IDS = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
GGUF_ID_RE = re.compile(r"[\w.-]+/[\w.-]+-GGUF")
FINAL_PROHIBITION_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6391_v550_terminal_handoff_and_queue_preflight "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6391_v550_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6391_v550_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6391_v550_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6391_v550_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6391_v550_terminal_handoff_and_queue_preflight.py"
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
    "results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json"
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
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6391_test_receipts.json")

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
    *[Path(path) for path in V549_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v549_active_roadmap_path_and_hash",
    "v549_task_ids",
    "v549_terminal_artifacts_by_task",
    "v549_artifact_verdicts",
    "v549_conductor_outcomes",
    "v549_adversarial_flags",
    "v549_duration_receipts_by_task",
    "v549_factor_boundary",
    "v549_arc_boundary",
    "v550_milestone_doc_and_queue_hashes",
    "v550_task_ids",
    "v550_id_and_deliverable_checks",
    "v550_dependency_and_gate_checks",
    "v550_gate_field_cross_reference_checks",
    "v550_prior_failure_checks",
    "v550_exclusion_manifest_checks",
    "v550_agent_model_and_llm_policy_checks",
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
    "status": "The status states whether the V550 queue is preflight-safe.",
    "v549_active_roadmap_path_and_hash": "The prior V549 active roadmap hash comes from the terminal handoff artifact.",
    "v549_task_ids": "The fixed V549 task denominator prevents same-number aliasing.",
    "v549_terminal_artifacts_by_task": "Each V549 task keeps its exact terminal class.",
    "v549_artifact_verdicts": "Artifact honest verdicts stay separate from conductor and verifier facts.",
    "v549_conductor_outcomes": "Conductor outcomes stay separate from artifact verdicts.",
    "v549_adversarial_flags": "Stamped and live adversarial flags stay visible.",
    "v549_duration_receipts_by_task": "Each task duration comes from its own artifact receipt.",
    "v549_factor_boundary": "The factor boundary preserves ready, null, blocked, absent, and control facts.",
    "v549_arc_boundary": "The ARC boundary preserves ready evidence and the nested scalar-gate failure.",
    "v550_milestone_doc_and_queue_hashes": "The V550 roadmap and planning sources are hash-pinned.",
    "v550_task_ids": "The audited V550 queue identity is explicit.",
    "v550_id_and_deliverable_checks": "The queue must contain thirteen unique ordered IDs and result JSON deliverables.",
    "v550_dependency_and_gate_checks": "Dependencies and structured gates must be ordered and valid.",
    "v550_gate_field_cross_reference_checks": "Gate fields must appear in upstream required artifact fields.",
    "v550_prior_failure_checks": "Prior failures must name the old verdict, changed mechanism, and retirement rule.",
    "v550_exclusion_manifest_checks": "Retired task reuse and retired upstream chains fail before execution.",
    "v550_agent_model_and_llm_policy_checks": "Agent routing and local GGUF policy are checked before live work.",
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
    "v549_active_roadmap_path_and_hash": "upstream",
    "v549_task_ids": "constant",
    "v549_terminal_artifacts_by_task": "derived",
    "v549_artifact_verdicts": "upstream",
    "v549_conductor_outcomes": "measured",
    "v549_adversarial_flags": "measured",
    "v549_duration_receipts_by_task": "upstream",
    "v549_factor_boundary": "derived",
    "v549_arc_boundary": "derived",
    "v550_milestone_doc_and_queue_hashes": "measured",
    "v550_task_ids": "upstream",
    "v550_id_and_deliverable_checks": "derived",
    "v550_dependency_and_gate_checks": "derived",
    "v550_gate_field_cross_reference_checks": "derived",
    "v550_prior_failure_checks": "derived",
    "v550_exclusion_manifest_checks": "derived",
    "v550_agent_model_and_llm_policy_checks": "derived",
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


def _conductor_rows(root: Path, task_id: str) -> list[JsonDict]:
    snippet = V549_TITLE_SNIPPETS.get(task_id, "").lower()
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


def v549_conductor_outcomes(root: Path) -> JsonDict:
    outcomes: JsonDict = {}
    for task_id in EXPECTED_V549_TASK_IDS:
        rows = _conductor_rows(root, task_id)
        counts = Counter(str(row["status"]) for row in rows)
        outcomes[task_id] = {
            **dict(sorted(counts.items())),
            "attempt_count": len(rows),
            "rows": rows,
        }
    return outcomes


def _base_terminal_class(payload: JsonMap, meta: JsonMap) -> str:
    if meta.get("error") == "missing":
        return "absent"
    if meta.get("error"):
        return "malformed"
    status = str(payload.get("status") or "").lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if payload.get("blocked_at_layer") == "conductor_pre_gate" or status.startswith("blocked"):
        return "blocked"
    if verdict.startswith("blocked") or "gate_check_failed" in verdict:
        return "blocked"
    if status.startswith("complete_null") or verdict.startswith("complete_null"):
        return "null"
    if status.startswith("complete_no_scope_change"):
        return "complete"
    if status.startswith("complete_positive") or verdict.startswith("complete_positive"):
        return "positive"
    if status.startswith("complete") or verdict.startswith("complete"):
        return "complete"
    return "unknown"


def _load_v549_inputs(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict], JsonDict]:
    payloads: dict[str, JsonDict] = {}
    metas: dict[str, JsonDict] = {}
    summaries: JsonDict = {}
    for task_id in EXPECTED_V549_TASK_IDS:
        rel = Path(V549_DELIVERABLES_BY_TASK[task_id])
        payload, meta = read_json_mapping(root / rel)
        payloads[task_id] = payload
        metas[task_id] = meta
        summaries[task_id] = _summarize_artifact(root, rel)
    return payloads, metas, summaries


def _adversarial_flags(
    root: Path,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    summaries: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task_id in EXPECTED_V549_TASK_IDS:
        rel = Path(V549_DELIVERABLES_BY_TASK[task_id])
        payload = payloads[task_id]
        present = metas[task_id].get("error") is None
        live = _live_adversarial(root, rel, present)
        rows[task_id] = {
            "path": rel.as_posix(),
            "present": present,
            "stamped_flagged_adversarial": payload.get("flagged_adversarial"),
            "stamped_corrigendum_pending": bool(payload.get("corrigendum_pending")),
            "live_verdict": live["verdict"],
            "live_has_critical": live["critical_count"] > 0,
            "live_flag_count": live["flag_count"],
            "live_flags": live["flags"],
            "summary_receipt": summaries.get(task_id),
        }
    return rows


def _terminal_artifacts_by_task(
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    flags: JsonMap,
    conductor: JsonMap,
) -> JsonDict:
    rows: JsonDict = {}
    for task_id in EXPECTED_V549_TASK_IDS:
        rel = Path(V549_DELIVERABLES_BY_TASK[task_id])
        payload = payloads[task_id]
        meta = metas[task_id]
        terminal_class = _base_terminal_class(payload, meta)
        if flags.get(task_id, {}).get("live_has_critical") is True:
            terminal_class = "flagged"
        rows[task_id] = {
            "task_id": task_id,
            "declared_deliverable": rel.as_posix(),
            "present": meta.get("present"),
            "loadable": meta.get("loadable"),
            "sha256": meta.get("sha256"),
            "terminal_class": terminal_class,
            "status_raw": payload.get("status"),
            "honest_verdict_raw": payload.get("honest_verdict"),
            "flagged_adversarial": payload.get("flagged_adversarial"),
            "corrigendum_pending": payload.get("corrigendum_pending"),
            "conductor_receipt": conductor.get(task_id),
        }
    return rows


def _duration_receipts(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    rows: JsonDict = {}
    for task_id in EXPECTED_V549_TASK_IDS:
        payload = payloads[task_id]
        value = payload.get("duration_s")
        source = "artifact.duration_s"
        if metas[task_id].get("error") == "missing":
            source = "artifact_absent"
            value = None
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            source = "duration_missing_or_non_numeric"
            value = None
        rows[task_id] = {
            "task_id": task_id,
            "duration_s": value,
            "source": source,
            "artifact_path": V549_DELIVERABLES_BY_TASK[task_id],
        }
    return rows


def _artifact_verdicts(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {task_id: payloads[task_id].get("honest_verdict") for task_id in EXPECTED_V549_TASK_IDS}


def _model_arm_counts(payload: JsonMap, key: str, model_id: str) -> JsonMap:
    counts = payload.get(key)
    if not isinstance(counts, Mapping):
        return {}
    by_model = counts.get("by_model_and_arm")
    if not isinstance(by_model, Mapping):
        return {}
    model_rows = by_model.get(model_id)
    if not isinstance(model_rows, Mapping):
        return {}
    arm = model_rows.get("canonical_prompt_computed_allowance")
    return arm if isinstance(arm, Mapping) else {}


def _v549_factor_boundary(
    payloads: Mapping[str, JsonDict],
    terminal_rows: JsonMap,
) -> JsonDict:
    exp6380 = payloads["exp6380-three-family-canonical-factor-transport-canary"]
    qualified: list[str] = []
    invalid: list[str] = []
    for model_id in (QWEN_MODEL_ID, *GEMMA_MODEL_IDS):
        parse_row = _model_arm_counts(
            exp6380, "parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm", model_id
        )
        exact_row = _model_arm_counts(exp6380, "exact_pass_fail_counts_by_model_and_arm", model_id)
        valid = int(parse_row.get("valid") or 0)
        exact_calls = int(exact_row.get("exact_calls") or 0)
        exact_pass = int(exact_row.get("exact_pass") or 0)
        if valid > 0 and exact_calls > 0 and exact_pass > 0:
            qualified.append(model_id)
        else:
            invalid.append(model_id)
    return {
        "exp6379_ready": payloads["exp6379-canonical-factor-edit-transport-contract"].get(
            "canonical_factor_transport_contract_ready_score"
        )
        == 1.0,
        "exp6380_global_null": payloads["exp6380-three-family-canonical-factor-transport-canary"]
        .get("honest_verdict", "")
        .startswith("complete_null"),
        "three_family_factor_transport_ready_score": exp6380.get(
            "three_family_factor_transport_ready_score"
        ),
        "qualified_gemma_models": [model for model in GEMMA_MODEL_IDS if model in qualified],
        "qualified_gemma_observation_count": len(
            [model for model in GEMMA_MODEL_IDS if model in qualified]
        ),
        "qwen_invalid": QWEN_MODEL_ID in invalid,
        "qwen_model_id": QWEN_MODEL_ID,
        "parse_valid_count": (
            exp6380.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm", {}) or {}
        ).get("total_valid"),
        "exact_checker_call_count": (
            exp6380.get("exact_pass_fail_counts_by_model_and_arm", {}) or {}
        ).get("total_exact_calls"),
        "exp6381_blocked": terminal_rows["exp6381-verified-frontier-live-factor-proposal-ab"][
            "terminal_class"
        ]
        == "blocked",
        "exp6382_blocked_or_absent": terminal_rows[
            "exp6382-chronological-verified-factor-self-learning"
        ]["terminal_class"]
        in {"blocked", "absent"},
        "exp6383_positive_control": payloads[
            "exp6383-dependency-guided-factor-rollback-stress"
        ].get("dependency_guided_rollback_ready_score")
        == 1.0,
        "exp6384_blocked": terminal_rows["exp6384-default-off-certified-factor-consumer-ab"][
            "terminal_class"
        ]
        == "blocked",
        "global_transport_promoted": False,
    }


def _v549_arc_boundary(payloads: Mapping[str, JsonDict], terminal_rows: JsonMap) -> JsonDict:
    delta = payloads["exp6388-arc-goal-evidence-response-calibration"].get(
        "delta_admission_precision"
    )
    return {
        "exp6386_ready": payloads["exp6386-arc-two-sided-goal-evidence-contract"].get(
            "arc_two_sided_goal_contract_ready_score"
        )
        == 1.0,
        "exp6387_ready": payloads["exp6387-arc-active-reward-machine-discriminator"].get(
            "arc_active_reward_machine_ready_score"
        )
        == 1.0,
        "exp6388_ready": payloads["exp6388-arc-goal-evidence-response-calibration"].get(
            "arc_evidence_calibration_ready_score"
        )
        == 1.0,
        "exp6388_delta_admission_precision_shape": type(delta).__name__,
        "exp6388_delta_admission_precision_pooled_unrounded": delta.get("pooled_unrounded")
        if isinstance(delta, Mapping)
        else None,
        "exp6388_delta_false_accept_count": payloads[
            "exp6388-arc-goal-evidence-response-calibration"
        ].get("delta_false_accept_count"),
        "exp6389_honest_verdict": payloads["exp6389-arc-default-off-active-goal-shadow"].get(
            "honest_verdict"
        ),
        "exp6389_blocked_gate_check_failed": payloads[
            "exp6389-arc-default-off-active-goal-shadow"
        ].get("honest_verdict")
        == "blocked_gate_check_failed",
        "exp6389_terminal_class": terminal_rows["exp6389-arc-default-off-active-goal-shadow"][
            "terminal_class"
        ],
        "arc_solve_claimed": any(
            payloads[task_id].get("arc_solve_claim") is True
            for task_id in (
                "exp6386-arc-two-sided-goal-evidence-contract",
                "exp6387-arc-active-reward-machine-discriminator",
                "exp6388-arc-goal-evidence-response-calibration",
                "exp6389-arc-default-off-active-goal-shadow",
            )
        ),
    }


def load_v550_queue(root: Path) -> tuple[JsonDict, JsonDict]:
    active_path = root / ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_data = read_yaml_mapping(active_path)
    next_data = read_yaml_mapping(next_path)
    if next_data.get("milestone") == MILESTONE_V550:
        data = next_data
        chosen = ROADMAP_NEXT_RELATIVE_PATH
        note = "research-roadmap-next.yaml contains V550 and was audited"
    else:
        data = active_data
        chosen = ACTIVE_ROADMAP_RELATIVE_PATH
        note = "active research-roadmap.yaml contains V550 and was audited"
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


def _risk_rows(risks: Sequence[Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for risk in risks:
        try:
            rows.append(asdict(risk))
        except TypeError:
            rows.append({"repr": repr(risk)})
    return rows


def _is_llm_task(task: JsonMap) -> bool:
    return task.get("requires_gpu") is True


def validate_v550_queue_data(
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
    missing_ids = [task_id for task_id in EXPECTED_V550_TASK_IDS if task_id not in ids]
    extra_ids = [task_id for task_id in ids if task_id not in EXPECTED_V550_TASK_IDS]
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
        if _is_llm_task(task):
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
            if "legacy headline cell" in prompt_lower:
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
        "v550_task_ids": ids,
        "v550_id_and_deliverable_checks": {
            "ok": ids == list(EXPECTED_V550_TASK_IDS)
            and not duplicate_ids
            and not deliverable_failures
            and not duplicate_deliverables
            and exp_numbers == sorted(exp_numbers)
            and None not in exp_numbers
            and not retired_task_ids,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V550_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V550_TASK_IDS),
            "missing_expected_task_ids": missing_ids,
            "extra_task_ids": extra_ids,
            "duplicate_task_ids": duplicate_ids,
            "unique_deliverables": not duplicate_deliverables,
            "duplicate_deliverables": duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
            "retired_task_ids": retired_task_ids,
        },
        "v550_dependency_and_gate_checks": {
            "ok": not dependency_failures and not gate_failures,
            "dependency_failures": dependency_failures,
            "gate_count": len(gate_expressions),
            "gate_failures": gate_failures,
            "structured_gate_expressions": gate_expressions,
            "capability_license_gate_fields": [
                "model_family_harness_freeze_ready_score",
                "held_factor_transport_license_ready_score",
                "licensed_model_count",
                "licensed_constraint_family_count",
            ],
            "arc_metric_gate_fields": [
                "arc_gate_metric_contract_ready_score",
                "delta_admission_precision_scalar",
                "delta_false_accept_count_scalar",
            ],
        },
        "v550_gate_field_cross_reference_checks": {
            "ok": not gate_cross_ref_failures,
            "failures": gate_cross_ref_failures,
            "checked_gate_count": len(gate_expressions),
        },
        "v550_prior_failure_checks": {
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
        "v550_exclusion_manifest_checks": {
            "ok": hard_exclusion_count == 0 and not retired_references and not retired_task_ids,
            "hard_risk_count": hard_exclusion_count,
            "risk_count": len(exclusion_risks),
            "risks": _risk_rows(exclusion_risks),
            "retired_task_ids": retired_task_ids,
            "retired_upstream_references": retired_references,
        },
        "v550_agent_model_and_llm_policy_checks": {
            "ok": not route_failures and not model_policy_failures,
            "route_failures": route_failures,
            "all_tasks_codex_gpt55": not route_failures,
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


def v549_active_roadmap_receipt(root: Path) -> JsonDict:
    exp6377_payload, _meta = read_json_mapping(
        root / V549_DELIVERABLES_BY_TASK["exp6377-v549-terminal-handoff-and-queue-preflight"]
    )
    upstream = exp6377_payload.get("v549_milestone_doc_and_queue_hashes", {})
    active_v549 = upstream.get("active_roadmap", {}) if isinstance(upstream, Mapping) else {}
    current = read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    return {
        "source": V549_DELIVERABLES_BY_TASK["exp6377-v549-terminal-handoff-and-queue-preflight"],
        "path": active_v549.get("path", ACTIVE_ROADMAP_RELATIVE_PATH.as_posix()),
        "sha256_at_exp6377": active_v549.get("sha256"),
        "milestone_at_exp6377": active_v549.get("milestone"),
        "current_active_roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "milestone": current.get("milestone"),
        },
        "same_as_current_active": active_v549.get("sha256")
        == path_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "boundary_note": "The V549 active roadmap has been consumed; Exp6377 supplies the V549 active hash.",
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
        "known_issues_sha256": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        "v549_expected_artifact_hashes": {
            task_id: {
                "path": rel,
                "present": (root / rel).exists(),
                "sha256": path_sha256(root / rel),
            }
            for task_id, rel in V549_DELIVERABLES_BY_TASK.items()
        },
        "protected_hashes_before": dict(before_hashes),
        "git_status_before": git_status_lines(root),
    }


def _field_principles(gate_expressions: Sequence[str]) -> JsonDict:
    principles = dict(FIELD_PRINCIPLES)
    for expression in gate_expressions:
        principles[expression] = "This structured V550 gate expression must remain auditable."
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
    payloads, metas, summaries = _load_v549_inputs(root)
    conductor = v549_conductor_outcomes(root)
    flags = _adversarial_flags(root, payloads, metas, summaries)
    terminal_rows = _terminal_artifacts_by_task(payloads, metas, flags, conductor)
    queue_data, queue_hashes = load_v550_queue(root)
    queue_checks = validate_v550_queue_data(queue_data, root, date)
    after_hashes = protected_hashes(root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    gate_expressions = queue_checks["v550_dependency_and_gate_checks"][
        "structured_gate_expressions"
    ]
    check_keys = (
        "v550_id_and_deliverable_checks",
        "v550_dependency_and_gate_checks",
        "v550_gate_field_cross_reference_checks",
        "v550_prior_failure_checks",
        "v550_exclusion_manifest_checks",
        "v550_agent_model_and_llm_policy_checks",
        "prompt_contract_checks",
    )
    all_v550_checks_ok = all(bool(queue_checks[key]["ok"]) for key in check_keys)
    status = (
        "complete_v550_queue_preflight_passed"
        if all_v550_checks_ok
        else "blocked_v550_queue_preflight_failed"
    )
    verdict = (
        "complete_v550_queue_preflight_passed: V549 evidence is bounded and the thirteen-task V550 queue validates"
        if all_v550_checks_ok
        else "blocked_v550_queue_preflight_failed: one or more V550 queue checks failed"
    )
    report: JsonDict = {
        "status": status,
        "v549_active_roadmap_path_and_hash": v549_active_roadmap_receipt(root),
        "v549_task_ids": list(EXPECTED_V549_TASK_IDS),
        "v549_terminal_artifacts_by_task": terminal_rows,
        "v549_artifact_verdicts": _artifact_verdicts(payloads),
        "v549_conductor_outcomes": conductor,
        "v549_adversarial_flags": flags,
        "v549_duration_receipts_by_task": _duration_receipts(payloads, metas),
        "v549_factor_boundary": _v549_factor_boundary(payloads, terminal_rows),
        "v549_arc_boundary": _v549_arc_boundary(payloads, terminal_rows),
        "v550_milestone_doc_and_queue_hashes": queue_hashes,
        "v550_task_ids": queue_checks["v550_task_ids"],
        "v550_id_and_deliverable_checks": queue_checks["v550_id_and_deliverable_checks"],
        "v550_dependency_and_gate_checks": queue_checks["v550_dependency_and_gate_checks"],
        "v550_gate_field_cross_reference_checks": queue_checks[
            "v550_gate_field_cross_reference_checks"
        ],
        "v550_prior_failure_checks": queue_checks["v550_prior_failure_checks"],
        "v550_exclusion_manifest_checks": queue_checks["v550_exclusion_manifest_checks"],
        "v550_agent_model_and_llm_policy_checks": queue_checks[
            "v550_agent_model_and_llm_policy_checks"
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
        gates = report.get("v550_dependency_and_gate_checks", {})
        expressions = (
            gates.get("structured_gate_expressions", []) if isinstance(gates, Mapping) else []
        )
        for expression in expressions:
            if expression not in principles:
                errors.append(f"missing field_principles entry: {expression}")
    else:
        errors.append("field_principles must be a mapping")
    provenance = report.get("field_provenance")
    allowed_provenance = {"measured", "derived", "constant", "upstream"}
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance must be a mapping")
    elif set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    elif any(value not in allowed_provenance for value in provenance.values()):
        errors.append("field_provenance has invalid classification")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("random_seed") is not None:
        errors.append("random_seed must be null")
    factor = report.get("v549_factor_boundary", {})
    if not isinstance(factor, Mapping) or factor.get("exp6380_global_null") is not True:
        errors.append("Exp6380 global null boundary must be preserved")
    if isinstance(factor, Mapping) and factor.get("qualified_gemma_observation_count") != 2:
        errors.append("two Gemma qualified observations must be preserved")
    arc = report.get("v549_arc_boundary", {})
    if not isinstance(arc, Mapping) or arc.get("exp6388_delta_admission_precision_shape") != "dict":
        errors.append("Exp6388 nested ARC metric boundary must be preserved")
    if isinstance(arc, Mapping) and arc.get("exp6389_blocked_gate_check_failed") is not True:
        errors.append("Exp6389 blocked gate verdict must be preserved")
    if report.get("protected_files_unchanged", {}).get("ok") is not True:
        errors.append("protected files changed")
    status = str(report.get("status") or "")
    if status.startswith("complete"):
        for key in (
            "v550_id_and_deliverable_checks",
            "v550_dependency_and_gate_checks",
            "v550_gate_field_cross_reference_checks",
            "v550_prior_failure_checks",
            "v550_exclusion_manifest_checks",
            "v550_agent_model_and_llm_policy_checks",
            "prompt_contract_checks",
        ):
            value = report.get(key)
            if not isinstance(value, Mapping) or value.get("ok") is not True:
                errors.append("complete report has failed V550 checks")
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
    parser = argparse.ArgumentParser(description="Write Exp6391 V550 handoff artifact.")
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
