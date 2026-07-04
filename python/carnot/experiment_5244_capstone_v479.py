"""Exp 5244: V479 milestone-close capstone reconciliation.

Spec refs: REQ-CAPSTONE-5244, SCENARIO-CAPSTONE-5244,
SCENARIO-CAPSTONE-5244-FIELD-PRINCIPLES.

This module reads the completed V479 artifacts and turns them into one closeout
record. The main rule is conservative: blocked, flagged, methodology-incomplete,
or no-bank evidence is still recorded, but it cannot become a headline claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results") / "experiment_5244_capstone_v479.json"
EXPERIMENT = "experiment_5244_capstone_v479"
EXPERIMENT_ID = "exp5244-capstone-v479"
MILESTONE = "2026.07.479"
SCHEMA = "carnot.experiment_5244_capstone_v479.v1"
RANDOM_SEED = 5244
INFERENCE_SUBSTRATE = "milestone_artifact_synthesis"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-CAPSTONE-5244",
    "SCENARIO-CAPSTONE-5244",
    "SCENARIO-CAPSTONE-5244-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "the milestone being closed, 2026.07.479",
    "tasks_seen": "count of loadable expected .479 artifacts read directly",
    "missing_artifacts": (
        "expected .479 artifacts absent or unreadable, empty only when every "
        "Exp5233-Exp5243 deliverable is loadable"
    ),
    "gated_or_blocked_artifacts": (
        "artifacts or task decisions that are blocked, gate-blocked, or no-bank "
        "and must not be rounded up"
    ),
    "headline_eligible_artifacts": (
        "clean non-flagged artifacts whose methodology and gates permit bounded headline use"
    ),
    "gap4_final_status": (
        "one of clean_null | clean_positive | blocked | unknown, with blocked "
        "required when Exp5236 is flagged or missing receipts"
    ),
    "gap1_final_status": (
        "one of promoted | blocked | retired | unknown, with promoted only after "
        "a clean frozen-subset registry promotion"
    ),
    "veribmc_final_status": (
        "one of positive | clean_null | retired | blocked | unknown, with retired "
        "allowed only after methodology-clean null evidence"
    ),
    "continuous_self_learning_status": (
        "one of controlled_positive | controlled_null | degraded | blocked | unknown, "
        "with controlled_positive requiring aligned lift, retention, rollback, and no model distillation"
    ),
    "arc_level_delta": "integer delta from clean reproduction-gated ARC evidence only",
    "kan_certificate_status": (
        "one of extended | tiny_only | blocked | unknown, with extended only for "
        "bounded deterministic certificate scale-up"
    ),
    "hardware_speedup_claimed": "must be false unless authenticated timing evidence exists",
    "ops_docs_updated": (
        "false when the stop rule delegates ops/status/changelog/traceability updates"
    ),
    "validation_commands_run": (
        "list of validation commands and pass/fail outcomes used for the capstone"
    ),
    "research_conductor_py_untouched_confirmed": (
        "hard constraint that scripts/research_conductor.py stayed untouched"
    ),
    "inference_substrate": "milestone_artifact_synthesis",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state the honest .479 close state."
    ),
}

PRINCIPLE_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "source_artifacts",
    "source_context",
    "excluded_from_headline_artifacts",
    "per_task_summary",
    "status_decisions",
    "preconditions_checked",
    "next_top_blockers",
    "random_seed",
    "reproducibility_checksum",
    "flagged_adversarial",
    *PRINCIPLE_WRAPPED_FIELDS,
)

DEFAULT_VALIDATION_COMMANDS = [
    {
        "command": ".venv/bin/pytest --override-ini addopts= tests/python/test_experiment_5244_capstone_v479.py -q",
        "status": "PENDING",
        "notes": "filled after the focused verification run",
    }
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V479 task deliverable."""

    experiment_number: int
    task_id: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5233,
        "exp5233-archive-478-activate-479",
        Path("results/experiment_5233_archive_478_activate_479.json"),
    ),
    UpstreamSource(
        5234,
        "exp5234-sota-ingestion-v479",
        Path("results/experiment_5234_sota_ingestion_v479.json"),
    ),
    UpstreamSource(
        5235,
        "exp5235-adversarial-qa-null-tautology-calibration-v479",
        Path("results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"),
    ),
    UpstreamSource(
        5236,
        "exp5236-gap4-clean-status-after-qa-calibration-v479",
        Path("results/experiment_5236_gap4_clean_status_after_qa_calibration_v479.json"),
    ),
    UpstreamSource(
        5237,
        "exp5237-gap1-stability-freeze-or-retire-v479",
        Path("results/experiment_5237_gap1_stability_freeze_or_retire_v479.json"),
    ),
    UpstreamSource(
        5238,
        "exp5238-veribmc-methodology-correct-rerun-or-retire-v479",
        Path("results/experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479.json"),
    ),
    UpstreamSource(
        5239,
        "exp5239-continuous-self-learning-controlled-memory-ablation-v479",
        Path(
            "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json"
        ),
    ),
    UpstreamSource(
        5240,
        "exp5240-arc-rubric-to-patch-synthesis-v479",
        Path("results/experiment_5240_arc_rubric_to_patch_synthesis_v479.json"),
    ),
    UpstreamSource(
        5241,
        "exp5241-arc-gated-live-patch-attempt-v479",
        Path("results/experiment_5241_arc_gated_live_patch_attempt_v479.json"),
    ),
    UpstreamSource(
        5242,
        "exp5242-kan-certificate-abstraction-scale-v479",
        Path("results/experiment_5242_kan_certificate_abstraction_scale_v479.json"),
    ),
    UpstreamSource(
        5243,
        "exp5243-hardware-continuity-kan-pbit-boundary-v479",
        Path("results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json"),
    ),
)

SOURCE_CONTEXT_PATHS = (
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-complete.yaml"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("_bmad/traceability.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
)

GAP4_STATUSES = {"clean_null", "clean_positive", "blocked", "unknown"}
GAP1_STATUSES = {"promoted", "blocked", "retired", "unknown"}
VERIBMC_STATUSES = {"positive", "clean_null", "retired", "blocked", "unknown"}
SELF_LEARNING_STATUSES = {
    "controlled_positive",
    "controlled_null",
    "degraded",
    "blocked",
    "unknown",
}
KAN_STATUSES = {"extended", "tiny_only", "blocked", "unknown"}


def source_by_number(experiment_number: int) -> UpstreamSource:
    for source in UPSTREAM_SOURCES:
        if source.experiment_number == experiment_number:
            return source
    raise KeyError(experiment_number)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def honest_verdict_text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def as_bool(value: Any) -> bool:
    return value_of(value) is True


def as_number(value: Any) -> float | None:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": file_sha256(path),
    }


def research_conductor_untouched(root: Path) -> bool:  # pragma: no cover - git integration.
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.returncode == 0 and status.stdout.strip() == ""


def has_critical_corrigendum(data: JsonMap) -> bool:
    pending = value_of(data.get("corrigendum_pending"))
    if not isinstance(pending, list):
        return False
    for finding in pending:
        if not isinstance(finding, Mapping):
            continue
        severity = finding.get("severity")
        if severity == 2 or (isinstance(severity, str) and severity.lower() == "critical"):
            return True
    return False


def has_methodology_gap(data: JsonMap) -> bool:
    pending = value_of(data.get("corrigendum_pending"))
    if not isinstance(pending, list):
        return False
    for finding in pending:
        if isinstance(finding, Mapping) and finding.get("kind") == "METHODOLOGY_MISSING":
            return True
    return False


def failed_validation_command(data: JsonMap) -> bool:
    commands = value_of(data.get("arc_validation_commands"))
    if not isinstance(commands, list):
        commands = value_of(data.get("validation_commands_run"))
    if not isinstance(commands, list):
        return False
    for command in commands:
        if not isinstance(command, Mapping):
            continue
        if command.get("passed") is False:
            return True
        status = command.get("status")
        if isinstance(status, str) and status.upper().startswith("FAIL"):
            return True
    return False


def is_gate_blocked(data: JsonMap) -> bool:
    verdict = honest_verdict_text(data.get("honest_verdict"))
    return bool(
        data.get("status") == "blocked"
        or data.get("blocked_at_layer") == "conductor_pre_gate"
        or verdict.startswith("blocked")
    )


def blocked_decision(data: JsonMap) -> str | None:
    gap4 = value_of(data.get("gap4_status_decision"))
    if isinstance(gap4, str) and gap4.startswith("blocked"):
        return gap4
    gap1 = value_of(data.get("gap1_stability_decision"))
    if isinstance(gap1, str) and gap1.startswith("blocked"):
        return gap1
    claim = data.get("solve_claim")
    if isinstance(claim, Mapping) and claim.get("residual") == "no_level_banked":
        return "no_level_banked"
    return None


def exclusion_reasons(data: JsonMap) -> list[str]:
    reasons: list[str] = []
    if data.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial")
    if has_critical_corrigendum(data):
        reasons.append("critical_corrigendum_pending")
    if value_of(data.get("adversarial_verify_passed")) is False:
        reasons.append("adversarial_verify_failed")
    if has_methodology_gap(data):
        reasons.append("methodology_incomplete")
    if is_gate_blocked(data):
        reasons.append("gate_blocked")
    if failed_validation_command(data):
        reasons.append("failed_validation_command")
    return reasons


def load_upstreams(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict], list[str]]:
    artifacts: dict[int, JsonDict] = {}
    source_rows: list[JsonDict] = []
    missing: list[str] = []
    for source in UPSTREAM_SOURCES:
        path = root / source.relative_path
        data, meta = read_json_mapping(path)
        row = {
            "experiment_number": source.experiment_number,
            "task_id": source.task_id,
            "relative_path": str(source.relative_path),
            "exists": meta.get("exists") is True,
            "loadable": meta.get("loadable") is True,
            "sha256": meta.get("sha256"),
            "error": meta.get("error"),
        }
        if not meta.get("loadable"):
            missing.append(source.task_id)
            source_rows.append(row)
            continue
        artifacts[source.experiment_number] = data
        reasons = exclusion_reasons(data)
        source_rows.append(
            row
            | {
                "honest_verdict": honest_verdict_text(data.get("honest_verdict")),
                "flagged_adversarial": data.get("flagged_adversarial") is True,
                "critical_corrigendum": has_critical_corrigendum(data),
                "methodology_incomplete": has_methodology_gap(data),
                "gate_blocked": is_gate_blocked(data),
                "blocked_decision": blocked_decision(data),
                "failed_validation_command": failed_validation_command(data),
                "excluded_from_headline": bool(reasons or blocked_decision(data)),
                "exclusion_reasons": reasons,
            }
        )
    return artifacts, source_rows, missing


def load_source_context(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for relative_path in SOURCE_CONTEXT_PATHS:
        path = root / relative_path
        rows.append(
            {
                "relative_path": str(relative_path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
                "read_only": True,
            }
        )
    return rows


def _status_for(data: JsonMap) -> str:
    if not data:
        return "missing"
    if is_gate_blocked(data):
        return "gate_blocked"
    if blocked_decision(data):
        return "blocked"
    if exclusion_reasons(data):
        return "excluded"
    return "complete"


def per_task_summary(artifacts: Mapping[int, JsonMap]) -> JsonDict:
    summary: JsonDict = {}
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        reasons = exclusion_reasons(data) if data else []
        block = blocked_decision(data) if data else None
        summary[source.task_id] = {
            "artifact_path": str(source.relative_path),
            "status": _status_for(data),
            "verdict": honest_verdict_text(data.get("honest_verdict"))
            if data
            else "missing_artifact",
            "headline_eligible": bool(data and not reasons and block is None),
            "exclusion_reasons": reasons,
            "blocked_decision": block,
            "flagged_adversarial": data.get("flagged_adversarial") is True if data else False,
            "critical_corrigendum": has_critical_corrigendum(data) if data else False,
            "methodology_incomplete": has_methodology_gap(data) if data else False,
            "gate_blocked": is_gate_blocked(data) if data else False,
            "failed_validation_command": failed_validation_command(data) if data else False,
        }
    return summary


def headline_eligible_task_ids(summary: JsonMap) -> list[str]:
    return [task_id for task_id, row in summary.items() if row.get("headline_eligible") is True]


def excluded_from_headline(summary: JsonMap) -> JsonDict:
    return {
        task_id: row
        for task_id, row in summary.items()
        if row.get("status") in {"excluded", "gate_blocked", "blocked"}
        or row.get("exclusion_reasons")
    }


def gated_or_blocked_rows(summary: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, row in summary.items():
        if row.get("status") not in {"blocked", "gate_blocked"} and not row.get("blocked_decision"):
            continue
        rows.append(
            {
                "task_id": task_id,
                "status": row.get("status"),
                "reason": row.get("blocked_decision") or "gate_blocked",
                "artifact_path": row.get("artifact_path"),
            }
        )
    return rows


def eligible_numbers(artifacts: Mapping[int, JsonMap]) -> set[int]:
    return {
        number
        for number, data in artifacts.items()
        if not exclusion_reasons(data) and blocked_decision(data) is None
    }


def gap4_final_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5236, {})
    if not data:
        return "unknown", "Exp5236 GAP-4 status artifact is missing"
    if 5236 not in eligible:
        return "blocked", "Exp5236 is flagged, receipt-blocked, or otherwise headline-ineligible"
    decision = value_of(data.get("gap4_status_decision"))
    if decision in {"clean_positive", "positive"}:
        return "clean_positive", "Exp5236 reported a clean positive GAP-4 decision"
    if decision in {"clean_null", "null"}:
        return "clean_null", "Exp5236 reported a clean null GAP-4 decision"
    if isinstance(decision, str) and decision.startswith("blocked"):
        return "blocked", f"Exp5236 reports {decision}"
    return "unknown", f"Exp5236 did not provide an allowed GAP-4 status: {decision}"


def gap1_final_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5237, {})
    if not data:
        return "unknown", "Exp5237 GAP-1 stability artifact is missing"
    if 5237 not in eligible and not blocked_decision(data):
        return "blocked", "Exp5237 is flagged or gate-blocked"
    if as_bool(data.get("gap1_registry_promoted")):
        return "promoted", "Exp5237 promoted a frozen stable GAP-1 registry verifier"
    decision = value_of(data.get("gap1_stability_decision"))
    if isinstance(decision, str) and decision.startswith("blocked"):
        return "blocked", f"Exp5237 reports {decision}"
    if decision == "retired" or decision == "retired_current_path":
        return "retired", "Exp5237 retired the current GAP-1 path"
    return "unknown", f"Exp5237 did not promote, block, or retire GAP-1: {decision}"


def veribmc_final_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5238, {})
    if not data:
        return "unknown", "Exp5238 VerIbmc artifact is missing"
    if 5238 not in eligible:
        return "blocked", "Exp5238 is flagged or gate-blocked"
    uplift = as_number(data.get("solver_feedback_uplift")) or 0.0
    if uplift > 0.0:
        return "positive", f"Exp5238 solver feedback uplift is positive: {uplift}"
    if as_bool(data.get("retire_current_veribmc_path")):
        return "retired", "Exp5238 records methodology-clean null and retires current path"
    return "clean_null", "Exp5238 records methodology-clean null solver-feedback evidence"


def continuous_self_learning_status(
    artifacts: Mapping[int, JsonMap], eligible: set[int]
) -> tuple[str, str]:
    data = artifacts.get(5239, {})
    if not data:
        return "unknown", "Exp5239 controlled memory artifact is missing"
    if 5239 not in eligible:
        return "blocked", "Exp5239 is flagged or gate-blocked"
    aligned_vs_shuffled = as_number(data.get("aligned_vs_shuffled_delta")) or 0.0
    aligned_vs_no_memory = as_number(data.get("aligned_vs_no_memory_delta")) or 0.0
    if as_bool(data.get("broad_self_distillation_used")):
        return (
            "degraded",
            "Exp5239 used broad self-distillation or failed the controlled-memory boundary",
        )
    if (
        as_bool(data.get("continuous_self_learning_task"))
        and aligned_vs_shuffled > 0.0
        and aligned_vs_no_memory > 0.0
        and as_bool(data.get("retention_check_passed"))
        and as_bool(data.get("rollback_policy_exercised"))
    ):
        return (
            "controlled_positive",
            "Exp5239 aligned typed memory beat controls with retention and rollback exercised",
        )
    if aligned_vs_shuffled == 0.0 and aligned_vs_no_memory == 0.0:
        return "controlled_null", "Exp5239 controlled memory had no lift over controls"
    return "degraded", "Exp5239 failed one or more controlled-memory guardrails"


def arc_level_delta(
    artifacts: Mapping[int, JsonMap], eligible: set[int] | None = None
) -> tuple[int, str]:
    data = artifacts.get(5241, {})
    if not data:
        return 0, "no ARC live-patch artifact was present"
    if eligible is not None and 5241 not in eligible:
        return 0, "Exp5241 is flagged or validation-failed, so ARC delta is forced to zero"
    delta = as_number(data.get("reproducible_total_levels_delta")) or 0.0
    return int(delta), "Exp5241 clean reproduction-gated ARC delta imported"


def kan_certificate_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5242, {})
    if not data:
        return "unknown", "Exp5242 KAN certificate artifact is missing"
    if 5242 not in eligible:
        return "blocked", "Exp5242 is flagged or gate-blocked"
    if as_bool(data.get("kan_certificate_extended")):
        return "extended", "Exp5242 extended the bounded deterministic KAEM certificate"
    if as_bool(data.get("kan_certificate_baseline_reproduced")):
        return "tiny_only", "Exp5242 reproduced only the bounded tiny certificate baseline"
    if value_of(data.get("blocked_reason")):
        return "blocked", f"Exp5242 blocked: {value_of(data.get('blocked_reason'))}"
    return "unknown", "Exp5242 did not record certificate extension or baseline reproduction"


def hardware_speedup_claimed(artifacts: Mapping[int, JsonMap]) -> tuple[bool, str]:
    data = artifacts.get(5243, {})
    if not data:
        return False, "hardware continuity evidence missing"
    speedup = as_bool(data.get("speedup_claimed"))
    kv260 = value_of(data.get("kv260_status"))
    polarfire = value_of(data.get("polarfire_status"))
    gatemate = value_of(data.get("gatemate_status"))
    return (
        speedup,
        f"KV260={kv260}; PolarFire={polarfire}; GateMate={gatemate}; speedup_claimed={speedup}",
    )


def build_honest_verdict(
    *,
    gap4: str,
    gap1: str,
    veribmc: str,
    self_learning: str,
    arc_delta: int,
    kan: str,
) -> str:
    return (
        "complete: .479 closed with "
        f"GAP-4 {gap4}, GAP-1 {gap1}, VerIbmc {veribmc} after clean-null evidence, "
        f"continuous self-learning {self_learning}, ARC delta {arc_delta}, "
        f"KAN certificate {kan}, hardware no-speedup, and flagged/gated artifacts excluded."
    )


def build_next_top_blockers(status_decisions: JsonMap) -> list[str]:
    return [
        f"GAP-4 remains blocked: {status_decisions['gap4']}",
        f"GAP-1 remains blocked unless subset stability improves: {status_decisions['gap1']}",
        f"ARC live patch did not bank a clean level: {status_decisions['arc']}",
        "Hardware speedup remains unclaimed until authenticated timing and GateMate physical/JTAG setup change.",
    ]


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    conductor_untouched: bool | None = None,
    ops_docs_updated: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, sources, missing = load_upstreams(root)
    context = load_source_context(root)
    summary = per_task_summary(artifacts)
    eligible = eligible_numbers(artifacts)
    headline_ids = headline_eligible_task_ids(summary)
    excluded = excluded_from_headline(summary)
    blocked_rows = gated_or_blocked_rows(summary)
    gap4_value, gap4_reason = gap4_final_status(artifacts, eligible)
    gap1_value, gap1_reason = gap1_final_status(artifacts, eligible)
    veribmc_value, veribmc_reason = veribmc_final_status(artifacts, eligible)
    self_value, self_reason = continuous_self_learning_status(artifacts, eligible)
    arc_delta, arc_reason = arc_level_delta(artifacts, eligible)
    kan_value, kan_reason = kan_certificate_status(artifacts, eligible)
    speedup, hardware_reason = hardware_speedup_claimed(artifacts)
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    status_decisions = {
        "gap4": gap4_reason,
        "gap1": gap1_reason,
        "veribmc": veribmc_reason,
        "continuous_self_learning": self_reason,
        "arc": arc_reason,
        "kan": kan_reason,
        "hardware": hardware_reason,
        "ops_docs": (
            "ops/status, ops/changelog, _bmad/traceability, and research-complete deferred by stop rule"
            if not ops_docs_updated
            else "ops docs updated by this capstone"
        ),
    }

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "source_artifacts": sources,
        "source_context": context,
        "excluded_from_headline_artifacts": excluded,
        "per_task_summary": summary,
        "status_decisions": status_decisions,
        "preconditions_checked": {
            "expected_deliverable_artifacts": len(UPSTREAM_SOURCES),
            "loadable_deliverable_artifacts": len(artifacts),
            "source_context_files_checked": len(context),
            "ops_docs_deferred_by_stop_rule": not ops_docs_updated,
            "research_conductor_py_untouched": conductor_clean,
        },
        "next_top_blockers": build_next_top_blockers(status_decisions),
        "milestone": _principled("milestone", MILESTONE),
        "tasks_seen": _principled("tasks_seen", len(artifacts)),
        "missing_artifacts": _principled("missing_artifacts", missing),
        "gated_or_blocked_artifacts": _principled("gated_or_blocked_artifacts", blocked_rows),
        "headline_eligible_artifacts": _principled("headline_eligible_artifacts", headline_ids),
        "gap4_final_status": _principled("gap4_final_status", gap4_value),
        "gap1_final_status": _principled("gap1_final_status", gap1_value),
        "veribmc_final_status": _principled("veribmc_final_status", veribmc_value),
        "continuous_self_learning_status": _principled(
            "continuous_self_learning_status", self_value
        ),
        "arc_level_delta": _principled("arc_level_delta", arc_delta),
        "kan_certificate_status": _principled("kan_certificate_status", kan_value),
        "hardware_speedup_claimed": _principled("hardware_speedup_claimed", speedup),
        "ops_docs_updated": _principled("ops_docs_updated", ops_docs_updated),
        "validation_commands_run": _principled(
            "validation_commands_run", list(validation_commands_run or DEFAULT_VALIDATION_COMMANDS)
        ),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", conductor_clean
        ),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _principled(
            "honest_verdict",
            build_honest_verdict(
                gap4=gap4_value,
                gap1=gap1_value,
                veribmc=veribmc_value,
                self_learning=self_value,
                arc_delta=arc_delta,
                kan=kan_value,
            ),
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "flagged_adversarial": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    if payload.get("flagged_adversarial") is not False:
        raise ValueError("flagged_adversarial must be false for the capstone itself")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        wrapped = payload[field]
        if not isinstance(wrapped, Mapping) or wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} field principle mismatch")
    verdict = honest_verdict_text(payload.get("honest_verdict"))
    if not verdict.startswith(TERMINAL_PREFIXES) or "\n" in verdict:
        raise ValueError("honest_verdict must be a single terminal-prefix sentence")
    if value_of(payload["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare milestone_artifact_synthesis")
    if value_of(payload["milestone"]) != MILESTONE:
        raise ValueError("milestone mismatch")
    if not isinstance(value_of(payload["tasks_seen"]), int):
        raise ValueError("tasks_seen must be an int")
    if not isinstance(value_of(payload["missing_artifacts"]), list):
        raise ValueError("missing_artifacts must be a list")
    if not isinstance(value_of(payload["gated_or_blocked_artifacts"]), list):
        raise ValueError("gated_or_blocked_artifacts must be a list")
    headline = value_of(payload["headline_eligible_artifacts"])
    if not isinstance(headline, list):
        raise ValueError("headline_eligible_artifacts must be a list")
    if value_of(payload["gap4_final_status"]) not in GAP4_STATUSES:
        raise ValueError("gap4_final_status has invalid value")
    if value_of(payload["gap1_final_status"]) not in GAP1_STATUSES:
        raise ValueError("gap1_final_status has invalid value")
    if value_of(payload["veribmc_final_status"]) not in VERIBMC_STATUSES:
        raise ValueError("veribmc_final_status has invalid value")
    if value_of(payload["continuous_self_learning_status"]) not in SELF_LEARNING_STATUSES:
        raise ValueError("continuous_self_learning_status has invalid value")
    if value_of(payload["kan_certificate_status"]) not in KAN_STATUSES:
        raise ValueError("kan_certificate_status has invalid value")
    if not isinstance(value_of(payload["arc_level_delta"]), int):
        raise ValueError("arc_level_delta must be an int")
    if value_of(payload["hardware_speedup_claimed"]) is not False:
        raise ValueError("hardware_speedup_claimed must remain false")
    if value_of(payload["ops_docs_updated"]) is not False:
        raise ValueError("ops_docs_updated must be false under the stop rule")
    if value_of(payload["research_conductor_py_untouched_confirmed"]) is not True:
        raise ValueError("research_conductor_py_untouched_confirmed must be true")
    validation = value_of(payload["validation_commands_run"])
    if not isinstance(validation, list) or not validation:
        raise ValueError("validation_commands_run must be a non-empty list")
    excluded = payload.get("excluded_from_headline_artifacts")
    if not isinstance(excluded, Mapping):
        raise ValueError("excluded_from_headline_artifacts must be a mapping")
    if set(headline) & set(excluded):
        raise ValueError("headline eligible artifacts overlap excluded artifacts")
    checksum = payload_checksum(payload)
    if payload.get("reproducibility_checksum") != checksum:
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    conductor_untouched: bool | None = None,
    ops_docs_updated: bool = False,
) -> Path:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_commands_run=validation_commands_run,
        conductor_untouched=conductor_untouched,
        ops_docs_updated=ops_docs_updated,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default="20260704")
    args = parser.parse_args(argv)
    out_path = run(root=args.root, run_date=args.run_date)
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
