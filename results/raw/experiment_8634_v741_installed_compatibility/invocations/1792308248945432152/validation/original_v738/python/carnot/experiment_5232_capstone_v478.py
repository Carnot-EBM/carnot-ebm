"""Exp 5232: V478 milestone-close capstone reconciliation.

Spec refs: REQ-CAPSTONE-5232, SCENARIO-CAPSTONE-5232,
SCENARIO-CAPSTONE-5232-FIELD-PRINCIPLES.

This module closes the milestone by aggregating existing artifacts. The
important behavior is conservative: a flagged, gate-blocked, or bounded
artifact is still recorded as evidence, but it cannot become a headline claim.
That distinction matters because several V478 artifacts contain useful audit
facts while also carrying adversarial or gate blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results") / "experiment_5232_capstone_v478.json"
EXPERIMENT = "experiment_5232_capstone_v478"
EXPERIMENT_ID = "exp5232-capstone-v478"
MILESTONE = "2026.07.478"
SCHEMA = "carnot.experiment_5232_capstone_v478.v1"
RANDOM_SEED = 5232
INFERENCE_SUBSTRATE = "aggregation_from_milestone_artifacts"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

TYPED_MEMORY_HEADS = ("constraints", "provenance", "failures", "skills_rubrics")
ARC_RUBRIC_FIELDS = (
    "skill_selection",
    "skill_following",
    "skill_composition",
    "reflection_retry_quality",
    "provenance_validity",
)

SPEC_REFS = [
    "REQ-CAPSTONE-5232",
    "SCENARIO-CAPSTONE-5232",
    "SCENARIO-CAPSTONE-5232-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "per_task_summary": (
        "map from task id to artifact path/status/verdict/headline eligibility, "
        "including flagged, blocked, gated, and bounded evidence without importing "
        "excluded metrics into headline claims"
    ),
    "gap1_final_status": (
        "one of promoted | blocked | unchanged; promoted only when a clean registry "
        "promotion artifact registers the verifier"
    ),
    "gap4_final_status": (
        "one of clean_positive | clean_null | blocked | unchanged; clean_* only when "
        "the validation artifact is not flagged and passes adversarial checks"
    ),
    "solver_feedback_status": (
        "one of positive | null | blocked | not_run; positive/null only when the "
        "VerIbmc artifact survives adversarial checks"
    ),
    "continuous_self_learning_satisfied": (
        "true only when typed memory, retention, promotion/rollback, and a "
        "consumer-ready handoff are present"
    ),
    "arc_new_levels_banked": (
        "list of clean reproduction-gated ARC level banks; empty is valid and must not be inflated"
    ),
    "arc_reproducible_total_levels_delta": (
        "integer delta from clean reproduction-gated ARC evidence only"
    ),
    "kan_certificate_status": (
        "one of produced | blocked | not_run; produced only for the bounded tiny "
        "KAEM certificate supported by Exp5230"
    ),
    "hardware_status": (
        "one-line summary of KV260, PolarFire, GateMate, p-bit plan, and no-speedup discipline"
    ),
    "flagged_artifacts_excluded": (
        "true when every flagged, critical-corrigendum, or gate-blocked artifact is "
        "excluded from headline aggregation"
    ),
    "docs_reconciled": (
        "false when the operator stop rule delegates ops/status/changelog/"
        "traceability/reference reconciliation to a later step"
    ),
    "validation_commands_run": (
        "list of commands and pass/fail/block status used to validate the capstone "
        "and changed surfaces"
    ),
    "research_conductor_py_untouched_confirmed": (
        "hard constraint that scripts/research_conductor.py stayed untouched"
    ),
    "inference_substrate": INFERENCE_SUBSTRATE,
    "honest_verdict": (
        "terminal-prefix single sentence that states the true .478 outcome without "
        "laundering flagged or gated artifacts"
    ),
}

PRINCIPLE_WRAPPED_FIELDS = (
    "per_task_summary",
    "gap1_final_status",
    "gap4_final_status",
    "solver_feedback_status",
    "continuous_self_learning_satisfied",
    "arc_new_levels_banked",
    "arc_reproducible_total_levels_delta",
    "kan_certificate_status",
    "hardware_status",
    "flagged_artifacts_excluded",
    "docs_reconciled",
    "validation_commands_run",
    "research_conductor_py_untouched_confirmed",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "source_artifacts",
    "source_context",
    "missing_artifacts",
    "excluded_from_headline_task_ids",
    "headline_eligible_task_ids",
    "status_decisions",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "flagged_adversarial",
    "inference_substrate",
    "honest_verdict",
    *PRINCIPLE_WRAPPED_FIELDS,
)

DEFAULT_VALIDATION_COMMANDS = [
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5232_capstone_v478.py -q --no-cov -o addopts=''",
        "status": "PENDING",
        "notes": "filled by the final run after implementation",
    }
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V478 task deliverable."""

    experiment_number: int
    task_id: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5220,
        "exp5220-archive-477-activate-478",
        Path("results/experiment_5220_archive_477_activate_478.json"),
    ),
    UpstreamSource(
        5221,
        "exp5221-sota-ingestion-v478",
        Path("results/experiment_5221_sota_ingestion_v478.json"),
    ),
    UpstreamSource(
        5222,
        "exp5222-gap1-gate-field-and-registry-promotion-v478",
        Path("results/experiment_5222_gap1_gate_field_registry_promotion_v478.json"),
    ),
    UpstreamSource(
        5223,
        "exp5223-gap4-flagged-pool-authenticity-audit-v478",
        Path("results/experiment_5223_gap4_flagged_pool_authenticity_audit_v478.json"),
    ),
    UpstreamSource(
        5224,
        "exp5224-gap4-canonical-pool-builder-v478",
        Path("results/experiment_5224_gap4_canonical_pool_builder_v478.json"),
    ),
    UpstreamSource(
        5225,
        "exp5225-gap4-clean-scale-validation-gated-v478",
        Path("results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"),
    ),
    UpstreamSource(
        5226,
        "exp5226-veribmc-local-solver-feedback-pilot-v478",
        Path("results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json"),
    ),
    UpstreamSource(
        5227,
        "exp5227-continuous-self-learning-multihead-memory-v478",
        Path("results/experiment_5227_continuous_self_learning_multihead_memory_v478.json"),
    ),
    UpstreamSource(
        5228,
        "exp5228-arc-provenance-skill-rubric-gate-v478",
        Path("results/experiment_5228_arc_provenance_skill_rubric_gate_v478.json"),
    ),
    UpstreamSource(
        5229,
        "exp5229-arc-gated-live-levelup-from-rubric-v478",
        Path("results/experiment_5229_arc_gated_live_levelup_from_rubric_v478.json"),
    ),
    UpstreamSource(
        5230,
        "exp5230-kan-milp-verifier-certificate-v478",
        Path("results/experiment_5230_kan_milp_verifier_certificate_v478.json"),
    ),
    UpstreamSource(
        5231,
        "exp5231-hardware-continuity-pbit-boundary-v478",
        Path("results/experiment_5231_hardware_continuity_pbit_boundary_v478.json"),
    ),
)

SOURCE_CONTEXT_PATHS = (
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/conductor-log.md"),
    Path("research-references.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/verifier_gaps.md"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("_bmad/traceability.md"),
)


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


def is_gate_blocked(data: JsonMap) -> bool:
    verdict = honest_verdict_text(data.get("honest_verdict"))
    return bool(
        data.get("status") == "blocked"
        or data.get("blocked_at_layer") == "conductor_pre_gate"
        or verdict.startswith("blocked")
    )


def exclusion_reasons(data: JsonMap) -> list[str]:
    reasons: list[str] = []
    if data.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial")
    if has_critical_corrigendum(data):
        reasons.append("critical_corrigendum_pending")
    if value_of(data.get("adversarial_verify_passed")) is False:
        reasons.append("adversarial_verify_failed")
    if is_gate_blocked(data):
        reasons.append("gate_blocked")
    return reasons


def is_excluded(data: JsonMap) -> bool:
    return bool(exclusion_reasons(data))


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
                "gate_blocked": is_gate_blocked(data),
                "excluded_from_headline": bool(reasons),
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
    if is_excluded(data):
        return "excluded"
    return "complete"


def per_task_summary(artifacts: Mapping[int, JsonMap]) -> JsonDict:
    summary: JsonDict = {}
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        reasons = exclusion_reasons(data) if data else []
        summary[source.task_id] = {
            "artifact_path": str(source.relative_path),
            "status": _status_for(data),
            "verdict": honest_verdict_text(data.get("honest_verdict"))
            if data
            else "missing_artifact",
            "headline_eligible": bool(data and not reasons),
            "exclusion_reasons": reasons,
            "flagged_adversarial": data.get("flagged_adversarial") is True if data else False,
            "critical_corrigendum": has_critical_corrigendum(data) if data else False,
            "gate_blocked": is_gate_blocked(data) if data else False,
        }
    return summary


def headline_eligible_task_ids(summary: JsonMap) -> list[str]:
    return [task_id for task_id, row in summary.items() if row.get("headline_eligible") is True]


def excluded_task_ids(summary: JsonMap) -> list[str]:
    return [
        task_id
        for task_id, row in summary.items()
        if row.get("status") in {"excluded", "gate_blocked"}
    ]


def eligible_numbers(artifacts: Mapping[int, JsonMap]) -> set[int]:
    return {number for number, data in artifacts.items() if not is_excluded(data)}


def excluded_numbers(artifacts: Mapping[int, JsonMap]) -> set[int]:
    return {number for number, data in artifacts.items() if is_excluded(data)}


def gap1_final_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5222, {})
    if not data:
        return "unchanged", "Exp5222 GAP-1 registry decision artifact is missing"
    if 5222 not in eligible:
        return "blocked", "Exp5222 exists but is flagged or gate-blocked"
    if as_bool(data.get("gap1_registry_promoted")):
        return "promoted", "Exp5222 promoted the GAP-1 registry verifier"
    decision = value_of(data.get("gap1_registry_decision"))
    if isinstance(decision, str) and decision.startswith("blocked"):
        return "blocked", f"Exp5222 blocked registry promotion: {decision}"
    return "unchanged", "Exp5222 did not promote or explicitly block GAP-1"


def gap4_final_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5225, {})
    if not data:
        return "unchanged", "Exp5225 GAP-4 validation artifact is missing"
    if 5225 not in eligible:
        return "blocked", "Exp5225 metrics are excluded because the artifact is flagged or gated"
    effect = value_of(data.get("effect_direction"))
    if effect == "positive":
        return "clean_positive", "Exp5225 reported a clean positive validation"
    if effect == "null":
        return "clean_null", "Exp5225 reported a clean null validation"
    return "blocked", f"Exp5225 did not provide a clean positive/null decision: {effect}"


def solver_feedback_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5226, {})
    if not data:
        return "not_run", "Exp5226 VerIbmc solver-feedback artifact is missing"
    if 5226 not in eligible:
        return (
            "blocked",
            "Exp5226 solver-feedback outcome is excluded because the artifact is flagged",
        )
    uplift = as_number(data.get("solver_feedback_uplift")) or 0.0
    if uplift > 0.0:
        return "positive", f"Exp5226 solver feedback uplift is positive: {uplift}"
    return "null", "Exp5226 solver feedback completed with no uplift over baselines"


def continuous_self_learning_satisfied(
    root: Path, artifacts: Mapping[int, JsonMap], eligible: set[int]
) -> tuple[bool, str]:
    data = artifacts.get(5227, {})
    if not data or 5227 not in eligible:
        return False, "Exp5227 typed-memory artifact missing or excluded"
    heads = value_of(data.get("typed_memory_heads"))
    consumer_rel = value_of(data.get("consumer_ready_path"))
    memory_rel = value_of(data.get("memory_artifact_path"))
    if not isinstance(consumer_rel, str) or not isinstance(memory_rel, str):
        return False, "Exp5227 did not record typed-memory and consumer paths"
    memory, memory_meta = read_json_mapping(root / memory_rel)
    consumer, consumer_meta = read_json_mapping(root / consumer_rel)
    memory_heads = memory.get("heads") if isinstance(memory.get("heads"), list) else []
    passed = bool(
        as_bool(data.get("continuous_self_learning_task"))
        and list(heads or []) == list(TYPED_MEMORY_HEADS)
        and memory_meta.get("loadable") is True
        and consumer_meta.get("loadable") is True
        and list(memory_heads) == list(TYPED_MEMORY_HEADS)
        and as_bool(data.get("retention_check_passed"))
        and as_bool(consumer.get("consumer_ready"))
        and not as_bool(data.get("broad_self_distillation_used"))
        and int(as_number(data.get("memory_entries_written")) or 0) >= 1
        and int(as_number(data.get("promotions")) or 0) >= 1
        and int(as_number(data.get("rollbacks")) or 0) >= 1
    )
    if passed:
        return (
            True,
            "Exp5227 wrote typed memory, retention passed, and Exp5228 consumer file is ready",
        )
    return False, "Exp5227 typed-memory, retention, or consumer-readiness checks did not pass"


def arc_status(artifacts: Mapping[int, JsonMap]) -> tuple[list[Any], int, str]:
    levelup = artifacts.get(5229, {})
    if levelup and not is_gate_blocked(levelup):
        levels = value_of(levelup.get("new_levels_banked"))
        banked = levels if isinstance(levels, list) else []
        delta = as_number(levelup.get("reproducible_total_levels_delta"))
        return banked, int(delta) if delta is not None else len(banked), "Exp5229 ran"
    rubric = artifacts.get(5228, {})
    known_nulls = rubric.get("known_arc_nulls_retained")
    if isinstance(known_nulls, Mapping):
        levels = known_nulls.get("new_levels_banked")
        banked = levels if isinstance(levels, list) else []
        delta = as_number(known_nulls.get("reproducible_total_levels_delta"))
        return (
            banked,
            int(delta) if delta is not None else len(banked),
            "Exp5228 retained zero-delta ARC memory and Exp5229 was gate-blocked",
        )
    return [], 0, "no ARC level-up artifact or rubric null evidence"


def kan_certificate_status(artifacts: Mapping[int, JsonMap], eligible: set[int]) -> tuple[str, str]:
    data = artifacts.get(5230, {})
    if not data:
        return "not_run", "Exp5230 KAN certificate artifact is missing"
    if 5230 not in eligible:
        return "blocked", "Exp5230 KAN certificate artifact is excluded"
    if as_bool(data.get("kan_certificate_produced")):
        return "produced", "Exp5230 produced the bounded tiny KAEM PWA/MILP certificate"
    return "blocked", "Exp5230 did not produce a certificate"


def hardware_status(
    artifacts: Mapping[int, JsonMap], excluded_numbers: set[int] | None = None
) -> str:
    if excluded_numbers and 5231 in excluded_numbers:
        return "hardware evidence excluded because Exp5231 is flagged or gate-blocked"
    data = artifacts.get(5231, {})
    if not data:
        return "hardware evidence missing"
    kv260 = "reachable" if as_bool(data.get("kv260_reachable")) else "unreachable"
    polarfire = "reachable" if as_bool(data.get("polarfire_reachable")) else "unreachable"
    speedup = "speedup claimed" if as_bool(data.get("speedup_claimed")) else "no speedup claim"
    return (
        f"KV260={kv260} via {value_of(data.get('kv260_check_method'))}; "
        f"PolarFire={polarfire}; "
        f"GateMate={value_of(data.get('gatemate_status'))} "
        f"IDCODE={value_of(data.get('gatemate_idcode_raw'))}; "
        f"p-bit plan={value_of(data.get('pbit_boundary_plan_path'))}; {speedup}"
    )


def build_honest_verdict(
    gap1: str,
    gap4: str,
    solver: str,
    self_learning: bool,
    arc_delta: int,
    kan: str,
) -> str:
    self_text = "typed memory satisfied" if self_learning else "typed memory not satisfied"
    return (
        "complete: v478 reconciled with "
        f"GAP-1 {gap1}, GAP-4 {gap4}, VerIbmc solver-feedback {solver}, "
        f"{self_text}, ARC delta {arc_delta}, KAN certificate {kan}, "
        "hardware continuity/no-speedup recorded, and flagged/gated artifacts excluded."
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    conductor_untouched: bool | None = None,
    docs_reconciled: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, sources, missing = load_upstreams(root)
    context = load_source_context(root)
    summary = per_task_summary(artifacts)
    eligible = eligible_numbers(artifacts)
    excluded = excluded_numbers(artifacts)
    eligible_ids = headline_eligible_task_ids(summary)
    excluded_ids = excluded_task_ids(summary)
    gap1_value, gap1_reason = gap1_final_status(artifacts, eligible)
    gap4_value, gap4_reason = gap4_final_status(artifacts, eligible)
    solver_value, solver_reason = solver_feedback_status(artifacts, eligible)
    self_learning_value, self_learning_reason = continuous_self_learning_satisfied(
        root, artifacts, eligible
    )
    arc_levels, arc_delta, arc_reason = arc_status(artifacts)
    kan_value, kan_reason = kan_certificate_status(artifacts, eligible)
    hardware_value = hardware_status(artifacts, excluded)
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    flagged_clean = not (set(excluded_ids) & set(eligible_ids))

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "source_artifacts": sources,
        "source_context": context,
        "missing_artifacts": missing,
        "excluded_from_headline_task_ids": excluded_ids,
        "headline_eligible_task_ids": eligible_ids,
        "status_decisions": {
            "gap1": gap1_reason,
            "gap4": gap4_reason,
            "solver_feedback": solver_reason,
            "continuous_self_learning": self_learning_reason,
            "arc": arc_reason,
            "kan": kan_reason,
            "hardware": hardware_value,
            "docs": (
                "docs reconciled in this capstone"
                if docs_reconciled
                else "ops/status, ops/changelog, _bmad/traceability, and research-references deferred by stop rule"
            ),
            "excluded_from_headline_task_ids": excluded_ids,
        },
        "preconditions_checked": {
            "expected_deliverable_artifacts": len(UPSTREAM_SOURCES),
            "loadable_deliverable_artifacts": len(artifacts),
            "source_context_files_checked": len(context),
            "docs_reconciliation_deferred_by_stop_rule": not docs_reconciled,
            "research_conductor_py_untouched": conductor_clean,
        },
        "per_task_summary": _principled("per_task_summary", summary),
        "gap1_final_status": _principled("gap1_final_status", gap1_value),
        "gap4_final_status": _principled("gap4_final_status", gap4_value),
        "solver_feedback_status": _principled("solver_feedback_status", solver_value),
        "continuous_self_learning_satisfied": _principled(
            "continuous_self_learning_satisfied", self_learning_value
        ),
        "arc_new_levels_banked": _principled("arc_new_levels_banked", arc_levels),
        "arc_reproducible_total_levels_delta": _principled(
            "arc_reproducible_total_levels_delta", arc_delta
        ),
        "kan_certificate_status": _principled("kan_certificate_status", kan_value),
        "hardware_status": _principled("hardware_status", hardware_value),
        "flagged_artifacts_excluded": _principled("flagged_artifacts_excluded", flagged_clean),
        "docs_reconciled": _principled("docs_reconciled", docs_reconciled),
        "validation_commands_run": _principled(
            "validation_commands_run",
            list(validation_commands_run or DEFAULT_VALIDATION_COMMANDS),
        ),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", conductor_clean
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": build_honest_verdict(
            gap1_value, gap4_value, solver_value, self_learning_value, arc_delta, kan_value
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
    verdict = honest_verdict_text(payload.get("honest_verdict"))
    if not verdict.startswith(TERMINAL_PREFIXES) or "\n" in verdict:
        raise ValueError("honest_verdict must be a single terminal-prefix sentence")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare V478 milestone aggregation")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    if payload.get("flagged_adversarial") is not False:
        raise ValueError("flagged_adversarial must be false for the capstone itself")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        wrapped = payload[field]
        if not isinstance(wrapped, Mapping) or wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} field principle mismatch")
    if value_of(payload["research_conductor_py_untouched_confirmed"]) is not True:
        raise ValueError("research_conductor_py_untouched_confirmed must be true")
    if value_of(payload["gap1_final_status"]) not in {"promoted", "blocked", "unchanged"}:
        raise ValueError("gap1_final_status has invalid value")
    if value_of(payload["gap4_final_status"]) not in {
        "clean_positive",
        "clean_null",
        "blocked",
        "unchanged",
    }:
        raise ValueError("gap4_final_status has invalid value")
    if value_of(payload["solver_feedback_status"]) not in {
        "positive",
        "null",
        "blocked",
        "not_run",
    }:
        raise ValueError("solver_feedback_status has invalid value")
    if value_of(payload["kan_certificate_status"]) not in {"produced", "blocked", "not_run"}:
        raise ValueError("kan_certificate_status has invalid value")
    if value_of(payload["flagged_artifacts_excluded"]) is not True:
        raise ValueError("flagged_artifacts_excluded must be true")
    summary = value_of(payload["per_task_summary"])
    if not isinstance(summary, Mapping):
        raise ValueError("per_task_summary must be a map")
    eligible = set(payload["headline_eligible_task_ids"])
    excluded = {
        task_id
        for task_id, row in summary.items()
        if isinstance(row, Mapping) and row.get("status") in {"excluded", "gate_blocked"}
    }
    if excluded & eligible:
        raise ValueError("flagged or gate-blocked artifacts cannot be headline eligible")
    levels = value_of(payload["arc_new_levels_banked"])
    delta = value_of(payload["arc_reproducible_total_levels_delta"])
    if not isinstance(levels, list) or not isinstance(delta, int) or delta != len(levels):
        raise ValueError("arc_reproducible_total_levels_delta must match banked levels")
    docs_value = value_of(payload["docs_reconciled"])
    if not isinstance(docs_value, bool):
        raise ValueError("docs_reconciled must be boolean")
    if not value_of(payload["validation_commands_run"]):
        raise ValueError("validation_commands_run must record verification commands")
    if not isinstance(value_of(payload["continuous_self_learning_satisfied"]), bool):
        raise ValueError("continuous_self_learning_satisfied must be boolean")
    if not isinstance(value_of(payload["hardware_status"]), str):
        raise ValueError("hardware_status must be a one-line string")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    conductor_untouched: bool | None = None,
    docs_reconciled: bool = False,
) -> Path:
    payload = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_commands_run=validation_commands_run,
        conductor_untouched=conductor_untouched,
        docs_reconciled=docs_reconciled,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, payload)
    return out_path


if __name__ == "__main__":  # pragma: no cover - direct module execution.
    run()
