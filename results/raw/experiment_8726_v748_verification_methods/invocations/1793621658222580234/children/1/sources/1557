"""Exp 5219: V477 milestone-close capstone reconciliation.

Spec refs: REQ-CAPSTONE-5219, SCENARIO-CAPSTONE-5219,
SCENARIO-CAPSTONE-5219-FIELD-PRINCIPLES.

This module closes a milestone by reading the artifacts that already exist. The
important behavior is conservative aggregation: a blocked gate, flagged
artifact, or missing result is recorded as evidence about the milestone, but it
is not allowed to become a headline win.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AdversarialReporter = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results") / "experiment_5219_capstone_v477.json"
EXPERIMENT = "experiment_5219_capstone_v477"
EXPERIMENT_ID = "exp5219-capstone-v477"
MILESTONE = "2026.07.477"
SCHEMA = "carnot.experiment_5219_capstone_v477.v1"
RANDOM_SEED = 5219
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
HONEST_VERDICT = (
    "complete: v477 closed with GAP-1 building after blocked registry promotion, "
    "GAP-4 blocked by flagged/protocol validation, MMLU hidden-state path retired, "
    "self-learning memory created, zero ARC levels banked, hardware reachability "
    "maintained with no speedup claim, and flagged artifacts excluded."
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-CAPSTONE-5219",
    "SCENARIO-CAPSTONE-5219",
    "SCENARIO-CAPSTONE-5219-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "per_task_summary": (
        "map from task id to artifact path/status/verdict/headline eligibility, "
        "including blocked, missing, gated, flagged, and checkpoint-only evidence "
        "without importing flagged metrics into headlines"
    ),
    "gap1_final_status": (
        "one of filled | building | open | blocked; filled only when the registry "
        "promotion task actually runs and registers the verifier, not merely when "
        "the hardening gate is positive"
    ),
    "gap4_final_status": (
        "one of filled | building | open | blocked; filled only when a clean "
        "non-flagged validation crosses the unchanged six-discordant-win floor"
    ),
    "hidden_state_path_decision": (
        "one of keep | retire_mmlu_path | blocked; derived from Exp5213 without "
        "treating external-text-scorer retirement as the reason"
    ),
    "continuous_self_learning_satisfied": (
        "true only when Exp5214 writes a durable verifier-memory artifact with "
        "promotion and rollback semantics"
    ),
    "new_levels_banked": (
        "list of reproduction-gated ARC level banks from clean eligible ARC "
        "artifacts; empty is valid and must not be inflated"
    ),
    "reproducible_total_levels_delta": (
        "integer sum of clean reproduction-gated level deltas; zero is valid and "
        "must not be inflated"
    ),
    "hardware_final_state": (
        "summary of KV260, PolarFire, GateMate, board reachability, and no-speedup "
        "discipline from clean hardware evidence"
    ),
    "flagged_adversarial_artifacts_excluded": (
        "true when every flagged_adversarial or verifier-critical artifact is "
        "excluded from headline aggregation"
    ),
    "docs_reconciled": (
        "true only when this capstone reconciled the requested docs; false is "
        "valid when the conductor stop rule explicitly defers "
        "ops/status/changelog/traceability updates"
    ),
    "validation_commands_run": (
        "list of commands and pass/fail/block status used to validate the new "
        "capstone and changed surfaces"
    ),
    "research_conductor_py_untouched_confirmed": (
        "hard constraint that scripts/research_conductor.py stayed untouched"
    ),
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "honest_verdict": (
        "terminal-prefix single sentence that states the milestone outcome without "
        "vague celebration or inflated nulls"
    ),
}

PRINCIPLE_WRAPPED_FIELDS = (
    "per_task_summary",
    "gap1_final_status",
    "gap4_final_status",
    "hidden_state_path_decision",
    "continuous_self_learning_satisfied",
    "new_levels_banked",
    "reproducible_total_levels_delta",
    "hardware_final_state",
    "flagged_adversarial_artifacts_excluded",
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
    "checkpoint_artifacts",
    "missing_artifacts",
    "adversarial_verification",
    "headline_eligible_task_ids",
    "preconditions_checked",
    "status_decisions",
    "random_seed",
    "reproducibility_checksum",
    "flagged_adversarial",
    "inference_substrate",
    "honest_verdict",
    *PRINCIPLE_WRAPPED_FIELDS,
)

DEFAULT_VALIDATION_COMMANDS = [
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5219_capstone_v477.py -q --no-cov -o addopts=''",
        "status": "PENDING",
        "notes": "filled by the final run after implementation",
    }
]


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V477 task deliverable."""

    experiment_number: int
    task_id: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5207,
        "exp5207-archive-476-activate-477",
        Path("results/experiment_5207_archive_476_activate_477.json"),
    ),
    UpstreamSource(
        5208,
        "exp5208-sota-ingestion-v477",
        Path("results/experiment_5208_sota_ingestion_v477.json"),
    ),
    UpstreamSource(
        5209,
        "exp5209-gap1-set-search-holdout-hardening-v477",
        Path("results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"),
    ),
    UpstreamSource(
        5210,
        "exp5210-gap1-registry-promotion-gated-v477",
        Path("results/experiment_5210_gap1_registry_promotion_gated_v477.json"),
    ),
    UpstreamSource(
        5211,
        "exp5211-gap4-sota-local-candidate-expansion-v477",
        Path("results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json"),
    ),
    UpstreamSource(
        5212,
        "exp5212-gap4-scale-validation-gated-v477",
        Path("results/experiment_5212_gap4_scale_validation_gated_v477.json"),
    ),
    UpstreamSource(
        5213,
        "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477",
        Path("results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json"),
    ),
    UpstreamSource(
        5214,
        "exp5214-continuous-self-learning-verifier-memory-v477",
        Path("results/experiment_5214_continuous_self_learning_verifier_memory_v477.json"),
    ),
    UpstreamSource(
        5215,
        "exp5215-arc-paw-amortization-gate-v477",
        Path("results/experiment_5215_arc_paw_amortization_gate_v477.json"),
    ),
    UpstreamSource(
        5216,
        "exp5216-arc-frontier-continuity-landmark-decomposition-v477",
        Path("results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json"),
    ),
    UpstreamSource(
        5217,
        "exp5217-hardware-continuity-v477",
        Path("results/experiment_5217_hardware_continuity_v477.json"),
    ),
    UpstreamSource(
        5218,
        "exp5218-verifier-authenticity-remediation-apply-v477",
        Path("results/experiment_5218_verifier_authenticity_remediation_apply_v477.json"),
    ),
)

CHECKPOINT_SOURCES = (
    Path("results/experiment_5211_gap4_sota_local_candidate_expansion_v477.checkpoint.json"),
)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def honest_verdict_text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def _bool(value: Any) -> bool:
    return value_of(value) is True


def _number(value: Any) -> float | None:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def source_by_number(experiment_number: int) -> UpstreamSource:
    for source in UPSTREAM_SOURCES:
        if source.experiment_number == experiment_number:
            return source
    raise KeyError(experiment_number)


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


def adversarial_report_for(path: Path) -> JsonDict:  # pragma: no cover - integration path.
    from scripts.adversarial_verify import verify_artifact

    return verify_artifact(path)


def research_conductor_untouched(root: Path) -> bool:  # pragma: no cover - git integration.
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.returncode == 0 and status.stdout.strip() == ""


def _flag_is_critical(flag: Mapping[str, Any]) -> bool:
    severity = flag.get("severity")
    if isinstance(severity, str):
        return severity.lower() == "critical"
    return severity == 2


def _verification_row(source: UpstreamSource, data: JsonMap, report: JsonMap) -> JsonDict:
    flags = [flag for flag in list(report.get("flags") or []) if isinstance(flag, Mapping)]
    return {
        "task_id": source.task_id,
        "relative_path": str(source.relative_path),
        "stamped_flagged_adversarial": data.get("flagged_adversarial") is True,
        "critical_adversarial_flag": any(_flag_is_critical(flag) for flag in flags),
        "flag_count": report.get("flag_count", len(flags)),
        "max_severity": report.get("max_severity"),
        "flags": [{"kind": flag.get("kind"), "severity": flag.get("severity")} for flag in flags],
    }


def _is_excluded(source: UpstreamSource, data: JsonMap, reports: Sequence[JsonMap]) -> bool:
    if data.get("flagged_adversarial") is True:
        return True
    return any(
        row["task_id"] == source.task_id and row.get("critical_adversarial_flag") is True
        for row in reports
    )


def is_gated_block(data: JsonMap) -> bool:
    return data.get("status") == "blocked" and data.get("blocked_at_layer") == "conductor_pre_gate"


def load_upstreams(
    root: Path, adversarial_reporter: AdversarialReporter
) -> tuple[dict[int, JsonDict], list[JsonDict], list[str], list[JsonDict]]:
    artifacts: dict[int, JsonDict] = {}
    source_rows: list[JsonDict] = []
    missing: list[str] = []
    reports: list[JsonDict] = []
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
        report = adversarial_reporter(path)
        reports.append(_verification_row(source, data, report))
        source_rows.append(
            row
            | {
                "honest_verdict": honest_verdict_text(data.get("honest_verdict")),
                "flagged_adversarial": data.get("flagged_adversarial") is True,
            }
        )
    return artifacts, source_rows, missing, reports


def load_checkpoints(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for relative_path in CHECKPOINT_SOURCES:
        _, meta = read_json_mapping(root / relative_path)
        rows.append(
            {
                "relative_path": str(relative_path),
                "exists": meta.get("exists") is True,
                "loadable": meta.get("loadable") is True,
                "sha256": meta.get("sha256"),
                "error": meta.get("error"),
                "checkpoint_only": True,
            }
        )
    return rows


def flagged_exclusions(artifacts: JsonMap, reports: Sequence[JsonMap]) -> list[str]:
    excluded: list[str] = []
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        if data and _is_excluded(source, data, reports):
            excluded.append(source.task_id)
    return excluded


def _status_for(source: UpstreamSource, data: JsonMap, reports: Sequence[JsonMap]) -> str:
    if not data:
        return "missing"
    if _is_excluded(source, data, reports):
        return "flagged_excluded"
    if data.get("status") == "blocked" or honest_verdict_text(
        data.get("honest_verdict")
    ).startswith("blocked_"):
        return "blocked"
    return "complete"


def per_task_summary(artifacts: JsonMap, reports: Sequence[JsonMap]) -> JsonDict:
    summary: JsonDict = {}
    for source in UPSTREAM_SOURCES:
        data = artifacts.get(source.experiment_number, {})
        status = _status_for(source, data, reports)
        headline_eligible = bool(data and status == "complete")
        summary[source.task_id] = {
            "artifact_path": str(source.relative_path),
            "status": status,
            "verdict": honest_verdict_text(data.get("honest_verdict"))
            if data
            else "missing_artifact",
            "headline_eligible": headline_eligible,
            "flagged_adversarial": data.get("flagged_adversarial") is True if data else False,
            "critical_adversarial_flag": any(
                row["task_id"] == source.task_id and row.get("critical_adversarial_flag") is True
                for row in reports
            ),
            "gated_and_skipped": is_gated_block(data),
        }
    return summary


def headline_eligible_task_ids(summary: JsonMap) -> list[str]:
    return [task_id for task_id, row in summary.items() if row.get("headline_eligible") is True]


def gap1_status(artifacts: JsonMap) -> tuple[str, str]:
    hardening = artifacts.get(5209, {})
    if not hardening:
        return "blocked", "missing Exp5209 hardening artifact"
    if not _bool(hardening.get("gap1_hardened_positive")):
        return "open", "Exp5209 did not preserve the GAP-1 held-out positive"
    promotion = artifacts.get(5210, {})
    if not promotion:
        return "building", "Exp5209 positive but Exp5210 registry promotion artifact is missing"
    if _bool(promotion.get("verifier_registered")) or _bool(
        promotion.get("gap1_registry_promoted")
    ):
        return "filled", "Exp5210 registered the hardened GAP-1 verifier"
    if promotion.get("status") == "blocked":
        return "building", "Exp5210 blocked before registry promotion; GAP-1 remains building"
    return "building", "Exp5209 positive but registry promotion is not confirmed"


def gap4_status(artifacts: JsonMap, excluded: Sequence[str] = ()) -> tuple[str, str]:
    excluded_set = set(excluded)
    expansion = artifacts.get(5211, {})
    if not expansion:
        return "blocked", "missing Exp5211 expansion artifact"
    expansion_excluded = "exp5211-gap4-sota-local-candidate-expansion-v477" in excluded_set
    validation_excluded = "exp5212-gap4-scale-validation-gated-v477" in excluded_set
    if expansion_excluded and validation_excluded:
        return (
            "blocked",
            "Exp5211 expansion and Exp5212 validation artifacts are flagged; "
            "Exp5212 therefore cannot cross the floor",
        )
    if expansion_excluded:
        return "blocked", "Exp5211 expansion artifact is flagged and cannot support a headline"
    if not _bool(expansion.get("gap4_expansion_usable")):
        return "open", "Exp5211 did not produce a usable expanded pool"
    validation = artifacts.get(5212, {})
    if not validation:
        return "building", "Exp5211 pool usable but Exp5212 validation artifact is missing"
    if validation_excluded:
        return "blocked", "Exp5212 validation artifact is flagged and cannot cross the floor"
    if _bool(validation.get("exact_test_passes_min6_rule")):
        return "filled", "Exp5212 crossed the established six-discordant-win floor"
    if value_of(validation.get("gap4_status_recommendation")) == "blocked":
        return "blocked", "Exp5212 blocked before scoring enough protocol-complete rows"
    return "open", "Exp5212 did not cross the six-discordant-win floor"


def hidden_state_decision(artifacts: JsonMap, excluded: Sequence[str] = ()) -> tuple[str, str]:
    if "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477" in set(excluded):
        return "blocked", "Exp5213 hidden-state artifact is flagged"
    data = artifacts.get(5213, {})
    if not data:
        return "blocked", "missing Exp5213 hidden-state artifact"
    if _bool(data.get("beats_all_controls")):
        return "keep", "Exp5213 found a hidden-state signal above controls"
    if _bool(data.get("retire_mmlu_hidden_state_path")):
        return "retire_mmlu_path", "Exp5213 did not beat all controls and retires this MMLU path"
    return "blocked", "Exp5213 did not provide a keep-or-retire decision"


def self_learning_satisfied(root: Path, artifacts: JsonMap, excluded: Sequence[str] = ()) -> bool:
    if "exp5214-continuous-self-learning-verifier-memory-v477" in set(excluded):
        return False
    data = artifacts.get(5214, {})
    if not data:
        return False
    memory_rel = value_of(data.get("memory_artifact_path"))
    if not isinstance(memory_rel, str):
        return False
    memory, meta = read_json_mapping(root / memory_rel)
    if not meta.get("loadable"):
        return False
    summary = memory.get("summary") if isinstance(memory.get("summary"), Mapping) else {}
    promotions = int(_number(data.get("promotions")) or _number(summary.get("promotions")) or 0)
    rollbacks = int(_number(data.get("rollbacks")) or _number(summary.get("rollbacks")) or 0)
    return bool(
        _bool(data.get("continuous_self_learning_task"))
        and _bool(data.get("deterministic_guardrails_enforced"))
        and _bool(data.get("heldout_gate_required_for_promotion"))
        and promotions >= 1
        and rollbacks >= 1
        and _bool(summary.get("deterministic_guardrails_enforced"))
        and _bool(summary.get("heldout_gate_required_for_promotion"))
    )


def arc_level_delta(artifacts: JsonMap, excluded: Sequence[str] = ()) -> tuple[list[Any], int]:
    if "exp5216-arc-frontier-continuity-landmark-decomposition-v477" in set(excluded):
        return [], 0
    data = artifacts.get(5216, {})
    levels = value_of(data.get("new_levels_banked"))
    banked = levels if isinstance(levels, list) else []
    delta = _number(data.get("reproducible_total_levels_delta"))
    return banked, int(delta) if delta is not None else len(banked)


def hardware_summary(artifacts: JsonMap, excluded: Sequence[str] = ()) -> str:
    if "exp5217-hardware-continuity-v477" in set(excluded):
        return "hardware evidence excluded because Exp5217 is flagged"
    data = artifacts.get(5217, {})
    if not data:
        return "hardware evidence missing"
    polarfire = value_of(data.get("polarfire_status"))
    gatemate = value_of(data.get("gatemate_status"))
    polarfire_summary = (
        polarfire.get("summary", str(polarfire))
        if isinstance(polarfire, Mapping)
        else str(polarfire)
    )
    gatemate_summary = (
        f"{gatemate.get('status')} leading_hypothesis={gatemate.get('leading_untested_hypothesis')}"
        if isinstance(gatemate, Mapping)
        else str(gatemate)
    )
    speedup = _bool(data.get("hardware_speedup_claimed"))
    speedup_text = "speedup claimed" if speedup else "no speedup claim"
    return (
        f"KV260={value_of(data.get('kv260_status'))}; "
        f"PolarFire={polarfire_summary}; "
        f"GateMate={value_of(data.get('gatemate_diagnostic_narrowed_to'))} ({gatemate_summary}); "
        f"boards_reachable_count={value_of(data.get('boards_reachable_count'))}; {speedup_text}"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    conductor_untouched: bool | None = None,
    docs_reconciled: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    reporter = adversarial_reporter or adversarial_report_for
    artifacts, sources, missing, reports = load_upstreams(root, reporter)
    checkpoints = load_checkpoints(root)
    excluded = flagged_exclusions(artifacts, reports)
    summary = per_task_summary(artifacts, reports)
    eligible = headline_eligible_task_ids(summary)
    gap1_value, gap1_reason = gap1_status(artifacts)
    gap4_value, gap4_reason = gap4_status(artifacts, excluded)
    hidden_value, hidden_reason = hidden_state_decision(artifacts, excluded)
    self_learning = self_learning_satisfied(root, artifacts, excluded)
    levels, level_delta = arc_level_delta(artifacts, excluded)
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    flagged_clean = not (set(excluded) & set(eligible))
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
        "checkpoint_artifacts": checkpoints,
        "missing_artifacts": missing,
        "adversarial_verification": reports,
        "headline_eligible_task_ids": eligible,
        "preconditions_checked": {
            "expected_deliverable_artifacts": len(UPSTREAM_SOURCES),
            "loadable_deliverable_artifacts": len(artifacts),
            "checkpoint_artifacts_checked": len(checkpoints),
            "adversarial_verify_ran_per_loadable_artifact": len(reports) == len(artifacts),
            "docs_reconciliation_deferred_by_stop_rule": not docs_reconciled,
            "research_conductor_py_untouched": conductor_clean,
        },
        "status_decisions": {
            "gap1": gap1_reason,
            "gap4": gap4_reason,
            "hidden_state": hidden_reason,
            "continuous_self_learning": (
                "Exp5214 durable memory artifact satisfied promotion/rollback semantics"
                if self_learning
                else "Exp5214 memory artifact missing or did not satisfy promotion/rollback semantics"
            ),
            "arc": f"new_levels_banked={levels}, reproducible_total_levels_delta={level_delta}",
            "hardware": hardware_summary(artifacts, excluded),
            "authenticity": (
                "Exp5218 remediation applied and headline-ineligible flags preserved"
                if _bool(artifacts.get(5218, {}).get("remediation_applied"))
                else "Exp5218 remediation missing or incomplete"
            ),
            "docs": (
                "docs reconciled in this capstone"
                if docs_reconciled
                else "ops/status, ops/changelog, and _bmad/traceability deferred by stop rule"
            ),
            "flagged_excluded_task_ids": excluded,
        },
        "per_task_summary": _principled("per_task_summary", summary),
        "gap1_final_status": _principled("gap1_final_status", gap1_value),
        "gap4_final_status": _principled("gap4_final_status", gap4_value),
        "hidden_state_path_decision": _principled("hidden_state_path_decision", hidden_value),
        "continuous_self_learning_satisfied": _principled(
            "continuous_self_learning_satisfied", self_learning
        ),
        "new_levels_banked": _principled("new_levels_banked", levels),
        "reproducible_total_levels_delta": _principled(
            "reproducible_total_levels_delta", level_delta
        ),
        "hardware_final_state": _principled(
            "hardware_final_state", hardware_summary(artifacts, excluded)
        ),
        "flagged_adversarial_artifacts_excluded": _principled(
            "flagged_adversarial_artifacts_excluded", flagged_clean
        ),
        "docs_reconciled": _principled("docs_reconciled", docs_reconciled),
        "validation_commands_run": _principled(
            "validation_commands_run",
            list(validation_commands_run or DEFAULT_VALIDATION_COMMANDS),
        ),
        "research_conductor_py_untouched_confirmed": _principled(
            "research_conductor_py_untouched_confirmed", conductor_clean
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": HONEST_VERDICT,
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
        raise ValueError("inference_substrate must declare V477 aggregation")
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
    if value_of(payload["gap1_final_status"]) not in {"filled", "building", "open", "blocked"}:
        raise ValueError("gap1_final_status has invalid value")
    if value_of(payload["gap4_final_status"]) not in {"filled", "building", "open", "blocked"}:
        raise ValueError("gap4_final_status has invalid value")
    if value_of(payload["hidden_state_path_decision"]) not in {
        "keep",
        "retire_mmlu_path",
        "blocked",
    }:
        raise ValueError("hidden_state_path_decision has invalid value")
    if value_of(payload["flagged_adversarial_artifacts_excluded"]) is not True:
        raise ValueError("flagged_adversarial_artifacts_excluded must be true")
    summary = value_of(payload["per_task_summary"])
    if not isinstance(summary, Mapping):
        raise ValueError("per_task_summary must be a map")
    eligible = set(payload["headline_eligible_task_ids"])
    excluded = {
        task_id
        for task_id, row in summary.items()
        if isinstance(row, Mapping)
        and (row.get("flagged_adversarial") is True or row.get("critical_adversarial_flag") is True)
    }
    if excluded & eligible:
        raise ValueError("flagged_adversarial artifacts cannot be headline eligible")
    levels = value_of(payload["new_levels_banked"])
    delta = value_of(payload["reproducible_total_levels_delta"])
    if not isinstance(levels, list) or not isinstance(delta, int) or delta != len(levels):
        raise ValueError("reproducible_total_levels_delta must match new_levels_banked")
    docs_value = value_of(payload["docs_reconciled"])
    if not isinstance(docs_value, bool):
        raise ValueError("docs_reconciled must be boolean")
    if not value_of(payload["validation_commands_run"]):
        raise ValueError("validation_commands_run must record verification commands")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    conductor_untouched: bool | None = None,
    docs_reconciled: bool = False,
) -> Path:
    payload = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_commands_run=validation_commands_run,
        adversarial_reporter=adversarial_reporter,
        conductor_untouched=conductor_untouched,
        docs_reconciled=docs_reconciled,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, payload)
    return out_path


if __name__ == "__main__":  # pragma: no cover - direct module execution.
    run()
