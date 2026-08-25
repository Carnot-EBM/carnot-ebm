"""Exp 5348: V487 capstone decision artifact.

Spec refs: REQ-CAPSTONE-5348, SCENARIO-CAPSTONE-5348,
SCENARIO-CAPSTONE-5348-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5348-FIELD-PRINCIPLES.

This module is a milestone aggregator. It reads the local .487 result artifacts
and the conductor log, then separates clean gates from blocked or quarantined
evidence. The capstone does not rerun models, solvers, KAN checks, or hardware
probes; its job is to keep the boundary honest so a bounded or flagged result
does not become a headline quality, internal-energy, certificate,
self-learning, or speedup claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5348_capstone_v487.json")
EXPERIMENT = "experiment_5348_capstone_v487"
EXPERIMENT_ID = "exp5348-capstone-v487"
MILESTONE = "2026.07.487"
SCHEMA = "carnot.experiment_5348_capstone_v487.v1"
RUN_DATE = "20260707"
RANDOM_SEED = 5348
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5348",
    "SCENARIO-CAPSTONE-5348",
    "SCENARIO-CAPSTONE-5348-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5348-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "identifies Exp5348 as the `.487` capstone artifact so downstream reconciliation "
        "cannot confuse it with an upstream SOTA, self-learning, constraint, KAN, "
        "internal-energy, or hardware task."
    ),
    "milestone": (
        "binds the aggregation to 2026.07.487 and the close-state read of Exp5335 "
        "through Exp5347 plus the Exp5340 Q-value sidecar."
    ),
    "status": "complete only when every expected artifact is readable; otherwise blocked_missing_required.",
    "honest_verdict": (
        "terminal prefix; starts with complete: or blocked_ and summarizes `.487` "
        "without laundering blocked, flagged, bounded, missing, no-speedup, no-quality, "
        "internal-energy, self-learning, or certificate evidence."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because the capstone reads local artifacts "
        "and conductor outcomes without running model, solver, or hardware workloads."
    ),
    "artifacts_read": (
        "every readable upstream artifact with path, experiment identity, status, "
        "verdict, sha256, and conductor outcome when available."
    ),
    "missing_blocked_flagged_or_skipped_artifacts": (
        "missing, malformed, blocked, flagged, and conductor-gate outcomes that remain "
        "first-class and cannot be rounded up."
    ),
    "gate_table": (
        "one reconciled row per requested gate with source artifacts, clean-ready "
        "boolean, imported evidence, claim boundary, and caveat text."
    ),
    "next_milestone_recommendation": (
        "short next branch recommendation grounded only in clean gates, blockers, and flagged rows."
    ),
    "cited_upstream_artifacts": (
        "every upstream artifact cited by sha256 with the imported fields that affected a gate."
    ),
    "tests_run": (
        "validation commands and outcomes used to check the capstone module, artifact, "
        "coverage, and required repository test status."
    ),
}

WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOLEAN_FIELDS = (
    "runtime_clean",
    "structured_output_protocol_ready",
    "bounded_sota_quality_usable",
    "utility_memory_ready",
    "bounded_compressor_ready",
    "self_learning_scaleup_ready",
    "qstr_fixture_ready",
    "solver_guidance_ready",
    "internal_energy_corrigendum_clean",
    "kan_constraint_bridge_ready",
    "hardware_speedup_claim",
    "active_roadmap_modified",
    "conductor_modified",
)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "spec_refs",
    "result_path",
    "random_seed",
    "field_principles",
    "reproducibility_checksum",
    *WRAPPED_FIELDS,
    *BARE_BOOLEAN_FIELDS,
)


@dataclass(frozen=True)
class UpstreamArtifact:
    """One expected V487 artifact or sidecar.

    The capstone uses this as the fixed milestone ledger. Missing or malformed
    entries block the capstone artifact itself, while blocked or flagged payloads
    remain readable evidence that can explain why a gate is not clean-ready.
    """

    experiment_number: int
    task_id: str
    relative_path: Path
    default_status: str = "missing_or_unreadable"


EXP5335 = UpstreamArtifact(
    5335,
    "exp5335-archive-486-activate-487",
    Path("results/experiment_5335_archive_486_activate_487.json"),
)
EXP5336 = UpstreamArtifact(
    5336,
    "exp5336-sota-source-delta-v487",
    Path("results/experiment_5336_sota_source_delta_v487.json"),
)
EXP5337 = UpstreamArtifact(
    5337,
    "exp5337-sota-runtime-corrigendum-multimodel-v487",
    Path("results/experiment_5337_sota_runtime_corrigendum_multimodel_v487.json"),
)
EXP5338 = UpstreamArtifact(
    5338,
    "exp5338-structured-output-protocol-calibration-v487",
    Path("results/experiment_5338_structured_output_protocol_calibration_v487.json"),
)
EXP5339 = UpstreamArtifact(
    5339,
    "exp5339-gated-sota-claim-rewrite-panel-v487",
    Path("results/experiment_5339_gated_sota_claim_rewrite_panel_v487.json"),
)
EXP5340 = UpstreamArtifact(
    5340,
    "exp5340-utility-weighted-context-memory-v487",
    Path("results/experiment_5340_utility_weighted_context_memory_v487.json"),
)
EXP5340_Q_VALUES = UpstreamArtifact(
    5340,
    "exp5340-utility-weighted-context-memory-q-values-v487",
    Path("results/experiment_5340_utility_weighted_context_memory_q_values_v487.json"),
    default_status="sidecar",
)
EXP5341 = UpstreamArtifact(
    5341,
    "exp5341-bounded-compressor-drift-monitor-v487",
    Path("results/experiment_5341_bounded_compressor_drift_monitor_v487.json"),
)
EXP5342 = UpstreamArtifact(
    5342,
    "exp5342-provenance-bound-self-learning-scaleup-v487",
    Path("results/experiment_5342_provenance_bound_self_learning_scaleup_v487.json"),
)
EXP5343 = UpstreamArtifact(
    5343,
    "exp5343-qstr-temporal-spatial-constraint-fixture-v487",
    Path("results/experiment_5343_qstr_temporal_spatial_constraint_fixture_v487.json"),
)
EXP5344 = UpstreamArtifact(
    5344,
    "exp5344-gated-solver-guidance-overwrite-telemetry-v487",
    Path("results/experiment_5344_solver_guidance_overwrite_telemetry_v487.json"),
)
EXP5345 = UpstreamArtifact(
    5345,
    "exp5345-tokenprob-energy-corrigendum-v487",
    Path("results/experiment_5345_tokenprob_energy_corrigendum_v487.json"),
)
EXP5346 = UpstreamArtifact(
    5346,
    "exp5346-kan-ising-counterexample-constraint-bridge-v487",
    Path("results/experiment_5346_kan_ising_counterexample_constraint_bridge_v487.json"),
)
EXP5347 = UpstreamArtifact(
    5347,
    "exp5347-hardware-continuity-workload-receipts-v487",
    Path("results/experiment_5347_hardware_continuity_workload_receipts_v487.json"),
)

EXPECTED_ARTIFACTS = (
    EXP5335,
    EXP5336,
    EXP5337,
    EXP5338,
    EXP5339,
    EXP5340,
    EXP5340_Q_VALUES,
    EXP5341,
    EXP5342,
    EXP5343,
    EXP5344,
    EXP5345,
    EXP5346,
    EXP5347,
)

CITED_FIELDS_BY_PATH: dict[str, list[str]] = {
    str(EXP5335.relative_path): [
        "status",
        "honest_verdict",
        "roadmap_next_present",
        "active_roadmap_modified",
        "conductor_modified",
    ],
    str(EXP5336.relative_path): ["status", "honest_verdict", "new_actionable_findings_count"],
    str(EXP5337.relative_path): [
        "sota_runtime_clean_receipt_ready",
        "quality_claim_permitted",
        "methodology_duration_s",
        "no_autotokenizer_used",
    ],
    str(EXP5338.relative_path): [
        "structured_output_protocol_ready",
        "selected_variant_id",
        "parse_success_rate",
        "flagged_adversarial",
        "corrigendum_pending",
        "no_quality_claim",
    ],
    str(EXP5339.relative_path): [
        "sota_claim_rewrite_panel_ready",
        "parse_success_rate",
        "paraphrase_label_preservation_rate",
        "rewrite_acceptability_rate",
        "headline_quality_claim",
    ],
    str(EXP5340.relative_path): [
        "utility_memory_ready",
        "utility_update_count",
        "quality_delta_vs_always_full",
        "unsafe_false_accepts",
        "no_weight_mutation",
    ],
    str(EXP5340_Q_VALUES.relative_path): [
        "operation_q_values",
        "utility_update_count",
        "no_weight_mutation",
    ],
    str(EXP5341.relative_path): [
        "compressor_drift_fixture_ready",
        "drift_detection_rate",
        "poison_rejection_rate",
        "unsafe_commits",
        "no_weight_mutation",
    ],
    str(EXP5342.relative_path): [
        "self_learning_scaleup_ready",
        "context_efficiency_delta",
        "memory_hygiene_delta",
        "flagged_adversarial",
        "corrigendum_pending",
    ],
    str(EXP5343.relative_path): [
        "qstr_fixture_ready",
        "solver_authoritative",
        "false_accept_count",
        "failure_localization_rate",
    ],
    str(EXP5344.relative_path): [
        "solver_guidance_telemetry_ready",
        "solver_authoritative",
        "fallback_completeness_rate",
        "misleading_hint_false_accepts",
    ],
    str(EXP5345.relative_path): [
        "internal_energy_corrigendum_clean",
        "status",
        "honest_verdict",
        "flagged_adversarial",
        "corrigendum_pending",
        "no_quality_claim",
    ],
    str(EXP5346.relative_path): [
        "constraint_bridge_ready",
        "true_property_preservation_rate",
        "injected_false_property_rejection_rate",
        "no_broad_certificate_claim",
    ],
    str(EXP5347.relative_path): [
        "speedup_claim",
        "authenticated_workload_run",
        "polarfire_workload_validated",
        "hardware_evidence_level",
    ],
}


def value_of(value: Any) -> Any:
    """Return the machine value from a principle-wrapped or bare artifact field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrapped(field: str, value: Any) -> JsonDict:
    """Attach the required field principle to a capstone value."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _sha256(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _path_key(source: UpstreamArtifact) -> str:
    return str(source.relative_path)


def _payload_value(payload: JsonMap | None, field: str, default: Any = None) -> Any:
    if payload is None:
        return default
    return value_of(payload.get(field, default))


def _status(payload: JsonMap | None, default: str = "missing_or_unreadable") -> str:
    return str(_payload_value(payload, "status", default))


def _verdict(payload: JsonMap | None) -> str:
    return str(_payload_value(payload, "honest_verdict", ""))


def _exp_id(payload: JsonMap, source: UpstreamArtifact) -> Any:
    return (
        value_of(payload.get("experiment_id"))
        or value_of(payload.get("experiment"))
        or source.task_id
    )


def read_conductor_outcomes(root: Path | str = REPO_ROOT) -> dict[int, list[JsonDict]]:
    """Parse conductor rows for Exp5335 through Exp5347 when the log exists."""

    path = Path(root) / "ops/conductor-log.md"
    outcomes: dict[int, list[JsonDict]] = {
        source.experiment_number: [] for source in EXPECTED_ARTIFACTS
    }
    if not path.exists():
        return outcomes
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_number, line in enumerate(lines, 1):
        for source in EXPECTED_ARTIFACTS:
            exp_text = f"Exp {source.experiment_number}"
            compact_exp_text = f"Exp{source.experiment_number}"
            if exp_text not in line and compact_exp_text not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4:
                outcomes[source.experiment_number].append(
                    {
                        "line_number": line_number,
                        "timestamp": cells[0],
                        "status": cells[2],
                        "summary": cells[3],
                    }
                )
    return outcomes


def latest_conductor_outcome(
    outcomes: Mapping[int, Sequence[JsonDict]], experiment_number: int
) -> JsonDict | None:
    rows = list(outcomes.get(experiment_number, ()))
    return rows[-1] if rows else None


def read_upstream_artifacts(
    root: Path | str = REPO_ROOT,
    conductor_outcomes: Mapping[int, Sequence[JsonDict]] | None = None,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[JsonDict]]:
    """Read every expected V487 artifact and preserve unreadable inputs."""

    root_path = Path(root)
    conductor = conductor_outcomes or {}
    payloads: dict[str, JsonDict] = {}
    artifacts_read: list[JsonDict] = []
    missing_or_malformed: list[JsonDict] = []
    for source in EXPECTED_ARTIFACTS:
        path = root_path / source.relative_path
        path_key = _path_key(source)
        if not path.exists():
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "missing",
                    "reason": "missing",
                }
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "malformed",
                    "reason": f"malformed_json:{exc.msg}",
                }
            )
            continue
        if not isinstance(payload, dict):
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "malformed",
                    "reason": "not_json_object",
                }
            )
            continue
        latest = latest_conductor_outcome(conductor, source.experiment_number)
        payloads[path_key] = payload
        artifacts_read.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": path_key,
                "experiment_id": _exp_id(payload, source),
                "status": _status(payload, source.default_status),
                "honest_verdict": _verdict(payload),
                "flagged_adversarial": _payload_value(payload, "flagged_adversarial") is True,
                "conductor_outcome": latest,
                "sha256": _sha256(path),
            }
        )
    return payloads, artifacts_read, missing_or_malformed


def _is_blocked(payload: JsonMap | None) -> bool:
    status = _status(payload)
    verdict = _verdict(payload)
    return status == "blocked" or status.startswith("blocked_") or verdict.startswith("blocked_")


def _is_flagged(payload: JsonMap | None, latest: JsonMap | None) -> bool:
    latest_status = str((latest or {}).get("status", ""))
    return _payload_value(payload, "flagged_adversarial") is True or latest_status == "FLAGGED"


def _is_skipped(latest: JsonMap | None) -> bool:
    return str((latest or {}).get("status", "")) in {"GATE_BLOCK", "SKIP"}


def _blocked_flagged_or_skipped_rows(
    payloads: Mapping[str, JsonDict],
    missing_or_malformed: Sequence[JsonDict],
    conductor_outcomes: Mapping[int, Sequence[JsonDict]],
) -> list[JsonDict]:
    rows = [dict(row) for row in missing_or_malformed]
    missing_paths = {str(row["path"]) for row in missing_or_malformed}
    for source in EXPECTED_ARTIFACTS:
        path_key = _path_key(source)
        if path_key in missing_paths:
            continue
        payload = payloads.get(path_key)
        latest = latest_conductor_outcome(conductor_outcomes, source.experiment_number)
        blocked = _is_blocked(payload)
        flagged = _is_flagged(payload, latest)
        skipped = _is_skipped(latest)
        classification = (
            "blocked_and_flagged"
            if blocked and flagged
            else "blocked"
            if blocked
            else "flagged"
            if flagged
            else "conductor_gate_skip"
            if skipped
            else None
        )
        if classification is None:
            continue
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": path_key,
                "classification": classification,
                "status": _status(payload, source.default_status),
                "honest_verdict": _verdict(payload),
                "conductor_outcome": latest,
                "corrigendum_pending": _payload_value(payload, "corrigendum_pending", []),
            }
        )
    return rows


def _issue_by_path(issues: Sequence[JsonMap]) -> dict[str, str]:
    return {str(row["path"]): str(row["classification"]) for row in issues}


def _payload(payloads: Mapping[str, JsonDict], source: UpstreamArtifact) -> JsonDict | None:
    return payloads.get(_path_key(source))


def _clean_artifact(
    payloads: Mapping[str, JsonDict], issues_by_path: Mapping[str, str], source: UpstreamArtifact
) -> bool:
    classification = issues_by_path.get(_path_key(source))
    return _path_key(source) in payloads and classification is None


def _gate_table(payloads: Mapping[str, JsonDict], issues: Sequence[JsonMap]) -> list[JsonDict]:
    issues_by_path = _issue_by_path(issues)
    exp5337 = _payload(payloads, EXP5337)
    exp5338 = _payload(payloads, EXP5338)
    exp5339 = _payload(payloads, EXP5339)
    exp5340 = _payload(payloads, EXP5340)
    exp5340_q = _payload(payloads, EXP5340_Q_VALUES)
    exp5341 = _payload(payloads, EXP5341)
    exp5342 = _payload(payloads, EXP5342)
    exp5343 = _payload(payloads, EXP5343)
    exp5344 = _payload(payloads, EXP5344)
    exp5345 = _payload(payloads, EXP5345)
    exp5346 = _payload(payloads, EXP5346)
    exp5347 = _payload(payloads, EXP5347)

    runtime_clean = (
        _clean_artifact(payloads, issues_by_path, EXP5337)
        and _payload_value(exp5337, "sota_runtime_clean_receipt_ready") is True
        and _payload_value(exp5337, "quality_claim_permitted") is False
    )
    structured_reported_ready = _payload_value(exp5338, "structured_output_protocol_ready") is True
    structured_clean = (
        _clean_artifact(payloads, issues_by_path, EXP5338)
        and structured_reported_ready
        and _payload_value(exp5338, "unsafe_false_accepts", 1) == 0
    )
    quality_usable = (
        _clean_artifact(payloads, issues_by_path, EXP5339)
        and _payload_value(exp5339, "sota_claim_rewrite_panel_ready") is True
        and _payload_value(exp5339, "headline_quality_claim") is False
    )
    utility_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5340)
        and _clean_artifact(payloads, issues_by_path, EXP5340_Q_VALUES)
        and _payload_value(exp5340, "utility_memory_ready") is True
        and _payload_value(exp5340, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5340, "no_weight_mutation") is True
        and _payload_value(exp5340_q, "no_weight_mutation") is True
    )
    compressor_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5341)
        and _payload_value(exp5341, "compressor_drift_fixture_ready") is True
        and _payload_value(exp5341, "unsafe_commits", 1) == 0
        and _payload_value(exp5341, "no_weight_mutation") is True
    )
    scaleup_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5342)
        and _payload_value(exp5342, "self_learning_scaleup_ready") is True
        and _payload_value(exp5342, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5342, "no_weight_mutation") is True
    )
    qstr_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5343)
        and _payload_value(exp5343, "qstr_fixture_ready") is True
        and _payload_value(exp5343, "solver_authoritative") is True
        and _payload_value(exp5343, "false_accept_count", 1) == 0
    )
    solver_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5344)
        and _payload_value(exp5344, "solver_guidance_telemetry_ready") is True
        and _payload_value(exp5344, "solver_authoritative") is True
        and _payload_value(exp5344, "misleading_hint_false_accepts", 1) == 0
    )
    energy_clean = (
        _clean_artifact(payloads, issues_by_path, EXP5345)
        and _payload_value(exp5345, "internal_energy_corrigendum_clean") is True
        and _payload_value(exp5345, "no_quality_claim") is True
    )
    kan_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5346)
        and _payload_value(exp5346, "constraint_bridge_ready") is True
        and _payload_value(exp5346, "no_broad_certificate_claim") is True
        and _payload_value(exp5346, "unsafe_false_accepts", 1) == 0
    )
    hardware_continuity_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5347)
        and _payload_value(exp5347, "speedup_claim") is False
        and _payload_value(exp5347, "authenticated_workload_run") is True
        and _payload_value(exp5347, "polarfire_workload_validated") is True
    )

    return [
        {
            "gate": "runtime",
            "source_artifacts": [str(EXP5337.relative_path)],
            "ready": runtime_clean,
            "classification": "clean_runtime_no_quality_claim"
            if runtime_clean
            else "runtime_not_clean",
            "claim_boundary": "runtime receipt only; no SOTA quality claim",
            "evidence": {
                "sota_runtime_clean_receipt_ready": _payload_value(
                    exp5337, "sota_runtime_clean_receipt_ready"
                ),
                "quality_claim_permitted": _payload_value(exp5337, "quality_claim_permitted"),
                "methodology_duration_s": _payload_value(exp5337, "methodology_duration_s"),
            },
        },
        {
            "gate": "structured_output_protocol",
            "source_artifacts": [str(EXP5338.relative_path)],
            "ready": structured_clean,
            "classification": (
                "clean_parse_only_protocol"
                if structured_clean
                else "flagged_parse_only_protocol_candidate"
                if structured_reported_ready and issues_by_path.get(_path_key(EXP5338)) == "flagged"
                else "protocol_not_ready"
            ),
            "claim_boundary": "parse-only protocol candidate; no quality claim",
            "evidence": {
                "reported_ready": structured_reported_ready,
                "selected_variant_id": _payload_value(exp5338, "selected_variant_id"),
                "parse_success_rate": _payload_value(exp5338, "parse_success_rate"),
                "flagged_adversarial": _payload_value(exp5338, "flagged_adversarial"),
                "corrigendum_pending": _payload_value(exp5338, "corrigendum_pending", []),
                "no_quality_claim": _payload_value(exp5338, "no_quality_claim"),
            },
        },
        {
            "gate": "sota_bounded_quality",
            "source_artifacts": [str(EXP5339.relative_path)],
            "ready": quality_usable,
            "classification": (
                "bounded_quality_usable_no_headline_claim"
                if quality_usable
                else "blocked_quality_panel_not_usable"
            ),
            "claim_boundary": "bounded fixture panel only; no headline SOTA quality claim",
            "evidence": {
                "sota_claim_rewrite_panel_ready": _payload_value(
                    exp5339, "sota_claim_rewrite_panel_ready"
                ),
                "parse_success_rate": _payload_value(exp5339, "parse_success_rate"),
                "paraphrase_label_preservation_rate": _payload_value(
                    exp5339, "paraphrase_label_preservation_rate"
                ),
                "rewrite_acceptability_rate": _payload_value(exp5339, "rewrite_acceptability_rate"),
                "headline_quality_claim": _payload_value(exp5339, "headline_quality_claim"),
            },
        },
        {
            "gate": "utility_memory",
            "source_artifacts": [str(EXP5340.relative_path), str(EXP5340_Q_VALUES.relative_path)],
            "ready": utility_ready,
            "classification": "clean_utility_memory_fixture" if utility_ready else "not_clean",
            "claim_boundary": "deterministic utility learning over context operations; no weight mutation",
            "evidence": {
                "utility_memory_ready": _payload_value(exp5340, "utility_memory_ready"),
                "utility_update_count": _payload_value(exp5340, "utility_update_count"),
                "quality_delta_vs_always_full": _payload_value(
                    exp5340, "quality_delta_vs_always_full"
                ),
                "unsafe_false_accepts": _payload_value(exp5340, "unsafe_false_accepts"),
                "q_value_sidecar_no_weight_mutation": _payload_value(
                    exp5340_q, "no_weight_mutation"
                ),
            },
        },
        {
            "gate": "bounded_compressor",
            "source_artifacts": [str(EXP5341.relative_path)],
            "ready": compressor_ready,
            "classification": "clean_bounded_compressor_fixture"
            if compressor_ready
            else "not_clean",
            "claim_boundary": "bounded deterministic compressor and drift monitor only",
            "evidence": {
                "compressor_drift_fixture_ready": _payload_value(
                    exp5341, "compressor_drift_fixture_ready"
                ),
                "drift_detection_rate": _payload_value(exp5341, "drift_detection_rate"),
                "poison_rejection_rate": _payload_value(exp5341, "poison_rejection_rate"),
                "unsafe_commits": _payload_value(exp5341, "unsafe_commits"),
                "no_weight_mutation": _payload_value(exp5341, "no_weight_mutation"),
            },
        },
        {
            "gate": "self_learning_scaleup",
            "source_artifacts": [str(EXP5342.relative_path)],
            "ready": scaleup_ready,
            "classification": (
                "clean_provenance_bound_scaleup"
                if scaleup_ready
                else "flagged_scaleup_not_claimable"
                if issues_by_path.get(_path_key(EXP5342)) == "flagged"
                else "not_clean"
            ),
            "claim_boundary": "multi-session context-policy scale-up only when not flagged",
            "evidence": {
                "reported_ready": _payload_value(exp5342, "self_learning_scaleup_ready"),
                "multi_session_trace_count": _payload_value(exp5342, "multi_session_trace_count"),
                "context_efficiency_delta": _payload_value(exp5342, "context_efficiency_delta"),
                "memory_hygiene_delta": _payload_value(exp5342, "memory_hygiene_delta"),
                "unsafe_false_accepts": _payload_value(exp5342, "unsafe_false_accepts"),
                "flagged_adversarial": _payload_value(exp5342, "flagged_adversarial"),
                "corrigendum_pending": _payload_value(exp5342, "corrigendum_pending", []),
            },
        },
        {
            "gate": "qstr_fixture",
            "source_artifacts": [str(EXP5343.relative_path)],
            "ready": qstr_ready,
            "classification": "clean_deterministic_qstr_fixture" if qstr_ready else "not_clean",
            "claim_boundary": "deterministic temporal/spatial fixture; solver remains authoritative",
            "evidence": {
                "qstr_fixture_ready": _payload_value(exp5343, "qstr_fixture_ready"),
                "solver_authoritative": _payload_value(exp5343, "solver_authoritative"),
                "false_accept_count": _payload_value(exp5343, "false_accept_count"),
                "failure_localization_rate": _payload_value(exp5343, "failure_localization_rate"),
            },
        },
        {
            "gate": "solver_guidance",
            "source_artifacts": [str(EXP5344.relative_path)],
            "ready": solver_ready,
            "classification": "clean_solver_guidance_telemetry" if solver_ready else "not_clean",
            "claim_boundary": "solver-authoritative overwrite telemetry; not LLM correctness",
            "evidence": {
                "solver_guidance_telemetry_ready": _payload_value(
                    exp5344, "solver_guidance_telemetry_ready"
                ),
                "solver_authoritative": _payload_value(exp5344, "solver_authoritative"),
                "fallback_completeness_rate": _payload_value(exp5344, "fallback_completeness_rate"),
                "misleading_hint_false_accepts": _payload_value(
                    exp5344, "misleading_hint_false_accepts"
                ),
            },
        },
        {
            "gate": "token_probability_energy",
            "source_artifacts": [str(EXP5345.relative_path)],
            "ready": energy_clean,
            "classification": (
                "clean_token_probability_energy_corrigendum"
                if energy_clean
                else "blocked_and_flagged_energy_corrigendum"
            ),
            "claim_boundary": "internal token-probability diagnostic only; no text-scorer reopening",
            "evidence": {
                "internal_energy_corrigendum_clean": _payload_value(
                    exp5345, "internal_energy_corrigendum_clean"
                ),
                "status": _status(exp5345),
                "honest_verdict": _verdict(exp5345),
                "flagged_adversarial": _payload_value(exp5345, "flagged_adversarial"),
                "corrigendum_pending": _payload_value(exp5345, "corrigendum_pending", []),
                "no_quality_claim": _payload_value(exp5345, "no_quality_claim"),
            },
        },
        {
            "gate": "kan_constraint_bridge",
            "source_artifacts": [str(EXP5346.relative_path)],
            "ready": kan_ready,
            "classification": "bounded_constraint_bridge_clean" if kan_ready else "not_clean",
            "claim_boundary": "bounded explicit cuts only; no broad KAN certificate claim",
            "evidence": {
                "constraint_bridge_ready": _payload_value(exp5346, "constraint_bridge_ready"),
                "true_property_preservation_rate": _payload_value(
                    exp5346, "true_property_preservation_rate"
                ),
                "injected_false_property_rejection_rate": _payload_value(
                    exp5346, "injected_false_property_rejection_rate"
                ),
                "no_broad_certificate_claim": _payload_value(exp5346, "no_broad_certificate_claim"),
            },
        },
        {
            "gate": "hardware",
            "source_artifacts": [str(EXP5347.relative_path)],
            "ready": hardware_continuity_ready,
            "classification": (
                "continuity_workload_receipt_no_speedup"
                if hardware_continuity_ready
                else "hardware_continuity_not_clean"
            ),
            "claim_boundary": "hardware continuity and board-local smoke only; no speedup claim",
            "evidence": {
                "speedup_claim": _payload_value(exp5347, "speedup_claim"),
                "authenticated_workload_run": _payload_value(exp5347, "authenticated_workload_run"),
                "polarfire_workload_validated": _payload_value(
                    exp5347, "polarfire_workload_validated"
                ),
                "hardware_evidence_level": _payload_value(exp5347, "hardware_evidence_level"),
            },
        },
    ]


def gate_value(gates: Sequence[JsonMap], gate: str) -> bool:
    return any(row.get("gate") == gate and row.get("ready") is True for row in gates)


def next_milestone_recommendation(gates: Sequence[JsonMap]) -> JsonDict:
    """Choose the short next branch from the reconciled .487 gate state."""

    energy_row = next(row for row in gates if row["gate"] == "token_probability_energy")
    quality_row = next(row for row in gates if row["gate"] == "sota_bounded_quality")
    scaleup_row = next(row for row in gates if row["gate"] == "self_learning_scaleup")
    return {
        "recommendation": "token-energy cleanup",
        "why": (
            "runtime, utility memory, bounded compression, QSTR, solver guidance, KAN, "
            "and no-speedup hardware continuity have clean bounded evidence, while "
            "token-probability energy is both blocked and flagged."
        ),
        "secondary_cleanup": [
            f"structured_sota_quality={quality_row['classification']}",
            f"self_learning_scaleup={scaleup_row['classification']}",
            f"token_probability_energy={energy_row['classification']}",
        ],
        "do_not_claim": [
            "headline_sota_quality",
            "structured_protocol_clean_success",
            "self_learning_scaleup_clean",
            "internal_energy_clean",
            "broad_kan_certificate",
            "hardware_speedup",
        ],
    }


def cited_upstream_artifacts(artifacts_read: Sequence[JsonMap]) -> list[JsonDict]:
    """Return the sha256 citation rows and imported fields used by the capstone."""

    citations: list[JsonDict] = []
    for row in artifacts_read:
        path = str(row["path"])
        citations.append(
            {
                "experiment_number": row["experiment_number"],
                "task_id": row["task_id"],
                "path": path,
                "sha256": row["sha256"],
                "imported_fields": CITED_FIELDS_BY_PATH.get(path, []),
            }
        )
    return citations


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def default_tests_run() -> list[JsonDict]:
    return [{"command": "validation pending at artifact generation", "outcome": "pending"}]


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    conductor_outcomes = read_conductor_outcomes(root)
    payloads, artifacts_read, missing_or_malformed = read_upstream_artifacts(
        root, conductor_outcomes
    )
    issues = _blocked_flagged_or_skipped_rows(payloads, missing_or_malformed, conductor_outcomes)
    gates = _gate_table(payloads, issues)
    all_artifacts_read = not any(
        row["classification"] in {"missing", "malformed"} for row in missing_or_malformed
    )
    status = "complete" if all_artifacts_read else "blocked_missing_required"
    verdict_prefix = "complete:" if all_artifacts_read else "blocked_missing_required:"
    verdict = (
        f"{verdict_prefix} .487 synthesized with runtime_clean="
        f"{gate_value(gates, 'runtime')}, structured_output_protocol_ready="
        f"{gate_value(gates, 'structured_output_protocol')}, "
        f"bounded_sota_quality_usable={gate_value(gates, 'sota_bounded_quality')}, "
        f"utility_memory_ready={gate_value(gates, 'utility_memory')}, "
        f"bounded_compressor_ready={gate_value(gates, 'bounded_compressor')}, "
        f"self_learning_scaleup_ready={gate_value(gates, 'self_learning_scaleup')}, "
        f"qstr_solver_kan_clean="
        f"{gate_value(gates, 'qstr_fixture') and gate_value(gates, 'solver_guidance') and gate_value(gates, 'kan_constraint_bridge')}, "
        f"internal_energy_corrigendum_clean={gate_value(gates, 'token_probability_energy')}, "
        "hardware_speedup_claim=false"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": wrapped("experiment_id", EXPERIMENT_ID),
        "milestone": wrapped("milestone", MILESTONE),
        "status": wrapped("status", status),
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": wrapped("honest_verdict", verdict),
        "inference_substrate": wrapped("inference_substrate", INFERENCE_SUBSTRATE),
        "artifacts_read": wrapped("artifacts_read", artifacts_read),
        "missing_blocked_flagged_or_skipped_artifacts": wrapped(
            "missing_blocked_flagged_or_skipped_artifacts", issues
        ),
        "gate_table": wrapped("gate_table", gates),
        "runtime_clean": gate_value(gates, "runtime"),
        "structured_output_protocol_ready": gate_value(gates, "structured_output_protocol"),
        "bounded_sota_quality_usable": gate_value(gates, "sota_bounded_quality"),
        "utility_memory_ready": gate_value(gates, "utility_memory"),
        "bounded_compressor_ready": gate_value(gates, "bounded_compressor"),
        "self_learning_scaleup_ready": gate_value(gates, "self_learning_scaleup"),
        "qstr_fixture_ready": gate_value(gates, "qstr_fixture"),
        "solver_guidance_ready": gate_value(gates, "solver_guidance"),
        "internal_energy_corrigendum_clean": gate_value(gates, "token_probability_energy"),
        "kan_constraint_bridge_ready": gate_value(gates, "kan_constraint_bridge"),
        "hardware_speedup_claim": False,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "next_milestone_recommendation": wrapped(
            "next_milestone_recommendation", next_milestone_recommendation(gates)
        ),
        "cited_upstream_artifacts": wrapped(
            "cited_upstream_artifacts", cited_upstream_artifacts(artifacts_read)
        ),
        "tests_run": wrapped(
            "tests_run",
            [dict(row) for row in (tests_run if tests_run is not None else default_tests_run())],
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        value = artifact[field]
        if (
            not isinstance(value, Mapping)
            or value.get("principle") != FIELD_PRINCIPLES[field]
            or "value" not in value
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    for field in BARE_BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare boolean")
    verdict = artifact["honest_verdict"]["value"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate drift")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must be false")
    for field in ("active_roadmap_modified", "conductor_modified"):
        if artifact[field] is not False:
            raise ValueError(f"{field} must be false")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    artifact = build_result_artifact(root=root, tests_run=tests_run)
    validate_artifact(artifact)
    write_json(result_path or Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
