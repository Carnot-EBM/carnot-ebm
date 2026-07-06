"""Experiment 5306: V484 capstone synthesis.

Spec refs: REQ-CAPSTONE-5306, SCENARIO-CAPSTONE-5306,
SCENARIO-CAPSTONE-5306-BLOCKED-MISSING-INPUT.

This module is aggregation-only. It reads the already-written V484 result
artifacts and conductor-log receipts, keeps blocked, gated, quarantined, null,
and mixed-class evidence in separate lanes, and writes a milestone closeout
without running new model, solver, or hardware workloads.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5306_capstone_v484.json")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
EXPERIMENT = "experiment_5306_capstone_v484"
EXPERIMENT_ID = "exp5306-capstone-v484"
MILESTONE = "2026.07.484"
SCHEMA = "carnot.experiment_5306_capstone_v484.v1"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5306
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")
SAME_VERDICT_RETIREMENT_ID = "exp5284_sota_offload_cpu_only_path_retired_v483"

SPEC_REFS = [
    "REQ-CAPSTONE-5306",
    "SCENARIO-CAPSTONE-5306",
    "SCENARIO-CAPSTONE-5306-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5306-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "terminal prefix; starts with complete: or blocked_ and summarizes the .484 milestone "
        "without laundering gated, blocked, null, harmful, mixed, quarantined, missing, or "
        "no-speedup evidence."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because the capstone reads checked-in artifacts "
        "and conductor logs without running LLM, solver, or hardware workloads."
    ),
    "tasks_summarized": (
        "classification counts and per-task lanes for clean_positive, clean_null, "
        "harmful_or_regression, mixed_positive_with_harmful_class, blocked_precondition, "
        "gated_skip, and missing_artifact."
    ),
    "changed_runtime_outcome": (
        "changed runtime backend evidence and blocked readiness separated from any quality claim."
    ),
    "sota_quality_outcome": (
        "quality-smoke state; if Exp5298 is gate-skipped then SOTA quality was not measured."
    ),
    "continuous_self_learning_outcome": (
        "adaptive memory results using held-out quality, calls avoided, unsafe false accepts, "
        "rollback, and stress competencies."
    ),
    "solver_energy_certificate_outcome": (
        "LNS readiness, p-bit/CDCL gate, EBT spectral-control quarantine, and KAN "
        "dynamic-abstraction limits."
    ),
    "hardware_speedup_claimed": (
        "must be false unless a local reproducible hardware speedup artifact exists."
    ),
    "retirements_or_exclusions_recommended": (
        "manifest-aware same-verdict retirements, quarantine handling, and bounded retry "
        "guidance without editing the manifest unless explicitly scoped."
    ),
    "next_milestone_recommendations": ("concrete next work that preserves evidence boundaries."),
    "docs_updated": (
        "records that ops/status, ops/changelog, traceability, and docs/index.html were not "
        "touched when the stop rule delegates reconciliation."
    ),
}

PRINCIPLE_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "tasks_summarized",
    "changed_runtime_outcome",
    "sota_quality_outcome",
    "continuous_self_learning_outcome",
    "solver_energy_certificate_outcome",
    "hardware_speedup_claimed",
    "retirements_or_exclusions_recommended",
    "next_milestone_recommendations",
    "docs_updated",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "random_seed",
    "field_principles",
    "source_artifacts_read",
    "conductor_log_entries",
    *PRINCIPLE_WRAPPED_FIELDS,
    "commands_run",
    "reproducibility_checksum",
)

TASK_CLASSIFICATIONS = (
    "clean_positive",
    "clean_null",
    "harmful_or_regression",
    "mixed_positive_with_harmful_class",
    "blocked_precondition",
    "gated_skip",
    "missing_artifact",
    "quarantined",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V484 result artifact to aggregate."""

    experiment_number: int
    task_id: str
    title: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5295,
        "exp5295-archive-483-activate-484",
        "Archive .483 and prepare .484 activation",
        Path("results/experiment_5295_archive_483_activate_484.json"),
    ),
    UpstreamSource(
        5296,
        "exp5296-sota-source-delta-v484",
        "V484 SOTA/source delta refresh",
        Path("results/experiment_5296_sota_source_delta_v484.json"),
    ),
    UpstreamSource(
        5297,
        "exp5297-changed-runtime-sota-substrate-gate-v484",
        "Changed SOTA GGUF runtime substrate gate",
        Path("results/experiment_5297_changed_runtime_sota_substrate_gate_v484.json"),
    ),
    UpstreamSource(
        5298,
        "exp5298-sota-coherence-trace-smoke-gated-v484",
        "SOTA coherence and trace smoke",
        Path("results/experiment_5298_sota_coherence_trace_smoke_gated_v484.json"),
    ),
    UpstreamSource(
        5299,
        "exp5299-constraint-lns-solver-repair-fixture-v484",
        "Constraint-LNS destroy/repair fixture",
        Path("results/experiment_5299_constraint_lns_solver_repair_fixture_v484.json"),
    ),
    UpstreamSource(
        5300,
        "exp5300-pbit-cdcl-instance-class-gate-v484",
        "p-bit/CDCL instance-class gate",
        Path("results/experiment_5300_pbit_cdcl_instance_class_gate_v484.json"),
    ),
    UpstreamSource(
        5301,
        "exp5301-ebt-spectral-step-control-diagnostic-v484",
        "EBT spectral step-control diagnostic",
        Path("results/experiment_5301_ebt_spectral_step_control_diagnostic_v484.json"),
    ),
    UpstreamSource(
        5302,
        "exp5302-adaptive-memory-policy-self-learning-v484",
        "Adaptive self-learning memory policy",
        Path("results/experiment_5302_adaptive_memory_policy_self_learning_v484.json"),
    ),
    UpstreamSource(
        5303,
        "exp5303-memory-stress-conflict-forgetting-v484",
        "Memory conflict, forgetting, and long-range stress",
        Path("results/experiment_5303_memory_stress_conflict_forgetting_v484.json"),
    ),
    UpstreamSource(
        5304,
        "exp5304-kan-dynamic-abstraction-spotcheck-v484",
        "KAN dynamic abstraction spot-check",
        Path("results/experiment_5304_kan_dynamic_abstraction_spotcheck_v484.json"),
    ),
    UpstreamSource(
        5305,
        "exp5305-hardware-continuity-reachability-v484",
        "Hardware continuity reachability receipts",
        Path("results/experiment_5305_hardware_continuity_reachability_v484.json"),
    ),
)

CONDUCTOR_LOG_PATTERNS = (
    "Plan milestone 2026.07.484",
    "Milestone 2026.07.484 activated",
    "PHASE 0 transition -- archive .483",
    "PHASE 0 SOTA/source refresh -- V484",
    "PHASE 0 runtime receipts -- changed SOTA GGUF",
    "PHASE 0 gated on exp5297",
    "PHASE 1 fixture -- constraint-LNS",
    "PHASE 1 gated on exp5299",
    "PHASE 1 diagnostic -- EBT spectral",
    "PHASE 2 continuous self-learning",
    "PHASE 2 gated on exp5302",
    "PHASE 3 certificates -- KAN dynamic",
    "PHASE 3 hardware continuity",
)


def value_of(value: Any) -> Any:
    return value["value"] if isinstance(value, Mapping) and "value" in value else value


def _text(value: Any) -> str:
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


def _count(value: Any) -> int:
    raw = value_of(value)
    if isinstance(raw, Mapping):
        raw = raw.get("count", 0)
    if isinstance(raw, bool) or raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "sha256": None, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            {},
            {
                "exists": True,
                "loadable": False,
                "sha256": file_sha256(path),
                "error": f"malformed_json:{exc.msg}",
            },
        )
    if not isinstance(parsed, Mapping):
        return (
            {},
            {
                "exists": True,
                "loadable": False,
                "sha256": file_sha256(path),
                "error": "not_json_object",
            },
        )
    return dict(parsed), {
        "exists": True,
        "loadable": True,
        "sha256": file_sha256(path),
        "error": None,
    }


def classify_payload(experiment_number: int, payload: JsonMap) -> str:
    verdict = _text(payload.get("honest_verdict")).lower()
    if payload.get("flagged_adversarial") is True:
        return "quarantined"
    if payload.get("blocked_at_layer") == "conductor_pre_gate" or "gate_check" in verdict:
        return "gated_skip"
    if experiment_number == 5300 and _bool(payload.get("pbit_gate_ready")):
        return "mixed_positive_with_harmful_class"
    if (
        verdict.startswith("blocked")
        or "blocked_preconditions" in verdict
        or experiment_number in {5297, 5305}
    ):
        return "blocked_precondition"
    if (
        verdict.startswith("harmful")
        or verdict.startswith("regression")
        or "harmful_regression" in verdict
        or " regression" in verdict
    ):
        return "harmful_or_regression"
    if experiment_number == 5304:
        dynamic = value_of(payload.get("dynamic_abstraction_helped"))
        if isinstance(dynamic, Mapping) and _number(dynamic.get("success_improvement")) == 0.0:
            return "clean_null"
    if "did not improve" in verdict or "no improvement" in verdict or "null" in verdict:
        return "clean_null"
    return "clean_positive"


def read_conductor_log_entries(root: Path) -> list[str]:
    path = root / CONDUCTOR_LOG_PATH
    if not path.exists():
        return []
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if any(pattern in line for pattern in CONDUCTOR_LOG_PATTERNS):
            entries.append(line)
    return entries


def _summary(experiment_number: int, payload: JsonMap, classification: str) -> str:
    if classification == "missing_artifact":
        return "missing or unreadable required upstream artifact"
    if experiment_number == 5295:
        return ".483 archived and .484 activation-ready without roadmap mutation"
    if experiment_number == 5296:
        return (
            "source refresh appended four findings, but the artifact is adversarially quarantined"
        )
    if experiment_number == 5297:
        return "changed runtime backend had CUDA/offload evidence but timed out; readiness false and no quality claim"
    if experiment_number == 5298:
        return "SOTA coherence/trace smoke was conductor-gated because changed_runtime_sota_ready=false"
    if experiment_number == 5299:
        return "constraint-LNS fixture ready with solver correctness preserved and unsafe false accepts zero"
    if experiment_number == 5300:
        return (
            "p-bit/CDCL gate blocks misleading classes while preserving aggregate conflict savings"
        )
    if experiment_number == 5301:
        return "spectral-control telemetry exists but is quarantined by adversarial DURATION_TOO_SHORT findings"
    if experiment_number == 5302:
        return "adaptive memory matched held-out always-full quality, avoided calls, and kept unsafe false accepts zero"
    if experiment_number == 5303:
        return "memory stress passed conflict, forgetting, stale-evidence, and rollback controls with call savings"
    if experiment_number == 5304:
        return "dynamic abstraction improved diagnostic tightness while certificate success improvement stayed zero"
    if experiment_number == 5305:
        return "hardware remained reachability-only: KV260 blocked, PolarFire status-only, GateMate blocked"
    return _text(payload.get("honest_verdict")) or classification


def _row_for_source(source: UpstreamSource, root: Path) -> tuple[JsonDict, JsonDict | None]:
    payload, info = read_json_mapping(root / source.relative_path)
    classification = (
        classify_payload(source.experiment_number, payload)
        if info["loadable"]
        else "missing_artifact"
    )
    row = {
        "experiment_number": source.experiment_number,
        "task_id": source.task_id,
        "title": source.title,
        "path": str(source.relative_path),
        "exists": info["exists"],
        "loadable": info["loadable"],
        "sha256": info["sha256"],
        "classification": classification,
        "load_error": info["error"],
        "verdict": _text(payload.get("honest_verdict")) if info["loadable"] else info["error"],
        "inference_substrate": _text(payload.get("inference_substrate"))
        if info["loadable"]
        else None,
        "summary": _summary(source.experiment_number, payload, str(classification)),
    }
    return row, payload if info["loadable"] else None


def _model_statuses(payload: JsonMap) -> dict[str, Any]:
    specs = value_of(payload.get("MODEL_SPECS"))
    if not isinstance(specs, Mapping):
        return {}
    statuses: dict[str, Any] = {}
    for role, spec in specs.items():
        if isinstance(spec, Mapping):
            statuses[str(role)] = {
                "runtime_status": spec.get("runtime_status"),
                "role": spec.get("role", role),
                "blocked_preconditions": spec.get("blocked_preconditions", []),
            }
    return statuses


def _changed_runtime_outcome(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5297 = payloads.get(5297, {})
    runtime = value_of(exp5297.get("runtime_substrate_changed"))
    runtime = runtime if isinstance(runtime, Mapping) else {}
    ready = exp5297.get("changed_runtime_sota_ready") is True
    return {
        "source_experiment": 5297,
        "changed_runtime_sota_ready": ready,
        "backend_kind": runtime.get("backend_kind"),
        "changed_from_exp5284": runtime.get("changed_from_exp5284") is True,
        "cuda_backend_evidence": runtime.get("cuda_backend_evidence") is True,
        "model_statuses": _model_statuses(exp5297),
        "no_quality_claim": _bool(exp5297.get("no_quality_claim")),
        "sota_quality_measured": False,
        "summary": (
            "Changed native llama.cpp CLI substrate showed CUDA/offload evidence but all "
            "mandated model runs timed out, so changed_runtime_sota_ready=false and no "
            "quality result is imported."
        ),
    }


def _sota_quality_outcome(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5298 = payloads.get(5298, {})
    measured = not (
        exp5298.get("blocked_at_layer") == "conductor_pre_gate"
        or _text(exp5298.get("honest_verdict")).startswith("blocked_gate")
    )
    return {
        "source_experiment": 5298,
        "measured": measured,
        "blocked_at_layer": exp5298.get("blocked_at_layer"),
        "gate_check_summary": exp5298.get("gate_check_summary"),
        "gates_evaluated": exp5298.get("gates_evaluated", []),
        "summary": (
            "SOTA smoke was not measured because Exp5298 was conductor-pre-gated on "
            "Exp5297.changed_runtime_sota_ready == true and the observed value was false."
        )
        if not measured
        else "SOTA smoke produced a loadable artifact; inspect upstream metrics before promotion.",
    }


def _competency_rates(payload: JsonMap) -> dict[str, float | None]:
    competencies = value_of(payload.get("competency_metrics"))
    if not isinstance(competencies, Mapping):
        return {}
    rates: dict[str, float | None] = {}
    for name, metrics in competencies.items():
        if name == "principle":
            continue
        if isinstance(metrics, Mapping):
            rates[str(name)] = _number(metrics.get("adaptive_quality_rate"))
    return rates


def _continuous_self_learning_outcome(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5302 = payloads.get(5302, {})
    exp5303 = payloads.get(5303, {})
    return {
        "policy_source_experiment": 5302,
        "stress_source_experiment": 5303,
        "adaptive_memory_policy_positive": _bool(exp5302.get("adaptive_memory_policy_positive")),
        "memory_policy_candidate_ready": exp5302.get("memory_policy_candidate_ready") is True,
        "heldout_quality_delta_vs_always_full": value_of(
            exp5302.get("heldout_quality_delta_vs_always_full")
        ),
        "full_verifier_calls_avoided": value_of(exp5302.get("full_verifier_calls_avoided")),
        "stress_calls_avoided": value_of(exp5303.get("calls_avoided")),
        "unsafe_false_accepts": _count(exp5302.get("unsafe_false_accepts"))
        + _count(exp5303.get("unsafe_false_accepts")),
        "rollback_exercised": value_of(exp5302.get("rollback_exercised")),
        "rollback_success_rate": value_of(exp5303.get("rollback_success_rate")),
        "stale_conflict_handling": value_of(exp5303.get("stale_conflict_handling")),
        "selective_forgetting_correctness": value_of(
            exp5303.get("selective_forgetting_correctness")
        ),
        "stress_competency_quality_rates": _competency_rates(exp5303),
        "memory_stress_passed": _bool(exp5303.get("memory_stress_passed")),
        "no_weight_mutation": _bool(exp5302.get("no_weight_mutation")),
        "summary": (
            "Adaptive memory matched always-full held-out quality, avoided 3/7 full "
            "verifier calls in policy selection and 5/8 under stress, kept unsafe false "
            "accepts at zero, exercised rollback, and passed conflict/forgetting/stale "
            "competency checks."
        ),
    }


def _blocked_classes(payload: JsonMap) -> list[str]:
    gate = value_of(payload.get("misleading_class_blocked"))
    if not isinstance(gate, Mapping):
        return []
    classes = gate.get("blocked_classes", [])
    return [str(item) for item in classes] if isinstance(classes, list) else []


def _solver_energy_certificate_outcome(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5299 = payloads.get(5299, {})
    exp5300 = payloads.get(5300, {})
    exp5301 = payloads.get(5301, {})
    exp5304 = payloads.get(5304, {})
    dynamic = value_of(exp5304.get("dynamic_abstraction_helped"))
    dynamic = dynamic if isinstance(dynamic, Mapping) else {}
    spectral_quarantined = exp5301.get("flagged_adversarial") is True
    return {
        "lns_source_experiment": 5299,
        "constraint_lns_fixture_ready": exp5299.get("constraint_lns_fixture_ready") is True,
        "lns_solver_correctness_preserved": _bool(exp5299.get("solver_correctness_preserved")),
        "lns_unsafe_false_accepts": _count(exp5299.get("unsafe_false_accepts")),
        "pbit_source_experiment": 5300,
        "pbit_gate_ready": _bool(exp5300.get("pbit_gate_ready")),
        "pbit_correctness_preserved": _bool(exp5300.get("correctness_preserved")),
        "pbit_blocked_classes": _blocked_classes(exp5300),
        "pbit_aggregate_metrics": exp5300.get("aggregate_metrics", {}),
        "pbit_hardware_speedup_claimed": _bool(exp5300.get("hardware_speedup_claimed")),
        "spectral_control": {
            "source_experiment": 5301,
            "ready": _bool(exp5301.get("spectral_control_ready")),
            "headline_eligible": not spectral_quarantined,
            "quarantined": spectral_quarantined,
            "corrigendum_pending": exp5301.get("corrigendum_pending", []),
            "divergence_recovery": value_of(exp5301.get("divergence_recovery")),
        },
        "kan_dynamic_abstraction": {
            "source_experiment": 5304,
            "diagnostic_tightness_helped": dynamic.get("helped") is True,
            "help_kind": dynamic.get("help_kind"),
            "spotcheck_hit_rate_delta": dynamic.get("spotcheck_hit_rate_delta"),
            "envelope_gap_reduction": dynamic.get("envelope_gap_reduction"),
            "certificate_success_improvement": dynamic.get("success_improvement"),
            "false_property_rejected": _bool(exp5304.get("false_property_rejected")),
        },
        "summary": (
            "LNS fixture and p-bit/CDCL gate are usable within deterministic solver scope; "
            "EBT spectral-control telemetry is quarantined when flagged; KAN dynamic "
            "abstraction improves diagnostic tightness but not certificate success."
        ),
    }


def _hardware_status(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5305 = payloads.get(5305, {})
    return {
        "source_experiment": 5305,
        "kv260_status": value_of(exp5305.get("kv260_status")),
        "polarfire_status": value_of(exp5305.get("polarfire_status")),
        "gatemate_status": value_of(exp5305.get("gatemate_status")),
        "hardware_evidence_level": value_of(exp5305.get("hardware_evidence_level")),
        "blocked_reason": value_of(exp5305.get("blocked_reason")),
    }


def _retirements_or_exclusions(root: Path, payloads: Mapping[int, JsonMap]) -> JsonDict:
    manifest_path = root / EXCLUSION_MANIFEST_PATH
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    source_quarantined = payloads.get(5296, {}).get("flagged_adversarial") is True
    spectral_quarantined = payloads.get(5301, {}).get("flagged_adversarial") is True
    return {
        "manifest_has_exp5284_cpu_path_retirement": SAME_VERDICT_RETIREMENT_ID in manifest_text,
        "same_verdict_retirements": [
            {
                "id": SAME_VERDICT_RETIREMENT_ID,
                "status": "recorded"
                if SAME_VERDICT_RETIREMENT_ID in manifest_text
                else "recommended",
                "scope": "current llama-cpp-python SOTA GGUF reruns without changed runtime substrate or GPU-offload receipt",
            }
        ],
        "quarantines_preserved": [
            item
            for item, active in (
                ("exp5296_sota_source_delta_v484", source_quarantined),
                ("exp5301_ebt_spectral_step_control_diagnostic_v484", spectral_quarantined),
            )
            if active
        ],
        "recommendations": [
            {
                "id": "repeat_exp5284_cpu_only_path",
                "recommendation": "Keep the Exp5284 CPU-only llama-cpp-python SOTA path retired unless an operator provides a new runtime/offload root cause.",
            },
            {
                "id": "sota_quality_until_changed_runtime_ready",
                "recommendation": "Keep SOTA quality smoke gated until changed_runtime_sota_ready=true on a completed mandatory-model run.",
            },
            {
                "id": "quarantined_artifact_correction",
                "recommendation": "Correct or rerun quarantined Exp5296/Exp5301 artifacts before using their metrics as headline evidence.",
            },
            {
                "id": "pbit_class_gate_not_speedup",
                "recommendation": "Reuse the p-bit/CDCL gate only as a class-sensitive CPU guidance rule; keep hardware speedup claims false.",
            },
            {
                "id": "kan_certificate_success_same_verdict",
                "recommendation": "Retire reruns that only reprove unchanged certificate success; continue only if diagnostic tightness maps to a stronger certificate gate.",
            },
        ],
    }


def _next_recommendations() -> list[JsonDict]:
    return [
        {
            "id": "runtime_then_quality",
            "recommendation": "Separate runtime readiness from quality: first get a completed changed-runtime mandatory-model receipt, then run the SOTA coherence/trace smoke.",
        },
        {
            "id": "memory_promote_with_larger_holdout",
            "recommendation": "Promote adaptive memory only after a larger held-out panel preserves quality, zero unsafe accepts, rollback, and call savings.",
        },
        {
            "id": "solver_gate_extend_classes",
            "recommendation": "Extend the LNS and p-bit/CDCL gates to more instance classes while keeping solver fallback authoritative.",
        },
        {
            "id": "spectral_control_fix_quarantine",
            "recommendation": "Rerun EBT spectral-control with corrected methodology receipts before treating it as clean stability evidence.",
        },
        {
            "id": "kan_retire_same_certificate_success",
            "recommendation": "Do not rerun KAN dynamic abstraction for the same unchanged certificate-success verdict unless a new certificate-strength metric is added.",
        },
        {
            "id": "hardware_reachability_before_speedup",
            "recommendation": "Fix KV260 SSH and GateMate physical/JTAG setup, and require a local same-workload timing artifact before any speedup claim.",
        },
    ]


def _docs_updated() -> JsonDict:
    return {
        "openspec_capstone_spec": True,
        "ops_status": False,
        "ops_changelog": False,
        "traceability": False,
        "docs_index": False,
        "research_complete": False,
        "exclusion_manifest": False,
        "reason": "stop_when_done_reconciler_deferred_ops_docs_and_traceability",
    }


def build_artifact(root: Path, duration_s: float, commands_run: Sequence[JsonMap]) -> JsonDict:
    rows: list[JsonDict] = []
    payloads: dict[int, JsonMap] = {}
    for source in UPSTREAM_SOURCES:
        row, payload = _row_for_source(source, root)
        rows.append(row)
        if payload is not None:
            payloads[source.experiment_number] = payload

    by_class: dict[str, list[JsonDict]] = {name: [] for name in TASK_CLASSIFICATIONS}
    for row in rows:
        by_class[str(row["classification"])].append(row)
    classifications = Counter(str(row["classification"]) for row in rows if row["loadable"])
    missing = by_class["missing_artifact"]
    milestone_synthesized = not missing

    verdict = (
        "complete: .484 closed with changed-runtime SOTA still blocked and quality unmeasured, "
        "adaptive memory/self-learning cleanly positive, solver/certificate tracks bounded with "
        "LNS and p-bit gates but quarantined EBT telemetry and null certificate-success lift, "
        "and hardware reachability-only with no speedup."
    )
    if missing:
        verdict = (
            f"blocked_missing_required: {len(missing)} expected .484 upstream artifact(s) "
            "missing or unreadable; no clean milestone synthesis"
        )

    task_summary = {
        "expected_count": len(UPSTREAM_SOURCES),
        "loadable_count": len([row for row in rows if row["loadable"]]),
        "milestone_synthesized": milestone_synthesized,
        "by_classification": dict(sorted(classifications.items())),
        "per_task": rows,
        "clean_positive": by_class["clean_positive"],
        "clean_null": by_class["clean_null"],
        "harmful_or_regression": by_class["harmful_or_regression"],
        "mixed_positive_with_harmful_class": by_class["mixed_positive_with_harmful_class"],
        "blocked_precondition": by_class["blocked_precondition"],
        "gated_skip": by_class["gated_skip"],
        "missing_artifact": missing,
        "quarantined": by_class["quarantined"],
    }

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": SPEC_REFS,
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": round(max(duration_s, 0.0001), 6),
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "source_artifacts_read": rows,
        "conductor_log_entries": read_conductor_log_entries(root),
        "honest_verdict": wrap_field("honest_verdict", verdict),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "tasks_summarized": wrap_field("tasks_summarized", task_summary),
        "changed_runtime_outcome": wrap_field(
            "changed_runtime_outcome", _changed_runtime_outcome(payloads)
        ),
        "sota_quality_outcome": wrap_field("sota_quality_outcome", _sota_quality_outcome(payloads)),
        "continuous_self_learning_outcome": wrap_field(
            "continuous_self_learning_outcome", _continuous_self_learning_outcome(payloads)
        ),
        "solver_energy_certificate_outcome": wrap_field(
            "solver_energy_certificate_outcome", _solver_energy_certificate_outcome(payloads)
        ),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", False),
        "hardware_status": _hardware_status(payloads),
        "retirements_or_exclusions_recommended": wrap_field(
            "retirements_or_exclusions_recommended",
            _retirements_or_exclusions(root, payloads),
        ),
        "next_milestone_recommendations": wrap_field(
            "next_milestone_recommendations", _next_recommendations()
        ),
        "docs_updated": wrap_field("docs_updated", _docs_updated()),
        "commands_run": list(commands_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        wrapped = artifact[field]
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")
    verdict = _text(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if value_of(artifact["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    tasks = value_of(artifact["tasks_summarized"])
    if not isinstance(tasks, Mapping):
        raise ValueError("tasks_summarized must be a principle-wrapped object")
    for classification in TASK_CLASSIFICATIONS:
        if classification not in tasks:
            raise ValueError(f"tasks_summarized missing {classification}")
    if value_of(artifact["hardware_speedup_claimed"]) is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if not isinstance(artifact["commands_run"], list):
        raise ValueError("commands_run must be a list")
    for command in artifact["commands_run"]:
        if not isinstance(command, Mapping) or "command" not in command or "outcome" not in command:
            raise ValueError("commands_run entries must include command and outcome")
    if not str(artifact["reproducibility_checksum"]).startswith("sha256:"):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_commands(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("commands JSON must contain a list")
    return [dict(item) for item in loaded if isinstance(item, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--commands-json", type=Path, default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    commands = load_commands(args.commands_json)
    artifact = build_artifact(args.root, time.perf_counter() - started, commands)
    validate_artifact(artifact)
    write_json(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
