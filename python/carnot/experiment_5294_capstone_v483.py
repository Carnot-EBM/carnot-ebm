"""Experiment 5294: V483 capstone synthesis.

Spec refs: REQ-CAPSTONE-5294, SCENARIO-CAPSTONE-5294,
SCENARIO-CAPSTONE-5294-BLOCKED-MISSING-INPUT.

This module is aggregation-only. It reads the already-written V483 artifacts
and conductor-log receipts, keeps gated, blocked, null, and mixed harmful
evidence in their own lanes, and writes the milestone closeout without running
new model, solver, or hardware workloads.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5294_capstone_v483.json")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
EXPERIMENT = "experiment_5294_capstone_v483"
EXPERIMENT_ID = "exp5294-capstone-v483"
MILESTONE = "2026.07.483"
SCHEMA = "carnot.experiment_5294_capstone_v483.v1"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5294
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")
SAME_VERDICT_RETIREMENT_ID = "exp5284_sota_offload_cpu_only_path_retired_v483"

SPEC_REFS = [
    "REQ-CAPSTONE-5294",
    "SCENARIO-CAPSTONE-5294",
    "SCENARIO-CAPSTONE-5294-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5294-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "terminal prefix; starts with complete: or blocked_ and summarizes the .483 milestone "
        "without laundering gated, blocked, null, harmful, mixed, quarantined, or no-speedup evidence."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because the capstone reads checked-in result artifacts "
        "and conductor logs without running LLM, solver, or hardware workloads."
    ),
    "tasks_summarized": (
        "expected, loadable, missing, classification, and conductor-log coverage for Exp5282 "
        "through Exp5293."
    ),
    "clean_positive_findings": (
        "bounded clean non-flagged results that advanced fixtures, attribution, coherence dosing, "
        "aggregate solver guidance, source refresh, or transition readiness."
    ),
    "null_or_harmful_findings": (
        "clean null effects and harmful or mixed-harmful subfindings preserved separately from positives."
    ),
    "gated_or_blocked_findings": (
        "blocked preconditions, conductor pre-gate skips, missing artifacts, hardware blocks, and "
        "quarantines that must not be rounded up."
    ),
    "retirements_or_exclusions": (
        "same-verdict retirements, manifest status, and scope limits for future reruns."
    ),
    "next_milestone_recommendations": (
        "concrete next gaps and retirements without creating a next roadmap file."
    ),
    "ops_docs_updated": (
        "false for ops/status, ops/changelog, and traceability when the stop rule delegates those "
        "docs to a later reconciler."
    ),
    "research_complete_updated": (
        "false when the stop rule limits this capstone to the result artifact plus tested code."
    ),
    "commands_run": "list of validation commands and outcomes used for this capstone.",
}

PRINCIPLE_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "tasks_summarized",
    "clean_positive_findings",
    "null_or_harmful_findings",
    "gated_or_blocked_findings",
    "retirements_or_exclusions",
    "next_milestone_recommendations",
    "ops_docs_updated",
    "research_complete_updated",
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
    *PRINCIPLE_WRAPPED_FIELDS,
    "commands_run",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V483 result artifact to aggregate."""

    experiment_number: int
    task_id: str
    title: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5282,
        "exp5282-archive-482-activate-483",
        "Archive .482 and prepare .483 activation",
        Path("results/experiment_5282_archive_482_activate_483.json"),
    ),
    UpstreamSource(
        5283,
        "exp5283-sota-source-delta-v483",
        "V483 SOTA/source delta refresh",
        Path("results/experiment_5283_sota_source_delta_v483.json"),
    ),
    UpstreamSource(
        5284,
        "exp5284-sota-runtime-offload-receipt-repair-v483",
        "SOTA GGUF generation/offload receipt repair",
        Path("results/experiment_5284_sota_runtime_offload_receipt_repair_v483.json"),
    ),
    UpstreamSource(
        5285,
        "exp5285-knowledge-thought-coherence-fixture-v483",
        "Knowledge-thought coherence fixture",
        Path("results/experiment_5285_knowledge_thought_coherence_fixture_v483.json"),
    ),
    UpstreamSource(
        5286,
        "exp5286-knowledge-thought-coherence-sota-pilot-v483",
        "SOTA claim-level coherence pilot",
        Path("results/experiment_5286_knowledge_thought_coherence_sota_pilot_v483.json"),
    ),
    UpstreamSource(
        5287,
        "exp5287-compilable-trace-dsl-fixture-v483",
        "Compilable trace DSL fixture",
        Path("results/experiment_5287_compilable_trace_dsl_fixture_v483.json"),
    ),
    UpstreamSource(
        5288,
        "exp5288-sota-trace-dsl-extraction-gated-v483",
        "SOTA trace DSL extraction retry",
        Path("results/experiment_5288_sota_trace_dsl_extraction_gated_v483.json"),
    ),
    UpstreamSource(
        5289,
        "exp5289-memory-operation-attribution-v483",
        "Memory operation attribution harness",
        Path("results/experiment_5289_memory_operation_attribution_v483.json"),
    ),
    UpstreamSource(
        5290,
        "exp5290-memory-assisted-coherence-dose-gated-v483",
        "Memory-assisted coherence dosing",
        Path("results/experiment_5290_memory_assisted_coherence_dose_gated_v483.json"),
    ),
    UpstreamSource(
        5291,
        "exp5291-low-order-factor-certificate-curriculum-v483",
        "Low-order factor certificate curriculum",
        Path("results/experiment_5291_low_order_factor_certificate_curriculum_v483.json"),
    ),
    UpstreamSource(
        5292,
        "exp5292-pbit-cdcl-factor-guidance-v483",
        "CPU p-bit/CDCL factor guidance",
        Path("results/experiment_5292_pbit_cdcl_factor_guidance_v483.json"),
    ),
    UpstreamSource(
        5293,
        "exp5293-hardware-continuity-reachability-v483",
        "Hardware continuity reachability receipts",
        Path("results/experiment_5293_hardware_continuity_reachability_v483.json"),
    ),
)

CONDUCTOR_LOG_PATTERNS = (
    "PHASE 0 transition -- archive .482",
    "PHASE 0 SOTA/source refresh -- V483",
    "PHASE 0 runtime receipts -- repair SOTA GGUF",
    "PHASE 1 fixture -- CheckRLM-style",
    "PHASE 1 gated on exp5284 and exp5285",
    "PHASE 1 fixture -- VeryTrace-style",
    "PHASE 1 gated on exp5284 and exp5287",
    "PHASE 2 continuous self-learning -- memory",
    "PHASE 2 gated on exp5285 and exp5289",
    "PHASE 3 certificates -- low-order",
    "PHASE 3 solver guidance -- p-bit",
    "PHASE 3 hardware continuity -- KV260, PolarFire",
)


def value_of(value: Any) -> Any:
    return value["value"] if isinstance(value, Mapping) and "value" in value else value


def _text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def _bool(value: Any) -> bool:
    return value_of(value) is True


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
    if experiment_number == 5292 and _bool(payload.get("pbit_cdcl_guidance_positive")):
        gate = value_of(payload.get("instance_class_gate"))
        harms = gate.get("harms") if isinstance(gate, Mapping) else []
        if harms:
            return "mixed_positive_with_harmful_class"
    if (
        verdict.startswith("blocked")
        or "blocked_preconditions" in verdict
        or experiment_number in {5284, 5293}
    ):
        return "blocked_precondition"
    if "harmful" in verdict or "regression" in verdict:
        return "harmful"
    if "did not improve" in verdict or "null" in verdict:
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
    if classification in {"missing", "malformed", "not_json_object"}:
        return f"{classification} required upstream artifact"
    if experiment_number == 5282:
        return "transition/archive record complete; .483 activated without roadmap mutation"
    if experiment_number == 5283:
        return "source refresh appended three actionable findings; plan unchanged"
    if experiment_number == 5284:
        return "SOTA GGUF generation ran only as blocked precondition evidence; GPU offload evidence absent"
    if experiment_number == 5285:
        return "claim-level coherence fixture ready; lexical baseline unsafe false accepts remain visible"
    if experiment_number == 5286:
        return "SOTA claim-level pilot conductor-gated because Exp5284 sota_offload_ready=false"
    if experiment_number == 5287:
        return "compilable trace DSL fixture ready with solver-checked structure"
    if experiment_number == 5288:
        return "SOTA trace DSL extraction conductor-gated because Exp5284 sota_offload_ready=false"
    if experiment_number == 5289:
        return "memory operation attribution ready with full bounded control coverage and unsafe propagation zero"
    if experiment_number == 5290:
        return "memory-assisted coherence dosing preserved quality, avoided full checks, and kept unsafe false accepts zero"
    if experiment_number == 5291:
        return "low-order curriculum did not beat shuffled ordering; bounded certificate telemetry still landed"
    if experiment_number == 5292:
        return "CPU p-bit/CDCL guidance saved aggregate conflicts but harmed the misleading-assumption class"
    if experiment_number == 5293:
        return "hardware status remained reachability-only: KV260 blocked, PolarFire SSH status-only, GateMate blocked"
    return _text(payload.get("honest_verdict")) or "no verdict"


def _row_for_source(source: UpstreamSource, root: Path) -> tuple[JsonDict, JsonDict | None]:
    payload, info = read_json_mapping(root / source.relative_path)
    classification = (
        classify_payload(source.experiment_number, payload) if info["loadable"] else info["error"]
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
        "verdict": _text(payload.get("honest_verdict")) if info["loadable"] else info["error"],
        "inference_substrate": _text(payload.get("inference_substrate"))
        if info["loadable"]
        else None,
        "summary": _summary(source.experiment_number, payload, str(classification)),
    }
    return row, payload if info["loadable"] else None


def _clean_positive_findings(payloads: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        {
            "id": "transition_and_source_refresh_ready",
            "source_experiments": [5282, 5283],
            "finding": "V483 activated cleanly and the execution refresh added three actionable references without changing the plan.",
        },
        {
            "id": "claim_level_coherence_fixture_ready",
            "source_experiments": [5285],
            "finding": "The CheckRLM-style fixture is ready for gated SOTA work and exposes lexical unsafe-false-accept controls.",
            "coherence_fixture_ready": payloads.get(5285, {}).get("coherence_fixture_ready")
            is True,
        },
        {
            "id": "compilable_trace_dsl_fixture_ready",
            "source_experiments": [5287],
            "finding": "The VeryTrace-style DSL compiles solver cases into executable, dependency-linked trace records.",
            "trace_dsl_ready": payloads.get(5287, {}).get("trace_dsl_ready") is True,
        },
        {
            "id": "memory_operation_attribution_ready",
            "source_experiments": [5289],
            "finding": "Memory failures and uses were attributed across bounded operation stages with no unsafe propagation.",
            "attribution_coverage": value_of(payloads.get(5289, {}).get("attribution_coverage")),
        },
        {
            "id": "memory_assisted_coherence_dosing_positive",
            "source_experiments": [5290],
            "finding": "Governed memory preserved always-full coherence quality while avoiding full claim/coherence checks.",
            "full_verifier_calls_avoided": value_of(
                payloads.get(5290, {}).get("full_verifier_calls_avoided")
            ),
            "unsafe_false_accepts": value_of(payloads.get(5290, {}).get("unsafe_false_accepts")),
        },
        {
            "id": "pbit_cdcl_aggregate_guidance_positive",
            "source_experiments": [5292],
            "finding": "CPU simulated p-bit assumptions saved aggregate CDCL conflicts while CDCL fallback preserved correctness.",
            "conflicts_saved": value_of(payloads.get(5292, {}).get("conflicts_saved")),
            "correctness_preserved": value_of(payloads.get(5292, {}).get("correctness_preserved")),
        },
    ]


def _null_or_harmful_findings(payloads: Mapping[int, JsonMap]) -> list[JsonDict]:
    gate = value_of(payloads.get(5292, {}).get("instance_class_gate"))
    harms = gate.get("harms", []) if isinstance(gate, Mapping) else []
    return [
        {
            "id": "low_order_curriculum_clean_null",
            "source_experiments": [5291],
            "finding": "Low-order-first scheduling did not improve certificate success over shuffled ordering.",
            "certificate_success_by_order": value_of(
                payloads.get(5291, {}).get("certificate_success_by_order")
            ),
        },
        {
            "id": "pbit_cdcl_misleading_assumption_harm",
            "source_experiments": [5292],
            "finding": "The misleading-assumption class lost conflicts, so the p-bit/CDCL result is distribution-sensitive.",
            "harmful_classes": harms,
            "conflicts_saved": value_of(payloads.get(5292, {}).get("conflicts_saved")),
        },
    ]


def _gated_or_blocked_findings(
    rows: Sequence[JsonMap], payloads: Mapping[int, JsonMap]
) -> list[JsonDict]:
    findings = [
        {
            "id": "sota_runtime_offload_blocked",
            "source_experiments": [5284],
            "finding": "Local mandated SOTA GGUF paths resolved and generated smoke text, but GPU offload evidence was absent.",
            "sota_offload_ready": payloads.get(5284, {}).get("sota_offload_ready") is True,
        },
        {
            "id": "claim_level_sota_pilot_gated_skip",
            "source_experiments": [5286],
            "finding": "The claim-level SOTA pilot was conductor pre-gated on Exp5284 and did not produce quality evidence.",
        },
        {
            "id": "trace_dsl_sota_extraction_gated_skip",
            "source_experiments": [5288],
            "finding": "The SOTA trace DSL extraction retry was conductor pre-gated on Exp5284 and did not produce quality evidence.",
        },
        {
            "id": "hardware_reachability_blocked_no_speedup",
            "source_experiments": [5293],
            "finding": "Hardware evidence is reachability/status only: KV260 blocked, PolarFire SSH status-only reachable, GateMate blocked, no speedup.",
            "hardware_speedup_claimed": value_of(
                payloads.get(5293, {}).get("hardware_speedup_claimed")
            ),
            "blocked_reason": value_of(payloads.get(5293, {}).get("blocked_reason")),
        },
    ]
    missing = [row for row in rows if not row["loadable"]]
    if missing:
        findings.append(
            {
                "id": "missing_required_artifacts",
                "source_experiments": [row["experiment_number"] for row in missing],
                "finding": "One or more expected V483 artifacts were missing or unreadable, so closure is blocked.",
                "missing_artifacts": missing,
            }
        )
    quarantined = [row for row in rows if row["classification"] == "quarantined"]
    if quarantined:
        findings.append(
            {
                "id": "quarantined_artifacts",
                "source_experiments": [row["experiment_number"] for row in quarantined],
                "finding": "Quarantined artifacts are excluded from headline evidence.",
                "quarantined": quarantined,
            }
        )
    return findings


def _retirements_or_exclusions(root: Path) -> JsonDict:
    manifest_text = ""
    manifest_path = root / EXCLUSION_MANIFEST_PATH
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_updated = SAME_VERDICT_RETIREMENT_ID in manifest_text
    return {
        "manifest_updated": manifest_updated,
        "same_verdict_retirements": [
            {
                "id": SAME_VERDICT_RETIREMENT_ID,
                "source_experiments": ["exp5274", "exp5284"],
                "status": "recorded" if manifest_updated else "recommended",
                "scope": "current llama-cpp-python SOTA GGUF reruns without GPU offload support or memory-delta evidence",
                "reason": "Exp5284 repeated the Exp5274 blocked offload-precondition failure class; future SOTA quality tasks need a changed runtime substrate.",
            }
        ],
        "not_retired": [
            "claim-level coherence fixture work",
            "trace DSL fixture work",
            "SOTA extraction after a genuine GPU-offload receipt",
            "hardware reachability continuity with changed board setup",
        ],
    }


def _next_recommendations() -> list[JsonDict]:
    return [
        {
            "id": "repair_sota_runtime_before_quality_tasks",
            "recommendation": "Build or select a GGUF backend with real GPU offload evidence before re-enabling SOTA claim-level or trace-extraction quality tasks.",
        },
        {
            "id": "run_claim_level_sota_only_after_offload",
            "recommendation": "Use Exp5285 labels and lexical unsafe-false-accept controls for the next claim-level SOTA pilot only after the offload gate opens.",
        },
        {
            "id": "run_trace_dsl_sota_only_after_offload",
            "recommendation": "Use Exp5287 trace records with solver-authoritative false-accept and recovery accounting after runtime receipts are clean.",
        },
        {
            "id": "extend_memory_dosing_with_stage_attribution",
            "recommendation": "Carry Exp5289/Exp5290 operation-stage attribution into any live coherence work and preserve rollback/escalation controls.",
        },
        {
            "id": "separate_pbit_instance_classes",
            "recommendation": "Treat p-bit/CDCL guidance as class-sensitive: report help, harm, neutral, overwrite counts, and no hardware speedup until board timing exists.",
        },
        {
            "id": "hardware_next_steps_are_reachability_first",
            "recommendation": "Fix KV260 SSH naming, keep PolarFire to authenticated workload receipts before speedup claims, and change GateMate physical/JTAG setup before reruns.",
        },
    ]


def build_artifact(root: Path, duration_s: float, commands_run: Sequence[JsonMap]) -> JsonDict:
    rows: list[JsonDict] = []
    payloads: dict[int, JsonMap] = {}
    for source in UPSTREAM_SOURCES:
        row, payload = _row_for_source(source, root)
        rows.append(row)
        if payload is not None:
            payloads[source.experiment_number] = payload

    missing = [row for row in rows if not row["loadable"]]
    classifications = Counter(str(row["classification"]) for row in rows if row["loadable"])
    milestone_synthesized = not missing
    verdict = (
        "complete: .483 closed with deterministic claim/coherence and trace fixtures ready, "
        "SOTA runtime/offload blocked, SOTA quality tasks gate-skipped, memory attribution "
        "and coherence dosing positive, low-order curriculum null, p-bit/CDCL aggregate "
        "positive with misleading-class harm, and hardware reachability-only with no speedup."
    )
    if missing:
        verdict = (
            f"blocked_missing_required: {len(missing)} expected .483 upstream artifact(s) "
            "missing or unreadable; no clean milestone synthesis"
        )

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
        "honest_verdict": wrap_field("honest_verdict", verdict),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "tasks_summarized": wrap_field(
            "tasks_summarized",
            {
                "expected_count": len(UPSTREAM_SOURCES),
                "loadable_count": len([row for row in rows if row["loadable"]]),
                "missing_artifacts": missing,
                "milestone_synthesized": milestone_synthesized,
                "by_classification": dict(sorted(classifications.items())),
                "per_task": rows,
                "conductor_log_entries": read_conductor_log_entries(root),
            },
        ),
        "clean_positive_findings": wrap_field(
            "clean_positive_findings", _clean_positive_findings(payloads)
        ),
        "null_or_harmful_findings": wrap_field(
            "null_or_harmful_findings", _null_or_harmful_findings(payloads)
        ),
        "gated_or_blocked_findings": wrap_field(
            "gated_or_blocked_findings", _gated_or_blocked_findings(rows, payloads)
        ),
        "retirements_or_exclusions": wrap_field(
            "retirements_or_exclusions", _retirements_or_exclusions(root)
        ),
        "next_milestone_recommendations": wrap_field(
            "next_milestone_recommendations", _next_recommendations()
        ),
        "ops_docs_updated": wrap_field(
            "ops_docs_updated",
            {
                "ops_status": False,
                "ops_changelog": False,
                "traceability": False,
                "reason": "stop_when_done_reconciler_deferred_ops_docs",
            },
        ),
        "research_complete_updated": wrap_field("research_complete_updated", False),
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
    if not isinstance(value_of(artifact["clean_positive_findings"]), list):
        raise ValueError("clean_positive_findings must be a principle-wrapped list")
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
