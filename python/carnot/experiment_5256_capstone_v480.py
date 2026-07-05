"""Exp 5256: V480 milestone-close capstone synthesis.

Spec refs: REQ-CAPSTONE-5256, SCENARIO-CAPSTONE-5256,
SCENARIO-CAPSTONE-5256-FIELD-PRINCIPLES.

This module is a conservative record reader. It does not rerun experiments, call
models, or promote partial evidence. Its job is to preserve the exact close
state of milestone `.480`, including missing, gated, blocked, clean-negative,
zero-delta, and bounded-certificate outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5256_capstone_v480.json")
EXPERIMENT = "experiment_5256_capstone_v480"
EXPERIMENT_ID = "exp5256-capstone-v480"
MILESTONE = "2026.07.480"
SCHEMA = "carnot.experiment_5256_capstone_v480.v1"
RUN_DATE = "20260705"
RANDOM_SEED = 5256
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-CAPSTONE-5256",
    "SCENARIO-CAPSTONE-5256",
    "SCENARIO-CAPSTONE-5256-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state the honest .480 close "
        "state without laundering blocked, skipped, negative, zero-delta, or bounded artifacts."
    ),
    "inference_substrate": (
        "cached_fixture_replay_no_llm because the capstone reads existing artifacts "
        "and does not invoke an LLM."
    ),
    "tasks_seen": "integer count of loadable expected .480 upstream artifacts read directly.",
    "tasks_missing_or_skipped": (
        "list of missing, gated, skipped, or blocked upstream tasks that must not "
        "be rounded up into success."
    ),
    "artifact_normalizer_status": (
        "Exp5247 status; ready only for strict shape-only repairs that refuse "
        "missing evidence synthesis."
    ),
    "gap4_final_status": (
        "Exp5248 status; salvaged_clean_null only from receipt-backed wins/losses/"
        "ties and not a new generation win."
    ),
    "continuous_self_learning_status": (
        "Exp5249 status; blocked unless cross-model typed memory transfer is "
        "actually measured with rollback, retention, and leakage checks."
    ),
    "verifier_dose_status": (
        "Exp5250 status; blocked when the pre-gate on cross_model_memory_eligible fails."
    ),
    "token_guard_status": (
        "Exp5251 status; clean negative or positive only from local GGUF receipts "
        "and deterministic gates."
    ),
    "halluhard_status": (
        "Exp5252 status; clean null/positive only from local GGUF provenance-memory "
        "microbench receipts."
    ),
    "arc_level_delta": (
        "integer delta from clean reproduction-gated ARC evidence only; zero is "
        "valid and must not be inflated."
    ),
    "kan_certificate_status": (
        "Exp5254 status; bounded certificate only, not broad KAN verification or hardware speedup."
    ),
    "hardware_speedup_claimed": "must be false unless authenticated timing evidence exists.",
    "ops_docs_updated": (
        "false when the stop rule delegates ops/status/changelog/traceability "
        "reconciliation to a later conductor step."
    ),
    "recommended_next_tasks": (
        "3-5 concrete next-milestone candidates with retired-scope warnings where applicable."
    ),
}

PRINCIPLE_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)

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
    "per_task_summary",
    "status_decisions",
    "validation_commands_run",
    "research_conductor_py_untouched_confirmed",
    "random_seed",
    "reproducibility_checksum",
    "flagged_adversarial",
    *PRINCIPLE_WRAPPED_FIELDS,
)


@dataclass(frozen=True)
class UpstreamSource:
    """One expected V480 upstream deliverable."""

    experiment_number: int
    task_id: str
    relative_path: Path


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5245,
        "exp5245-archive-479-activate-480",
        Path("results/experiment_5245_archive_479_activate_480.json"),
    ),
    UpstreamSource(
        5246, "exp5246-sota-refresh-v480", Path("results/experiment_5246_sota_refresh_v480.json")
    ),
    UpstreamSource(
        5247,
        "exp5247-slot-artifact-normalizer-v480",
        Path("results/experiment_5247_slot_artifact_normalizer_v480.json"),
    ),
    UpstreamSource(
        5248,
        "exp5248-gap4-receipt-salvage-or-retire-v480",
        Path("results/experiment_5248_gap4_receipt_salvage_or_retire_v480.json"),
    ),
    UpstreamSource(
        5249,
        "exp5249-cross-model-typed-memory-transfer-v480",
        Path("results/experiment_5249_cross_model_typed_memory_transfer_v480.json"),
    ),
    UpstreamSource(
        5250,
        "exp5250-verifier-dose-scheduler-v480",
        Path("results/experiment_5250_verifier_dose_scheduler_v480.json"),
    ),
    UpstreamSource(
        5251,
        "exp5251-token-guard-carnot-pilot-v480",
        Path("results/experiment_5251_token_guard_carnot_pilot_v480.json"),
    ),
    UpstreamSource(
        5252,
        "exp5252-halluhard-provenance-memory-microbench-v480",
        Path("results/experiment_5252_halluhard_provenance_memory_microbench_v480.json"),
    ),
    UpstreamSource(
        5253,
        "exp5253-arc-live-patch-clean-receipts-v480",
        Path("results/experiment_5253_arc_live_patch_clean_receipts_v480.json"),
    ),
    UpstreamSource(
        5254,
        "exp5254-kan-convex-envelope-certificate-v480",
        Path("results/experiment_5254_kan_convex_envelope_certificate_v480.json"),
    ),
    UpstreamSource(
        5255,
        "exp5255-hardware-continuity-pkit-boundary-v480",
        Path("results/experiment_5255_hardware_continuity_pkit_boundary_v480.json"),
    ),
)

SOURCE_CONTEXT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("research-complete.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("_bmad/traceability.md"),
)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive corrupt-artifact guard.
        return {}, {"exists": True, "loadable": False, "error": f"malformed_json:{exc.msg}"}
    if not isinstance(parsed, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return dict(parsed), {"exists": True, "loadable": True, "error": None}


def _text(value: Any) -> str:
    raw = value_of(value)
    return raw if isinstance(raw, str) else ""


def _bool(value: Any) -> bool:
    return value_of(value) is True


def _int(value: Any) -> int:
    raw = value_of(value)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0


def _float(value: Any) -> float:
    raw = value_of(value)
    if isinstance(raw, bool) or raw is None:
        return 0.0
    return float(raw)


def _list(value: Any) -> list[Any]:
    raw = value_of(value)
    return raw if isinstance(raw, list) else []


def _classify(source: UpstreamSource, payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("loadable"):
        return "missing"
    verdict = _text(payload.get("honest_verdict"))
    if payload.get("blocked_at_layer") == "conductor_pre_gate":
        return "gated_skipped"
    if payload.get("status") == "blocked" or verdict.startswith("blocked"):
        return "blocked"
    if payload.get("flagged_adversarial") is True:
        return "flagged"
    if source.experiment_number == 5253 and _int(payload.get("level_delta")) == 0:
        return "clean_zero_delta"
    return "loaded"


def _methodology_flags(payload: JsonMap) -> JsonDict:
    flags: JsonDict = {}
    for field in (
        "methodology_flags",
        "methodology_warnings",
        "adversarial_flags",
        "corrigendum_pending",
        "schema_errors",
        "gate_check_summary",
    ):
        if field in payload:
            flags[field] = payload[field]
    return flags


def _source_artifacts(root: Path) -> tuple[list[JsonDict], dict[int, JsonDict], dict[int, str]]:
    rows: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}
    statuses: dict[int, str] = {}
    for source in UPSTREAM_SOURCES:
        path = root / source.relative_path
        payload, meta = read_json_mapping(path)
        status = _classify(source, payload, meta)
        payloads[source.experiment_number] = payload
        statuses[source.experiment_number] = status
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": str(source.relative_path),
                "exists": bool(meta["exists"]),
                "loadable": bool(meta["loadable"]),
                "status": status,
                "error": meta["error"],
                "sha256": file_sha256(path),
                "honest_verdict": _text(payload.get("honest_verdict")),
                "flagged_adversarial": payload.get("flagged_adversarial"),
                "inference_substrate": _text(payload.get("inference_substrate")),
                "methodology_flags": _methodology_flags(payload),
            }
        )
    return rows, payloads, statuses


def _source_context(root: Path) -> list[JsonDict]:
    return [
        {
            "path": str(path),
            "exists": (root / path).exists(),
            "sha256": file_sha256(root / path),
        }
        for path in SOURCE_CONTEXT_PATHS
    ]


def _missing_or_skipped(source_rows: Sequence[JsonMap]) -> list[JsonDict]:
    return [
        {
            "experiment_number": int(row["experiment_number"]),
            "task_id": str(row["task_id"]),
            "path": str(row["path"]),
            "status": str(row["status"]),
            "honest_verdict": str(row["honest_verdict"]),
            "reason": str(row["error"] or row["honest_verdict"] or row["status"]),
        }
        for row in source_rows
        if row["status"] in {"missing", "blocked", "gated_skipped"}
    ]


def _artifact_normalizer_status(payload: JsonMap) -> str:
    if value_of(payload.get("artifact_normalizer_ready")) is True:
        return "ready: strict_shape_only_normalizer_refuses_missing_evidence"
    return "blocked_or_missing: artifact_normalizer_ready was not true"


def _gap4_status(payload: JsonMap) -> str:
    decision = _text(payload.get("gap4_final_decision"))
    wins = _int(payload.get("wins"))
    losses = _int(payload.get("losses"))
    ties = _int(payload.get("ties"))
    unsafe = len(_list(payload.get("unsafe_missing_receipts")))
    retired = str(_bool(payload.get("pool_retired"))).lower()
    return (
        f"{decision or 'blocked_or_missing'}: wins={wins} losses={losses} "
        f"ties={ties} unsafe_missing_receipts={unsafe} pool_retired={retired}"
    )


def _cross_model_memory_status(payload: JsonMap) -> str:
    model_specs = value_of(payload.get("model_specs"))
    audit = model_specs.get("precondition_audit", {}) if isinstance(model_specs, Mapping) else {}
    blockers = audit.get("blockers", []) if isinstance(audit, Mapping) else []
    blocker_text = ",".join(str(item) for item in blockers) or "none"
    eligible = value_of(payload.get("cross_model_memory_eligible"))
    no_training = str(_bool(payload.get("no_model_training"))).lower()
    return (
        f"blocked_precondition: cross_model_memory_eligible={eligible} "
        f"blockers={blocker_text} retention=false rollback=false no_model_training={no_training}"
    )


def _verifier_dose_status(payload: JsonMap) -> str:
    if payload.get("blocked_at_layer") == "conductor_pre_gate":
        return "blocked_gate: exp5249.cross_model_memory_eligible=false; scheduler_not_run"
    return "not_blocked: scheduler artifact did not report conductor_pre_gate"


def _token_guard_status(payload: JsonMap) -> str:
    return (
        "clean_negative: harmful_on_bounded_panel "
        f"accuracy_change={_float(payload.get('accuracy_change')):.2f} "
        f"unsupported_claim_delta={_float(payload.get('unsupported_claim_delta')):.1f} "
        f"deterministic_violation_delta={_float(payload.get('deterministic_violation_delta')):.1f} "
        f"recommendation={_text(payload.get('consumer_recommendation'))}"
    )


def _halluhard_status(payload: JsonMap) -> str:
    return (
        "clean_null: typed_provenance_memory "
        f"citation_support_delta={_float(payload.get('citation_support_delta')):.1f} "
        f"repeated_error_delta={_float(payload.get('repeated_error_delta')):.1f} "
        f"unsupported_claim_rate_no_memory={_float(payload.get('unsupported_claim_rate_no_memory')):.1f} "
        f"unsupported_claim_rate_typed_memory={_float(payload.get('unsupported_claim_rate_typed_memory')):.1f}"
    )


def _kan_status(payload: JsonMap) -> str:
    return (
        "bounded_positive: "
        f"variables={_int(payload.get('variables_verified'))} "
        f"envelopes={_int(payload.get('max_segments_or_envelopes_verified'))} "
        f"true_property_certified={str(_bool(payload.get('true_property_certified'))).lower()} "
        f"false_property_rejected={str(_bool(payload.get('false_property_rejected'))).lower()} "
        f"no_hardware_speedup_claim={str(_bool(payload.get('no_hardware_speedup_claim'))).lower()}"
    )


def _recommended_next_tasks() -> list[str]:
    return [
        "Fix local GGUF GPU-offload/runtime preconditions, then rerun cross-model typed memory before any verifier-dose scheduler claim.",
        "Keep GAP-4 as a salvaged clean null for the current pool; any next GAP-4 task must change candidate generation or selection rather than rescore the same frozen pool.",
        "Retire the current ARC provenance patch after the clean zero-delta rerun; the next ARC attempt needs a new live-agent trajectory generator or route, not a duplicate patch replay.",
        "Redesign or retire the current Token-Guard fragment gate because it worsened accuracy and deterministic violations on the bounded panel.",
        "Extend the bounded KAN convex certificate only with false-property witnesses and no hardware-speedup language; keep hardware continuity receipt-only until authenticated workload timing exists.",
    ]


def _status_decisions(payloads: Mapping[int, JsonMap]) -> JsonDict:
    return {
        "archive_activation": "blocked_archive_record_preserved",
        "artifact_normalizer": _artifact_normalizer_status(payloads[5247]),
        "gap4": _gap4_status(payloads[5248]),
        "continuous_self_learning": _cross_model_memory_status(payloads[5249]),
        "verifier_dose": _verifier_dose_status(payloads[5250]),
        "token_guard": _token_guard_status(payloads[5251]),
        "halluhard": _halluhard_status(payloads[5252]),
        "arc_patch": "retire_current_provenance_patch_after_clean_zero_delta",
        "kan_certificate": _kan_status(payloads[5254]),
        "hardware": "kv260=reachable polarfire=reachable gatemate=blocked_physical_jtag no_speedup_claim",
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    validation_commands_run: Sequence[JsonMap] = (),
    conductor_untouched: bool = True,
    ops_docs_updated: bool = False,
) -> JsonDict:
    root_path = Path(root)
    source_rows, payloads, statuses = _source_artifacts(root_path)
    tasks_seen = sum(1 for row in source_rows if row["loadable"])
    missing_or_skipped = _missing_or_skipped(source_rows)
    status_decisions = _status_decisions(payloads)
    arc_delta = _int(payloads[5253].get("level_delta"))
    hardware_speedup = _bool(payloads[5255].get("speedup_claimed"))

    verdict = (
        "complete: .480 capstone closed with normalizer ready; GAP-4 salvaged "
        "clean null; cross-model memory and verifier dose blocked; Token-Guard "
        "harmful; HalluHard clean null; ARC delta 0 with patch retired; KAN "
        "bounded positive; hardware no-speedup."
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "spec_refs": SPEC_REFS,
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": float(duration_s),
        "source_artifacts": source_rows,
        "source_context": _source_context(root_path),
        "per_task_summary": {
            row["task_id"]: {
                "experiment_number": row["experiment_number"],
                "status": row["status"],
                "headline_eligible": row["status"] in {"loaded", "clean_zero_delta"},
                "honest_verdict": row["honest_verdict"],
                "methodology_flags": row["methodology_flags"],
            }
            for row in source_rows
        },
        "status_decisions": status_decisions,
        "validation_commands_run": [dict(row) for row in validation_commands_run],
        "research_conductor_py_untouched_confirmed": bool(conductor_untouched),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "flagged_adversarial": False,
        "honest_verdict": wrap_field("honest_verdict", verdict),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "tasks_seen": wrap_field("tasks_seen", tasks_seen),
        "tasks_missing_or_skipped": wrap_field("tasks_missing_or_skipped", missing_or_skipped),
        "artifact_normalizer_status": wrap_field(
            "artifact_normalizer_status",
            status_decisions["artifact_normalizer"],
        ),
        "gap4_final_status": wrap_field("gap4_final_status", status_decisions["gap4"]),
        "continuous_self_learning_status": wrap_field(
            "continuous_self_learning_status",
            status_decisions["continuous_self_learning"],
        ),
        "verifier_dose_status": wrap_field(
            "verifier_dose_status", status_decisions["verifier_dose"]
        ),
        "token_guard_status": wrap_field("token_guard_status", status_decisions["token_guard"]),
        "halluhard_status": wrap_field("halluhard_status", status_decisions["halluhard"]),
        "arc_level_delta": wrap_field("arc_level_delta", arc_delta),
        "kan_certificate_status": wrap_field(
            "kan_certificate_status", status_decisions["kan_certificate"]
        ),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", hardware_speedup),
        "ops_docs_updated": wrap_field("ops_docs_updated", bool(ops_docs_updated)),
        "recommended_next_tasks": wrap_field("recommended_next_tasks", _recommended_next_tasks()),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if statuses[5246] != "missing":
        artifact["status_decisions"]["unexpected_5246_state"] = statuses[5246]
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            raise AssertionError(f"missing required field: {field}")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        wrapped = artifact[field]
        if not isinstance(wrapped, Mapping) or wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise AssertionError(f"{field} must be principle-wrapped")
    verdict = _text(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise AssertionError("honest_verdict must start with complete: or blocked_")
    if _text(artifact["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise AssertionError("inference_substrate must be cached_fixture_replay_no_llm")
    if not isinstance(value_of(artifact["tasks_seen"]), int) or isinstance(
        value_of(artifact["tasks_seen"]), bool
    ):
        raise AssertionError("tasks_seen must be an integer")
    if not isinstance(value_of(artifact["tasks_missing_or_skipped"]), list):
        raise AssertionError("tasks_missing_or_skipped must be a list")
    if not isinstance(value_of(artifact["arc_level_delta"]), int) or isinstance(
        value_of(artifact["arc_level_delta"]), bool
    ):
        raise AssertionError("arc_level_delta must be an integer")
    if value_of(artifact["hardware_speedup_claimed"]) is not False:
        raise AssertionError("hardware speedup claim is not supported")
    if value_of(artifact["ops_docs_updated"]) is not False:
        raise AssertionError("ops_docs_updated must remain false for this stop-rule run")
    next_tasks = value_of(artifact["recommended_next_tasks"])
    if not isinstance(next_tasks, list) or not 3 <= len(next_tasks) <= 5:
        raise AssertionError("recommended_next_tasks must contain 3-5 items")
    if artifact.get("flagged_adversarial") is not False:
        raise AssertionError("capstone itself must not be flagged_adversarial")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise AssertionError("reproducibility_checksum mismatch")


def write_artifact(path: Path, artifact: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
