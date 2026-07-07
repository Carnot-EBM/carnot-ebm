"""Exp 5334: V486 capstone gate reconciliation.

Spec refs: REQ-CAPSTONE-5334, SCENARIO-CAPSTONE-5334,
SCENARIO-CAPSTONE-5334-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5334-FIELD-PRINCIPLES.

This module is deliberately an artifact synthesizer. It reads the checked-in
.486 results and conductor notes, then separates stable runtime, bounded
quality smoke, clean deterministic gates, flagged diagnostics, bounded KAN
localization, and reachability-only hardware so none of them can be rounded up
into a quality, speedup, or broad certificate claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5334_capstone_v486.json")
EXPERIMENT = "experiment_5334_capstone_v486"
EXPERIMENT_ID = "exp5334-capstone-v486"
MILESTONE = "2026.07.486"
SCHEMA = "carnot.experiment_5334_capstone_v486.v1"
RUN_DATE = "20260707"
RANDOM_SEED = 5334
INFERENCE_SUBSTRATE = "artifact_synthesis_and_gate_reconciliation"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5334",
    "SCENARIO-CAPSTONE-5334",
    "SCENARIO-CAPSTONE-5334-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5334-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "identifies Exp5334 as the `.486` capstone artifact so downstream reconciliation "
        "cannot confuse it with an upstream runtime, self-learning, KAN, or hardware task."
    ),
    "milestone": (
        "binds the aggregation to 2026.07.486 and the close-state read of Exp5321 through "
        "Exp5333."
    ),
    "status": "complete only when every expected artifact is readable; otherwise blocked_missing_required.",
    "honest_verdict": (
        "terminal prefix; starts with complete: or blocked_ and summarizes the milestone "
        "without laundering blocked, flagged, bounded, missing, no-speedup, or "
        "no-headline-quality evidence."
    ),
    "inference_substrate": (
        "artifact_synthesis_and_gate_reconciliation because the capstone reads local "
        "artifacts and conductor notes without running model, solver, or hardware workloads."
    ),
    "artifacts_read": (
        "every readable upstream artifact with path, experiment identity, status, verdict, "
        "sha256, and conductor outcome when available."
    ),
    "missing_or_blocked_artifacts": (
        "missing, malformed, blocked, flagged, and conductor-gate outcomes that must remain "
        "first-class and cannot be rounded up."
    ),
    "gate_table": (
        "one reconciled row per requested gate with source artifacts, boolean outcome, claim "
        "boundary, and blocker or caveat text."
    ),
    "next_milestone_recommendation": (
        "short next branch recommendation grounded only in the reconciled gates."
    ),
    "tests_run": (
        "validation commands and outcomes used to check the capstone module, artifact, "
        "coverage, and required repository test status."
    ),
}

WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOLEAN_FIELDS = (
    "runtime_stable",
    "sota_quality_measured",
    "rewrite_state_ready",
    "smt_corrigendum_clean",
    "context_lifecycle_ready",
    "certificate_self_learning_ready",
    "internal_signal_path_open",
    "kan_localization_ready",
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
    """One expected V486 result artifact.

    The capstone treats this list as the milestone ledger. Missing files and
    blocked or flagged results are recorded as outcomes rather than skipped, so
    downstream readers can see exactly why a gate is or is not claimable.
    """

    experiment_number: int
    task_id: str
    relative_path: Path


EXP5321 = UpstreamArtifact(
    5321,
    "exp5321-archive-485-activate-486",
    Path("results/experiment_5321_archive_485_activate_486.json"),
)
EXP5322 = UpstreamArtifact(
    5322,
    "exp5322-sota-source-delta-v486",
    Path("results/experiment_5322_sota_source_delta_v486.json"),
)
EXP5323 = UpstreamArtifact(
    5323,
    "exp5323-native-gguf-backend-flag-bisect-v486",
    Path("results/experiment_5323_native_gguf_backend_flag_bisect_v486.json"),
)
EXP5324 = UpstreamArtifact(
    5324,
    "exp5324-runtime-receipt-stabilization-v486",
    Path("results/experiment_5324_runtime_receipt_stabilization_v486.json"),
)
EXP5325 = UpstreamArtifact(
    5325,
    "exp5325-theoria-rewrite-state-fixture-v486",
    Path("results/experiment_5325_theoria_rewrite_state_fixture_v486.json"),
)
EXP5326 = UpstreamArtifact(
    5326,
    "exp5326-gated-sota-paraphrase-rewrite-smoke-v486",
    Path("results/experiment_5326_gated_sota_paraphrase_rewrite_smoke_v486.json"),
)
EXP5327 = UpstreamArtifact(
    5327,
    "exp5327-smt-hint-corrigendum-reemit-v486",
    Path("results/experiment_5327_smt_hint_corrigendum_reemit_v486.json"),
)
EXP5328 = UpstreamArtifact(
    5328,
    "exp5328-context-object-lifecycle-self-learning-v486",
    Path("results/experiment_5328_context_object_lifecycle_self_learning_v486.json"),
)
EXP5329 = UpstreamArtifact(
    5329,
    "exp5329-memory-context-policy-rollout-v486",
    Path("results/experiment_5329_memory_context_policy_rollout_v486.json"),
)
EXP5330 = UpstreamArtifact(
    5330,
    "exp5330-sea-anytime-certificate-gate-v486",
    Path("results/experiment_5330_sea_anytime_certificate_gate_v486.json"),
)
EXP5331 = UpstreamArtifact(
    5331,
    "exp5331-internal-energy-receipt-harness-v486",
    Path("results/experiment_5331_internal_energy_receipt_harness_v486.json"),
)
EXP5332 = UpstreamArtifact(
    5332,
    "exp5332-kan-counterexample-localization-v486",
    Path("results/experiment_5332_kan_counterexample_localization_v486.json"),
)
EXP5333 = UpstreamArtifact(
    5333,
    "exp5333-hardware-continuity-no-speedup-v486",
    Path("results/experiment_5333_hardware_continuity_no_speedup_v486.json"),
)

EXPECTED_ARTIFACTS = (
    EXP5321,
    EXP5322,
    EXP5323,
    EXP5324,
    EXP5325,
    EXP5326,
    EXP5327,
    EXP5328,
    EXP5329,
    EXP5330,
    EXP5331,
    EXP5332,
    EXP5333,
)


def value_of(value: Any) -> Any:
    """Return the machine value from a principle-wrapped or bare artifact field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrapped(field: str, value: Any) -> JsonDict:
    """Attach the field principle required by the capstone schema."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_value(payload: JsonMap | None, field: str, default: Any = None) -> Any:
    if payload is None:
        return default
    return value_of(payload.get(field, default))


def _status(payload: JsonMap | None) -> str:
    return str(_payload_value(payload, "status", "missing_or_unreadable"))


def _verdict(payload: JsonMap | None) -> str:
    return str(_payload_value(payload, "honest_verdict", ""))


def read_conductor_outcomes(root: Path | str = REPO_ROOT) -> dict[int, list[JsonDict]]:
    """Parse milestone conductor rows for Exp5321 through Exp5333 when present."""

    path = Path(root) / "ops/conductor-log.md"
    outcomes: dict[int, list[JsonDict]] = {source.experiment_number: [] for source in EXPECTED_ARTIFACTS}
    if not path.exists():
        return outcomes
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for source in EXPECTED_ARTIFACTS:
            if f"Exp {source.experiment_number}" not in line and f"Exp{source.experiment_number}" not in line:
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


def latest_conductor_outcome(outcomes: Mapping[int, Sequence[JsonDict]], experiment_number: int) -> JsonDict | None:
    rows = list(outcomes.get(experiment_number, ()))
    return rows[-1] if rows else None


def read_upstream_artifacts(
    root: Path | str = REPO_ROOT,
    conductor_outcomes: Mapping[int, Sequence[JsonDict]] | None = None,
) -> tuple[dict[int, JsonDict], list[JsonDict], list[JsonDict]]:
    """Read every expected V486 artifact and preserve missing or malformed inputs."""

    root_path = Path(root)
    conductor = conductor_outcomes or {}
    payloads: dict[int, JsonDict] = {}
    artifacts_read: list[JsonDict] = []
    missing: list[JsonDict] = []
    for source in EXPECTED_ARTIFACTS:
        path = root_path / source.relative_path
        if not path.exists():
            missing.append(
                {
                    "experiment_number": source.experiment_number,
                    "path": str(source.relative_path),
                    "classification": "missing",
                    "reason": "missing",
                }
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing.append(
                {
                    "experiment_number": source.experiment_number,
                    "path": str(source.relative_path),
                    "classification": "malformed",
                    "reason": f"malformed_json:{exc.msg}",
                }
            )
            continue
        if not isinstance(payload, dict):
            missing.append(
                {
                    "experiment_number": source.experiment_number,
                    "path": str(source.relative_path),
                    "classification": "malformed",
                    "reason": "not_json_object",
                }
            )
            continue
        latest = latest_conductor_outcome(conductor, source.experiment_number)
        payloads[source.experiment_number] = payload
        artifacts_read.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": str(source.relative_path),
                "experiment_id": value_of(payload.get("experiment_id")) or source.task_id,
                "status": _status(payload),
                "honest_verdict": _verdict(payload),
                "flagged_adversarial": value_of(payload.get("flagged_adversarial")) is True,
                "conductor_outcome": latest,
                "sha256": _sha256(path),
            }
        )
    return payloads, artifacts_read, missing


def _blocked_or_flagged_rows(
    payloads: Mapping[int, JsonDict],
    missing: Sequence[JsonDict],
    conductor_outcomes: Mapping[int, Sequence[JsonDict]],
) -> list[JsonDict]:
    rows = [dict(row) for row in missing]
    missing_numbers = {int(row["experiment_number"]) for row in missing}
    for source in EXPECTED_ARTIFACTS:
        if source.experiment_number in missing_numbers:
            continue
        payload = payloads.get(source.experiment_number)
        verdict = _verdict(payload)
        status = _status(payload)
        latest = latest_conductor_outcome(conductor_outcomes, source.experiment_number)
        latest_status = str((latest or {}).get("status", ""))
        flagged = _payload_value(payload, "flagged_adversarial") is True or latest_status == "FLAGGED"
        blocked = status == "blocked" or status.startswith("blocked_") or verdict.startswith("blocked_")
        conductor_gate = latest_status in {"GATE_BLOCK", "SKIP"}
        classification = (
            "blocked"
            if blocked
            else "flagged"
            if flagged
            else "conductor_gate_skip"
            if conductor_gate
            else None
        )
        if classification is None:
            continue
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "path": str(source.relative_path),
                "classification": classification,
                "status": status,
                "honest_verdict": verdict,
                "conductor_outcome": latest,
            }
        )
    return rows


def _gate_table(payloads: Mapping[int, JsonDict], missing_numbers: set[int]) -> list[JsonDict]:
    exp5324 = payloads.get(5324)
    exp5325 = payloads.get(5325)
    exp5326 = payloads.get(5326)
    exp5327 = payloads.get(5327)
    exp5328 = payloads.get(5328)
    exp5329 = payloads.get(5329)
    exp5330 = payloads.get(5330)
    exp5331 = payloads.get(5331)
    exp5332 = payloads.get(5332)
    exp5333 = payloads.get(5333)

    runtime_ready = 5324 not in missing_numbers and _payload_value(exp5324, "sota_runtime_unblocked_stable") is True
    quality_measured = 5326 not in missing_numbers and _payload_value(exp5326, "sota_quality_measured") is True
    rewrite_ready = 5325 not in missing_numbers and _payload_value(exp5325, "rewrite_state_fixture_ready") is True
    smt_clean = 5327 not in missing_numbers and _payload_value(exp5327, "smt_hint_protocol_clean") is True
    lifecycle_ready = bool(
        5328 not in missing_numbers
        and 5329 not in missing_numbers
        and _payload_value(exp5328, "context_lifecycle_fixture_ready") is True
        and _payload_value(exp5329, "policy_rollout_ready") is True
        and _payload_value(exp5329, "unsafe_false_accepts", 1) == 0
    )
    certificate_ready = bool(
        5330 not in missing_numbers
        and _payload_value(exp5330, "anytime_certificate_gate_ready") is True
        and _payload_value(exp5330, "unsafe_promotions", 1) == 0
    )
    internal_open = bool(
        5331 not in missing_numbers
        and _payload_value(exp5331, "internal_signal_receipt_ready") is True
        and _payload_value(exp5331, "token_probability_available") is True
    )
    internal_flagged = _payload_value(exp5331, "flagged_adversarial") is True or bool(
        _payload_value(exp5331, "corrigendum_pending", [])
    )
    kan_ready = bool(
        5332 not in missing_numbers
        and _payload_value(exp5332, "counterexample_localization_ready") is True
        and _payload_value(exp5332, "no_broad_certificate_claim") is True
    )
    hardware_speedup = _payload_value(exp5333, "speedup_claim") is True

    return [
        {
            "gate": "runtime",
            "source_experiments": [5323, 5324],
            "ready": runtime_ready,
            "classification": "stable_runtime_no_quality_claim" if runtime_ready else "not_stable",
            "claim_boundary": "runtime stability only; no quality claim",
            "evidence": {
                "sota_runtime_unblocked_stable": _payload_value(exp5324, "sota_runtime_unblocked_stable"),
                "quality_claim_permitted": _payload_value(exp5324, "quality_claim_permitted"),
            },
        },
        {
            "gate": "sota_quality",
            "source_experiments": [5326],
            "ready": quality_measured,
            "classification": "bounded_smoke_measured_no_headline_claim"
            if quality_measured
            else "not_measured",
            "claim_boundary": "bounded fixture-scored smoke; no headline quality claim",
            "evidence": {
                "sota_quality_measured": _payload_value(exp5326, "sota_quality_measured"),
                "headline_quality_claim": _payload_value(exp5326, "headline_quality_claim"),
                "paraphrase_label_preservation_rate": _payload_value(
                    exp5326, "paraphrase_label_preservation_rate"
                ),
                "rewrite_acceptability_rate": _payload_value(exp5326, "rewrite_acceptability_rate"),
            },
        },
        {
            "gate": "rewrite_state_verification",
            "source_experiments": [5325],
            "ready": rewrite_ready,
            "classification": "deterministic_fixture_ready" if rewrite_ready else "fixture_not_ready",
            "claim_boundary": "typed deterministic rewrite-state fixture only",
            "evidence": {
                "rewrite_state_fixture_ready": _payload_value(exp5325, "rewrite_state_fixture_ready"),
                "rewrite_case_count": _payload_value(exp5325, "rewrite_case_count"),
                "unsafe_rewrite_rejection_rate": _payload_value(
                    exp5325, "unsafe_rewrite_rejection_rate"
                ),
            },
        },
        {
            "gate": "smt_corrigendum",
            "source_experiments": [5327],
            "ready": smt_clean,
            "classification": "clean_deterministic_solver_corrigendum" if smt_clean else "not_clean",
            "claim_boundary": "deterministic SMT solver/protocol validation; no LLM proposer claim",
            "evidence": {
                "smt_hint_protocol_clean": _payload_value(exp5327, "smt_hint_protocol_clean"),
                "solver_fallback_complete": _payload_value(exp5327, "solver_fallback_complete"),
                "llm_invoked": _payload_value(exp5327, "llm_invoked"),
                "unsound_hint_rejection_rate": _payload_value(
                    exp5327, "unsound_hint_rejection_rate"
                ),
            },
        },
        {
            "gate": "context_lifecycle",
            "source_experiments": [5328, 5329],
            "ready": lifecycle_ready,
            "classification": "lifecycle_policy_rollout_ready" if lifecycle_ready else "not_ready",
            "claim_boundary": "deterministic context policy rollout with frozen model weights",
            "evidence": {
                "context_lifecycle_fixture_ready": _payload_value(
                    exp5328, "context_lifecycle_fixture_ready"
                ),
                "policy_rollout_ready": _payload_value(exp5329, "policy_rollout_ready"),
                "quality_delta_vs_always_full": _payload_value(
                    exp5329, "quality_delta_vs_always_full"
                ),
                "unsafe_false_accepts": _payload_value(exp5329, "unsafe_false_accepts"),
                "no_weight_mutation": _payload_value(exp5329, "no_weight_mutation"),
            },
        },
        {
            "gate": "certificate_self_learning",
            "source_experiments": [5330],
            "ready": certificate_ready,
            "classification": "certificate_gate_ready" if certificate_ready else "not_ready",
            "claim_boundary": "policy promotion certificate only; no foundation-weight mutation",
            "evidence": {
                "anytime_certificate_gate_ready": _payload_value(
                    exp5330, "anytime_certificate_gate_ready"
                ),
                "policy_promotions": _payload_value(exp5330, "policy_promotions"),
                "policy_rejections": _payload_value(exp5330, "policy_rejections"),
                "unsafe_promotions": _payload_value(exp5330, "unsafe_promotions"),
                "no_weight_mutation": _payload_value(exp5330, "no_weight_mutation"),
            },
        },
        {
            "gate": "internal_signal_receipts",
            "source_experiments": [5331],
            "ready": internal_open,
            "classification": "open_but_flagged" if internal_open and internal_flagged else "open_clean" if internal_open else "closed",
            "claim_boundary": "token-probability receipt path only; flagged diagnostics are not quality claims",
            "evidence": {
                "internal_signal_receipt_ready": _payload_value(
                    exp5331, "internal_signal_receipt_ready"
                ),
                "token_probability_available": _payload_value(
                    exp5331, "token_probability_available"
                ),
                "logits_available": _payload_value(exp5331, "logits_available"),
                "attention_available": _payload_value(exp5331, "attention_available"),
                "flagged_adversarial": _payload_value(exp5331, "flagged_adversarial"),
                "no_quality_claim": _payload_value(exp5331, "no_quality_claim"),
            },
        },
        {
            "gate": "kan_localization",
            "source_experiments": [5332],
            "ready": kan_ready,
            "classification": "bounded_localization_ready" if kan_ready else "not_ready",
            "claim_boundary": "bounded false-property localization; no broad KAN certificate claim",
            "evidence": {
                "counterexample_localization_ready": _payload_value(
                    exp5332, "counterexample_localization_ready"
                ),
                "counterexample_localization_accuracy": _payload_value(
                    exp5332, "counterexample_localization_accuracy"
                ),
                "no_broad_certificate_claim": _payload_value(
                    exp5332, "no_broad_certificate_claim"
                ),
                "certificate_success_delta": _payload_value(exp5332, "certificate_success_delta"),
            },
        },
        {
            "gate": "hardware",
            "source_experiments": [5333],
            "ready": False,
            "classification": "speedup_claim_present" if hardware_speedup else "reachability_only_no_speedup",
            "claim_boundary": "reachability receipts only; no authenticated workload or speedup claim",
            "evidence": {
                "speedup_claim": _payload_value(exp5333, "speedup_claim"),
                "authenticated_workload_run": _payload_value(exp5333, "authenticated_workload_run"),
                "hardware_evidence_level": _payload_value(exp5333, "hardware_evidence_level"),
                "status": _status(exp5333),
            },
        },
    ]


def gate_value(gates: Sequence[JsonMap], gate: str) -> bool:
    return any(row.get("gate") == gate and row.get("ready") is True for row in gates)


def next_milestone_recommendation(gates: Sequence[JsonMap]) -> JsonDict:
    """Choose the short next branch from the reconciled gate state."""

    internal_row = next(row for row in gates if row["gate"] == "internal_signal_receipts")
    quality_row = next(row for row in gates if row["gate"] == "sota_quality")
    return {
        "recommendation": "self_learning_scale_up",
        "why": (
            "runtime is stable, bounded quality smoke has run without headline claim, "
            "rewrite/SMT/context/certificate gates are clean, and self-learning now has "
            "the clearest clean scale-up path."
        ),
        "carry_forward_caveats": [
            f"internal_signal={internal_row['classification']}",
            f"sota_quality={quality_row['classification']}",
            "hardware=reachability_only_no_speedup",
            "kan=bounded_localization_no_broad_certificate_claim",
        ],
        "do_not_claim": ["hardware_speedup", "headline_sota_quality", "broad_kan_certificate"],
    }


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
    payloads, artifacts_read, missing = read_upstream_artifacts(root, conductor_outcomes)
    missing_numbers = {int(row["experiment_number"]) for row in missing}
    gates = _gate_table(payloads, missing_numbers)
    blocked_rows = _blocked_or_flagged_rows(payloads, missing, conductor_outcomes)
    all_artifacts_read = not any(row["classification"] in {"missing", "malformed"} for row in missing)
    status = "complete" if all_artifacts_read else "blocked_missing_required"
    verdict_prefix = "complete:" if all_artifacts_read else "blocked_missing_required:"
    verdict = (
        f"{verdict_prefix} .486 synthesized with runtime_stable={gate_value(gates, 'runtime')}, "
        f"sota_quality_measured={gate_value(gates, 'sota_quality')}, "
        f"rewrite_smt_self_learning_clean={gate_value(gates, 'rewrite_state_verification') and gate_value(gates, 'smt_corrigendum') and gate_value(gates, 'certificate_self_learning')}, "
        f"internal_signal={next(row for row in gates if row['gate'] == 'internal_signal_receipts')['classification']}, "
        f"kan_localization_ready={gate_value(gates, 'kan_localization')}, "
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
        "missing_or_blocked_artifacts": wrapped("missing_or_blocked_artifacts", blocked_rows),
        "gate_table": wrapped("gate_table", gates),
        "runtime_stable": gate_value(gates, "runtime"),
        "sota_quality_measured": gate_value(gates, "sota_quality"),
        "rewrite_state_ready": gate_value(gates, "rewrite_state_verification"),
        "smt_corrigendum_clean": gate_value(gates, "smt_corrigendum"),
        "context_lifecycle_ready": gate_value(gates, "context_lifecycle"),
        "certificate_self_learning_ready": gate_value(gates, "certificate_self_learning"),
        "internal_signal_path_open": gate_value(gates, "internal_signal_receipts"),
        "kan_localization_ready": gate_value(gates, "kan_localization"),
        "hardware_speedup_claim": False,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "next_milestone_recommendation": wrapped(
            "next_milestone_recommendation", next_milestone_recommendation(gates)
        ),
        "tests_run": wrapped(
            "tests_run", [dict(row) for row in (tests_run if tests_run is not None else default_tests_run())]
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
        if not isinstance(value, Mapping) or value.get("principle") != FIELD_PRINCIPLES[field] or "value" not in value:
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
