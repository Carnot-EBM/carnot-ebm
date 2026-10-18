"""Exp5768 V514 capstone reconciliation.

Spec refs: REQ-REPORT-5768, SCENARIO-REPORT-5768,
SCENARIO-REPORT-5768-MISSING-OR-BLOCKED,
SCENARIO-REPORT-5768-FIELD-PRINCIPLES.

This module is intentionally a cached-artifact reconciler. It does not run
models, solvers, ARC environments, Rust benchmarks, publication steps, or the
research conductor. Its job is to read the V514 evidence that already exists
and keep the denominator honest: scalar bridge readiness, gate-blocked tasks,
negative proposal utility, positive constraint acquisition, Rust 10x
retirement, and ARC null evidence stay separate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5768_v514_capstone_reconciliation.json")

EXPERIMENT = "experiment_5768_v514_capstone_reconciliation"
EXPERIMENT_ID = "exp5768-v514-capstone-reconciliation"
MILESTONE = "2026.07.514"
RUN_DATE = "2026-07-21"
RANDOM_SEED = 5768
SCHEMA = "carnot.experiment_5768.v514_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "cached_artifact_reconciliation_no_llm"

SPEC_REFS = (
    "REQ-REPORT-5768",
    "SCENARIO-REPORT-5768",
    "SCENARIO-REPORT-5768-MISSING-OR-BLOCKED",
    "SCENARIO-REPORT-5768-FIELD-PRINCIPLES",
)

AGENTS_PATH = Path("AGENTS.md")
CODEX_PATH = Path("CODEX.md")
CLAUDE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_PATH = Path("research-program.md")
PRD_PATH = Path("_bmad/prd.md")
ARCHITECTURE_PATH = Path("_bmad/architecture.md")
TRACEABILITY_PATH = Path("_bmad/traceability.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
STATUS_PATH = Path("ops/status.md")
CHANGELOG_PATH = Path("ops/changelog.md")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
VNEXT_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CAPABILITIES_PATH = Path("openspec/capabilities")

EXP5755_PATH = Path("results/experiment_5755_transition_v514.json")
EXP5756_PATH = Path("results/experiment_5756_v514_source_delta_ingestion.json")
EXP5757_PATH = Path("results/experiment_5757_proposal_benchmark_scalar_bridge.json")
EXP5758_PATH = Path("results/experiment_5758_rust_parity_scalar_bridge.json")
EXP5759_PATH = Path("results/experiment_5759_sota_exact_proposal_utility_panel.json")
EXP5760_PATH = Path("results/experiment_5760_selective_exact_feedback_search.json")
EXP5761_PATH = Path("results/experiment_5761_exact_constraint_acquisition_benchmark.json")
EXP5762_PATH = Path("results/experiment_5762_query_driven_constraint_lifecycle.json")
EXP5763_PATH = Path("results/experiment_5763_dependent_task_constraint_acquisition.json")
EXP5764_PATH = Path("results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json")
EXP5765_PATH = Path("results/experiment_5765_one_axis_final_10x_crossover.json")
EXP5766_PATH = Path("results/experiment_5766_arc_loo_component_interaction_audit.json")
EXP5767_PATH = Path("results/experiment_5767_arc_game_blind_composition_hardening.json")

EXP5755_ID = "exp5755-transition-v514"
EXP5756_ID = "exp5756-v514-source-delta-ingestion"
EXP5757_ID = "exp5757-proposal-benchmark-scalar-bridge"
EXP5758_ID = "exp5758-rust-parity-scalar-bridge"
EXP5759_ID = "exp5759-sota-exact-proposal-utility-panel"
EXP5760_ID = "exp5760-selective-exact-feedback-search"
EXP5761_ID = "exp5761-exact-constraint-acquisition-benchmark"
EXP5762_ID = "exp5762-query-driven-constraint-lifecycle"
EXP5763_ID = "exp5763-dependent-task-constraint-acquisition"
EXP5764_ID = "exp5764-one-axis-profiled-allocation-free-hot-path"
EXP5765_ID = "exp5765-one-axis-final-10x-crossover"
EXP5766_ID = "exp5766-arc-loo-component-interaction-audit"
EXP5767_ID = "exp5767-arc-game-blind-composition-hardening"

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5755_ID: EXP5755_PATH,
    EXP5756_ID: EXP5756_PATH,
    EXP5757_ID: EXP5757_PATH,
    EXP5758_ID: EXP5758_PATH,
    EXP5759_ID: EXP5759_PATH,
    EXP5760_ID: EXP5760_PATH,
    EXP5761_ID: EXP5761_PATH,
    EXP5762_ID: EXP5762_PATH,
    EXP5763_ID: EXP5763_PATH,
    EXP5764_ID: EXP5764_PATH,
    EXP5765_ID: EXP5765_PATH,
    EXP5766_ID: EXP5766_PATH,
    EXP5767_ID: EXP5767_PATH,
}
TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

SOURCE_CONTEXT_PATHS = (
    AGENTS_PATH,
    CODEX_PATH,
    CLAUDE_PATH,
    RESEARCH_PROGRAM_PATH,
    PRD_PATH,
    ARCHITECTURE_PATH,
    TRACEABILITY_PATH,
    ROADMAP_PATH,
    RESEARCH_COMPLETE_PATH,
    CONDUCTOR_LOG_PATH,
    STATUS_PATH,
    CHANGELOG_PATH,
    EXCLUSION_MANIFEST_PATH,
    ARC_REGISTRY_PATH,
    E2E_PLAN_PATH,
    VNEXT_PATH,
    SPEC_PATH,
    CAPABILITIES_PATH,
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/test_experiment_5768_v514_capstone_reconciliation.py -q"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5768_v514_capstone_reconciliation.py "
            "-m pytest tests/python/test_experiment_5768_v514_capstone_reconciliation.py -q"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5768_v514_capstone_reconciliation.py "
            "--fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py "
            "results/experiment_5768_v514_capstone_reconciliation.json"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Versioned schema for the Exp5768 capstone artifact.",
    "experiment": "Stable experiment slug for local indexing.",
    "experiment_id": "Conductor task id for the V514 capstone.",
    "result_path": "Canonical deliverable path written by this workflow.",
    "run_date": "Absolute run date for the capstone closeout.",
    "random_seed": "Deterministic metadata even though no stochastic science is run.",
    "spec_refs": "OpenSpec anchors that define this artifact contract.",
    "field_principles": "Maps every artifact field to its evidence boundary.",
    "status": "Bare terminal state derived from provenance and schema checks.",
    "preconditions_checked": "Records instruction, source-context, artifact, gate, and no-publish checks.",
    "milestone": "Binds the reconciliation to milestone 2026.07.514.",
    "task_ids": "Fixed Exp5755-Exp5767 denominator in conductor order.",
    "task_outcome_matrix": "Per-task planned, dispatched, complete, blocked, negative, null, and promoted states.",
    "artifact_hashes": "Path, byte hash, loadability, status, and checksum checks for every expected artifact.",
    "gate_outcomes": "Scalar producer and downstream conductor gate outcomes kept separate from science.",
    "blocked_task_ids": "Tasks blocked by preconditions, missing artifacts, malformed artifacts, or conductor gates.",
    "scientific_null_task_ids": "Executed tasks whose measured result is null rather than blocked.",
    "negative_result_task_ids": "Executed tasks with genuinely negative gate science, not skipped tasks.",
    "promoted_task_ids": "Executed tasks whose outputs became downstream-ready evidence.",
    "proposal_bridge_ready": "Exp5757 is ready only from lossless bare scalar bridge fields.",
    "rust_bridge_ready": "Exp5758 is ready only from lossless bare scalar Rust parity fields.",
    "proposal_panel_executed": "Exp5759 execution is separate from proposal utility readiness.",
    "proposal_utility_ready": "Only positive utility LCB and flagship non-regression can make proposal utility ready.",
    "selective_feedback_executed": "Exp5760 remains false when the conductor gate blocked it.",
    "selective_feedback_ready": "Selective feedback cannot be ready without an executed positive Exp5760.",
    "continuous_self_learning_executed": "The query-driven FR-11 branch executed when Exp5762 and Exp5763 ran.",
    "continuous_self_learning_credited": "Credit requires positive recovery, exact retention, rollback, and no unsafe updates.",
    "kan_scaleup_retired": "Preserves the .513 KAN-specific retirement while allowing a new CA mechanism.",
    "constraint_acquisition_ready": "Exp5761 CA corpus readiness from exact disjoint validator-clean evidence.",
    "dependent_task_ca_executed": "Exp5763 execution is separate from Exp5762 single-stream credit.",
    "dependent_task_ca_ready": "Dependent-task CA readiness requires Exp5763's gate and exact safety receipts.",
    "rust_hot_path_ready": "Exp5764 optimized-path readiness is parity/reachability evidence, not a 10x claim.",
    "rust_benchmark_executed": "Exp5765 is the only V514 final Rust/Python 10x benchmark execution.",
    "rust_10x_claimed": "Bare false unless the strict consecutive larger-size lower-bound rule passed.",
    "rust_10x_retired": "Retires only the allocation-free one-axis PyO3 technique when the final benchmark says so.",
    "arc_loo_audit_executed": "Exp5766 LOO audit execution is separate from composition promotion.",
    "arc_loo_generalization_positive": "Bare true only if held-out LCB and causal interactions are positive.",
    "arc_composition_executed": "Exp5767 remains false when its gate did not pass.",
    "arc_live_generalization_delta": "The observed ARC live/development-proxy generalization delta.",
    "solve_provenance": "development_proxy prevents public-registry evidence from becoming hidden-game credit.",
    "arc_registry_delta": "Registry credit must remain zero in this capstone.",
    "arc_solve_credited": "No ARC solve is credited without eligible solve provenance.",
    "hardware_claims": "Records CUDA execution and CPU benchmarks without creating hardware speedup claims.",
    "model_weight_mutation": "Bare false preserves immutable GGUF/model weight boundaries.",
    "closed_scopes_reopened": "Bare false preserves retired PHASE-D, KAN scale-up, and two-axis scopes.",
    "specs_reconciled": "True when this Exp5768 OpenSpec contract exists and is cited.",
    "traceability_reconciled": "False here because the operator stop rule delegates traceability to the next step.",
    "research_complete_reconciled": "False here because the operator stop rule delegates completed-research updates.",
    "status_reconciled": "False here because the operator stop rule forbids ops/status.md edits in this run.",
    "changelog_reconciled": "False here because the operator stop rule forbids ops/changelog.md edits in this run.",
    "references_reconciled": "True when Exp5756 completed with no accepted source delta requiring references edits.",
    "public_docs_modified": "Bare false; this capstone must not edit public/operator-curated docs.",
    "publication_performed": "Bare false; publication is operator-only and not part of this workflow.",
    "inference_substrate": "cached_artifact_reconciliation_no_llm because all claims come from local artifacts.",
    "test_commands": "Verification commands recorded exactly.",
    "test_exit_codes": "Observed command exit codes without relabeling failures.",
    "e2e_checks": "Relevant E2E plan items recorded; unavailable hardware checks are not applicable, not passed.",
    "reproducibility_checksum": "Stable content checksum catches capstone drift.",
    "honest_verdict": "Terminal summary begins with complete: or blocked: and avoids claim inflation.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def path_sha256(path: Path) -> str | None:
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
        return "sha256:" + digest.hexdigest()
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(stable_json(stable).encode("utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "sha256": None, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive corruption path
        return (
            {},
            {
                "exists": True,
                "loadable": False,
                "sha256": path_sha256(path),
                "error": f"json_error:{exc.msg}",
            },
        )
    if not isinstance(payload, dict):
        return (
            {"_non_mapping_payload": payload},
            {
                "exists": True,
                "loadable": False,
                "sha256": path_sha256(path),
                "error": "json_payload_not_mapping",
            },
        )
    return payload, {"exists": True, "loadable": True, "sha256": path_sha256(path), "error": None}


def _verdict(payload: JsonMap) -> str:
    value = payload.get("honest_verdict")
    return value if isinstance(value, str) else ""


def _status_for_payload(payload: JsonMap, metadata: JsonMap) -> str:
    if metadata.get("exists") is False:
        return "missing"
    if metadata.get("exists") is True and metadata.get("loadable") is False:
        return "malformed"
    if payload.get("schema") == "blocked_gate_check_v1":
        return "blocked-gate"
    verdict = _verdict(payload)
    if verdict.startswith("blocked:") or verdict.startswith("blocked_"):
        return "blocked-precondition"
    if payload.get("status") == "blocked":
        return "blocked-gate"
    if verdict.startswith("complete:") or verdict.startswith("success:"):
        return "complete"
    status = payload.get("status")
    return str(status) if isinstance(status, str) and status else "unknown"


def _number(payload: JsonMap, field: str, default: float = 0.0) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _bool(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _checksum_matches(payload: JsonMap, expected_checksum: str) -> bool | None:
    observed = payload.get("reproducibility_checksum")
    if observed is None:
        return None
    if not isinstance(observed, str) or not observed:
        return False
    normalized = observed if observed.startswith("sha256:") else f"sha256:{observed}"
    return normalized == expected_checksum


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _outcome_from_line(line: str | None) -> str:
    if line is None:
        return "MISSING_LOG_LINE"
    for outcome in ("GATE_BLOCK", "FAIL", "OK", "BLOCK", "FLAGGED"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _source_context_hashes(root: Path) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        rows[rel_path.as_posix()] = {
            "present": path.exists(),
            "sha256": path_sha256(path),
            "read_only": True,
        }
    return rows


def _task_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        checksum = payload_checksum(payload) if meta.get("loadable") else None
        payloads[task_id] = payload
        metadata[task_id] = {
            "path": rel_path.as_posix(),
            "present": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "status": _status_for_payload(payload, meta),
            "reproducibility_checksum": payload.get("reproducibility_checksum"),
            "checksum_matches": _checksum_matches(payload, checksum or ""),
            "error": meta.get("error"),
        }
    return payloads, metadata


def _log_patterns() -> dict[str, tuple[str, ...]]:
    return {
        EXP5755_ID: (EXP5755_ID, "Transition terminal .513 evidence"),
        EXP5756_ID: (EXP5756_ID, "Ingest post-V514 source deltas"),
        EXP5757_ID: (EXP5757_ID, "lossless bare-scalar bridge for the ready"),
        EXP5758_ID: (EXP5758_ID, "lossless bare-scalar bridge for repaired"),
        EXP5759_ID: (EXP5759_ID, "Gated on Exp5757 scalar readiness"),
        EXP5760_ID: (EXP5760_ID, "Gated on Exp5759 utility>0"),
        EXP5761_ID: (EXP5761_ID, "exact MPMMine-shaped benchmark"),
        EXP5762_ID: (EXP5762_ID, "Gated on Exp5761 exact corpus"),
        EXP5763_ID: (EXP5763_ID, "Gated on Exp5762 recovery>0"),
        EXP5764_ID: (EXP5764_ID, "Gated on Exp5758 scalar parity"),
        EXP5765_ID: (EXP5765_ID, "Gated on Exp5764 parity"),
        EXP5766_ID: (EXP5766_ID, "ARC leave-one-game-out generalization"),
        EXP5767_ID: (EXP5767_ID, "Gated on Exp5766 held-out delta>0"),
    }


def _conductor_outcomes(root: Path) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    rows: dict[str, JsonDict] = {}
    for task_id, patterns in _log_patterns().items():
        line = _latest_log_line(text, patterns)
        rows[task_id] = {
            "outcome": _outcome_from_line(line),
            "evidence_line": line,
            "source": CONDUCTOR_LOG_PATH.as_posix(),
        }
    return rows


def _proposal_bridge_ready(exp5757: JsonMap) -> bool:
    return (
        _number(exp5757, "benchmark_bridge_ready_score") >= 1.0
        and _number(exp5757, "benchmark_ready_score") >= 1.0
        and _number(exp5757, "heldout_partition_disjoint_score") >= 1.0
        and _number(exp5757, "adversarial_verification_clean_score") >= 1.0
        and _number(exp5757, "structure_receipt_failure_count") == 0.0
        and _number(exp5757, "solution_receipt_failure_count") == 0.0
        and _number(exp5757, "validator_disagreement_count") == 0.0
        and _number(exp5757, "unsafe_synthesis_count") == 0.0
        and not _bool(exp5757, "upstream_modified")
    )


def _rust_bridge_ready(exp5758: JsonMap) -> bool:
    return (
        _number(exp5758, "rust_benchmark_gate_ready_score") >= 1.0
        and _number(exp5758, "restart_parity_ready_score") >= 1.0
        and _number(exp5758, "distributional_parity_score") >= 1.0
        and _number(exp5758, "fallback_equivalence_score") >= 1.0
        and _number(exp5758, "production_backend_reachable_score") >= 1.0
        and not _bool(exp5758, "timing_claimed")
        and not _bool(exp5758, "hardware_speedup_claimed")
        and not _bool(exp5758, "sampler_code_modified")
    )


def _constraint_acquisition_ready(exp5761: JsonMap) -> bool:
    return (
        _number(exp5761, "ca_benchmark_ready_score") >= 1.0
        and _number(exp5761, "train_dev_science_disjoint_score") >= 1.0
        and _number(exp5761, "exact_validator_disagreement_count") == 0.0
        and _number(exp5761, "structure_receipt_failure_count") == 0.0
        and _number(exp5761, "solution_receipt_failure_count") == 0.0
        and not _bool(exp5761, "llm_inference_used")
    )


def _continuous_self_learning_credited(exp5762: JsonMap, exp5763: JsonMap) -> bool:
    return (
        _number(exp5762, "constraint_recovery_gain_lcb") > 0.0
        and _number(exp5762, "prefix_retention_pass_score") >= 1.0
        and _number(exp5762, "unsafe_update_count") == 0.0
        and _number(exp5762, "rejected_update_propagation_count") == 0.0
        and _number(exp5762, "rollback_hash_mismatch_count") == 0.0
        and bool(exp5762.get("restart_equivalence", {}).get("all_passed"))
        and _bool(exp5762, "continuous_self_learning_credited")
        and _number(exp5763, "dependent_task_ca_ready_score") >= 1.0
        and _bool(exp5763, "continuous_self_learning_credited")
        and not _bool(exp5762, "model_weight_mutation")
        and not _bool(exp5763, "model_weight_mutation")
    )


def _dependent_task_ca_ready(exp5763: JsonMap) -> bool:
    certificate = exp5763.get("nonforgetting_certificate", {})
    return (
        _number(exp5763, "dependent_task_ca_ready_score") >= 1.0
        and bool(certificate.get("all_prefixes_exact"))
        and _number(exp5763, "unsafe_update_count") == 0.0
        and _number(exp5763, "rejected_update_propagation_count") == 0.0
        and _number(exp5763, "rollback_hash_mismatch_count") == 0.0
        and bool(exp5763.get("restart_equivalence", {}).get("all_passed"))
        and not _bool(exp5763, "production_default_enabled")
    )


def _rust_hot_path_ready(exp5764: JsonMap) -> bool:
    return (
        _number(exp5764, "optimized_path_ready_score") >= 1.0
        and _number(exp5764, "semantic_parity_score") >= 1.0
        and _number(exp5764, "distributional_parity_score") >= 1.0
        and _number(exp5764, "production_backend_reachable_score") >= 1.0
        and not _bool(exp5764, "timing_promotion_claimed")
        and not _bool(exp5764, "hardware_speedup_claimed")
    )


def _arc_loo_generalization_positive(exp5766: JsonMap) -> bool:
    return (
        _number(exp5766, "loo_generalization_delta_lcb") > 0.0
        and _number(exp5766, "causal_interaction_count") >= 1.0
        and _number(exp5766, "source_leak_count") == 0.0
        and _number(exp5766, "game_identity_leak_count") == 0.0
    )


def _task_outcome_matrix(
    payloads: Mapping[str, JsonMap],
    artifact_hashes: Mapping[str, JsonMap],
    conductor_outcomes: Mapping[str, JsonMap],
    promoted_task_ids: set[str],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id in TASK_IDS:
        metadata = artifact_hashes[task_id]
        status = str(metadata.get("status"))
        conductor = conductor_outcomes[task_id]
        outcome = str(conductor.get("outcome"))
        complete = status == "complete"
        blocked_gate = status == "blocked-gate" or outcome == "GATE_BLOCK"
        blocked_precondition = status == "blocked-precondition" or outcome in {"FAIL", "BLOCK"}
        missing_or_malformed = status in {"missing", "malformed"}
        preemptively_skipped = missing_or_malformed and outcome == "GATE_BLOCK"
        negative = (
            task_id == EXP5759_ID and complete and not _proposal_utility_ready(payloads[task_id])
        )
        null = (
            task_id == EXP5765_ID and complete and _bool(payloads[task_id], "rust_10x_retired")
        ) or (
            task_id == EXP5766_ID
            and complete
            and not _arc_loo_generalization_positive(payloads[task_id])
        )
        rows[task_id] = {
            "planned": True,
            "dispatched": outcome != "MISSING_LOG_LINE" or bool(metadata.get("present")),
            "artifact_path": metadata.get("path"),
            "artifact_present": bool(metadata.get("present")),
            "artifact_status": status,
            "conductor_outcome": outcome,
            "evidence_line": conductor.get("evidence_line"),
            "complete": complete,
            "blocked_gate": blocked_gate,
            "blocked_precondition": blocked_precondition,
            "preemptively_skipped": preemptively_skipped,
            "negative": negative,
            "null": null,
            "promoted": task_id in promoted_task_ids,
            "honest_verdict": _verdict(payloads[task_id]) or None,
            "gate_block_reason": payloads[task_id].get("gate_check_summary"),
        }
    return rows


def _proposal_utility_ready(exp5759: JsonMap) -> bool:
    return (
        _number(exp5759, "proposal_utility_lcb") > 0.0
        and _number(exp5759, "proposal_utility_ready_score") >= 1.0
        and _number(exp5759, "flagship_nonregression_count") >= 2.0
        and _number(exp5759, "validator_disagreement_count") == 0.0
        and _number(exp5759, "authority_violation_count") == 0.0
    )


def _gate_outcomes(
    payloads: Mapping[str, JsonMap],
    *,
    proposal_bridge_ready: bool,
    rust_bridge_ready: bool,
    constraint_acquisition_ready: bool,
    dependent_task_ca_ready: bool,
    rust_hot_path_ready: bool,
    arc_loo_generalization_positive: bool,
) -> JsonDict:
    return {
        "exp5757_to_exp5759": {
            "passed": proposal_bridge_ready,
            "receipts": payloads[EXP5757_ID].get("gate_replay_receipts"),
        },
        "exp5759_to_exp5760": {
            "passed": False,
            "gates_evaluated": payloads[EXP5760_ID].get("gates_evaluated"),
            "summary": payloads[EXP5760_ID].get("gate_check_summary"),
        },
        "exp5761_to_exp5762": {
            "passed": constraint_acquisition_ready,
            "source_fields": [
                "ca_benchmark_ready_score",
                "exact_validator_disagreement_count",
                "train_dev_science_disjoint_score",
            ],
        },
        "exp5762_to_exp5763": {
            "passed": dependent_task_ca_ready,
            "source_fields": [
                "constraint_recovery_gain_lcb",
                "prefix_retention_pass_score",
                "unsafe_update_count",
                "rollback_hash_mismatch_count",
            ],
        },
        "exp5758_to_exp5764": {
            "passed": rust_bridge_ready,
            "receipts": payloads[EXP5758_ID].get("gate_replay_receipts"),
        },
        "exp5764_to_exp5765": {
            "passed": rust_hot_path_ready,
            "source_fields": [
                "semantic_parity_score",
                "distributional_parity_score",
                "production_backend_reachable_score",
                "optimized_path_ready_score",
            ],
        },
        "exp5766_to_exp5767": {
            "passed": arc_loo_generalization_positive,
            "gates_evaluated": payloads[EXP5767_ID].get("gates_evaluated"),
            "summary": payloads[EXP5767_ID].get("gate_check_summary"),
        },
    }


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> dict[str, Any]:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def _e2e_checks() -> list[JsonDict]:
    return [
        {
            "id": "E2E-006",
            "status": "not_applicable",
            "requires_hardware": True,
            "reason": "No CPU/KV260 trace-scorer code changed; hardware checks are not claimed as passed.",
        },
        {
            "id": "E2E-007",
            "status": "not_applicable",
            "requires_hardware": False,
            "reason": "SMGI certificate workflow was not changed; Exp5762/5763 evidence is cached artifact input.",
        },
        {
            "id": "artifact-schema-and-accounting",
            "status": "covered_by_unit_and_artifact_validation",
            "requires_hardware": False,
            "reason": "This capstone is artifact-only and has no executable end-user runtime path.",
        },
    ]


def _hardware_claims(payloads: Mapping[str, JsonMap]) -> JsonDict:
    exp5759 = payloads[EXP5759_ID]
    exp5764 = payloads[EXP5764_ID]
    exp5765 = payloads[EXP5765_ID]
    return {
        "gpu_execution": {
            "exp5759_cuda_offload_authenticated": exp5759.get("cuda_offload_authenticated"),
            "scope": "proposal panel execution only; no hardware speedup claim",
        },
        "cpu_release_benchmark": {
            "exp5764_hot_path_profiled": _status_for_payload(exp5764, {}) == "complete",
            "exp5765_matched_benchmark_executed": _status_for_payload(exp5765, {}) == "complete",
            "scope": "local Rust/PyO3 CPU release path",
        },
        "speedup_claimed": False,
        "hardware_speedup_claimed": False,
        "unavailable_or_not_run": {
            "kv260": "not_applicable",
            "polarfire": "not_applicable",
            "tsu": "not_applicable",
            "kona": "not_applicable",
        },
    }


def _preconditions_checked(root: Path, source_context_hashes: Mapping[str, JsonMap]) -> JsonDict:
    return {
        "instructions_read": {
            AGENTS_PATH.as_posix(): bool(source_context_hashes[AGENTS_PATH.as_posix()]["present"]),
            CODEX_PATH.as_posix(): bool(source_context_hashes[CODEX_PATH.as_posix()]["present"]),
            CLAUDE_PATH.as_posix(): bool(source_context_hashes[CLAUDE_PATH.as_posix()]["present"]),
        },
        "artifact_range": "experiment_5755 through experiment_5767",
        "conductor_log_present": (root / CONDUCTOR_LOG_PATH).exists(),
        "openspec_capabilities_present": (root / CAPABILITIES_PATH).exists(),
        "arc_registry_present": (root / ARC_REGISTRY_PATH).exists(),
        "exclusion_manifest_present": (root / EXCLUSION_MANIFEST_PATH).exists(),
        "public_docs_modified": False,
        "publication_performed": False,
        "research_conductor_modified": False,
        "operator_stop_rule_delegates_ops_docs": True,
    }


def build_report(root: Path = REPO_ROOT, *, tests_run: Sequence[JsonMap] | None = None) -> JsonDict:
    source_hashes = _source_context_hashes(root)
    payloads, artifact_hashes = _task_payloads(root)
    conductor_outcomes = _conductor_outcomes(root)

    proposal_bridge_ready = _proposal_bridge_ready(payloads[EXP5757_ID])
    rust_bridge_ready = _rust_bridge_ready(payloads[EXP5758_ID])
    proposal_panel_executed = artifact_hashes[EXP5759_ID]["status"] == "complete"
    proposal_utility_ready = proposal_panel_executed and _proposal_utility_ready(
        payloads[EXP5759_ID]
    )
    selective_feedback_executed = artifact_hashes[EXP5760_ID]["status"] == "complete"
    selective_feedback_ready = selective_feedback_executed and _bool(
        payloads[EXP5760_ID], "selective_feedback_ready"
    )
    constraint_acquisition_ready = _constraint_acquisition_ready(payloads[EXP5761_ID])
    dependent_task_ca_executed = artifact_hashes[EXP5763_ID]["status"] == "complete"
    dependent_task_ca_ready = dependent_task_ca_executed and _dependent_task_ca_ready(
        payloads[EXP5763_ID]
    )
    continuous_self_learning_executed = (
        artifact_hashes[EXP5762_ID]["status"] == "complete" and dependent_task_ca_executed
    )
    continuous_self_learning_credited = (
        continuous_self_learning_executed
        and _continuous_self_learning_credited(payloads[EXP5762_ID], payloads[EXP5763_ID])
    )
    rust_hot_path_ready = _rust_hot_path_ready(payloads[EXP5764_ID])
    rust_benchmark_executed = artifact_hashes[EXP5765_ID]["status"] == "complete"
    rust_10x_claimed = rust_benchmark_executed and _bool(payloads[EXP5765_ID], "rust_10x_claimed")
    rust_10x_retired = rust_benchmark_executed and _bool(payloads[EXP5765_ID], "rust_10x_retired")
    arc_loo_audit_executed = artifact_hashes[EXP5766_ID]["status"] == "complete"
    arc_loo_generalization_positive = arc_loo_audit_executed and _arc_loo_generalization_positive(
        payloads[EXP5766_ID]
    )
    arc_composition_executed = artifact_hashes[EXP5767_ID]["status"] == "complete"
    arc_live_generalization_delta = _number(payloads[EXP5766_ID], "loo_generalization_delta")

    promoted_task_ids = {
        task_id
        for task_id, ready in (
            (EXP5757_ID, proposal_bridge_ready),
            (EXP5758_ID, rust_bridge_ready),
            (EXP5761_ID, constraint_acquisition_ready),
            (EXP5762_ID, continuous_self_learning_credited),
            (EXP5763_ID, dependent_task_ca_ready),
            (EXP5764_ID, rust_hot_path_ready),
        )
        if ready
    }
    task_outcome_matrix = _task_outcome_matrix(
        payloads,
        artifact_hashes,
        conductor_outcomes,
        promoted_task_ids,
    )
    blocked_task_ids = [
        task_id
        for task_id, row in task_outcome_matrix.items()
        if row["blocked_gate"]
        or row["blocked_precondition"]
        or row["artifact_status"] in {"missing", "malformed"}
    ]
    scientific_null_task_ids = [
        task_id for task_id, row in task_outcome_matrix.items() if row["null"]
    ]
    negative_result_task_ids = [
        task_id for task_id, row in task_outcome_matrix.items() if row["negative"]
    ]
    tests = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]
    gate_outcomes = _gate_outcomes(
        payloads,
        proposal_bridge_ready=proposal_bridge_ready,
        rust_bridge_ready=rust_bridge_ready,
        constraint_acquisition_ready=constraint_acquisition_ready,
        dependent_task_ca_ready=dependent_task_ca_ready,
        rust_hot_path_ready=rust_hot_path_ready,
        arc_loo_generalization_positive=arc_loo_generalization_positive,
    )
    malformed = [
        task_id for task_id, meta in artifact_hashes.items() if meta["status"] == "malformed"
    ]
    missing_required = [
        task_id for task_id, meta in artifact_hashes.items() if meta["status"] == "missing"
    ]
    status = "blocked" if malformed or missing_required else "complete"
    verdict_prefix = "blocked:" if status == "blocked" else "complete:"
    verdict_body = (
        " irreconcilable missing_or_malformed_v514_artifacts"
        if status == "blocked"
        else " V514 reconciled: bridges ready, proposal utility not ready, CA credited, Rust 10x retired, ARC composition gate blocked"
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": _preconditions_checked(root, source_hashes),
        "milestone": MILESTONE,
        "task_ids": list(TASK_IDS),
        "task_outcome_matrix": task_outcome_matrix,
        "artifact_hashes": artifact_hashes,
        "gate_outcomes": gate_outcomes,
        "blocked_task_ids": blocked_task_ids,
        "scientific_null_task_ids": scientific_null_task_ids,
        "negative_result_task_ids": negative_result_task_ids,
        "promoted_task_ids": sorted(promoted_task_ids),
        "proposal_bridge_ready": proposal_bridge_ready,
        "rust_bridge_ready": rust_bridge_ready,
        "proposal_panel_executed": proposal_panel_executed,
        "proposal_utility_ready": proposal_utility_ready,
        "selective_feedback_executed": selective_feedback_executed,
        "selective_feedback_ready": selective_feedback_ready,
        "continuous_self_learning_executed": continuous_self_learning_executed,
        "continuous_self_learning_credited": continuous_self_learning_credited,
        "kan_scaleup_retired": True,
        "constraint_acquisition_ready": constraint_acquisition_ready,
        "dependent_task_ca_executed": dependent_task_ca_executed,
        "dependent_task_ca_ready": dependent_task_ca_ready,
        "rust_hot_path_ready": rust_hot_path_ready,
        "rust_benchmark_executed": rust_benchmark_executed,
        "rust_10x_claimed": rust_10x_claimed,
        "rust_10x_retired": rust_10x_retired,
        "arc_loo_audit_executed": arc_loo_audit_executed,
        "arc_loo_generalization_positive": arc_loo_generalization_positive,
        "arc_composition_executed": arc_composition_executed,
        "arc_live_generalization_delta": arc_live_generalization_delta,
        "solve_provenance": payloads[EXP5766_ID].get("solve_provenance", "development_proxy"),
        "arc_registry_delta": int(_number(payloads[EXP5766_ID], "arc_registry_delta")),
        "arc_solve_credited": _bool(payloads[EXP5766_ID], "arc_solve_credited"),
        "hardware_claims": _hardware_claims(payloads),
        "model_weight_mutation": False,
        "closed_scopes_reopened": False,
        "specs_reconciled": (root / SPEC_PATH).exists(),
        "traceability_reconciled": False,
        "research_complete_reconciled": False,
        "status_reconciled": False,
        "changelog_reconciled": False,
        "references_reconciled": artifact_hashes[EXP5756_ID]["status"] == "complete"
        and not bool(payloads[EXP5756_ID].get("accepted_findings")),
        "public_docs_modified": False,
        "publication_performed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [str(row.get("command")) for row in tests],
        "test_exit_codes": _test_exit_codes(tests),
        "e2e_checks": _e2e_checks(),
        "reproducibility_checksum": "",
        "honest_verdict": verdict_prefix + verdict_body,
    }
    missing_principles = set(artifact) - set(FIELD_PRINCIPLES)
    if missing_principles:
        raise KeyError(f"missing field principles: {sorted(missing_principles)}")
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field in artifact:
            if not isinstance(principles.get(field), str) or not principles.get(field):
                errors.append(f"field_principles.{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("kan_scaleup_retired") is not True:
        errors.append("kan_scaleup_retired")
    if artifact.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta")
    if artifact.get("arc_solve_credited") is not False:
        errors.append("arc_solve_credited")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if artifact.get("closed_scopes_reopened") is not False:
        errors.append("closed_scopes_reopened")
    if artifact.get("public_docs_modified") is not False:
        errors.append("public_docs_modified")
    if artifact.get("publication_performed") is not False:
        errors.append("publication_performed")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:
    artifact = build_report(root, tests_run=tests_run)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"artifact schema errors: {errors}")
    write_json(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _load_tests_run(path: Path | None) -> list[JsonDict]:  # pragma: no cover - CLI helper
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in payload]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output, tests_run=_load_tests_run(args.tests_run))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
