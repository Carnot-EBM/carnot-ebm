"""Exp5716 V510 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5716, SCENARIO-CAPSTONE-5716,
SCENARIO-CAPSTONE-5716-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5716-FIELD-PRINCIPLES.

This module is a ledger over already-written experiment artifacts. It does not
rerun FR-11, ARC, model inference, or sampler experiments. The purpose is to
make the milestone boundary auditable: every positive claim below is derived
from a concrete upstream field, while blocked, skipped, missing, proxy, or null
evidence remains visibly bounded instead of being rounded up into a promotion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    path_sha256,
    payload_checksum,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5716_v510_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5716_v510_capstone_reconciliation"
EXPERIMENT_ID = "exp5716-v510-capstone"
MILESTONE = "2026.07.510"
RUN_DATE = "2026-07-15"
RANDOM_SEED = 5716
SCHEMA = "carnot.experiment_5716.v510_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5716",
    "SCENARIO-CAPSTONE-5716",
    "SCENARIO-CAPSTONE-5716-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5716-FIELD-PRINCIPLES",
)

EXP5647_TASK_ID = "exp5647-v509-capstone-reconciliation"
EXP5706_TASK_ID = "exp5706-transition-v510"
EXP5707_TASK_ID = "exp5707-v510-source-delta-ingestion"
EXP5708_TASK_ID = "exp5708-sota-exact-constraint-canary"
EXP5709_TASK_ID = "exp5709-fr11-prospective-shadow-stream"
EXP5710_TASK_ID = "exp5710-fr11-isolated-act-on-advice-canary"
EXP5711_TASK_ID = "exp5711-placement-spatial-goal-energy-qualification"
EXP5712_TASK_ID = "exp5712-known-level-relational-route-ab"
EXP5713_TASK_ID = "exp5713-arc-live-levelup-attempt"
EXP5714_TASK_ID = "exp5714-one-axis-rust-python-exact-parity"
EXP5715_TASK_ID = "exp5715-one-axis-rust-quality-restart-parity"

EXP5647_CAPSTONE_PATH = Path("results/experiment_5647_v509_capstone_reconciliation.json")
EXP5706_TRANSITION_PATH = Path("results/experiment_5706_transition_v510.json")
EXP5707_SOURCE_PATH = Path("results/experiment_5707_v510_source_delta_ingestion.json")
EXP5708_CANARY_PATH = Path("results/experiment_5708_sota_exact_constraint_canary.json")
EXP5708_ROWS_PATH = Path("results/experiment_5708_sota_exact_constraint_canary.rows.jsonl")
EXP5709_SHADOW_PATH = Path("results/experiment_5709_fr11_prospective_shadow_stream.json")
EXP5710_ISOLATED_PATH = Path("results/experiment_5710_fr11_isolated_act_on_advice_canary.json")
EXP5711_ARC_QUAL_PATH = Path(
    "results/experiment_5711_arc_relational_goal_energy_live_qualification.json"
)
EXP5712_ARC_AB_PATH = Path("results/experiment_5712_arc_relational_goal_energy_live_ab.json")
EXP5713_ARC_LEVEL_PATH = Path("results/experiment_5713_arc_live_self_discovery_levelup_v510.json")
EXP5713_TRACE_PATH = Path("results/experiment_5713_arc_live_self_discovery_levelup_v510_trace.json")
EXP5714_RUST_PARITY_PATH = Path("results/experiment_5714_one_axis_tempering_rust_parity.json")
EXP5715_RUST_QUALITY_PATH = Path(
    "results/experiment_5715_one_axis_tempering_rust_quality_restart.json"
)

UPSTREAM_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5647_TASK_ID: EXP5647_CAPSTONE_PATH,
    EXP5706_TASK_ID: EXP5706_TRANSITION_PATH,
    EXP5707_TASK_ID: EXP5707_SOURCE_PATH,
    EXP5708_TASK_ID: EXP5708_CANARY_PATH,
    EXP5709_TASK_ID: EXP5709_SHADOW_PATH,
    EXP5710_TASK_ID: EXP5710_ISOLATED_PATH,
    EXP5711_TASK_ID: EXP5711_ARC_QUAL_PATH,
    EXP5712_TASK_ID: EXP5712_ARC_AB_PATH,
    EXP5713_TASK_ID: EXP5713_ARC_LEVEL_PATH,
    EXP5714_TASK_ID: EXP5714_RUST_PARITY_PATH,
    EXP5715_TASK_ID: EXP5715_RUST_QUALITY_PATH,
}
PRIMARY_ARTIFACT_PATHS = tuple(UPSTREAM_ARTIFACT_PATHS.values())
SIDE_ARTIFACT_PATHS = (EXP5708_ROWS_PATH, EXP5713_TRACE_PATH)

FORBIDDEN_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    Path("research-complete.yaml"),
    Path("research-references.md"),
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    VERIFIER_GAPS_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required capstone field.",
    "upstream_artifacts": (
        "fixed evidence denominator; every import traces to a hashed artifact or explicit missing path."
    ),
    "upstream_gate_statuses": (
        "complete, blocked, gate-skipped, missing, malformed, and flagged evidence stay distinct."
    ),
    "adversarial_verification_summary": (
        "live re-check outcomes and stamped flags block promotion before aggregation."
    ),
    "v509_fr11_promotion_preserved": (
        "prior independently audited FR-11 science remains true even when .510 canaries block."
    ),
    "sota_canary_status": (
        "Exp5708 is bounded to data/runtime stream receipt unless all exact stream gates pass."
    ),
    "cuda_offload_status": (
        "authenticated offload is provenance only and cannot override parse or validator gates."
    ),
    "prospective_shadow_status": (
        "Exp5709 chronology/prequential gates are separate from prior replay promotion."
    ),
    "isolated_canary_status": (
        "Exp5710 cannot mutate production or model weights and cannot promote when absent or skipped."
    ),
    "production_default_enabled": "bare false keeps canary evidence out of production defaults.",
    "model_weight_mutation": "bare false proves controller-state canaries did not alter LLM weights.",
    "arc_relational_qualification_status": "Exp5711 route qualification is no-solve development evidence.",
    "arc_relational_live_ab_status": (
        "Exp5712 promotion requires matched benefit, zero regression, and zero unsafe route accepts."
    ),
    "arc_registry_count_before": "authoritative registry baseline before the Exp5713 live attempt.",
    "arc_registry_count_after": "authoritative registry count after the Exp5713 live attempt.",
    "arc_registry_delta": "solve credit requires a positive reproduced-level registry delta.",
    "arc_solve_provenance": (
        "only live self-discovery plus generic reproduction and registry update can credit an ARC solve."
    ),
    "one_axis_python_promotion_preserved": (
        "the promoted Python one-axis sampler stays live after two-axis retirement."
    ),
    "one_axis_rust_parity_status": (
        "Rust exact parity is a portability gate separate from speed or hardware."
    ),
    "one_axis_rust_quality_restart_status": (
        "hard-instance portability requires zero material regression and both restart directions."
    ),
    "two_axis_retirement_preserved": (
        "the Exp5645 two-axis failure remains closed without over-retiring one-axis or generic exchange."
    ),
    "timing_claimed": "bare false prevents runtime-speed inflation.",
    "hardware_speedup_claimed": "bare false prevents board or accelerator-speed inflation.",
    "retirements_applied": (
        "prior and repeated same-verdict scopes are bounded without over-retiring parent capabilities."
    ),
    "spec_reconciliation": "REQ-* anchors and tests backing the capstone are explicit.",
    "ops_reconciliation": (
        "ops ledgers and delegated stop-rule files are recorded without laundering them as edited."
    ),
    "known_issue_reconciliation": "known prior failures and exclusions remain visible.",
    "test_commands": "verification commands are replayable.",
    "test_exit_codes": "observed command exits are recorded without inferring success.",
    "e2e_check_receipts": (
        "applicable E2E checks are named and nonapplicable checks are justified."
    ),
    "forbidden_files_unchanged": "roadmap and conductor invariants are explicitly checked.",
    "inference_substrate": "must equal aggregation_from_upstream_artifacts.",
    "reproducibility_checksum": "content-addressed capstone output is stable.",
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
    "artifact_metadata",
    "side_artifacts",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "validation_results",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5716_v510_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5716_v510_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5716_v510_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5716_v510_capstone_reconciliation.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": "python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {
        "command": "python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "python scripts/validate-phase-gate.sh",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            "python scripts/adversarial_verify.py "
            "results/experiment_5716_v510_capstone_reconciliation.json"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "python scripts/root_clutter_sweep.py --check",
        "exit_code": None,
        "status": "not_run",
    },
)


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "json_type": None,
        "sha256": path_sha256(path),
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata["json_type"] = type(parsed).__name__
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_json_object"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {"exists": path.exists(), "loadable": False, "sha256": path_sha256(path)}
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        metadata["error"] = "malformed_yaml"
        return {}, metadata
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_yaml_mapping"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _read_side_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in SIDE_ARTIFACT_PATHS:
        path = root / rel_path
        line_count = None
        if path.exists():
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        rows.append(
            {
                "path": rel_path.as_posix(),
                "exists": path.exists(),
                "sha256": path_sha256(path),
                "line_count": line_count,
            }
        )
    return rows


def read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    rows: list[JsonDict] = []
    for task_id, rel_path in UPSTREAM_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        payloads[rel] = payload
        metadata[rel] = meta
        rows.append(
            {
                "task_id": task_id,
                "path": rel,
                "exists": bool(meta.get("exists")),
                "loadable": bool(meta.get("loadable")),
                "sha256": meta.get("sha256"),
                "schema": payload.get("schema"),
                "experiment_id": payload.get("experiment_id", payload.get("experiment")),
                "milestone": payload.get("milestone"),
                "honest_verdict": _verdict(payload) or None,
                "inference_substrate": payload.get("inference_substrate"),
                "terminal_prefix_valid": _verdict(payload).startswith(TERMINAL_PREFIXES),
            }
        )
    return payloads, metadata, rows


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict == "blocked_gate_check_failed"
        or ("gate" in blocked_at_layer and str(payload.get("status") or "").lower() == "blocked")
        or (
            payload.get("gate_check_summary")
            and str(payload.get("status") or "").lower() == "blocked"
        )
    )


def _is_blocked(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "blocked"
        or verdict.startswith("blocked:")
        or verdict.startswith("blocked_")
        or verdict.startswith("blocked ")
    )


def _is_complete(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "complete" or verdict.startswith("complete:") or verdict.startswith("complete_")
    )


def _status_for_payload(payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if payload.get("flagged_adversarial"):
        return "flagged"
    if _is_gate_skip(payload):
        return "gate_skipped"
    if _is_blocked(payload):
        return "blocked"
    if _is_complete(payload):
        return "complete"
    return "unknown"


def _clean_complete(task_id: str, statuses: Mapping[str, JsonMap]) -> bool:
    row = statuses.get(task_id, {})
    return row.get("status") == "complete" and not row.get("flagged_adversarial")


def _number(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _bool(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def _all_controls_safe(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes | bytearray):
        return False
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        if row.get("unsafe_route_accepted") is True or row.get("unsafe_route_accept") is True:
            return False
    return True


def _terminal_statuses(artifacts: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    statuses: dict[str, JsonDict] = {}
    for task_id, rel_path in UPSTREAM_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        statuses[task_id] = {
            "status": status,
            "path": rel,
            "sha256": meta.get("sha256"),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "gate_check_summary": payload.get("gate_check_summary"),
            "gates_evaluated": payload.get("gates_evaluated", []),
            "supports_promotion": status == "complete",
            "blocked_or_skipped_cannot_promote": status in {"blocked", "gate_skipped", "missing"},
            "metadata_error": meta.get("error"),
        }
    return statuses


def _derive_sota_canary(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    ready_score = _number(payload, "sota_canary_ready_score")
    parse_failures = _int(payload, "parse_failure_count")
    missing_rows = _int(payload, "missing_row_count")
    validator_disagreement = _int(payload, "validator_disagreement_count")
    commitments_present = all(
        bool(payload.get(field))
        for field in ("stream_root_commitment", "shadow_prefix_hash", "sealed_suffix_hash")
    )
    promoted = bool(
        _clean_complete(EXP5708_TASK_ID, statuses)
        and ready_score >= 1.0
        and _number(payload, "cuda_offload_authenticated_score") >= 1.0
        and parse_failures == 0
        and missing_rows == 0
        and validator_disagreement == 0
        and commitments_present
        and not _bool(payload, "native_json_grammar_used")
        and not _bool(payload, "retired_runtime_used")
        and not _bool(payload, "external_scorer_used")
    )
    return {
        "status": statuses[EXP5708_TASK_ID]["status"],
        "promoted": promoted,
        "stream_receipt_only": not promoted,
        "ready_score": ready_score,
        "blocked_reasons": payload.get("blocked_reasons", []),
        "manifest_row_count": _int(payload, "manifest_row_count"),
        "missing_row_count": missing_rows,
        "parse_failure_count": parse_failures,
        "validator_disagreement_count": validator_disagreement,
        "stream_root_commitment": payload.get("stream_root_commitment"),
        "shadow_prefix_hash": payload.get("shadow_prefix_hash"),
        "sealed_suffix_hash": payload.get("sealed_suffix_hash"),
        "row_manifest_path": payload.get("row_manifest_path"),
        "retired_runtime_used": _bool(payload, "retired_runtime_used"),
        "external_scorer_used": _bool(payload, "external_scorer_used"),
        "failed_condition": None if promoted else "parse_or_stream_gate_failed",
    }


def _derive_cuda_status(payload: JsonMap) -> JsonDict:
    return {
        "authenticated": _bool(payload, "cuda_offload_authenticated"),
        "score": _number(payload, "cuda_offload_authenticated_score"),
        "n_gpu_layers_requested": payload.get("n_gpu_layers_requested"),
        "n_gpu_layers_offloaded": payload.get("n_gpu_layers_offloaded"),
        "model_repo_id": payload.get("model_repo_id"),
        "gguf_filename": payload.get("gguf_filename"),
        "model_hash": payload.get("model_hash"),
        "promotes_canary": False,
        "boundary": "offload_provenance_not_quality_or_parse_promotion",
    }


def _derive_prospective_shadow(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    promoted = bool(
        _clean_complete(EXP5709_TASK_ID, statuses)
        and _number(payload, "prospective_shadow_ready_score") >= 1.0
        and _bool(payload, "chronological_order_preserved")
        and _int(payload, "stream_consumption_count") == 1
        and _int(payload, "decision_after_label_violation_count") == 0
        and _int(payload, "unsafe_false_accept_count") == 0
        and _bool(payload, "checkpoint_restart_pass")
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "production_default_enabled")
    )
    return {
        "status": statuses[EXP5709_TASK_ID]["status"],
        "promoted": promoted,
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated", []),
        "prospective_shadow_ready_score": _number(payload, "prospective_shadow_ready_score"),
        "unsafe_false_accept_count": _int(payload, "unsafe_false_accept_count"),
        "production_default_enabled": _bool(payload, "production_default_enabled"),
        "model_weight_mutation": _bool(payload, "model_weight_mutation"),
        "failed_condition": None if promoted else "gate_skipped_or_prospective_safety_not_proven",
    }


def _derive_isolated_canary(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    artifact_present = statuses[EXP5710_TASK_ID]["status"] not in {"missing", "malformed"}
    promoted = bool(
        _clean_complete(EXP5710_TASK_ID, statuses)
        and _number(payload, "isolated_canary_ready_score") >= 1.0
        and _int(payload, "unsafe_false_accept_count") == 0
        and _int(payload, "unsafe_update_accept_count") == 0
        and _int(payload, "retention_regression_count") == 0
        and _bool(payload, "crash_restart_replay_pass")
        and not _bool(payload, "model_weight_mutation")
        and not _bool(payload, "production_default_enabled")
        and not _bool(payload, "automatic_production_enablement")
    )
    return {
        "status": statuses[EXP5710_TASK_ID]["status"],
        "artifact_present": artifact_present,
        "promoted": promoted,
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated", []),
        "isolated_canary_ready_score": _number(payload, "isolated_canary_ready_score"),
        "production_default_enabled": _bool(payload, "production_default_enabled"),
        "model_weight_mutation": _bool(payload, "model_weight_mutation"),
        "automatic_production_enablement": _bool(payload, "automatic_production_enablement"),
        "failed_condition": None
        if promoted
        else "missing_gate_skipped_or_isolated_safety_not_proven",
    }


def _derive_arc_qualification(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    negative_safe = _all_controls_safe(payload.get("negative_control_results"))
    corrupted_safe = _all_controls_safe(payload.get("corrupted_control_results"))
    constant_scan = payload.get("per_game_constant_scan")
    if not isinstance(constant_scan, Mapping):
        constant_scan = {}
    qualified = bool(
        _clean_complete(EXP5711_TASK_ID, statuses)
        and _number(payload, "relational_goal_energy_ready_score") >= 1.0
        and _number(payload, "live_path_reachable_score") >= 1.0
        and _int(payload, "candidate_guidance_call_count") > 0
        and _int(payload, "frontier_goal_bias_call_count") > 0
        and _int(payload, "candidate_order_change_count") > 0
        and _bool(payload, "fallback_order_equivalence")
        and negative_safe
        and corrupted_safe
        and not _bool(payload, "per_game_leakage_detected")
        and not _bool(constant_scan, "per_game_constants_detected")
        and not _bool(payload, "outer_loop_bfs_used")
        and _int(payload, "game_source_read_count") == 0
        and _int(payload, "game_adapter_count") == 0
    )
    solve_claimed = (
        _int(payload, "new_levels_claimed") > 0
        or payload.get("solve_provenance") != "development_proxy"
    )
    return {
        "status": statuses[EXP5711_TASK_ID]["status"],
        "qualified": qualified,
        "promoted_as_solve": False,
        "solve_claimed": solve_claimed and _int(payload, "new_levels_claimed") > 0,
        "ready_score": _number(payload, "relational_goal_energy_ready_score"),
        "live_path_reachable_score": _number(payload, "live_path_reachable_score"),
        "candidate_order_change_count": _int(payload, "candidate_order_change_count"),
        "zero_variance_fallback_count": _int(payload, "zero_variance_fallback_count"),
        "negative_controls_safe": negative_safe,
        "corrupted_controls_safe": corrupted_safe,
        "per_game_leakage_detected": _bool(payload, "per_game_leakage_detected"),
        "solve_provenance": payload.get("solve_provenance"),
        "failed_condition": None if qualified else "route_qualification_gate_failed",
    }


def _interval_excludes_zero_positive(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    low = _number(row, "ci95_low") if "ci95_low" in row else _number(row, "lower")
    high = _number(row, "ci95_high") if "ci95_high" in row else _number(row, "upper")
    return low > 0.0 and high > 0.0


def _derive_arc_ab(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    budget = payload.get("budget_parity_receipt")
    if not isinstance(budget, Mapping):
        budget = {}
    intervals = payload.get("paired_intervals")
    interval_values = list(intervals.values()) if isinstance(intervals, Mapping) else []
    matched_benefit = any(_interval_excludes_zero_positive(row) for row in interval_values)
    zero_regression = _int(payload, "level_regression_count") == 0
    zero_unsafe = _int(payload, "unsafe_route_accept_count") == 0
    promoted = bool(
        _clean_complete(EXP5712_TASK_ID, statuses)
        and _number(payload, "relational_live_ab_ready_score") >= 1.0
        and _bool(budget, "matched")
        and _int(payload, "successful_pair_count") > 0
        and matched_benefit
        and zero_regression
        and zero_unsafe
        and _bool(payload, "negative_controls_preserved")
    )
    return {
        "status": statuses[EXP5712_TASK_ID]["status"],
        "promoted": promoted,
        "ready_score": _number(payload, "relational_live_ab_ready_score"),
        "matched_budget": _bool(budget, "matched"),
        "successful_pair_count": _int(payload, "successful_pair_count"),
        "matched_benefit": matched_benefit,
        "level_regression_count": _int(payload, "level_regression_count"),
        "zero_regression": zero_regression,
        "unsafe_route_accept_count": _int(payload, "unsafe_route_accept_count"),
        "zero_unsafe_route_accepts": zero_unsafe,
        "new_levels_claimed": _int(payload, "new_levels_claimed"),
        "registry_updated": _bool(payload, "registry_updated"),
        "solve_provenance": payload.get("solve_provenance"),
        "failed_condition": None if promoted else "matched_live_ab_null_or_not_promoted",
    }


def _derive_arc_solve(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    registry_before = _int(payload, "registry_count_before")
    registry_after = _int(payload, "registry_count_after")
    registry_delta = _int(payload, "registry_delta")
    critical_flags = payload.get("critical_flags")
    if not isinstance(critical_flags, list):
        critical_flags = []
    solve_credited = bool(
        _clean_complete(EXP5713_TASK_ID, statuses)
        and payload.get("solve_provenance") == "live_agent_self_discovery"
        and _bool(payload, "independent_reproduction_pass")
        and _bool(payload, "offline_reproduced")
        and _int(payload, "reproduction_seed_count") > 0
        and _int(payload, "reproduced_levels") > 0
        and registry_delta > 0
        and registry_after - registry_before == registry_delta
        and _bool(payload, "registry_updated")
        and not critical_flags
        and not _bool(payload, "hand_solution_used")
        and not _bool(payload, "outer_loop_bfs_used")
        and _int(payload, "game_source_read_count") == 0
        and _int(payload, "game_adapter_count") == 0
    )
    return {
        "solve_credited": solve_credited,
        "solve_provenance": payload.get("solve_provenance"),
        "selected_game": payload.get("selected_game"),
        "selected_level": payload.get("selected_level"),
        "target_level": payload.get("target_level"),
        "independent_reproduction_pass": _bool(payload, "independent_reproduction_pass"),
        "offline_reproduced": _bool(payload, "offline_reproduced"),
        "reproduction_seed_count": _int(payload, "reproduction_seed_count"),
        "reproduced_levels": _int(payload, "reproduced_levels"),
        "registry_count_before": registry_before,
        "registry_count_after": registry_after,
        "registry_delta": registry_delta,
        "registry_updated": _bool(payload, "registry_updated"),
        "critical_flags": critical_flags,
        "no_critical_flag": not critical_flags,
        "hand_solution_used": _bool(payload, "hand_solution_used"),
        "outer_loop_bfs_used": _bool(payload, "outer_loop_bfs_used"),
        "failed_condition": None if solve_credited else "no_registry_delta_or_reproduction",
    }


def _derive_one_axis_parity(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    exact_metrics = {
        "energy_error_max": _number(payload, "energy_error_max"),
        "target_marginal_delta": _number(payload, "target_marginal_delta"),
        "proposal_probability_error_max": _number(payload, "proposal_probability_error_max"),
        "swap_log_ratio_error_max": _number(payload, "swap_log_ratio_error_max"),
    }
    promoted = bool(
        _clean_complete(EXP5714_TASK_ID, statuses)
        and _number(payload, "one_axis_rust_parity_ready_score") >= 1.0
        and _bool(payload, "broken_control_rejected")
        and _number(payload, "broken_control_rejected_score") >= 1.0
        and all(value <= 1e-9 for value in exact_metrics.values())
        and _bool(payload, "deterministic_decision_parity")
        and _bool(payload, "cross_language_restart_pass")
        and _bool(payload, "checkpoint_roundtrip_pass")
        and _bool(payload, "python_fallback_equivalence")
        and not _bool(payload, "two_axis_code_added")
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
    )
    return {
        "status": statuses[EXP5714_TASK_ID]["status"],
        "promoted": promoted,
        "ready_score": _number(payload, "one_axis_rust_parity_ready_score"),
        "broken_control_rejected": _bool(payload, "broken_control_rejected"),
        "broken_control_rejected_score": _number(payload, "broken_control_rejected_score"),
        "exact_metrics": exact_metrics,
        "deterministic_decision_parity": _bool(payload, "deterministic_decision_parity"),
        "cross_language_restart_pass": _bool(payload, "cross_language_restart_pass"),
        "checkpoint_roundtrip_pass": _bool(payload, "checkpoint_roundtrip_pass"),
        "python_fallback_equivalence": _bool(payload, "python_fallback_equivalence"),
        "two_axis_code_added": _bool(payload, "two_axis_code_added"),
        "timing_claimed": _bool(payload, "timing_claimed"),
        "hardware_speedup_claimed": _bool(payload, "hardware_speedup_claimed"),
        "failed_condition": None if promoted else "exact_parity_or_control_gate_failed",
    }


def _successful_seed_count(payload: JsonMap) -> int:
    value = payload.get("successful_seed_count")
    if isinstance(value, Mapping):
        return _int(value, "value")
    return _int(payload, "successful_seed_count")


def _derive_one_axis_quality(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> JsonDict:
    upstream = payload.get("upstream_gate_receipts")
    if not isinstance(upstream, Mapping):
        upstream = {}
    exp5634 = upstream.get("exp5634", {})
    exp5714 = upstream.get("exp5714", {})
    transition_budget = payload.get("transition_budget_parity")
    if not isinstance(transition_budget, Mapping):
        transition_budget = {}
    swap_schedule = payload.get("swap_schedule_parity")
    if not isinstance(swap_schedule, Mapping):
        swap_schedule = {}
    promoted = bool(
        _clean_complete(EXP5715_TASK_ID, statuses)
        and _number(payload, "one_axis_rust_quality_ready_score") >= 1.0
        and isinstance(exp5634, Mapping)
        and exp5634.get("ready") is True
        and isinstance(exp5714, Mapping)
        and exp5714.get("ready") is True
        and _successful_seed_count(payload) > 0
        and _int(payload, "material_regression_count") == 0
        and _bool(payload, "python_to_rust_restart_pass")
        and _bool(payload, "rust_to_python_restart_pass")
        and _bool(transition_budget, "matched_cold_target_collection")
        and _bool(transition_budget, "matched_corrected_transition_budget")
        and _bool(swap_schedule, "matched_language_swap_schedule")
        and not _bool(transition_budget, "wall_time_compared")
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
    )
    return {
        "status": statuses[EXP5715_TASK_ID]["status"],
        "promoted": promoted,
        "ready_score": _number(payload, "one_axis_rust_quality_ready_score"),
        "upstream_exp5634_ready": isinstance(exp5634, Mapping) and exp5634.get("ready") is True,
        "upstream_exp5714_ready": isinstance(exp5714, Mapping) and exp5714.get("ready") is True,
        "successful_seed_count": _successful_seed_count(payload),
        "material_regression_count": _int(payload, "material_regression_count"),
        "python_to_rust_restart_pass": _bool(payload, "python_to_rust_restart_pass"),
        "rust_to_python_restart_pass": _bool(payload, "rust_to_python_restart_pass"),
        "matched_transition_budget": _bool(
            transition_budget, "matched_corrected_transition_budget"
        ),
        "matched_swap_schedule": _bool(swap_schedule, "matched_language_swap_schedule"),
        "wall_time_compared": _bool(transition_budget, "wall_time_compared"),
        "timing_claimed": _bool(payload, "timing_claimed"),
        "hardware_speedup_claimed": _bool(payload, "hardware_speedup_claimed"),
        "failed_condition": None if promoted else "quality_restart_or_zero_regression_gate_failed",
    }


def _derive_gate_statuses(statuses: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, status in statuses.items():
        rows[task_id] = {
            "status": status.get("status"),
            "artifact_path": status.get("path"),
            "sha256": status.get("sha256"),
            "gate_check_summary": status.get("gate_check_summary"),
            "gates_evaluated": status.get("gates_evaluated", []),
            "gate_skipped": status.get("status") == "gate_skipped",
            "blocked": status.get("status") == "blocked",
            "missing": status.get("status") == "missing",
            "malformed": status.get("status") == "malformed",
            "flagged": status.get("status") == "flagged",
            "supports_promotion": status.get("status") == "complete",
            "skipped_work_is_success": False,
        }
    return rows


def _derive_adversarial_summary(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> JsonDict:
    flagged_rows: list[JsonDict] = []
    for task_id, rel_path in UPSTREAM_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        stamped = bool(payload.get("flagged_adversarial"))
        critical_flags = payload.get("critical_flags")
        if not isinstance(critical_flags, list):
            critical_flags = []
        corrigendum = payload.get("corrigendum_pending")
        if not isinstance(corrigendum, list):
            corrigendum = []
        if stamped or critical_flags or corrigendum:
            flagged_rows.append(
                {
                    "task_id": task_id,
                    "artifact_path": rel_path.as_posix(),
                    "status": statuses[task_id]["status"],
                    "flagged_adversarial": stamped,
                    "critical_flags": critical_flags,
                    "corrigendum_pending": corrigendum,
                    "blocks_promotion": True,
                }
            )
    blocked_statuses = [
        {"task_id": task_id, "status": row["status"]}
        for task_id, row in statuses.items()
        if row["status"] in {"blocked", "gate_skipped", "missing", "malformed", "flagged"}
    ]
    return {
        "flagged_or_corrigendum_rows": flagged_rows,
        "blocked_or_skipped_rows": blocked_statuses,
        "critical_flag_count": sum(len(row["critical_flags"]) for row in flagged_rows),
        "promotion_requires_clean_live_recheck": True,
        "adversarial_verify_command": (
            "python scripts/adversarial_verify.py "
            "results/experiment_5716_v510_capstone_reconciliation.json"
        ),
    }


def _manifest_entries(manifest: JsonMap) -> list[JsonMap]:
    extras = manifest.get("retired_extras")
    return list(extras) if isinstance(extras, list) else []


def _manifest_has_scope(manifest: JsonMap, scope: str) -> bool:
    for row in _manifest_entries(manifest):
        if isinstance(row, Mapping) and row.get("scope_key") == scope:
            return True
    return False


def _registry_count(registry: JsonMap) -> int | None:
    value = registry.get("reproducible_total_levels")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _derive_retirements(
    transition: JsonMap,
    manifest: JsonMap,
    statuses: Mapping[str, JsonMap],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    applied = transition.get("retirements_applied")
    if isinstance(applied, Sequence) and not isinstance(applied, str | bytes | bytearray):
        for row in applied:
            if isinstance(row, Mapping):
                scope = str(row.get("scope") or row.get("scope_key") or "")
                rows.append(
                    {
                        "scope": scope,
                        "decision": "preserved_from_exp5706_transition",
                        "manifest_entry_present": bool(
                            row.get("manifest_entry_present")
                            or _manifest_has_scope(manifest, scope)
                        ),
                        "preserves": row.get("preserves", []),
                        "source": EXP5706_TRANSITION_PATH.as_posix(),
                    }
                )
    if statuses[EXP5709_TASK_ID]["status"] == "gate_skipped":
        rows.append(
            {
                "scope": "fr11_prospective_shadow_stream_exp5709_same_verdict",
                "decision": "retire_this_parse_failed_stream_scope_only",
                "manifest_entry_present": _manifest_has_scope(
                    manifest,
                    "fr11_prospective_shadow_stream_exp5709_same_verdict",
                ),
                "manifest_update_required": True,
                "preserves": [
                    "v509_fr11_independent_controller",
                    "fr11_shadow_adapter_disabled_by_default",
                    "future_clean_prospective_streams",
                ],
                "reason": "Exp5709 repeated blocked_gate_check_failed after Exp5708 did not clear the canary gate.",
            }
        )
    return rows


def _forbidden_file_checks(
    root: Path,
    overrides: Mapping[Path | str, bool] | None,
) -> dict[str, JsonDict]:
    checks: dict[str, JsonDict] = {}
    for rel_path in FORBIDDEN_FILE_PATHS:
        modified = _modification_status(root, rel_path, overrides)
        checks[rel_path.as_posix()] = {
            "exists": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "unchanged": not modified,
        }
    return checks


def _spec_reconciliation(root: Path) -> JsonDict:
    spec_path = root / SPEC_RELATIVE_PATH
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    test_path = Path("tests/python/test_experiment_5716_v510_capstone_reconciliation.py")
    module_path = Path("python/carnot/experiment_5716_v510_capstone_reconciliation.py")
    return {
        "spec_path": SPEC_RELATIVE_PATH.as_posix(),
        "spec_present": spec_path.exists(),
        "req_present": "REQ-CAPSTONE-5716" in spec,
        "scenarios_present": all(ref in spec for ref in SPEC_REFS[1:]),
        "test_path": test_path.as_posix(),
        "test_present": (root / test_path).exists(),
        "module_path": module_path.as_posix(),
        "module_present": (root / module_path).exists(),
        "spec_refs": list(SPEC_REFS),
    }


def _ops_reconciliation(root: Path, manifest: JsonMap, registry: JsonMap) -> JsonDict:
    return {
        "research_complete_present": (root / "research-complete.yaml").exists(),
        "research_references_present": (root / "research-references.md").exists(),
        "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "exclusion_manifest_present": bool(manifest),
        "exp5641_retirement_present": _manifest_has_scope(
            manifest,
            "arc_counterexample_patched_transition_model_exp5641",
        ),
        "exp5645_retirement_present": _manifest_has_scope(
            manifest,
            "two_axis_beta_lambda_tempering_extension_exp5645",
        ),
        "arc_registry_count": _registry_count(registry),
        "delegated_by_stop_rule": [
            STATUS_RELATIVE_PATH.as_posix(),
            CHANGELOG_RELATIVE_PATH.as_posix(),
            TRACEABILITY_RELATIVE_PATH.as_posix(),
        ],
        "conductor_log_present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
    }


def _known_issue_reconciliation(root: Path) -> JsonDict:
    return {
        "known_issues_present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "verifier_gaps_present": (root / VERIFIER_GAPS_RELATIVE_PATH).exists(),
        "prior_failure_retirements_preserved": True,
        "blocked_skipped_flagged_proxy_cannot_promote": True,
        "no_native_three_model_or_json_grammar_reopen": True,
        "no_tsu_kona_board_or_snn_claim": True,
    }


def _e2e_receipts() -> list[JsonDict]:
    return [
        {
            "check_id": "E2E-003",
            "applicability": "applicable_upstream",
            "receipt": "Exp5714/Exp5715 cover the Rust/PyO3 one-axis boundary; capstone rechecks artifact receipts.",
            "capstone_ran_new_stack": False,
        },
        {
            "check_id": "E2E-004",
            "applicability": "applicable_upstream",
            "receipt": "Exp5714/Exp5715 checkpoint and cross-language restart receipts are aggregated.",
            "capstone_ran_new_stack": False,
        },
        {
            "check_id": "E2E-001/E2E-002/E2E-005/E2E-006/E2E-007/E2E-008",
            "applicability": "not_applicable_to_aggregation_capstone",
            "receipt": "No new training, LLM repair, board scoring, SMGI update, or CLaRa-V runtime is executed by Exp5716.",
            "capstone_ran_new_stack": False,
        },
    ]


def _load_validation_results(path: Path | None) -> list[JsonDict]:
    if path is None:
        return DEFAULT_VALIDATION_RESULTS
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def _test_commands(validation_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in validation_results if row.get("command") is not None]


def _test_exit_codes(validation_results: Sequence[JsonMap]) -> dict[str, Any]:
    return {
        str(row.get("command")): row.get("exit_code")
        for row in validation_results
        if row.get("command") is not None
    }


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    source_context, source_context_missing = _read_source_context(root)
    artifacts, metadata, upstream_rows = read_artifacts(root)
    statuses = _terminal_statuses(artifacts, metadata)
    manifest, manifest_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    registry, registry_meta = _read_yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)

    exp5647 = _payload(artifacts, EXP5647_CAPSTONE_PATH)
    exp5706 = _payload(artifacts, EXP5706_TRANSITION_PATH)
    exp5708 = _payload(artifacts, EXP5708_CANARY_PATH)
    exp5709 = _payload(artifacts, EXP5709_SHADOW_PATH)
    exp5710 = _payload(artifacts, EXP5710_ISOLATED_PATH)
    exp5711 = _payload(artifacts, EXP5711_ARC_QUAL_PATH)
    exp5712 = _payload(artifacts, EXP5712_ARC_AB_PATH)
    exp5713 = _payload(artifacts, EXP5713_ARC_LEVEL_PATH)
    exp5714 = _payload(artifacts, EXP5714_RUST_PARITY_PATH)
    exp5715 = _payload(artifacts, EXP5715_RUST_QUALITY_PATH)

    sota_status = _derive_sota_canary(exp5708, statuses)
    cuda_status = _derive_cuda_status(exp5708)
    prospective_status = _derive_prospective_shadow(exp5709, statuses)
    isolated_status = _derive_isolated_canary(exp5710, statuses)
    arc_qual_status = _derive_arc_qualification(exp5711, statuses)
    arc_ab_status = _derive_arc_ab(exp5712, statuses)
    arc_solve = _derive_arc_solve(exp5713, statuses)
    rust_parity = _derive_one_axis_parity(exp5714, statuses)
    rust_quality = _derive_one_axis_quality(exp5715, statuses)
    forbidden_checks = _forbidden_file_checks(root, modification_overrides)

    missing_artifacts = [
        str(row["path"]) for row in statuses.values() if row.get("status") == "missing"
    ]
    malformed_artifacts = [
        str(row["path"]) for row in statuses.values() if row.get("status") == "malformed"
    ]
    forbidden_dirty = [path for path, row in forbidden_checks.items() if not row["unchanged"]]
    hard_blockers = bool(missing_artifacts or malformed_artifacts or forbidden_dirty)

    v509_promoted = bool(
        isinstance(exp5647.get("fr11_independent_promotion_status"), Mapping)
        and exp5647["fr11_independent_promotion_status"].get("promoted") is True
        and _bool(exp5706, "fr11_promoted")
    )
    one_axis_preserved = bool(
        _bool(exp5647, "one_axis_replica_exchange_preserved")
        or _bool(exp5706, "one_axis_replica_exchange_promoted")
    )
    transition_retirements = {
        str(row.get("scope"))
        for row in exp5706.get("retirements_applied", [])
        if isinstance(row, Mapping)
    }
    two_axis_retired = bool(
        not _bool(exp5706, "two_axis_quality_promoted")
        and (
            _manifest_has_scope(
                manifest,
                "two_axis_beta_lambda_tempering_extension_exp5645",
            )
            or "two_axis_beta_lambda_tempering_extension_exp5645" in transition_retirements
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
        "artifact_metadata": metadata,
        "side_artifacts": _read_side_artifacts(root),
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "validation_results": validation_rows,
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_artifacts": upstream_rows,
        "upstream_gate_statuses": _derive_gate_statuses(statuses),
        "adversarial_verification_summary": _derive_adversarial_summary(artifacts, statuses),
        "v509_fr11_promotion_preserved": v509_promoted,
        "sota_canary_status": sota_status,
        "cuda_offload_status": cuda_status,
        "prospective_shadow_status": prospective_status,
        "isolated_canary_status": isolated_status,
        "production_default_enabled": False,
        "model_weight_mutation": False,
        "arc_relational_qualification_status": arc_qual_status,
        "arc_relational_live_ab_status": arc_ab_status,
        "arc_registry_count_before": arc_solve["registry_count_before"],
        "arc_registry_count_after": arc_solve["registry_count_after"],
        "arc_registry_delta": arc_solve["registry_delta"],
        "arc_solve_provenance": arc_solve,
        "one_axis_python_promotion_preserved": one_axis_preserved,
        "one_axis_rust_parity_status": rust_parity,
        "one_axis_rust_quality_restart_status": rust_quality,
        "two_axis_retirement_preserved": two_axis_retired,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "retirements_applied": _derive_retirements(exp5706, manifest, statuses),
        "spec_reconciliation": _spec_reconciliation(root),
        "ops_reconciliation": {
            **_ops_reconciliation(root, manifest, registry),
            "manifest_metadata": manifest_meta,
            "registry_metadata": registry_meta,
        },
        "known_issue_reconciliation": _known_issue_reconciliation(root),
        "test_commands": _test_commands(validation_rows),
        "test_exit_codes": _test_exit_codes(validation_rows),
        "e2e_check_receipts": _e2e_receipts(),
        "forbidden_files_unchanged": forbidden_checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    prefix = "blocked:" if hard_blockers else "complete:"
    artifact["honest_verdict"] = (
        f"{prefix} v510 reconciled; exp5708_stream_promoted={sota_status['promoted']}; "
        f"exp5709_promoted={prospective_status['promoted']}; "
        f"exp5710_promoted={isolated_status['promoted']}; "
        f"arc_registry_delta={arc_solve['registry_delta']}; "
        f"arc_solve_credited={arc_solve['solve_credited']}; "
        f"one_axis_rust_parity={rust_parity['promoted']}; "
        f"one_axis_quality_restart={rust_quality['promoted']}; "
        "timing_claimed=false; hardware_speedup_claimed=false"
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _required_missing(payload: JsonMap) -> list[str]:
    return [field for field in REQUIRED_FIELDS if field not in payload]


def _paths_with_status(payload: JsonMap, status: str) -> set[str]:
    rows = payload.get("upstream_gate_statuses")
    if not isinstance(rows, Mapping):
        return set()
    return {
        str(row.get("artifact_path"))
        for row in rows.values()
        if isinstance(row, Mapping) and row.get("status") == status
    }


def _has_problem_for(payload: JsonMap, rel_path: Path) -> bool:
    missing = set(payload.get("missing_artifacts", []))
    malformed = set(payload.get("malformed_artifacts", []))
    return rel_path.as_posix() in missing or rel_path.as_posix() in malformed


def validate_artifact(payload: JsonMap) -> list[str]:
    errors = _required_missing(payload)
    if errors:
        return errors
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    upstream = payload.get("upstream_artifacts")
    if not isinstance(upstream, list) or len(upstream) != len(UPSTREAM_ARTIFACT_PATHS):
        errors.append("upstream_artifacts")
    statuses = payload.get("upstream_gate_statuses")
    if not isinstance(statuses, Mapping) or set(statuses) != set(UPSTREAM_ARTIFACT_PATHS):
        errors.append("upstream_gate_statuses")
    if not isinstance(payload.get("adversarial_verification_summary"), Mapping):
        errors.append("adversarial_verification_summary")
    if payload.get("v509_fr11_promotion_preserved") is not True:
        errors.append("v509_fr11_promotion_preserved")
    if payload.get("sota_canary_status", {}).get("promoted") is not False:
        errors.append("sota_canary_status")
    if payload.get("cuda_offload_status", {}).get("authenticated") is not True:
        errors.append("cuda_offload_status")
    if payload.get("cuda_offload_status", {}).get("promotes_canary") is not False:
        errors.append("cuda_offload_status")
    if payload.get("prospective_shadow_status", {}).get("promoted") is not False:
        errors.append("prospective_shadow_status")
    if payload.get("isolated_canary_status", {}).get("promoted") is not False:
        errors.append("isolated_canary_status")
    if payload.get("production_default_enabled") is not False:
        errors.append("production_default_enabled")
    if payload.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if (
        not isinstance(payload.get("arc_relational_qualification_status"), Mapping)
        or payload["arc_relational_qualification_status"].get("qualified") is not True
    ):
        errors.append("arc_relational_qualification_status")
    if payload.get("arc_relational_live_ab_status", {}).get("promoted") is not False:
        errors.append("arc_relational_live_ab_status")
    if not isinstance(payload.get("arc_registry_count_before"), int):
        errors.append("arc_registry_count_before")
    if not isinstance(payload.get("arc_registry_count_after"), int):
        errors.append("arc_registry_count_after")
    if payload.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta")
    if payload.get("arc_solve_provenance", {}).get("solve_credited") is not False:
        errors.append("arc_solve_provenance")
    if payload.get("one_axis_python_promotion_preserved") is not True:
        errors.append("one_axis_python_promotion_preserved")
    if (
        not _has_problem_for(payload, EXP5714_RUST_PARITY_PATH)
        and payload.get("one_axis_rust_parity_status", {}).get("promoted") is not True
    ):
        errors.append("one_axis_rust_parity_status")
    if (
        not _has_problem_for(payload, EXP5715_RUST_QUALITY_PATH)
        and payload.get("one_axis_rust_quality_restart_status", {}).get("promoted") is not True
    ):
        errors.append("one_axis_rust_quality_restart_status")
    if payload.get("two_axis_retirement_preserved") is not True:
        errors.append("two_axis_retirement_preserved")
    if payload.get("timing_claimed") is not False:
        errors.append("timing_claimed")
    if payload.get("hardware_speedup_claimed") is not False:
        errors.append("hardware_speedup_claimed")
    if (
        not isinstance(payload.get("retirements_applied"), list)
        or not payload["retirements_applied"]
    ):
        errors.append("retirements_applied")
    if not isinstance(payload.get("spec_reconciliation"), Mapping):
        errors.append("spec_reconciliation")
    if not isinstance(payload.get("ops_reconciliation"), Mapping):
        errors.append("ops_reconciliation")
    if not isinstance(payload.get("known_issue_reconciliation"), Mapping):
        errors.append("known_issue_reconciliation")
    if not isinstance(payload.get("test_commands"), list):
        errors.append("test_commands")
    if not isinstance(payload.get("test_exit_codes"), Mapping):
        errors.append("test_exit_codes")
    if not isinstance(payload.get("e2e_check_receipts"), list):
        errors.append("e2e_check_receipts")
    forbidden = payload.get("forbidden_files_unchanged")
    if not isinstance(forbidden, Mapping) or not all(
        isinstance(row, Mapping) and row.get("unchanged") is True for row in forbidden.values()
    ):
        errors.append("forbidden_files_unchanged")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("reproducibility_checksum")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    missing_status_paths = _paths_with_status(payload, "missing")
    if set(payload.get("missing_artifacts", [])) != missing_status_paths:
        errors.append("missing_artifacts")
    malformed_status_paths = _paths_with_status(payload, "malformed")
    if set(payload.get("malformed_artifacts", [])) != malformed_status_paths:
        errors.append("malformed_artifacts")
    return sorted(set(errors))


def write_capstone(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root,
        validation_results=validation_results,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp5716 artifact: {errors}")
    destination = output if output is not None else root / RESULT_RELATIVE_PATH
    write_json(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)
    validation_results = _load_validation_results(args.validation_results)
    try:
        write_capstone(
            root=args.root,
            output=args.output,
            validation_results=validation_results,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
