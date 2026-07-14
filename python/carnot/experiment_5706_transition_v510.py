"""Exp5706 transition receipt from milestone .509 into .510.

Spec refs: REQ-CAPSTONE-5706, SCENARIO-CAPSTONE-5706,
SCENARIO-CAPSTONE-5706-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5706-FIELD-PRINCIPLES.

This module is a reconciliation ledger. It does not run inference, tune a
model, or reinterpret scientific results. Its job is to bind the `.510`
dependency map to the terminal `.509` capstone, close exactly the two manifest
retirement debts named there, and prevent the already-written Exp5700-Exp5705
outer-loop artifacts from being overwritten by the new task range.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5706_transition_v510.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")

EXPERIMENT = "experiment_5706_transition_v510"
EXPERIMENT_ID = "exp5706-transition-v510"
PREVIOUS_MILESTONE = "2026.07.509"
CURRENT_MILESTONE = "2026.07.510"
PREVIOUS_TASK_RANGE = "exp5636-exp5647"
CURRENT_TASK_RANGE = "exp5706-exp5716"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5706
SCHEMA = "carnot.experiment_5706.transition_v510.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5706",
    "SCENARIO-CAPSTONE-5706",
    "SCENARIO-CAPSTONE-5706-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5706-FIELD-PRINCIPLES",
)

EXP5636_TRANSITION_PATH = Path("results/experiment_5636_transition_v509.json")
EXP5637_SOURCE_PATH = Path("results/experiment_5637_v509_source_delta_ingestion.json")
EXP5638_SCHEMA_PATH = Path("results/experiment_5638_fr11_gate_schema_corrigendum.json")
EXP5639_AUDIT_PATH = Path("results/experiment_5639_anytime_valid_csl_independent_audit.json")
EXP5640_SHADOW_PATH = Path("results/experiment_5640_fr11_shadow_pipeline_integration.json")
EXP5641_ARC_MODEL_PATH = Path("results/experiment_5641_arc_counterexample_executable_model.json")
EXP5642_ARC_LIVE_AB_PATH = Path("results/experiment_5642_arc_executable_model_live_ab.json")
EXP5643_ARC_LEVEL_PATH = Path(
    "results/experiment_5643_arc_live_self_discovery_levelup_v509.json"
)
EXP5644_TWO_AXIS_EXACT_PATH = Path(
    "results/experiment_5644_two_axis_parallel_tempering_exact_audit.json"
)
EXP5645_TWO_AXIS_QUALITY_PATH = Path(
    "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json"
)
EXP5646_RUST_PARITY_PATH = Path("results/experiment_5646_two_axis_tempering_rust_parity.json")
EXP5647_CAPSTONE_PATH = Path("results/experiment_5647_v509_capstone_reconciliation.json")

V509_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5636-transition-v509": EXP5636_TRANSITION_PATH,
    "exp5637-v509-source-delta-ingestion": EXP5637_SOURCE_PATH,
    "exp5638-fr11-gate-schema-corrigendum": EXP5638_SCHEMA_PATH,
    "exp5639-anytime-valid-csl-independent-audit": EXP5639_AUDIT_PATH,
    "exp5640-fr11-shadow-pipeline-integration": EXP5640_SHADOW_PATH,
    "exp5641-arc-counterexample-executable-model": EXP5641_ARC_MODEL_PATH,
    "exp5642-arc-executable-model-live-ab": EXP5642_ARC_LIVE_AB_PATH,
    "exp5643-arc-live-self-discovery-levelup-v509": EXP5643_ARC_LEVEL_PATH,
    "exp5644-two-axis-parallel-tempering-exact-audit": EXP5644_TWO_AXIS_EXACT_PATH,
    "exp5645-two-axis-tempering-hard-constraint-quality": EXP5645_TWO_AXIS_QUALITY_PATH,
    "exp5646-two-axis-tempering-rust-parity": EXP5646_RUST_PARITY_PATH,
    "exp5647-v509-capstone-reconciliation": EXP5647_CAPSTONE_PATH,
}
V509_TASK_IDS = tuple(V509_ARTIFACT_PATHS)

OUTER_LOOP_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5700-goal-predicate-veto-live-integration": Path(
        "results/experiment_5700_goal_predicate_veto_live_integration.json"
    ),
    "exp5701-candidate-scoring-stack-bare-control-ab-headroom": Path(
        "results/experiment_5701_candidate_scoring_stack_bare_control_ab_headroom.json"
    ),
    "exp5702-dynamics-gate-pass-rate-survey": Path(
        "results/experiment_5702_dynamics_gate_pass_rate_survey.json"
    ),
    "exp5703-sp80-candidate-stack-mechanism-trace": Path(
        "results/experiment_5703_sp80_candidate_stack_mechanism_trace.json"
    ),
    "exp5704-dynamics-gate-relaxed-threshold-ab": Path(
        "results/experiment_5704_dynamics_gate_relaxed_threshold_ab.json"
    ),
    "exp5705-full-precision-27b-vs-4bit-quant-ab": Path(
        "results/experiment_5705_full_precision_27b_vs_4bit_quant_ab.json"
    ),
}
OUTER_LOOP_TASK_IDS = tuple(OUTER_LOOP_ARTIFACT_PATHS)

CURRENT_TASK_IDS = (
    "exp5706-transition-v510",
    "exp5707-v510-source-delta-ingestion",
    "exp5708-sota-exact-constraint-canary",
    "exp5709-fr11-prospective-shadow-stream",
    "exp5710-fr11-isolated-act-on-advice-canary",
    "exp5711-placement-spatial-goal-energy-qualification",
    "exp5712-known-level-relational-route-ab",
    "exp5713-arc-live-levelup-attempt",
    "exp5714-one-axis-rust-python-exact-parity",
    "exp5715-one-axis-rust-quality-restart-parity",
    "exp5716-v510-capstone",
)

OUTER_LOOP_IMPLICATIONS: dict[str, str] = {
    "exp5700-goal-predicate-veto-live-integration": (
        "Goal-predicate veto can catch a real miscalibrated induced predicate."
    ),
    "exp5701-candidate-scoring-stack-bare-control-ab-headroom": (
        "Candidate stack tied level progress but improved efficiency versus bare control."
    ),
    "exp5702-dynamics-gate-pass-rate-survey": (
        "Strict dynamics gate pass rate is low and must be accounted for downstream."
    ),
    "exp5703-sp80-candidate-stack-mechanism-trace": (
        "GAP-5703: sp80 candidate stack mechanisms were inert; goal energy was constant."
    ),
    "exp5704-dynamics-gate-relaxed-threshold-ab": (
        "Relaxed dynamics threshold produced no clean relaxed-only attempt band."
    ),
    "exp5705-full-precision-27b-vs-4bit-quant-ab": (
        "Full-precision 27B/Q8 path was less reliable than the current 9B stack."
    ),
}

REQUIRED_MANIFEST_RETIREMENTS: tuple[JsonDict, ...] = (
    {
        "id": "exp5641_arc_counterexample_transition_model_retired_v510",
        "scope_key": "arc_counterexample_patched_transition_model_exp5641",
        "experiment_scope": (
            "Exp5641 counterexample-patched ARC executable transition model only"
        ),
        "reason": (
            "Exp5641 accepted safe patches but executable_model_ready_score stayed 0.0 "
            "and the .509 capstone recorded terminal retirement; do not retune or reuse "
            "this counterexample patcher without operator authorization."
        ),
        "experiment_ids": ["exp5641"],
        "retired_milestone": CURRENT_MILESTONE,
        "retired_by_artifact": EXP5641_ARC_MODEL_PATH.as_posix(),
        "recorded_by_artifact": RESULT_RELATIVE_PATH.as_posix(),
        "operator_reopen_required": True,
        "retire_if_same_verdict": True,
        "blocked_patterns": [
            "Exp5641 counterexample-patched ARC executable transition model",
            "counterexample-patched ARC transition patcher exp5641",
            "counterexample_patched_executable_model_retired_terminal",
        ],
    },
    {
        "id": "exp5645_two_axis_beta_lambda_tempering_retired_v510",
        "scope_key": "two_axis_beta_lambda_tempering_extension_exp5645",
        "experiment_scope": (
            "Exp5645 two-axis beta-lambda tempering hard-constraint quality extension only"
        ),
        "reason": (
            "Exp5645 failed every preregistered quality-promotion gate and regressed "
            "hard-instance quality; preserve the promoted one-axis exchange sampler."
        ),
        "experiment_ids": ["exp5645"],
        "retired_milestone": CURRENT_MILESTONE,
        "retired_by_artifact": EXP5645_TWO_AXIS_QUALITY_PATH.as_posix(),
        "recorded_by_artifact": RESULT_RELATIVE_PATH.as_posix(),
        "operator_reopen_required": True,
        "retire_if_same_verdict": True,
        "blocked_patterns": [
            "Exp5645 two-axis beta-lambda tempering hard-constraint quality extension",
            "two-axis beta-lambda tempering extension exp5645",
            "two_axis_quality_extension_exp5645",
        ],
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required transition field.",
    "source_capstone_hash": "binds the transition to the terminal .509 capstone bytes.",
    "v509_task_verdicts": "every Exp5636-Exp5647 verdict is explicit before carry-forward.",
    "fr11_promoted": "preserves the independent FR-11 promotion only from Exp5647.",
    "fr11_shadow_default_enabled": "keeps the shadow adapter disabled by default.",
    "arc_registry_count": "authoritative reproduced-level count after .509.",
    "arc_registry_delta": "the .509 live attempt banked no ARC level.",
    "one_axis_replica_exchange_promoted": "preserves the prior one-axis sampler promotion.",
    "two_axis_quality_promoted": "records the terminal two-axis quality failure.",
    "missing_retirements_before": "the manifest debt named by Exp5647 before this task.",
    "retirements_applied": "narrow manifest retirements closed by this transition.",
    "outer_loop_snapshot": "Exp5700-Exp5705 cannot be overwritten or reused.",
    "current_task_range": "canonical allocation is exp5706-exp5716.",
    "dependency_map": "successors and prerequisites are reconstructable.",
    "gate_map": "structured gates are auditable and ID-valid.",
    "retired_scopes": "terminal negative scopes are bounded narrowly.",
    "preserved_scopes": "non-retired scopes stay live and unbroadened.",
    "timing_claimed": "bare false prevents runtime inflation.",
    "hardware_speedup_claimed": "bare false prevents hardware inflation.",
    "inference_substrate": "artifact_reconciliation_only -- no inference occurred.",
    "test_commands": "verification commands are replayable.",
    "test_exit_codes": "observed command exits are recorded without laundering failures.",
    "reproducibility_checksum": "content-addressed transition output is stable.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "previous_milestone",
    "current_milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "artifact_metadata",
    "missing_artifacts",
    "malformed_artifacts",
    "manifest_metadata",
    "manifest_debt_after",
    "protected_file_checks",
    "dependency_id_validation",
    "model_hardware_prerequisites",
    "unconditional_arc_path",
    "expected_deliverables",
    "pre_existing_broad_suite_debt",
    *FIELD_PRINCIPLES,
)

DEFAULT_VALIDATION_RESULTS: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5706_transition_v510.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage erase && .venv/bin/coverage run "
            "--include=python/carnot/experiment_5706_transition_v510.py "
            "-m pytest tests/python/test_experiment_5706_transition_v510.py "
            "-q --no-cov -n 0 && .venv/bin/coverage report "
            "--include=python/carnot/experiment_5706_transition_v510.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": "python scripts/exclusion_manifest_lint.py",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": "python scripts/adversarial_verify.py results/experiment_5706_transition_v510.json",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
)


def _terminal_prefix_ok(verdict: str) -> bool:
    return str(verdict).startswith(TERMINAL_PREFIXES)


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


def _read_manifest(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "yaml_type": None,
        "sha256": path_sha256(path),
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        metadata.update({"error": "malformed_yaml", "message": str(exc)})
        return {}, metadata
    metadata["yaml_type"] = type(parsed).__name__
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_yaml_mapping"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


def _status_for_meta(meta: JsonMap) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    return "present"


def _registry_count_from_yaml(payload: JsonMap) -> int | None:
    value = payload.get("reproducible_total_levels")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _payload(payloads: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = payloads.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    return str(payload.get("honest_verdict") or "")


def _read_artifacts(
    root: Path,
) -> tuple[dict[str, JsonDict], dict[str, JsonDict], list[str], list[str]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    missing: list[str] = []
    malformed: list[str] = []
    for rel_path in [*V509_ARTIFACT_PATHS.values(), *OUTER_LOOP_ARTIFACT_PATHS.values()]:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        payloads[rel] = payload
        metadata[rel] = meta
        status = _status_for_meta(meta)
        if status == "missing":
            missing.append(rel)
        elif status == "malformed":
            malformed.append(rel)
    return payloads, metadata, missing, malformed


def _v509_task_verdicts(payloads: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in V509_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(payloads, rel_path)
        meta = metadata.get(rel, {})
        rows[task_id] = {
            "path": rel,
            "status": _status_for_meta(meta),
            "sha256": meta.get("sha256"),
            "schema": payload.get("schema"),
            "experiment_id": payload.get("experiment_id", payload.get("experiment")),
            "honest_verdict": _verdict(payload) or None,
            "terminal_prefix_valid": _terminal_prefix_ok(_verdict(payload)),
        }
    return rows


def _outer_loop_snapshot(
    payloads: Mapping[str, JsonMap], metadata: JsonMap
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in OUTER_LOOP_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(payloads, rel_path)
        row: JsonDict = {
            "path": rel,
            "status": _status_for_meta(metadata.get(rel, {})),
            "sha256": metadata.get(rel, {}).get("sha256"),
            "schema": payload.get("schema"),
            "experiment_id": payload.get("experiment_id", payload.get("experiment", task_id)),
            "honest_verdict": _verdict(payload) or None,
            "implication": OUTER_LOOP_IMPLICATIONS[task_id],
        }
        if task_id == "exp5703-sp80-candidate-stack-mechanism-trace":
            row["gap_id"] = "GAP-5703"
        rows[task_id] = row
    return rows


def _manifest_entries(manifest: JsonMap) -> list[JsonMap]:
    entries = manifest.get("retired_extras", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _manifest_has_required(entry: JsonMap, required: JsonMap) -> bool:
    return entry.get("id") == required["id"] and entry.get("experiment_scope") == required[
        "experiment_scope"
    ]


def _manifest_retirement_status(manifest: JsonMap) -> tuple[list[JsonDict], list[JsonDict]]:
    entries = _manifest_entries(manifest)
    applied: list[JsonDict] = []
    debt: list[JsonDict] = []
    for required in REQUIRED_MANIFEST_RETIREMENTS:
        present = any(_manifest_has_required(entry, required) for entry in entries)
        row = {
            "id": required["id"],
            "scope": required["scope_key"],
            "experiment_scope": required["experiment_scope"],
            "retired_by_artifact": required["retired_by_artifact"],
            "recorded_by_artifact": required["recorded_by_artifact"],
            "manifest_entry_present": present,
            "operator_reopen_required": required["operator_reopen_required"],
            "preserves": [
                "one_axis_temperature_exchange_replica_exchange",
                "generic_replica_exchange",
                "generic_arc_models",
            ],
        }
        applied.append(row)
        if not present:
            debt.append(row)
    return applied, debt


def _missing_retirements_before(capstone: JsonMap) -> list[JsonDict]:
    rows = capstone.get("retirements_applied", [])
    if not isinstance(rows, list):
        return []
    return [
        {
            "scope": str(row.get("scope")),
            "reason": row.get("reason"),
            "manifest_update_required": True,
            "manifest_updated": False,
        }
        for row in rows
        if isinstance(row, Mapping)
        and row.get("manifest_update_required") is True
        and row.get("manifest_updated") is False
    ]


def _dependency_map() -> dict[str, JsonDict]:
    return {
        "exp5706-transition-v510": {
            "depends_on": [
                "exp5647-v509-capstone-reconciliation",
                *OUTER_LOOP_TASK_IDS,
            ],
            "deliverable": RESULT_RELATIVE_PATH.as_posix(),
        },
        "exp5707-v510-source-delta-ingestion": {
            "depends_on": ["exp5706-transition-v510"],
            "deliverable": "results/experiment_5707_v510_source_delta_ingestion.json",
        },
        "exp5708-sota-exact-constraint-canary": {
            "depends_on": ["exp5706-transition-v510", "exp5707-v510-source-delta-ingestion"],
            "deliverable": "results/experiment_5708_sota_exact_constraint_canary.json",
        },
        "exp5709-fr11-prospective-shadow-stream": {
            "depends_on": [
                "exp5639-anytime-valid-csl-independent-audit",
                "exp5640-fr11-shadow-pipeline-integration",
                "exp5708-sota-exact-constraint-canary",
            ],
            "deliverable": "results/experiment_5709_fr11_prospective_shadow_stream.json",
        },
        "exp5710-fr11-isolated-act-on-advice-canary": {
            "depends_on": ["exp5640-fr11-shadow-pipeline-integration", "exp5709-fr11-prospective-shadow-stream"],
            "deliverable": "results/experiment_5710_fr11_isolated_act_on_advice_canary.json",
        },
        "exp5711-placement-spatial-goal-energy-qualification": {
            "depends_on": ["exp5703-sp80-candidate-stack-mechanism-trace", "exp5706-transition-v510"],
            "deliverable": "results/experiment_5711_placement_spatial_goal_energy_qualification.json",
        },
        "exp5712-known-level-relational-route-ab": {
            "depends_on": ["exp5711-placement-spatial-goal-energy-qualification"],
            "deliverable": "results/experiment_5712_known_level_relational_route_ab.json",
        },
        "exp5713-arc-live-levelup-attempt": {
            "depends_on": ["exp5706-transition-v510"],
            "optional_prerequisites": ["exp5712-known-level-relational-route-ab"],
            "unconditional": True,
            "deliverable": "results/experiment_5713_arc_live_self_discovery_levelup_v510.json",
        },
        "exp5714-one-axis-rust-python-exact-parity": {
            "depends_on": ["exp5636-transition-v509", "exp5647-v509-capstone-reconciliation"],
            "deliverable": "results/experiment_5714_one_axis_rust_python_exact_parity.json",
        },
        "exp5715-one-axis-rust-quality-restart-parity": {
            "depends_on": ["exp5714-one-axis-rust-python-exact-parity"],
            "deliverable": "results/experiment_5715_one_axis_rust_quality_restart_parity.json",
        },
        "exp5716-v510-capstone": {
            "depends_on": [
                "exp5706-transition-v510",
                "exp5707-v510-source-delta-ingestion",
                "exp5710-fr11-isolated-act-on-advice-canary",
                "exp5713-arc-live-levelup-attempt",
                "exp5715-one-axis-rust-quality-restart-parity",
            ],
            "deliverable": "results/experiment_5716_v510_capstone.json",
        },
    }


def _gate_map() -> dict[str, list[JsonDict]]:
    return {
        "exp5708-sota-exact-constraint-canary": [
            {"upstream": "exp5706-transition-v510", "field": "current_task_range", "op": "==", "value": CURRENT_TASK_RANGE}
        ],
        "exp5709-fr11-prospective-shadow-stream": [
            {"upstream": "exp5708-sota-exact-constraint-canary", "field": "sota_canary_ready_score", "op": ">=", "value": 1.0},
            {"upstream": "exp5708-sota-exact-constraint-canary", "field": "validator_disagreement_count", "op": "==", "value": 0},
            {"upstream": "exp5708-sota-exact-constraint-canary", "field": "cuda_offload_authenticated_score", "op": ">=", "value": 1.0},
        ],
        "exp5710-fr11-isolated-act-on-advice-canary": [
            {"upstream": "exp5709-fr11-prospective-shadow-stream", "field": "prospective_shadow_ready_score", "op": ">=", "value": 1.0},
            {"upstream": "exp5709-fr11-prospective-shadow-stream", "field": "unsafe_false_accept_count", "op": "==", "value": 0},
        ],
        "exp5712-known-level-relational-route-ab": [
            {"upstream": "exp5711-placement-spatial-goal-energy-qualification", "field": "goal_energy_route_ready_score", "op": ">=", "value": 1.0}
        ],
        "exp5713-arc-live-levelup-attempt": [],
        "exp5715-one-axis-rust-quality-restart-parity": [
            {"upstream": "exp5714-one-axis-rust-python-exact-parity", "field": "rust_python_parity_ready_score", "op": ">=", "value": 1.0}
        ],
    }


def _valid_dependency_ids(dependency_map: JsonMap, gate_map: JsonMap) -> JsonDict:
    allowed = set(V509_TASK_IDS) | set(OUTER_LOOP_TASK_IDS) | set(CURRENT_TASK_IDS)
    seen: list[str] = []
    invalid: list[str] = []
    for task_id, row in dependency_map.items():
        seen.append(str(task_id))
        if task_id not in allowed:
            invalid.append(str(task_id))
        if isinstance(row, Mapping):
            for field in ("depends_on", "optional_prerequisites"):
                values = row.get(field, [])
                if isinstance(values, list):
                    seen.extend(str(value) for value in values)
                    invalid.extend(str(value) for value in values if str(value) not in allowed)
    for task_id, gates in gate_map.items():
        seen.append(str(task_id))
        if task_id not in allowed:
            invalid.append(str(task_id))
        if isinstance(gates, list):
            for gate in gates:
                if isinstance(gate, Mapping):
                    upstream = str(gate.get("upstream"))
                    seen.append(upstream)
                    if upstream not in allowed:
                        invalid.append(upstream)
    return {
        "valid": not invalid,
        "invalid_ids": sorted(set(invalid)),
        "checked_ids": sorted(set(seen)),
        "allowed_ranges": [PREVIOUS_TASK_RANGE, "exp5700-exp5705", CURRENT_TASK_RANGE],
    }


def _retired_scopes() -> list[JsonDict]:
    return [
        {
            "scope": "arc_counterexample_patched_transition_model_exp5641",
            "boundary": "only the Exp5641 counterexample patcher is retired",
            "retired_by_artifact": EXP5641_ARC_MODEL_PATH.as_posix(),
            "preserves": ["generic_arc_models", "relational_goal_energy", "live_agent_self_discovery"],
        },
        {
            "scope": "two_axis_beta_lambda_tempering_extension_exp5645",
            "boundary": "only the two-axis beta-lambda extension is retired",
            "retired_by_artifact": EXP5645_TWO_AXIS_QUALITY_PATH.as_posix(),
            "preserves": ["one_axis_temperature_exchange", "generic_replica_exchange"],
        },
    ]


def _preserved_scopes() -> list[JsonDict]:
    return [
        {"scope": "fr11_independent_controller", "preserved_fact": "promoted"},
        {"scope": "fr11_shadow_adapter", "preserved_fact": "disabled_by_default"},
        {"scope": "arc_registry", "preserved_fact": "177_reproduced_levels_delta_0"},
        {"scope": "one_axis_temperature_exchange", "preserved_fact": "promoted"},
        {"scope": "generic_arc_models", "preserved_fact": "not_retired_by_exp5641"},
        {"scope": "generic_replica_exchange", "preserved_fact": "not_retired_by_exp5645"},
    ]


def _model_hardware_prerequisites() -> JsonDict:
    return {
        "exp5708_model": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "exp5708_runtime": "llama-cpp-python CUDA with authenticated GPU offload",
        "exp5714_portability": "Rust/PyO3 one-axis corrected cDLS parity only",
        "timing_claim_allowed": False,
        "hardware_speedup_claim_allowed": False,
    }


def _expected_deliverables() -> dict[str, str]:
    return {
        task_id: str(row["deliverable"])
        for task_id, row in _dependency_map().items()
        if isinstance(row, Mapping) and "deliverable" in row
    }


def _pre_existing_debt(validation_rows: Sequence[JsonMap]) -> list[JsonDict]:
    debt: list[JsonDict] = []
    for row in validation_rows:
        command = str(row.get("command", ""))
        exit_code = row.get("exit_code")
        if command == ".venv/bin/pytest tests/python -q" and exit_code not in (0, None):
            debt.append(
                {
                    "command": command,
                    "exit_code": exit_code,
                    "status": row.get("status", "failed"),
                    "classification": "pre_existing_broad_suite_debt",
                }
            )
    return debt


def run_transition(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    payloads, metadata, missing, malformed = _read_artifacts(root)
    manifest, manifest_meta = _read_manifest(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    registry, registry_meta = _read_manifest(root / ARC_REGISTRY_RELATIVE_PATH)
    capstone = _payload(payloads, EXP5647_CAPSTONE_PATH)
    arc_registry_count = int(capstone.get("arc_registry_count_after") or 0)
    registry_yaml_count = _registry_count_from_yaml(registry)
    arc_solve = capstone.get("arc_solve_provenance", {})
    if not isinstance(arc_solve, Mapping):
        arc_solve = {}
    fr11 = capstone.get("fr11_independent_promotion_status", {})
    shadow = capstone.get("fr11_shadow_integration_status", {})
    two_axis = capstone.get("two_axis_quality_status", {})
    rust = capstone.get("rust_parity_status", {})
    if not isinstance(fr11, Mapping):
        fr11 = {}
    if not isinstance(shadow, Mapping):
        shadow = {}
    if not isinstance(two_axis, Mapping):
        two_axis = {}
    if not isinstance(rust, Mapping):
        rust = {}

    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    test_commands = [str(row.get("command", "")) for row in validation_rows]
    test_exit_codes = {str(row.get("command", "")): row.get("exit_code") for row in validation_rows}
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    manifest_retirements, manifest_debt_after = _manifest_retirement_status(manifest)
    dependency_map = _dependency_map()
    gate_map = _gate_map()
    dependency_validation = _valid_dependency_ids(dependency_map, gate_map)

    blocked = bool(
        missing
        or malformed
        or manifest_debt_after
        or not manifest_meta.get("loadable")
        or not registry_meta.get("loadable")
        or roadmap_modified
        or conductor_modified
        or not dependency_validation["valid"]
    )
    honest_verdict = (
        "blocked: v510 transition incomplete because evidence, manifest, or protected-file checks failed"
        if blocked
        else (
            "complete: v510 transition archived .509 evidence; fr11_promoted=True; "
            "shadow_default_enabled=False; arc_registry_count=177; arc_registry_delta=0; "
            "one_axis_promoted=True; two_axis_quality_promoted=False; "
            "current_task_range=exp5706-exp5716"
        )
    )

    result: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "current_task_range": CURRENT_TASK_RANGE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifact_metadata": metadata,
        "missing_artifacts": missing,
        "malformed_artifacts": malformed,
        "manifest_metadata": manifest_meta,
        "arc_registry_metadata": registry_meta,
        "arc_registry_yaml_count": registry_yaml_count,
        "source_capstone_hash": path_sha256(root / EXP5647_CAPSTONE_PATH),
        "v509_task_verdicts": _v509_task_verdicts(payloads, metadata),
        "fr11_promoted": fr11.get("promoted") is True,
        "fr11_shadow_default_enabled": shadow.get("default_enabled") is True,
        "arc_registry_count": arc_registry_count,
        "arc_registry_delta": int(arc_solve.get("registry_delta") or 0),
        "one_axis_replica_exchange_promoted": capstone.get("one_axis_replica_exchange_preserved")
        is True,
        "two_axis_quality_promoted": two_axis.get("promoted") is True,
        "two_axis_rust_parity_gate_skipped": rust.get("gate_skipped") is True,
        "missing_retirements_before": _missing_retirements_before(capstone),
        "retirements_applied": manifest_retirements,
        "manifest_debt_after": manifest_debt_after,
        "outer_loop_snapshot": _outer_loop_snapshot(payloads, metadata),
        "dependency_map": dependency_map,
        "gate_map": gate_map,
        "dependency_id_validation": dependency_validation,
        "retired_scopes": _retired_scopes(),
        "preserved_scopes": _preserved_scopes(),
        "unconditional_arc_path": {
            "task_id": "exp5713-arc-live-levelup-attempt",
            "structured_gate_required": False,
            "registry_precheck_required": True,
            "uses_exp5712_if_promoted": True,
            "baseline_runs_if_not_promoted": True,
        },
        "model_hardware_prerequisites": _model_hardware_prerequisites(),
        "expected_deliverables": _expected_deliverables(),
        "protected_file_checks": {
            ROADMAP_RELATIVE_PATH.as_posix(): {"unchanged": not roadmap_modified},
            CONDUCTOR_RELATIVE_PATH.as_posix(): {"unchanged": not conductor_modified},
        },
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
        "validation_results": validation_rows,
        "pre_existing_broad_suite_debt": _pre_existing_debt(validation_rows),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    result["reproducibility_checksum"] = payload_checksum(result)
    return result


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    principles = payload.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(FIELD_PRINCIPLES)
        or any(
            principles.get(field) != principle for field, principle in FIELD_PRINCIPLES.items()
        )
    ):
        errors.append("field_principles")
    for field in (
        "v509_task_verdicts",
        "outer_loop_snapshot",
        "dependency_map",
        "gate_map",
        "test_exit_codes",
        "protected_file_checks",
        "dependency_id_validation",
    ):
        if not isinstance(payload.get(field), Mapping):
            errors.append(field)
    for field in (
        "missing_retirements_before",
        "retirements_applied",
        "retired_scopes",
        "preserved_scopes",
        "test_commands",
        "validation_results",
        "pre_existing_broad_suite_debt",
    ):
        if not isinstance(payload.get(field), list):
            errors.append(field)
    if isinstance(payload.get("v509_task_verdicts"), Mapping) and set(
        payload["v509_task_verdicts"]
    ) != set(V509_TASK_IDS):
        errors.append("v509_task_verdicts")
    if isinstance(payload.get("outer_loop_snapshot"), Mapping) and set(
        payload["outer_loop_snapshot"]
    ) != set(OUTER_LOOP_TASK_IDS):
        errors.append("outer_loop_snapshot")
    if payload.get("fr11_promoted") is not True:
        errors.append("fr11_promoted")
    if payload.get("fr11_shadow_default_enabled") is not False:
        errors.append("fr11_shadow_default_enabled")
    if payload.get("arc_registry_count") != 177:
        errors.append("arc_registry_count")
    if payload.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta")
    if payload.get("one_axis_replica_exchange_promoted") is not True:
        errors.append("one_axis_replica_exchange_promoted")
    if payload.get("two_axis_quality_promoted") is not False:
        errors.append("two_axis_quality_promoted")
    if payload.get("current_task_range") != CURRENT_TASK_RANGE:
        errors.append("current_task_range")
    if payload.get("timing_claimed") is not False:
        errors.append("timing_claimed")
    if payload.get("hardware_speedup_claimed") is not False:
        errors.append("hardware_speedup_claimed")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not payload.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum")
    if not _terminal_prefix_ok(str(payload.get("honest_verdict", ""))):
        errors.append("honest_verdict")
    retired_scopes = payload.get("retired_scopes")
    if isinstance(retired_scopes, list):
        retired_text = json.dumps(retired_scopes, sort_keys=True).lower()
        if "generic_arc_models" in retired_text and "preserves" not in retired_text:
            errors.append("retired_scopes")
        if "generic_replica_exchange" in retired_text and "preserves" not in retired_text:
            errors.append("retired_scopes")
        for row in retired_scopes:
            if isinstance(row, Mapping) and str(row.get("scope", "")).startswith("generic_"):
                errors.append("retired_scopes")
    dependency_map = payload.get("dependency_map")
    gate_map = payload.get("gate_map")
    if isinstance(dependency_map, Mapping) and isinstance(gate_map, Mapping):
        if not _valid_dependency_ids(dependency_map, gate_map)["valid"]:
            errors.append("dependency_map")
            errors.append("gate_map")
    return sorted(set(errors))


def write_transition(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    payload = run_transition(
        root=root,
        validation_results=validation_results,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(payload)
    if errors:
        raise ValueError(f"invalid Exp5706 transition artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def _load_validation_results(path: Path | None) -> Sequence[JsonMap]:
    if path is None:
        return DEFAULT_VALIDATION_RESULTS
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)

    validation_results = _load_validation_results(args.validation_results)
    payload = run_transition(root=args.root, validation_results=validation_results)
    errors = validate_artifact(payload)
    if errors:
        raise SystemExit(f"invalid Exp5706 transition artifact fields: {', '.join(errors)}")
    output = args.output or args.root / RESULT_RELATIVE_PATH
    write_json(output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
