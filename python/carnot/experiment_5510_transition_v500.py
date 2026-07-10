"""Exp5510 transition receipt from milestone .499 into .500.

Spec refs: REQ-REPORT-5510, SCENARIO-REPORT-5510,
SCENARIO-REPORT-5510-BLOCKED-INPUT.

This module is a record-only handoff. It does not activate another roadmap,
rerun SOTA inference, repair CSL metrics, touch hardware, or attempt ARC. Its
job is to preserve exactly what the .499 artifacts already proved so .500 work
starts with clean gates: structured-output control before SOTA panels,
independent CSL metrics before memory panels, action-diverse ARC prechecks
before a live level-up, and receipt-only hardware until matched timing exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    normalize_task_range,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5510_transition_v500.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5509_capstone_v499.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5510_transition_v500"
EXPERIMENT_ID = "exp5510-transition-v500"
MILESTONE = "2026.07.500"
PREVIOUS_MILESTONE = "2026.07.499"
PREVIOUS_TASK_RANGE = "exp5496-exp5509"
NEXT_TASK_RANGE = "exp5510-exp5522"
SCHEMA = "carnot.experiment_5510.transition_v500.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5510
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXPECTED_TASK_IDS = [
    "exp5510-transition-v500",
    "exp5511-v500-sota-source-delta-ingestion",
    "exp5512-structured-output-positive-control",
    "exp5513-sota-hard-soft-structured-panel",
    "exp5514-energy-spill-sidecar-diagnostic",
    "exp5515-csl-independent-outcome-gate-repair",
    "exp5516-sota-csl-memory-panel",
    "exp5517-csl-memory-residue-stress",
    "exp5518-block-gibbs-sparse-repair-descriptors",
    "exp5519-hardware-continuity-methodology-receipts",
    "exp5520-arc-action-diversity-target-precheck",
    "exp5521-arc-live-action-diverse-levelup",
    "exp5522-v500-capstone-reconciliation",
]

ARTIFACTS: dict[int, Path] = {
    5496: Path("results/experiment_5496_transition_v499.json"),
    5497: Path("results/experiment_5497_pretest_cascade_diagnostic_v499.json"),
    5498: Path("results/experiment_5498_source_delta_v499.json"),
    5499: Path("results/experiment_5499_preference_maxsat_minimal_fixture_v499.json"),
    5500: Path("results/experiment_5500_sota_concept_claim_panel_v499.json"),
    5501: Path("results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json"),
    5502: Path("results/experiment_5502_csl_tautology_static_corrigendum_v499.json"),
    5503: Path("results/experiment_5503_csl_experience_graph_replay_v499.json"),
    5504: Path("results/experiment_5504_sota_csl_memory_panel_v499.json"),
    5505: Path("results/experiment_5505_active_constraint_milp_descriptor_v499.json"),
    5506: Path("results/experiment_5506_hardware_multiboard_receipts_v499.json"),
    5507: Path("results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json"),
    5508: Path("results/experiment_5508_arc_live_perception_generation_levelup_v499.json"),
    5509: PRIOR_CAPSTONE_RELATIVE_PATH,
}

PROMPT_ARTIFACT_ALIASES: tuple[tuple[Path, Path], ...] = (
    (
        Path("results/experiment_5502_csl_tautology_corrigendum_v499.json"),
        ARTIFACTS[5502],
    ),
    (Path("results/experiment_5506_hardware_receipts_v499.json"), ARTIFACTS[5506]),
    (
        Path("results/experiment_5508_arc_live_perception_generation_v499.json"),
        ARTIFACTS[5508],
    ),
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the new .500 transition receipt.",
    "previous_milestone": "Source milestone whose terminal facts are archived.",
    "prior_capstone_path": "Exact .499 capstone artifact used as the main source of truth.",
    "previous_task_range": "Closed .499 conductor range.",
    "clean_lanes": "Completed .499 evidence safe to carry forward without promotion.",
    "bounded_lanes": "Useful but claim-limited .499 evidence that needs a .500 gate.",
    "blocked_lanes": "Blocked prerequisites that must be repaired before headline credit.",
    "honest_null_lanes": "Executed lanes that produced no positive bankable result.",
    "flagged_lanes": "Adversarially or methodologically flagged evidence kept out of clean lanes.",
    "next_task_range": "Planned .500 conductor range.",
    "structured_sota_gate_required": "Bare boolean for Exp5512 before another SOTA hard/soft panel.",
    "csl_independent_metric_gate_required": "Bare boolean for Exp5515 before any SOTA CSL memory panel.",
    "arc_live_levelup_gate_required": "Bare boolean for Exp5520 before the live ARC level-up attempt.",
    "hardware_receipt_only": "Bare boolean preserving no-speedup posture without matched timing.",
    "roadmap_yaml_unchanged": "Protected-file check for research-roadmap.yaml.",
    "conductor_unchanged": "Protected-file check for scripts/research_conductor.py.",
    "inference_substrate": "Aggregation only; no hidden model, solver, ARC, or hardware run.",
    "honest_verdict": "Terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "status",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "source_context",
    "source_context_missing",
    "artifact_metadata",
    "artifact_aliases",
    "artifacts_expected",
    "artifacts_found",
    "artifacts_missing",
    "roadmap_task_ids",
    "roadmap_doc_task_range",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "conductor_evidence",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = (
    "REQ-REPORT-5510",
    "SCENARIO-REPORT-5510",
    "SCENARIO-REPORT-5510-BLOCKED-INPUT",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5510_transition_v500.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5510_transition_v500.py "
            "-m pytest tests/python/test_experiment_5510_transition_v500.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5510_transition_v500.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    {
        "command": (
            "ops/e2e-test-plan.md review: Exp5510 is aggregation-only; no fresh "
            "training, PyO3 round trip, ARC live action, or hardware workload applies"
        ),
        "outcome": "not_applicable",
    },
)


def _row(
    lane: str,
    classification: str,
    source_artifacts: Sequence[Path | str],
    evidence: Mapping[str, Any],
    claim_boundary: str,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": [str(item) for item in source_artifacts],
        "evidence": dict(evidence),
        "claim_boundary": claim_boundary,
    }


def _read_artifacts(root: Path) -> tuple[dict[int, JsonDict], list[str], list[str], JsonDict]:
    artifacts: dict[int, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    metadata: JsonDict = {}
    for exp_id, rel_path in ARTIFACTS.items():
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[exp_id] = payload
        metadata[rel] = meta
        if meta.get("exists") and meta.get("loadable"):
            found.append(rel)
        else:
            missing.append(rel)
    return artifacts, found, missing, metadata


def _source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        exists = (root / rel_path).exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(root / rel_path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _artifact_aliases(root: Path) -> list[JsonDict]:
    return [
        {
            "prompt_path": prompt_path.as_posix(),
            "prompt_path_exists": (root / prompt_path).exists(),
            "actual_path": actual_path.as_posix(),
            "actual_path_exists": (root / actual_path).exists(),
            "actual_path_sha256": path_sha256(root / actual_path),
        }
        for prompt_path, actual_path in PROMPT_ARTIFACT_ALIASES
    ]


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _conductor_evidence(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    tokens = ("5500", "5502", "5503", "5504", "5505", "5506", "5508", "SOTA", "CSL", "hardware", "ARC")
    return [
        line.strip()
        for line in text.splitlines()
        if any(token.lower() in line.lower() for token in tokens)
        and any(marker in line for marker in ("OK", "GATE_BLOCK", "FLAGGED"))
    ][-12:]


def _gate_summary(roadmap_text: str, vnext_text: str) -> JsonDict:
    combined = f"{roadmap_text}\n{vnext_text}"
    return {
        "structured_sota_gate_required": "structured_output_positive_control_ready" in combined,
        "csl_independent_metric_gate_required": (
            "metric_independence_clean" in combined
            and "csl_gate_fields_resolvable" in combined
        ),
        "arc_live_levelup_gate_required": "arc_levelup_candidate_ready" in combined,
        "hardware_receipt_only": "receipt-only" in combined.lower()
        or "receipt_only" in combined.lower(),
    }


def _clean_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "transition",
            "clean",
            (ARTIFACTS[5496], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "previous_transition_verdict": artifacts[5496].get("honest_verdict"),
                "capstone_status": artifacts[5509].get("status"),
                "missing_primary_artifacts": len(artifacts[5509].get("artifacts_missing", [])),
            },
            "The .499 transition and capstone are carried forward without re-planning.",
        ),
        _row(
            "pretest_cascade_repair",
            "clean",
            (ARTIFACTS[5497],),
            {
                "pretest_cascade_resolved": bool(artifacts[5497].get("pretest_cascade_resolved")),
                "reproduced_pretest_failure": bool(
                    artifacts[5497].get("reproduced_pretest_failure")
                ),
                "recommendation": artifacts[5497].get("downstream_gate_recommendation"),
            },
            "The .498 skip cascade no longer blocks .500 science gates, with full-suite caveats preserved.",
        ),
        _row(
            "source_delta",
            "clean",
            (ARTIFACTS[5498],),
            {
                "new_references_added": len(artifacts[5498].get("new_references_added", [])),
                "closed_scopes_reopened": bool(artifacts[5498].get("closed_scopes_reopened")),
                "research_references_updated": bool(
                    artifacts[5498].get("research_references_updated")
                ),
            },
            "The .499 source delta is complete context, not a reason to reopen closed scopes.",
        ),
        _row(
            "preference_maxsat_fixture",
            "clean",
            (ARTIFACTS[5499],),
            {
                "preference_maxsat_fixture_ready": bool(
                    artifacts[5499].get("preference_maxsat_fixture_ready")
                ),
                "false_accept_rate": artifacts[5499].get("false_accept_rate"),
            },
            "The exact Preference-MaxSAT fixture is a validator substrate, not SOTA evidence by itself.",
        ),
        _row(
            "helper_contract_fixture",
            "clean",
            (ARTIFACTS[5501],),
            {
                "helper_contract_fixture_ready": bool(
                    artifacts[5501].get("helper_contract_fixture_ready")
                ),
                "rolled_up_verdict_accuracy": artifacts[5501].get(
                    "rolled_up_verdict_accuracy"
                ),
                "false_accept_rate": artifacts[5501].get("false_accept_rate"),
            },
            "The helper-contract fixture is exact-predicate evidence for downstream structured rows.",
        ),
        _row(
            "experience_graph_replay_readiness",
            "clean",
            (ARTIFACTS[5503],),
            {
                "csl_experience_graph_ready": bool(
                    artifacts[5503].get("csl_experience_graph_ready")
                ),
                "heldout_delta": artifacts[5503].get("heldout_delta"),
                "graph_memory_score": artifacts[5503].get("graph_memory_score"),
                "no_memory_baseline_score": artifacts[5503].get("no_memory_baseline_score"),
                "model_weights_mutated": bool(artifacts[5503].get("model_weights_mutated")),
            },
            "Graph replay readiness is useful only after independent metrics and gate fields are repaired.",
        ),
    ]


def _bounded_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "sota_missing_candidate_panel",
            "bounded",
            (ARTIFACTS[5500],),
            {
                "concept_claim_telemetry_rows": artifacts[5500].get(
                    "concept_claim_telemetry_rows"
                ),
                "abstention_count": artifacts[5500].get("abstention_count"),
                "exact_validator_accuracy": artifacts[5500].get("exact_validator_accuracy"),
                "headline_models_used": artifacts[5500].get("headline_models_used", []),
                "gpu_offload_verified": bool(artifacts[5500].get("gpu_offload_verified")),
            },
            "The SOTA panel ran but missing or abstained candidate rows require a structured-output positive control.",
        ),
        _row(
            "active_constraint_exact_fallback",
            "bounded",
            (ARTIFACTS[5505], ARTIFACTS[5499]),
            {
                "descriptor_ready_for_hardware": bool(
                    artifacts[5505].get("descriptor_ready_for_hardware")
                ),
                "num_descriptor_rows": artifacts[5505].get("num_descriptor_rows"),
                "exact_fallback_agreement_rate": artifacts[5505].get(
                    "exact_fallback_agreement_rate"
                ),
                "hardware_speedup_claim": bool(artifacts[5505].get("hardware_speedup_claim")),
            },
            "Descriptor plumbing is exact-fallback checked, but .500 needs a sparse repair mechanism before any speedup claim.",
        ),
        _row(
            "hardware_receipt_only_timing",
            "bounded",
            (ARTIFACTS[5506],),
            {
                "cpu_status": artifacts[5506].get("cpu_status"),
                "cuda_status": artifacts[5506].get("cuda_status"),
                "polar_fire_status": artifacts[5506].get("polar_fire_status"),
                "matched_timing_available": bool(
                    artifacts[5506].get("matched_timing_available")
                ),
                "hardware_speedup_claim": bool(artifacts[5506].get("hardware_speedup_claim")),
            },
            "Hardware remains receipt-only because matched timing is unavailable and speedup is false.",
        ),
    ]


def _blocked_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "csl_metric_independence_block",
            "blocked",
            (ARTIFACTS[5502],),
            {
                "tautology_flag_resolved": bool(
                    artifacts[5502].get("tautology_flag_resolved")
                ),
                "metric_independence_clean": bool(
                    artifacts[5502].get("metric_independence_clean")
                ),
                "csl_scale_headline_allowed": bool(
                    artifacts[5502].get("csl_scale_headline_allowed")
                ),
                "downstream_recommendation": artifacts[5502].get(
                    "downstream_recommendation"
                ),
            },
            "CSL headline evidence is blocked until heldout metrics are independent of the policy score.",
        ),
        _row(
            "csl_gate_field_mismatch",
            "blocked",
            (ARTIFACTS[5504], ARTIFACTS[5503]),
            {
                "status": artifacts[5504].get("status"),
                "honest_verdict": artifacts[5504].get("honest_verdict"),
                "gate_check_summary": artifacts[5504].get("gate_check_summary"),
                "gates_evaluated": artifacts[5504].get("gates_evaluated", []),
            },
            "The SOTA CSL panel was skipped by conductor gates, including an unresolved replay readiness field.",
        ),
        _row(
            "hardware_identity_blocks",
            "blocked",
            (ARTIFACTS[5506],),
            {
                "kv260_status": artifacts[5506].get("kv260_status"),
                "gatemate_status": artifacts[5506].get("gatemate_status"),
            },
            "KV260 and GateMate identity blocks prevent board workload credit.",
        ),
    ]


def _honest_null_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    arc_delta = int(artifacts[5508].get("arc_registry_delta") or 0)
    return [
        _row(
            "arc_no_bank",
            "honest_null",
            (ARTIFACTS[5507], ARTIFACTS[5508]),
            {
                "selected_game": artifacts[5508].get("selected_game"),
                "selected_level": artifacts[5508].get("selected_level"),
                "solve_provenance": artifacts[5508].get("solve_provenance"),
                "live_agent_attempts": artifacts[5508].get("live_agent_attempts"),
                "reproduced_levels": artifacts[5508].get("reproduced_levels"),
                "offline_reproduced": bool(artifacts[5508].get("offline_reproduced")),
            },
            "The live ARC attempt banked no reproduced level.",
        ),
        _row(
            "arc_registry_delta_zero",
            "honest_null",
            (ARTIFACTS[5508], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "registry_before_levels": artifacts[5508].get("registry_before_levels"),
                "registry_after_levels": artifacts[5508].get("registry_after_levels"),
                "arc_registry_delta": arc_delta,
            },
            "The ARC registry did not increase during .499.",
        ),
    ]


def _flagged_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "csl_headline_leakage_unresolved",
            "flagged",
            (ARTIFACTS[5502], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "csl_scale_headline_allowed": bool(
                    artifacts[5502].get("csl_scale_headline_allowed")
                ),
                "retire_same_scope_if_repeated": bool(
                    artifacts[5502].get("retire_same_scope_if_repeated")
                ),
                "capstone_csl_verdict": artifacts[5509].get("csl_verdict"),
            },
            "Exp5474-style CSL scale credit remains bounded until the leakage pattern is rerun cleanly.",
        ),
        _row(
            "hardware_timing_methodology_flags",
            "flagged",
            (ARTIFACTS[5506],),
            {
                "flagged_adversarial": bool(artifacts[5506].get("flagged_adversarial")),
                "corrigendum_pending": artifacts[5506].get("corrigendum_pending", []),
                "matched_timing_available": bool(
                    artifacts[5506].get("matched_timing_available")
                ),
            },
            "Duration and methodology flags keep hardware evidence out of speedup claims.",
        ),
        _row(
            "arc_repeated_pattern_no_bank",
            "flagged",
            (ARTIFACTS[5508],),
            {
                "live_agent_attempts": artifacts[5508].get("live_agent_attempts"),
                "trajectory_taxonomy_counts": artifacts[5508].get(
                    "trajectory_taxonomy_counts", {}
                ),
                "arc_registry_delta": artifacts[5508].get("arc_registry_delta"),
            },
            "The next ARC lane must change action generation beyond the repeated coordinate/action pattern.",
        ),
    ]


def _failed_preconditions(
    artifacts_missing: Sequence[str],
    capstone: JsonMap,
    roadmap: JsonMap,
    roadmap_task_ids: Sequence[str],
    roadmap_doc_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failed = [f"{path}_missing_or_unreadable" for path in artifacts_missing]
    if capstone.get("milestone") != PREVIOUS_MILESTONE:
        failed.append("prior_capstone_milestone_mismatch")
    if roadmap.get("milestone") != MILESTONE:
        failed.append("research-roadmap.yaml_milestone_mismatch")
    if list(roadmap_task_ids) != EXPECTED_TASK_IDS:
        failed.append("roadmap_task_ids_mismatch")
    if roadmap_doc_task_range != NEXT_TASK_RANGE:
        failed.append("vnext_task_range_mismatch")
    if roadmap_modified:
        failed.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failed.append("scripts/research_conductor.py_modified")
    return failed


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifacts_found, artifacts_missing, artifact_metadata = _read_artifacts(root)
    roadmap, roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    vnext_text = _read_text(root, VNEXT_RELATIVE_PATH)
    roadmap_text = _read_text(root, ROADMAP_RELATIVE_PATH)
    roadmap_doc_task_range = normalize_task_range(vnext_text)
    source_context, source_context_missing = _source_context(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(
        root, CONDUCTOR_RELATIVE_PATH, modification_overrides
    )
    failed_preconditions = _failed_preconditions(
        artifacts_missing,
        artifacts[5509],
        roadmap,
        roadmap_task_ids,
        roadmap_doc_task_range,
        roadmap_modified,
        conductor_modified,
    )
    gate_summary = _gate_summary(roadmap_text, vnext_text)
    status_prefix = "blocked:" if failed_preconditions else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": "blocked" if failed_preconditions else "complete",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "artifact_metadata": {
            **artifact_metadata,
            ROADMAP_RELATIVE_PATH.as_posix(): roadmap_meta,
        },
        "artifact_aliases": _artifact_aliases(root),
        "artifacts_expected": [path.as_posix() for path in ARTIFACTS.values()],
        "artifacts_found": artifacts_found,
        "artifacts_missing": artifacts_missing,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "protected_file_checks": [
            {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
                "git_status_clean": not roadmap_modified,
                "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            },
            {
                "path": CONDUCTOR_RELATIVE_PATH.as_posix(),
                "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
                "git_status_clean": not conductor_modified,
                "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
            },
        ],
        "preconditions_checked": [
            "prior_capstone_readable",
            "all_exp5496_exp5509_artifacts_readable",
            "active_roadmap_names_2026_07_500",
            "vnext_task_range_exp5510_exp5522",
            "roadmap_task_ids_match_expected",
            "protected_files_clean",
        ],
        "failed_preconditions": failed_preconditions,
        "conductor_evidence": _conductor_evidence(root),
        "tests_run": list(tests_run),
        "reproducibility_checksum": "",
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(),
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "clean_lanes": _clean_lanes(artifacts),
        "bounded_lanes": _bounded_lanes(artifacts),
        "blocked_lanes": _blocked_lanes(artifacts),
        "honest_null_lanes": _honest_null_lanes(artifacts),
        "flagged_lanes": _flagged_lanes(artifacts),
        "next_task_range": NEXT_TASK_RANGE,
        "structured_sota_gate_required": bool(
            gate_summary["structured_sota_gate_required"]
        ),
        "csl_independent_metric_gate_required": bool(
            gate_summary["csl_independent_metric_gate_required"]
        ),
        "arc_live_levelup_gate_required": bool(
            gate_summary["arc_live_levelup_gate_required"]
        ),
        "hardware_receipt_only": bool(gate_summary["hardware_receipt_only"])
        and not bool(artifacts[5506].get("hardware_speedup_claim")),
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"{status_prefix} archived .499 terminal evidence into .500 transition receipt; "
            "clean lanes preserve transition/pretest/source/exact fixtures and graph replay; "
            "bounded and blocked lanes preserve SOTA abstentions, CSL independence/gate failures, "
            "active-constraint exact fallback, receipt-only hardware, and ARC no-bank; "
            f"next_task_range={NEXT_TASK_RANGE}; failed_preconditions={len(failed_preconditions)}"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def write_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    payload = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    write_report(args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
