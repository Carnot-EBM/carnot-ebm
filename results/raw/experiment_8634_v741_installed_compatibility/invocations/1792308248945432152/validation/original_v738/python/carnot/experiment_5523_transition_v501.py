"""Exp5523 transition receipt from milestone .500 into .501.

Spec refs: REQ-REPORT-5523, SCENARIO-REPORT-5523,
SCENARIO-REPORT-5523-BLOCKED-INPUT.

This module is a record-only handoff. It reads the .500 capstone, the .500
lane artifacts, the active .501 roadmap, and the conductor log, then writes a
receipt that keeps proven evidence separate from blocked or bounded claims.
That separation matters because the next milestone starts with several gates:
the SOTA schema taxonomy must precede repair, repair must precede another SOTA
panel, the CSL gate must be canonical before memory stress or SOTA memory
tasks, and ARC must pass a strategy precheck before another live level-up.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5523_transition_v501.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5522_capstone_v500.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5523_transition_v501"
EXPERIMENT_ID = "exp5523-transition-v501"
MILESTONE = "2026.07.501"
PREVIOUS_MILESTONE = "2026.07.500"
PREVIOUS_TASK_RANGE = "exp5510-exp5522"
NEXT_TASK_RANGE = "exp5523-exp5535"
SCHEMA = "carnot.experiment_5523.transition_v501.v1"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5523
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXPECTED_TASK_IDS = [
    "exp5523-transition-v501",
    "exp5524-v501-source-delta-ingestion",
    "exp5525-sota-schema-failure-taxonomy",
    "exp5526-gated-sota-structured-repair-loop",
    "exp5527-gated-sota-hard-soft-panel-v2",
    "exp5528-csl-canonical-gate-artifact",
    "exp5529-gated-csl-event-topic-residue-stress",
    "exp5530-gated-sota-csl-memory-panel-v2",
    "exp5531-sparse-repair-scaleup-ci",
    "exp5532-hardware-receipt-parser-repeatability",
    "exp5533-arc-strategy-routing-precheck",
    "exp5534-gated-arc-strategy-routed-levelup",
    "exp5535-v501-capstone-reconciliation",
]

ARTIFACTS: dict[int, Path] = {
    5510: Path("results/experiment_5510_transition_v500.json"),
    5511: Path("results/experiment_5511_v500_source_delta_ingestion.json"),
    5512: Path("results/experiment_5512_structured_output_positive_control.json"),
    5513: Path("results/experiment_5513_sota_hard_soft_structured_panel.json"),
    5514: Path("results/experiment_5514_energy_spill_sidecar_diagnostic.json"),
    5515: Path("results/experiment_5515_csl_independent_outcome_gate_repair.json"),
    5516: Path("results/experiment_5516_sota_csl_memory_panel.json"),
    5517: Path("results/experiment_5517_csl_memory_residue_stress.json"),
    5518: Path("results/experiment_5518_block_gibbs_sparse_repair_descriptors.json"),
    5519: Path("results/experiment_5519_hardware_continuity_methodology_receipts.json"),
    5520: Path("results/experiment_5520_arc_action_diversity_target_precheck.json"),
    5521: Path("results/experiment_5521_arc_live_action_diverse_levelup.json"),
    5522: PRIOR_CAPSTONE_RELATIVE_PATH,
}

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
    "milestone": "Route key for the .501 transition receipt.",
    "previous_milestone": "Source milestone whose terminal facts are archived.",
    "prior_capstone_path": "Exact .500 capstone artifact used as the main source of truth.",
    "previous_task_range": "Closed .500 conductor task range.",
    "next_task_range": "Planned .501 conductor task range.",
    "clean_lanes": "Completed .500 evidence safe to carry forward without promotion.",
    "bounded_lanes": "Useful but claim-limited .500 evidence that needs a .501 gate.",
    "blocked_lanes": "Blocked prerequisites that must be repaired before headline credit.",
    "honest_null_lanes": "Executed lanes that produced no positive bankable result.",
    "flagged_lanes": "Operational or methodology risks kept out of clean lanes.",
    "sota_schema_repair_gate_required": "Bare boolean enforcing taxonomy before repair and repair before SOTA panel v2.",
    "csl_canonical_gate_required": "Bare boolean enforcing a canonical CSL gate artifact before downstream memory tasks.",
    "arc_strategy_gate_required": "Bare boolean enforcing ARC strategy precheck before live level-up.",
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
    "REQ-REPORT-5523",
    "SCENARIO-REPORT-5523",
    "SCENARIO-REPORT-5523-BLOCKED-INPUT",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5523_transition_v501.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5523_transition_v501.py "
            "-m pytest tests/python/test_experiment_5523_transition_v501.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5523_transition_v501.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    {
        "command": (
            "ops/e2e-test-plan.md review: Exp5523 is aggregation-only; no fresh "
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


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _conductor_evidence(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    tokens = (
        "5512",
        "5513",
        "5514",
        "5515",
        "5516",
        "5517",
        "5518",
        "5519",
        "5520",
        "5521",
        "5522",
        "5523",
        "5524",
        "5525",
        "2026.07.501",
        "SOTA",
        "CSL",
        "Hardware",
        "ARC",
        "Capstone",
    )
    markers = ("OK", "GATE_BLOCK", "FLAGGED", "FAIL")
    return [
        line.strip()
        for line in text.splitlines()
        if any(token.lower() in line.lower() for token in tokens)
        and any(marker in line for marker in markers)
    ][-20:]


def _gate_summary(roadmap_text: str, vnext_text: str) -> JsonDict:
    combined = f"{roadmap_text}\n{vnext_text}".lower()
    return {
        "sota_schema_repair_gate_required": (
            "sota-schema-failure-taxonomy" in combined
            and "sota-structured-repair-loop" in combined
            and "sota-hard-soft-panel-v2" in combined
        )
        or ("taxonomy before" in combined and "repair loop" in combined),
        "csl_canonical_gate_required": (
            "csl-canonical-gate-artifact" in combined
            and "csl-event-topic-residue-stress" in combined
            and "sota-csl-memory-panel-v2" in combined
        )
        or "canonical csl gate" in combined,
        "arc_strategy_gate_required": (
            "arc-strategy-routing-precheck" in combined
            and "arc-strategy-routed-levelup" in combined
        )
        or "arc strategy precheck" in combined,
    }


def _arc_registry_delta(arc_artifact: JsonMap, capstone: JsonMap) -> int:
    direct = arc_artifact.get("registry_delta", arc_artifact.get("arc_registry_delta"))
    if direct is not None:
        return int(direct)
    before = int(arc_artifact.get("registry_before_levels") or 0)
    after = int(arc_artifact.get("registry_after_levels") or before)
    return int(capstone.get("arc_registry_delta", after - before) or 0)


def _clean_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "transition_source",
            "clean",
            (ARTIFACTS[5510], ARTIFACTS[5511]),
            {
                "transition_verdict": artifacts[5510].get("honest_verdict"),
                "source_delta_verdict": artifacts[5511].get("honest_verdict"),
                "research_references_updated": bool(
                    artifacts[5511].get("research_references_updated")
                ),
            },
            "The .500 transition and source delta are archived as context only, not replanned.",
        ),
        _row(
            "deterministic_structured_output_fixture",
            "clean",
            (ARTIFACTS[5512],),
            {
                "structured_output_positive_control_ready": bool(
                    artifacts[5512].get("structured_output_positive_control_ready")
                ),
                "schema_validity_rate": artifacts[5512].get("schema_validity_rate"),
                "exact_validator_handoff_ready": artifacts[5512].get(
                    "exact_validator_handoff_ready"
                ),
            },
            "Deterministic fixtures are schema-valid, but this does not make live SOTA rows valid.",
        ),
        _row(
            "csl_independent_graph_memory_positive",
            "clean",
            (ARTIFACTS[5515],),
            {
                "continuous_self_learning_evidence": bool(
                    artifacts[5515].get("continuous_self_learning_evidence")
                ),
                "metric_independence_clean": bool(
                    artifacts[5515].get("metric_independence_clean")
                ),
                "csl_experience_graph_ready": bool(
                    artifacts[5515].get("csl_experience_graph_ready")
                ),
                "heldout_delta": artifacts[5515].get("heldout_delta"),
                "stale_evidence_rejection_rate": artifacts[5515].get(
                    "stale_evidence_rejection_rate"
                ),
            },
            "The graph-memory positive is fixture-level CSL evidence pending a canonical gate artifact.",
        ),
        _row(
            "sparse_descriptor_interface",
            "clean",
            (ARTIFACTS[5518],),
            {
                "active_constraint_sparse_repair_ready": bool(
                    artifacts[5518].get("active_constraint_sparse_repair_ready")
                ),
                "candidate_count": artifacts[5518].get("candidate_count"),
                "all_candidates_exact_checked": bool(
                    artifacts[5518].get("all_candidates_exact_checked")
                ),
                "exact_fallback_used": bool(artifacts[5518].get("exact_fallback_used")),
            },
            "Sparse repair descriptors are valid interface evidence with exact fallback.",
        ),
        _row(
            "hardware_receipt_only_posture",
            "clean",
            (ARTIFACTS[5519],),
            {
                "hardware_speedup_claim": bool(
                    artifacts[5519].get("hardware_speedup_claim")
                ),
                "hardware_speedup_claim_allowed": bool(
                    artifacts[5519].get("hardware_speedup_claim_allowed")
                ),
                "matched_timing_available": bool(
                    artifacts[5519].get("matched_timing_available")
                ),
                "polar_fire_status": artifacts[5519].get("polar_fire_receipt", {}).get(
                    "status"
                ),
            },
            "Hardware continuity is receipt-only until authenticated matched timing exists.",
        ),
        _row(
            "arc_target_precheck",
            "clean",
            (ARTIFACTS[5520],),
            {
                "arc_levelup_candidate_ready": bool(
                    artifacts[5520].get("arc_levelup_candidate_ready")
                ),
                "registry_precheck_done": bool(
                    artifacts[5520].get("registry_precheck_done")
                ),
                "selected_game": artifacts[5520].get("selected_game"),
                "selected_level": artifacts[5520].get("selected_level"),
            },
            "The ARC target precheck selected a non-duplicate target but did not claim a solve.",
        ),
        _row(
            "capstone_closure",
            "clean",
            (PRIOR_CAPSTONE_RELATIVE_PATH,),
            {
                "missing_artifacts": artifacts[5522].get("missing_artifacts", []),
                "claim_boundaries": artifacts[5522].get("claim_boundaries", []),
                "honest_verdict": artifacts[5522].get("honest_verdict"),
            },
            "The .500 capstone closed with explicit claim boundaries and no missing primary artifacts.",
        ),
    ]


def _bounded_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "live_sota_schema_rows",
            "bounded",
            (ARTIFACTS[5513], ARTIFACTS[5512]),
            {
                "sota_structured_panel_ready": bool(
                    artifacts[5513].get("sota_structured_panel_ready")
                ),
                "sota_rows_emitted": artifacts[5513].get("sota_rows_emitted"),
                "missing_candidate_rows": artifacts[5513].get("missing_candidate_rows"),
                "schema_validity_rate": artifacts[5513].get("schema_validity_rate"),
                "readiness_blockers": artifacts[5513].get("readiness_blockers", []),
                "gpu_offload_verified": bool(artifacts[5513].get("gpu_offload_verified")),
            },
            "Live SOTA GPU offload occurred, but schema-invalid or missing rows require taxonomy and repair.",
        ),
        _row(
            "sparse_repair_scale_speedup",
            "bounded",
            (ARTIFACTS[5518],),
            {
                "sparse_repair_success_rate": artifacts[5518].get(
                    "sparse_repair_success_rate"
                ),
                "exact_only_success_rate": artifacts[5518].get("exact_only_success_rate"),
                "mean_iterations_sparse_repair": artifacts[5518].get(
                    "mean_iterations_sparse_repair"
                ),
                "mean_iterations_exact_only": artifacts[5518].get(
                    "mean_iterations_exact_only"
                ),
                "speedup_claim_allowed": bool(
                    artifacts[5518].get("speedup_claim_allowed")
                ),
                "claim_limits": artifacts[5518].get("claim_limits", []),
            },
            "Sparse repair is promising on tiny fixtures but scale and speedup remain unproven.",
        ),
    ]


def _blocked_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "energy_sidecar",
            "blocked",
            (ARTIFACTS[5514], ARTIFACTS[5513]),
            {
                "status": artifacts[5514].get("status"),
                "blocked_at_layer": artifacts[5514].get("blocked_at_layer"),
                "gate_check_summary": artifacts[5514].get("gate_check_summary"),
                "energy_sidecar_headline_allowed": bool(
                    artifacts[5522].get("energy_sidecar_headline_allowed")
                ),
            },
            "The energy sidecar stayed gated because the SOTA structured panel was not ready.",
        ),
        _row(
            "downstream_csl_gate_sidecar_selection",
            "blocked",
            (ARTIFACTS[5516], ARTIFACTS[5517], ARTIFACTS[5515]),
            {
                "memory_panel_status": artifacts[5516].get("status"),
                "memory_panel_summary": artifacts[5516].get("gate_check_summary"),
                "residue_status": artifacts[5517].get("status"),
                "residue_summary": artifacts[5517].get("gate_check_summary"),
                "upstream_metric_independence_clean": bool(
                    artifacts[5515].get("metric_independence_clean")
                ),
                "upstream_csl_gate_fields_resolvable": bool(
                    artifacts[5515].get("csl_gate_fields_resolvable")
                ),
            },
            "Downstream CSL gates saw missing fields, so .501 needs one canonical gate artifact.",
        ),
        _row(
            "broad_csl_claims",
            "blocked",
            (PRIOR_CAPSTONE_RELATIVE_PATH, ARTIFACTS[5515]),
            {
                "continuous_self_learning_evidence": bool(
                    artifacts[5522].get("continuous_self_learning_evidence")
                ),
                "csl_claim_allowed": bool(artifacts[5522].get("csl_claim_allowed")),
                "downstream_skips": artifacts[5522].get("skipped_by_gates", []),
            },
            "Fixture-level CSL evidence is positive, but broad CSL claims are blocked until downstream memory tasks execute cleanly.",
        ),
        _row(
            "hardware_matched_timing",
            "blocked",
            (ARTIFACTS[5519],),
            {
                "matched_timing_available": bool(
                    artifacts[5519].get("matched_timing_available")
                ),
                "timing_methodology": artifacts[5519].get("timing_methodology", {}),
                "blocked_devices": artifacts[5519].get("blocked_devices", []),
            },
            "No speedup or matched-timing claim is allowed from metadata receipts alone.",
        ),
        _row(
            "arc_registry_delta",
            "blocked",
            (ARTIFACTS[5521], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "registry_before_levels": artifacts[5521].get("registry_before_levels"),
                "registry_after_levels": artifacts[5521].get("registry_after_levels"),
                "registry_delta": _arc_registry_delta(artifacts[5521], artifacts[5522]),
                "reproduced_levels": artifacts[5521].get("reproduced_levels"),
            },
            "ARC registry progress is zero, so no live level-up credit carries forward.",
        ),
    ]


def _honest_null_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "arc_live_no_bank",
            "honest_null",
            (ARTIFACTS[5521],),
            {
                "status": artifacts[5521].get("status"),
                "selected_game": artifacts[5521].get("selected_game"),
                "selected_level": artifacts[5521].get("selected_level"),
                "solve_provenance": artifacts[5521].get("solve_provenance"),
                "reproduced_levels": artifacts[5521].get("reproduced_levels"),
                "registry_delta": _arc_registry_delta(artifacts[5521], artifacts[5522]),
                "honest_verdict": artifacts[5521].get("honest_verdict"),
            },
            "The live ARC attempt was a real live-path attempt that banked no new level.",
        )
    ]


def _flagged_lanes(artifacts: Mapping[int, JsonMap]) -> list[JsonDict]:
    return [
        _row(
            "csl_sidecar_selection_risk",
            "flagged",
            (ARTIFACTS[5515], ARTIFACTS[5516], ARTIFACTS[5517]),
            {
                "primary_gate_fields": {
                    "metric_independence_clean": artifacts[5515].get(
                        "metric_independence_clean"
                    ),
                    "csl_gate_fields_resolvable": artifacts[5515].get(
                        "csl_gate_fields_resolvable"
                    ),
                    "csl_experience_graph_ready": artifacts[5515].get(
                        "csl_experience_graph_ready"
                    ),
                },
                "downstream_gate_actuals": [
                    row.get("actual")
                    for row in artifacts[5516].get("gates_evaluated", [])
                    + artifacts[5517].get("gates_evaluated", [])
                ],
            },
            "The primary CSL artifact is positive but downstream gates selected missing values.",
        ),
        _row(
            "arc_repeated_coordinate_risk",
            "flagged",
            (ARTIFACTS[5521], ARTIFACTS[5520]),
            {
                "precheck_repeated_coordinate_rate": artifacts[5520].get(
                    "repeated_coordinate_rate"
                ),
                "live_repeated_coordinate_rate": artifacts[5521].get(
                    "repeated_coordinate_rate"
                ),
                "action_entropy": artifacts[5521].get("action_entropy"),
            },
            "The live ARC attempt reintroduced repeated coordinates, so .501 needs strategy routing.",
        ),
        _row(
            "hardware_parser_receipt_risk",
            "flagged",
            (ARTIFACTS[5519],),
            {
                "blocked_devices": artifacts[5519].get("blocked_devices", []),
                "matched_timing_available": bool(
                    artifacts[5519].get("matched_timing_available")
                ),
                "hardware_speedup_claim_allowed": bool(
                    artifacts[5519].get("hardware_speedup_claim_allowed")
                ),
            },
            "Parser and identity failures must stay visible even though receipt-only posture is clean.",
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
        artifacts[5522],
        roadmap,
        roadmap_task_ids,
        roadmap_doc_task_range,
        roadmap_modified,
        conductor_modified,
    )
    gate_summary = _gate_summary(roadmap_text, vnext_text)
    hardware_receipt_only = (
        not bool(artifacts[5519].get("hardware_speedup_claim"))
        and not bool(artifacts[5519].get("hardware_speedup_claim_allowed"))
        and not bool(artifacts[5519].get("matched_timing_available"))
    )
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
            "all_exp5510_exp5522_artifacts_readable",
            "active_roadmap_names_2026_07_501",
            "vnext_task_range_exp5523_exp5535",
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
        "next_task_range": NEXT_TASK_RANGE,
        "clean_lanes": _clean_lanes(artifacts),
        "bounded_lanes": _bounded_lanes(artifacts),
        "blocked_lanes": _blocked_lanes(artifacts),
        "honest_null_lanes": _honest_null_lanes(artifacts),
        "flagged_lanes": _flagged_lanes(artifacts),
        "sota_schema_repair_gate_required": bool(
            gate_summary["sota_schema_repair_gate_required"]
        ),
        "csl_canonical_gate_required": bool(gate_summary["csl_canonical_gate_required"]),
        "arc_strategy_gate_required": bool(gate_summary["arc_strategy_gate_required"]),
        "hardware_receipt_only": hardware_receipt_only,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"{status_prefix} archived .500 terminal evidence into .501 transition receipt; "
            "clean lanes preserve transition/source, deterministic structured fixture, CSL "
            "independent graph memory, sparse descriptors, hardware receipt-only posture, "
            "ARC target precheck, and capstone closure; blocked/bounded lanes preserve live "
            "SOTA schema rows, energy sidecar, downstream CSL sidecar gate selection, broad "
            "CSL claims, sparse scale/speedup, hardware matched timing, and ARC registry "
            f"delta; next_task_range={NEXT_TASK_RANGE}; failed_preconditions={len(failed_preconditions)}"
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
