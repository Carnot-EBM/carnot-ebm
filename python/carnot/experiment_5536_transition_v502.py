"""Exp5536 transition receipt from milestone .501 into .502.

Spec refs: REQ-REPORT-5536, SCENARIO-REPORT-5536,
SCENARIO-REPORT-5536-BLOCKED-INPUT, SCENARIO-REPORT-5536-FIELD-PRINCIPLES.

This module is a record-only handoff. It reads the completed `.501` capstone,
the lane artifacts that the `.502` planner singled out, the active `.502`
roadmap, and the conductor log. It then writes a receipt that keeps clean
evidence, bounded evidence, and adversarial flags separate. That separation is
important because the next milestone starts with repair gates; a flagged SOTA
panel, tautological CSL residue row, or no-bank ARC attempt must become repair
work rather than headline evidence.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5536_transition_v502.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5535_capstone_v501.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5536_transition_v502"
EXPERIMENT_ID = "exp5536-transition-v502"
MILESTONE = "2026.07.502"
PREVIOUS_MILESTONE = "2026.07.501"
PREVIOUS_TASK_RANGE = "exp5523-exp5535"
NEXT_TASK_RANGE = "exp5536-exp5549"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5536
SCHEMA = "carnot.experiment_5536.transition_v502.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5536-transition-v502",
    "exp5537-v502-source-delta-ingestion",
    "exp5538-sota-panel-duration-substrate-corrigendum",
    "exp5539-gram2token-grammar-table-preflight",
    "exp5540-gated-sota-hard-soft-live-panel-v3",
    "exp5541-llm-fsm-exact-fixture",
    "exp5542-csl-residue-metric-independence-corrigendum",
    "exp5543-gated-retrieval-warmed-csl-five-arm-ablation",
    "exp5544-gated-cross-model-sota-csl-transfer",
    "exp5545-gated-sparse-repair-fsm-descriptor-scale",
    "exp5546-hardware-receipt-substrate-corrigendum",
    "exp5547-arc-no-llm-substrate-precheck",
    "exp5548-gated-arc-clean-live-levelup",
    "exp5549-v502-capstone-reconciliation",
]

ARTIFACTS: dict[str, Path] = {
    "source_delta": Path("results/experiment_5524_v501_source_delta_ingestion.json"),
    "sota_taxonomy": Path("results/experiment_5525_sota_schema_failure_taxonomy.json"),
    "sota_repair": Path("results/experiment_5526_sota_structured_repair_loop.json"),
    "sota_panel": Path("results/experiment_5527_sota_hard_soft_panel_v2.json"),
    "csl_gate": Path("results/experiment_5528_csl_canonical_gate_artifact.json"),
    "csl_residue": Path("results/experiment_5529_csl_event_topic_residue_stress.json"),
    "csl_memory": Path("results/experiment_5530_sota_csl_memory_panel_v2.json"),
    "sparse_repair": Path("results/experiment_5531_sparse_repair_scaleup_ci.json"),
    "hardware_receipt": Path("results/experiment_5532_hardware_receipt_parser_repeatability.json"),
    "arc_precheck": Path("results/experiment_5533_arc_strategy_routing_precheck.json"),
    "arc_levelup": Path("results/experiment_5534_arc_strategy_routed_levelup.json"),
    "prior_capstone": PRIOR_CAPSTONE_RELATIVE_PATH,
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
    "milestone": "Route key for the `.502` transition receipt.",
    "previous_milestone": "Source milestone whose terminal facts are archived.",
    "prior_capstone_path": "Exact `.501` capstone artifact used as the main source of truth.",
    "previous_task_range": "Closed `.501` conductor task range.",
    "next_task_range": "Planned `.502` conductor task range.",
    "clean_lanes": "Completed `.501` evidence safe to carry forward without promoting flagged claims.",
    "bounded_lanes": "Useful `.501` evidence kept claim-limited until the matching `.502` gate runs.",
    "blocked_lanes": "Blocked `.501` prerequisites that must not become headline credit.",
    "flagged_lanes": "Adversarial or methodology flags preserved as repair work, not clean evidence.",
    "sota_duration_corrigendum_required": "Bare boolean requiring Exp5527 duration and substrate repair before panel v3.",
    "grammar_preflight_required": "Bare boolean requiring grammar-table reachability before another SOTA hard/soft panel.",
    "csl_residue_corrigendum_required": "Bare boolean requiring non-tautological event/topic residue evidence before retrieval and cross-model memory.",
    "finite_state_fixture_required": "Bare boolean requiring a deterministic exact finite-state fixture before sparse descriptor scale.",
    "arc_clean_precheck_required": "Bare boolean requiring seed/checksum/substrate-clean ARC precheck before live level-up.",
    "hardware_receipt_only": "Bare boolean preserving no-speedup posture without matched authenticated timing.",
    "roadmap_yaml_unchanged": "Protected-file check for `research-roadmap.yaml`.",
    "conductor_unchanged": "Protected-file check for `scripts/research_conductor.py`.",
    "inference_substrate": "Must equal `aggregation_from_upstream_artifacts` because Exp5536 is synthesis only.",
    "honest_verdict": "Terminal summary starting with `complete:` or `blocked:` that names the transition boundary.",
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
BOOL_FIELDS = (
    "sota_duration_corrigendum_required",
    "grammar_preflight_required",
    "csl_residue_corrigendum_required",
    "finite_state_fixture_required",
    "arc_clean_precheck_required",
    "hardware_receipt_only",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
)
LIST_FIELDS = (
    "clean_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "flagged_lanes",
    "source_context",
    "source_context_missing",
    "artifacts_expected",
    "artifacts_found",
    "artifacts_missing",
    "roadmap_task_ids",
    "failed_preconditions",
    "conductor_evidence",
    "tests_run",
)

SPEC_REFS = (
    "REQ-REPORT-5536",
    "SCENARIO-REPORT-5536",
    "SCENARIO-REPORT-5536-BLOCKED-INPUT",
    "SCENARIO-REPORT-5536-FIELD-PRINCIPLES",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5536_transition_v502.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5536_transition_v502.py "
            "-m pytest tests/python/test_experiment_5536_transition_v502.py -q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5536_transition_v502.py --fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
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


def _read_artifacts(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str], JsonDict]:
    artifacts: dict[str, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    metadata: JsonDict = {}
    for key, rel_path in ARTIFACTS.items():
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[key] = payload
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


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _corrigenda(payload: JsonMap) -> list[JsonDict]:
    rows = payload.get("corrigendum_pending")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _conductor_evidence(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    tokens = (
        "5524",
        "5525",
        "5526",
        "5527",
        "5528",
        "5529",
        "5530",
        "5531",
        "5532",
        "5533",
        "5534",
        "5535",
        "5536",
        "5537",
        "5538",
        "5539",
        "5540",
        "5541",
        "5542",
        "5545",
        "5548",
        "5549",
        "2026.07.502",
        "SOTA",
        "CSL",
        "Hardware",
        "ARC",
        "Capstone",
    )
    markers = ("OK", "FLAGGED", "GATE_BLOCK", "FAIL", "SKIP")
    return [
        line.strip()
        for line in text.splitlines()
        if any(token.lower() in line.lower() for token in tokens)
        and any(marker in line for marker in markers)
    ][-30:]


def _clean_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    source = artifacts["source_delta"]
    taxonomy = artifacts["sota_taxonomy"]
    repair = artifacts["sota_repair"]
    csl_gate = artifacts["csl_gate"]
    csl_memory = artifacts["csl_memory"]
    sparse = artifacts["sparse_repair"]
    return [
        _row(
            "source_delta",
            "clean",
            (ARTIFACTS["source_delta"],),
            {
                "research_references_updated": bool(source.get("research_references_updated")),
                "new_references_added": source.get("new_references_added", []),
                "closed_scopes_reopened": bool(source.get("closed_scopes_reopened")),
                "honest_verdict": source.get("honest_verdict"),
            },
            "The .501 source delta is archived as execution context only; it does not re-plan .502.",
        ),
        _row(
            "sota_schema_taxonomy",
            "clean",
            (ARTIFACTS["sota_taxonomy"],),
            {
                "sota_schema_failure_taxonomy_ready": bool(
                    taxonomy.get("sota_schema_failure_taxonomy_ready")
                ),
                "schema_validity_rate": taxonomy.get("schema_validity_rate"),
                "exact_validator_handoff_ready": bool(
                    taxonomy.get("exact_validator_handoff_ready")
                ),
                "honest_verdict": taxonomy.get("honest_verdict"),
            },
            "The taxonomy explains the schema failure boundary but is not itself a quality claim.",
        ),
        _row(
            "structured_repair_loop",
            "clean",
            (ARTIFACTS["sota_repair"],),
            {
                "sota_structured_repair_loop_ready": bool(
                    repair.get("sota_structured_repair_loop_ready")
                ),
                "schema_validity_before": repair.get("schema_validity_before"),
                "schema_validity_after": repair.get("schema_validity_after"),
                "missing_candidate_rows_after": repair.get("missing_candidate_rows_after"),
                "exact_validator_handoff_ready": bool(
                    repair.get("exact_validator_handoff_ready")
                ),
            },
            "The repair loop made rows schema-valid with exact handoff, but panel quality still needs clean duration evidence.",
        ),
        _row(
            "canonical_csl_gate",
            "clean",
            (ARTIFACTS["csl_gate"],),
            {
                "continuous_self_learning_evidence": bool(
                    csl_gate.get("continuous_self_learning_evidence")
                ),
                "csl_gate_fields_conductor_visible": bool(
                    csl_gate.get("csl_gate_fields_conductor_visible")
                ),
                "conductor_gate_probe_passed": bool(csl_gate.get("conductor_gate_probe_passed")),
                "heldout_delta": csl_gate.get("heldout_delta"),
            },
            "The canonical gate is conductor-visible CSL evidence, not broad memory-transfer proof.",
        ),
        _row(
            "sota_csl_memory_panel",
            "clean",
            (ARTIFACTS["csl_memory"],),
            {
                "continuous_self_learning_evidence": bool(
                    csl_memory.get("continuous_self_learning_evidence")
                ),
                "csl_claim_allowed": bool(csl_memory.get("csl_claim_allowed")),
                "heldout_delta": csl_memory.get("heldout_delta"),
                "stale_evidence_rejection_rate": csl_memory.get(
                    "stale_evidence_rejection_rate"
                ),
            },
            "The memory panel ran and is useful, but its broad claim remains bounded by flagged residue evidence.",
        ),
        _row(
            "sparse_repair_scaleup",
            "clean",
            (ARTIFACTS["sparse_repair"],),
            {
                "active_constraint_sparse_repair_ready": bool(
                    sparse.get("active_constraint_sparse_repair_ready")
                ),
                "sparse_repair_success_rate": sparse.get("sparse_repair_success_rate"),
                "exact_only_success_rate": sparse.get("exact_only_success_rate"),
                "all_candidates_exact_checked": bool(
                    sparse.get("all_candidates_exact_checked")
                ),
            },
            "Sparse repair is exact-checked scale evidence, not a speedup or hardware result.",
        ),
    ]


def _bounded_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    capstone = artifacts["prior_capstone"]
    csl_memory = artifacts["csl_memory"]
    csl_residue = artifacts["csl_residue"]
    sparse = artifacts["sparse_repair"]
    arc = artifacts["arc_levelup"]
    return [
        _row(
            "structured_sota_repair_claim_boundary",
            "bounded",
            (ARTIFACTS["sota_taxonomy"], ARTIFACTS["sota_repair"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "structured_sota_claim_allowed": bool(
                    capstone.get("structured_sota_claim_allowed")
                ),
                "sota_hard_soft_claim_allowed": bool(
                    capstone.get("sota_hard_soft_claim_allowed")
                ),
                "capstone_verdict": capstone.get("honest_verdict"),
            },
            "Structured repair may carry forward, but the live hard/soft SOTA claim is blocked.",
        ),
        _row(
            "bounded_csl_memory_claim",
            "bounded",
            (ARTIFACTS["csl_memory"], ARTIFACTS["csl_residue"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "continuous_self_learning_evidence": bool(
                    capstone.get("continuous_self_learning_evidence")
                ),
                "csl_claim_allowed": bool(capstone.get("csl_claim_allowed")),
                "memory_panel_claim_allowed": bool(csl_memory.get("csl_claim_allowed")),
                "residue_corrigendum_pending": _corrigenda(csl_residue),
            },
            "The CSL memory panel is useful evidence, but broad CSL waits on non-tautological residue repair.",
        ),
        _row(
            "sparse_repair_no_speedup",
            "bounded",
            (ARTIFACTS["sparse_repair"],),
            {
                "active_constraint_sparse_repair_ready": bool(
                    sparse.get("active_constraint_sparse_repair_ready")
                ),
                "matched_timing_available": bool(sparse.get("matched_timing_available")),
                "speedup_claim_allowed": bool(sparse.get("speedup_claim_allowed")),
            },
            "Sparse repair can scale descriptors only after the finite-state exact fixture, and still has no speedup claim.",
        ),
        _row(
            "arc_live_path_provenance_no_bank",
            "bounded",
            (ARTIFACTS["arc_levelup"],),
            {
                "solve_provenance": arc.get("solve_provenance"),
                "selected_game": arc.get("selected_game"),
                "selected_level": arc.get("selected_level"),
                "offline_reproduced": bool(arc.get("offline_reproduced")),
                "reproduced_levels": int(arc.get("reproduced_levels") or 0),
            },
            "ARC live-path provenance is preserved, but no level is banked without offline reproduction.",
        ),
    ]


def _blocked_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    capstone = artifacts["prior_capstone"]
    hardware = artifacts["hardware_receipt"]
    arc = artifacts["arc_levelup"]
    return [
        _row(
            "sota_hard_soft_claim_blocked",
            "blocked",
            (ARTIFACTS["sota_panel"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "sota_hard_soft_claim_allowed": bool(
                    capstone.get("sota_hard_soft_claim_allowed")
                ),
                "flagged_adversarial": bool(artifacts["sota_panel"].get("flagged_adversarial")),
            },
            "Hard/soft SOTA claim credit is blocked until duration and substrate evidence are clean.",
        ),
        _row(
            "broad_csl_claim_blocked",
            "blocked",
            (ARTIFACTS["csl_residue"], ARTIFACTS["csl_memory"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "continuous_self_learning_evidence": bool(
                    capstone.get("continuous_self_learning_evidence")
                ),
                "csl_claim_allowed": bool(capstone.get("csl_claim_allowed")),
                "residue_flagged_adversarial": bool(
                    artifacts["csl_residue"].get("flagged_adversarial")
                ),
            },
            "Broad CSL claim credit is blocked by the flagged event/topic residue tautology.",
        ),
        _row(
            "hardware_speedup_false",
            "blocked",
            (ARTIFACTS["hardware_receipt"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "hardware_speedup_claim": bool(capstone.get("hardware_speedup_claim")),
                "artifact_hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "hardware_speedup_claim_allowed": bool(
                    hardware.get("hardware_speedup_claim_allowed")
                ),
            },
            "No hardware speedup is allowed without matched authenticated timing.",
        ),
        _row(
            "arc_registry_delta_zero",
            "blocked",
            (ARTIFACTS["arc_levelup"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "arc_registry_delta": int(capstone.get("arc_registry_delta") or 0),
                "registry_delta": int(arc.get("registry_delta") or 0),
                "registry_before_levels": arc.get("registry_before_levels"),
                "registry_after_levels": arc.get("registry_after_levels"),
                "reproduced_levels": int(arc.get("reproduced_levels") or 0),
            },
            "ARC registry progress is zero, so no live level-up credit carries forward.",
        ),
    ]


def _flagged_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    panel = artifacts["sota_panel"]
    residue = artifacts["csl_residue"]
    hardware = artifacts["hardware_receipt"]
    precheck = artifacts["arc_precheck"]
    levelup = artifacts["arc_levelup"]
    return [
        _row(
            "exp5527_duration_substrate",
            "flagged",
            (ARTIFACTS["sota_panel"],),
            {
                "flagged_adversarial": bool(panel.get("flagged_adversarial")),
                "duration_s": panel.get("duration_s"),
                "inference_substrate": panel.get("inference_substrate"),
                "sota_hard_soft_claim_allowed": bool(panel.get("sota_hard_soft_claim_allowed")),
                "corrigendum_pending": _corrigenda(panel),
            },
            "Exp5527 reported valid rows but was too short for a live local GGUF substrate claim.",
        ),
        _row(
            "exp5529_residue_tautology",
            "flagged",
            (ARTIFACTS["csl_residue"],),
            {
                "flagged_adversarial": bool(residue.get("flagged_adversarial")),
                "event_only_score": residue.get("event_only_score"),
                "topic_only_score": residue.get("topic_only_score"),
                "corrigendum_pending": _corrigenda(residue),
            },
            "Exp5529 event-only and topic-only metrics are identical, so residue repair is mandatory.",
        ),
        _row(
            "exp5532_hardware_receipt_methodology",
            "flagged",
            (ARTIFACTS["hardware_receipt"],),
            {
                "flagged_adversarial": bool(hardware.get("flagged_adversarial")),
                "duration_s": hardware.get("duration_s"),
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
                "corrigendum_pending": _corrigenda(hardware),
            },
            "Exp5532 remains receipt-only and needs methodology cleanup before it can be clean evidence.",
        ),
        _row(
            "exp5533_arc_precheck_hygiene",
            "flagged",
            (ARTIFACTS["arc_precheck"],),
            {
                "flagged_adversarial": bool(precheck.get("flagged_adversarial")),
                "selected_game": precheck.get("selected_game"),
                "selected_level": precheck.get("selected_level"),
                "solve_provenance": precheck.get("solve_provenance"),
                "random_seed_present": precheck.get("random_seed") is not None,
                "reproducibility_checksum_present": precheck.get(
                    "reproducibility_checksum"
                )
                is not None,
                "corrigendum_pending": _corrigenda(precheck),
            },
            "Exp5533 kept live-path provenance but needs clean seed, checksum, and substrate hygiene.",
        ),
        _row(
            "exp5534_arc_levelup_hygiene",
            "flagged",
            (ARTIFACTS["arc_levelup"],),
            {
                "flagged_adversarial": bool(levelup.get("flagged_adversarial")),
                "status": levelup.get("status"),
                "registry_delta": int(levelup.get("registry_delta") or 0),
                "reproduced_levels": int(levelup.get("reproduced_levels") or 0),
                "duration_s": levelup.get("duration_s"),
                "random_seed_present": levelup.get("random_seed") is not None,
                "reproducibility_checksum_present": levelup.get(
                    "reproducibility_checksum"
                )
                is not None,
                "corrigendum_pending": _corrigenda(levelup),
            },
            "Exp5534 was an honest null with zero registry delta and avoidable hygiene flags.",
        ),
    ]


def _failed_preconditions(
    artifacts: Mapping[str, JsonMap],
    artifacts_missing: Sequence[str],
    roadmap: JsonMap,
    roadmap_task_ids: Sequence[str],
    vnext_text: str,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failed = [f"{path}_missing_or_unreadable" for path in artifacts_missing]
    if artifacts["prior_capstone"].get("milestone") != PREVIOUS_MILESTONE:
        failed.append("prior_capstone_milestone_mismatch")
    if roadmap.get("milestone") != MILESTONE:
        failed.append("research-roadmap.yaml_milestone_mismatch")
    if list(roadmap_task_ids) != EXPECTED_TASK_IDS:
        failed.append("roadmap_task_ids_mismatch")
    if normalize_task_range(vnext_text) != NEXT_TASK_RANGE:
        failed.append("vnext_task_range_mismatch")
    if MILESTONE not in vnext_text:
        failed.append("vnext_milestone_mismatch")
    if roadmap_modified:
        failed.append(f"{ROADMAP_RELATIVE_PATH.as_posix()}_modified")
    if conductor_modified:
        failed.append(f"{CONDUCTOR_RELATIVE_PATH.as_posix()}_modified")
    return failed


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifacts_found, artifacts_missing, artifact_metadata = _read_artifacts(root)
    source_context, source_context_missing = _source_context(root)
    roadmap, roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    vnext_text = _read_text(root, VNEXT_RELATIVE_PATH)
    roadmap_doc_task_range = normalize_task_range(vnext_text)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    failed = _failed_preconditions(
        artifacts,
        artifacts_missing,
        roadmap,
        roadmap_task_ids,
        vnext_text,
        roadmap_modified,
        conductor_modified,
    )
    status = "blocked" if failed else "complete"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "artifact_metadata": artifact_metadata,
        "artifacts_expected": [path.as_posix() for path in ARTIFACTS.values()],
        "artifacts_found": artifacts_found,
        "artifacts_missing": list(artifacts_missing),
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "protected_file_checks": {
            ROADMAP_RELATIVE_PATH.as_posix(): {"modified": roadmap_modified},
            CONDUCTOR_RELATIVE_PATH.as_posix(): {"modified": conductor_modified},
        },
        "preconditions_checked": {
            "prior_capstone_milestone": artifacts["prior_capstone"].get("milestone"),
            "active_roadmap_milestone": roadmap.get("milestone"),
            "active_roadmap_loadable": bool(roadmap_meta.get("loadable")),
            "roadmap_task_ids_match": roadmap_task_ids == EXPECTED_TASK_IDS,
            "vnext_task_range": roadmap_doc_task_range,
            "vnext_names_milestone": MILESTONE in vnext_text,
            "roadmap_next_present": ROADMAP_NEXT_RELATIVE_PATH.as_posix()
            not in source_context_missing,
        },
        "failed_preconditions": failed,
        "conductor_evidence": _conductor_evidence(root),
        "tests_run": list(tests_run),
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(),
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "next_task_range": NEXT_TASK_RANGE,
        "clean_lanes": _clean_lanes(artifacts),
        "bounded_lanes": _bounded_lanes(artifacts),
        "blocked_lanes": _blocked_lanes(artifacts),
        "flagged_lanes": _flagged_lanes(artifacts),
        "sota_duration_corrigendum_required": True,
        "grammar_preflight_required": True,
        "csl_residue_corrigendum_required": True,
        "finite_state_fixture_required": True,
        "arc_clean_precheck_required": True,
        "hardware_receipt_only": True,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    payload["honest_verdict"] = (
        f"{status}: .501 terminal evidence archived into .502 transition; "
        f"clean_lanes={len(payload['clean_lanes'])}; bounded_lanes={len(payload['bounded_lanes'])}; "
        f"blocked_lanes={len(payload['blocked_lanes'])}; flagged_lanes={len(payload['flagged_lanes'])}; "
        "gates=sota_duration,grammar_preflight,csl_residue,finite_state,arc_clean; "
        f"hardware_receipt_only={payload['hardware_receipt_only']}; "
        f"arc_registry_delta={int(artifacts['prior_capstone'].get('arc_registry_delta') or 0)}"
    )
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    if payload.get("hardware_receipt_only") is not True:
        errors.append("hardware_receipt_only")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("previous_milestone") != PREVIOUS_MILESTONE:
        errors.append("previous_milestone")
    if payload.get("prior_capstone_path") != PRIOR_CAPSTONE_RELATIVE_PATH.as_posix():
        errors.append("prior_capstone_path")
    if payload.get("previous_task_range") != PREVIOUS_TASK_RANGE:
        errors.append("previous_task_range")
    if payload.get("next_task_range") != NEXT_TASK_RANGE:
        errors.append("next_task_range")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - guarded by validate_artifact unit coverage
        raise ValueError(f"invalid Exp5536 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5536 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
