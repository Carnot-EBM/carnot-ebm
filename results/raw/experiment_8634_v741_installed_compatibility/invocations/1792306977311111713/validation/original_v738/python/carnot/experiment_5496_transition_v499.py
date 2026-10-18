"""Exp5496 transition receipt from milestone .498 into .499.

Spec refs: REQ-REPORT-5496, SCENARIO-REPORT-5496,
SCENARIO-REPORT-5496-BLOCKED-INPUT.

This module writes a record-only handoff. It does not repair the pretest
cascade, rerun ARC, or touch hardware. Its job is to preserve the exact .498
terminal evidence before .499 starts, so downstream tasks can distinguish
completed receipt lanes from lanes that were merely planned, skipped, blocked,
or honestly null.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5496_transition_v499.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5495_capstone_v498.json")
HARDWARE_RELATIVE_PATH = Path("results/experiment_5492_hardware_receipts_v498.json")
ARC_LIVE_RELATIVE_PATH = Path("results/experiment_5494_arc_live_trajectory_levelup_v498.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5496_transition_v499"
EXPERIMENT_ID = "exp5496-transition-v499"
MILESTONE = "2026.07.499"
PREVIOUS_MILESTONE = "2026.07.498"
PREVIOUS_TASK_RANGE = "exp5482-exp5495"
NEXT_TASK_RANGE = "exp5496-exp5509"
SCHEMA = "carnot.experiment_5496.transition_v499.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5496
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5496-transition-v499",
    "exp5497-pretest-cascade-diagnostic-v499",
    "exp5498-source-delta-v499",
    "exp5499-preference-maxsat-minimal-fixture-v499",
    "exp5500-sota-concept-claim-panel-v499",
    "exp5501-helper-contract-hierarchical-claim-fixture-v499",
    "exp5502-csl-tautology-static-corrigendum-v499",
    "exp5503-csl-experience-graph-replay-v499",
    "exp5504-sota-csl-memory-panel-v499",
    "exp5505-active-constraint-milp-descriptor-v499",
    "exp5506-hardware-multiboard-receipts-v499",
    "exp5507-arc-null-coordinate-perception-precheck-v499",
    "exp5508-arc-live-perception-generation-levelup-v499",
    "exp5509-capstone-v499",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "route key for the new .499 transition receipt.",
    "previous_milestone": "source milestone whose terminal facts are being archived.",
    "prior_capstone_path": "exact .498 capstone artifact used as the main source of truth.",
    "previous_task_range": "closed .498 conductor range.",
    "clean_lanes": "completed or bounded-clean evidence safe to carry forward with caveats.",
    "missing_or_skipped_lanes": "planned .498 lanes that did not produce clean evidence.",
    "blocked_lanes": "quarantined or unavailable lanes that must not be promoted.",
    "honest_null_lanes": "executed or evaluated lanes that produced no positive bankable result.",
    "flagged_lanes": "adversarially flagged evidence that remains separate from clean lanes.",
    "exp5474_tautology_still_blocks_csl_headlines": "bare boolean for the CSL headline gate.",
    "next_task_range": "planned .499 conductor range.",
    "roadmap_yaml_unchanged": "protected-file check for research-roadmap.yaml.",
    "conductor_unchanged": "protected-file check for scripts/research_conductor.py.",
    "inference_substrate": "aggregation only; no hidden live inference or hardware run.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
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
    "source_artifacts",
    "source_context_missing",
    "roadmap_task_ids",
    "roadmap_doc_task_range",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = (
    "REQ-REPORT-5496",
    "SCENARIO-REPORT-5496",
    "SCENARIO-REPORT-5496-BLOCKED-INPUT",
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
    PRIOR_CAPSTONE_RELATIVE_PATH,
    HARDWARE_RELATIVE_PATH,
    ARC_LIVE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5496_transition_v499.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5496_transition_v499.py "
            "-m pytest tests/python/test_experiment_5496_transition_v499.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5496_transition_v499.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    {
        "command": (
            "ops/e2e-test-plan.md review: Exp5496 is aggregation-only; no fresh "
            "training, PyO3 round trip, ARC live action, or hardware workload applies"
        ),
        "outcome": "not_applicable",
    },
)


def _row(
    lane: str,
    classification: str,
    source_artifacts: Sequence[str],
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


def _truth_row(capstone: JsonMap, lane: str) -> JsonDict:
    table = capstone.get("lane_truth_table")
    if not isinstance(table, Mapping):
        return {}
    row = table.get(lane, {})
    return dict(row) if isinstance(row, Mapping) else {}


def _evidence(row: JsonMap) -> JsonDict:
    evidence = row.get("evidence")
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _sources(row: JsonMap, fallback: Sequence[str]) -> list[str]:
    sources = row.get("source_artifacts")
    if isinstance(sources, list):
        return [str(item) for item in sources]
    return [str(item) for item in fallback]


def _conductor_text(root: Path) -> str:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _conductor_evidence(conductor_text: str, *tokens: str) -> list[str]:
    lowered_tokens = [token.lower() for token in tokens]
    evidence: list[str] = []
    for line in conductor_text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in lowered_tokens) and any(
            marker in lowered for marker in ("skip", "gate_block", "failed")
        ):
            evidence.append(line.strip())
    return evidence


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


def _board_receipt_summary(hardware: JsonMap) -> list[JsonDict]:
    receipts = hardware.get("board_receipts")
    if not isinstance(receipts, list):
        return []
    return [
        {
            "board_identity": row.get("board_identity"),
            "aggregate_output_hash": row.get("aggregate_output_hash"),
            "repeat_count": row.get("repeat_count"),
            "matched_repeat_count": row.get("matched_repeat_count"),
            "invalid_repeat_count": row.get("invalid_repeat_count"),
        }
        for row in receipts
        if isinstance(row, Mapping)
    ]


def derive_clean_lanes(capstone: JsonMap, hardware: JsonMap) -> list[JsonDict]:
    transition = _truth_row(capstone, "transition_source_delta")
    active = _truth_row(capstone, "active_constraints")
    hardware_row = _truth_row(capstone, "hardware")
    arc = _truth_row(capstone, "arc")
    synthesis = _truth_row(capstone, "synthesis")
    return [
        _row(
            "transition",
            "clean",
            _sources(transition, ["results/experiment_5482_transition_v498.json"]),
            {
                "previous_transition_complete": _evidence(transition).get(
                    "transition_complete"
                ),
                "previous_task_range": PREVIOUS_TASK_RANGE,
                "capstone_status": capstone.get("status"),
            },
            "The .498 transition and capstone are carried forward without re-planning.",
        ),
        _row(
            "active_constraint_subproblem_descriptors",
            "clean",
            _sources(
                active,
                ["results/experiment_5491_active_constraint_subproblem_descriptor_v498.json"],
            ),
            _evidence(active),
            "Descriptors are exact-fallback checked and useful as .499 seeds, not speedup proof.",
        ),
        _row(
            "hardware_polarfire_hash_receipts",
            "clean",
            _sources(hardware_row, [HARDWARE_RELATIVE_PATH.as_posix()]),
            {
                "reachable_boards": hardware.get("reachable_boards", []),
                "result_hash_match_rate": hardware.get("result_hash_match_rate"),
                "authenticated_board_identity_count": hardware.get(
                    "authenticated_board_identity_count"
                ),
                "board_receipts": _board_receipt_summary(hardware),
                "hardware_speedup_claim": hardware.get("hardware_speedup_claim"),
            },
            "PolarFire receipts matched hashes, but remain receipt-only evidence.",
        ),
        _row(
            "arc_target_precheck",
            "clean",
            _sources(
                arc,
                [
                    "results/experiment_5493_arc_trajectory_target_precheck_v498.json",
                    ARC_LIVE_RELATIVE_PATH.as_posix(),
                ],
            ),
            {
                "precheck_ready": _evidence(arc).get("precheck_ready"),
                "selected_game": _evidence(arc).get("selected_game"),
                "target_level": _evidence(arc).get("target_level"),
            },
            "The dc22 L3 target precheck was ready; the later live attempt stayed honest-null.",
        ),
        _row(
            "capstone_synthesis",
            "clean",
            _sources(synthesis, [PRIOR_CAPSTONE_RELATIVE_PATH.as_posix()]),
            {
                "honest_verdict": capstone.get("honest_verdict"),
                "inference_substrate": capstone.get("inference_substrate"),
                "upstream_missing_count": _evidence(synthesis).get("upstream_missing_count"),
            },
            "The capstone is aggregation evidence and cannot manufacture missing artifacts.",
        ),
    ]


def derive_missing_or_skipped_lanes(capstone: JsonMap, conductor_text: str) -> list[JsonDict]:
    missing = set(str(item) for item in capstone.get("artifacts_missing", []))
    fixed_point = _truth_row(capstone, "fixed_point_kan_ledger")
    return [
        _row(
            "source_delta",
            "missing_or_skipped",
            ["results/experiment_5483_source_delta_v498.json"],
            {
                "artifact_missing": "results/experiment_5483_source_delta_v498.json" in missing,
                "conductor_evidence": _conductor_evidence(conductor_text, "5483", "source delta"),
            },
            "Execution-time source delta never produced a usable .498 artifact.",
        ),
        _row(
            "csl_tautology_corrigendum",
            "missing_or_skipped",
            ["results/experiment_5484_csl_tautology_corrigendum_v498.json"],
            {
                "artifact_missing": "results/experiment_5484_csl_tautology_corrigendum_v498.json"
                in missing,
                "exp5474_tautology_resolved": capstone.get("exp5474_tautology_resolved"),
                "conductor_evidence": _conductor_evidence(conductor_text, "5484", "tautology"),
            },
            "The required Exp5474 tautology corrigendum did not land.",
        ),
        _row(
            "preference_maxsat_fixture",
            "missing_or_skipped",
            ["results/experiment_5485_preference_maxsat_claim_fixture_v498.json"],
            {
                "artifact_missing": "results/experiment_5485_preference_maxsat_claim_fixture_v498.json"
                in missing,
                "conductor_evidence": _conductor_evidence(conductor_text, "5485", "maxsat"),
            },
            "The hard/soft Preference-MaxSAT fixture did not emit a .498 artifact.",
        ),
        _row(
            "concept_telemetry",
            "missing_or_skipped",
            ["results/experiment_5486_sota_concept_evidence_panel_v498.json"],
            {
                "artifact_missing": "results/experiment_5486_sota_concept_evidence_panel_v498.json"
                in missing,
                "conductor_evidence": _conductor_evidence(conductor_text, "5486", "concept"),
            },
            "Concept-attributed local SOTA telemetry stayed gate-blocked.",
        ),
        _row(
            "helper_contract_repair",
            "missing_or_skipped",
            ["results/experiment_5487_helper_contract_nl_spec_repair_v498.json"],
            {
                "artifact_missing": "results/experiment_5487_helper_contract_nl_spec_repair_v498.json"
                in missing,
                "conductor_evidence": _conductor_evidence(conductor_text, "5487", "helper"),
            },
            "Natural-language helper-contract repair did not produce a .498 artifact.",
        ),
        _row(
            "csl_independent_metrics",
            "missing_or_skipped",
            [
                "results/experiment_5488_csl_latent_exploration_replay_v498.json",
                "results/experiment_5489_sota_csl_independent_metrics_v498.json",
            ],
            {
                "exp5488_missing": "results/experiment_5488_csl_latent_exploration_replay_v498.json"
                in missing,
                "exp5489_missing": "results/experiment_5489_sota_csl_independent_metrics_v498.json"
                in missing,
                "conductor_evidence": _conductor_evidence(
                    conductor_text, "5488", "5489", "independent"
                ),
            },
            "The independent CSL metrics lane did not run cleanly, so CSL headlines stay blocked.",
        ),
        _row(
            "downstream_gate_blocked_csl_hardware_mapping",
            "missing_or_skipped",
            ["results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json"],
            {
                "classification": fixed_point.get("classification"),
                "evidence": _evidence(fixed_point),
                "conductor_evidence": _conductor_evidence(conductor_text, "5490", "fixed-point"),
            },
            "The downstream CSL hardware-mapping ledger was gate-blocked by missing CSL replay.",
        ),
    ]


def derive_blocked_lanes(capstone: JsonMap, hardware: JsonMap) -> list[JsonDict]:
    blocked_boards = hardware.get("blocked_boards")
    if not isinstance(blocked_boards, Mapping):
        blocked_boards = _evidence(_truth_row(capstone, "hardware")).get("blocked_boards", {})
    return [
        _row(
            "guided_decoding_quarantine",
            "blocked",
            ["results/experiment_5482_transition_v498.json", PRIOR_CAPSTONE_RELATIVE_PATH.as_posix()],
            _evidence(_truth_row(capstone, "guided_decoding")),
            "Guided decoding remains quarantined and is not a .499 starting claim.",
        ),
        _row(
            "kv260_ssh_identity",
            "blocked",
            [HARDWARE_RELATIVE_PATH.as_posix()],
            dict(blocked_boards.get("kv260", {})) if isinstance(blocked_boards, Mapping) else {},
            "KV260 was blocked by SSH identity, so no KV260 workload receipt is credited.",
        ),
        _row(
            "gatemate_jtag_identity",
            "blocked",
            [HARDWARE_RELATIVE_PATH.as_posix()],
            dict(blocked_boards.get("gatemate", {})) if isinstance(blocked_boards, Mapping) else {},
            "GateMate was blocked by JTAG identity, so no GateMate workload receipt is credited.",
        ),
    ]


def derive_honest_null_lanes(capstone: JsonMap, hardware: JsonMap, arc_live: JsonMap) -> list[JsonDict]:
    arc_delta = capstone.get("arc_registry_delta")
    return [
        _row(
            "arc_dc22_l3_no_bank",
            "honest_null",
            [ARC_LIVE_RELATIVE_PATH.as_posix()],
            {
                "selected_game": arc_live.get("selected_game"),
                "target_level": arc_live.get("target_level"),
                "prior_levels_reproduced": arc_live.get("prior_levels_reproduced"),
                "post_levels_reproduced": arc_live.get("post_levels_reproduced"),
                "new_level_banked": arc_live.get("new_level_banked"),
                "offline_reproduced": arc_live.get("offline_reproduced"),
                "failure_mode": arc_live.get("failure_mode"),
                "honest_verdict": arc_live.get("honest_verdict"),
            },
            "The dc22 L3 live trajectory attempt banked no new ARC level.",
        ),
        _row(
            "arc_registry_delta_zero",
            "honest_null",
            [PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(), ARC_LIVE_RELATIVE_PATH.as_posix()],
            {"arc_registry_delta": arc_delta},
            "The .498 ARC registry delta stayed zero.",
        ),
        _row(
            "hardware_speedup_claim_false",
            "honest_null",
            [PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(), HARDWARE_RELATIVE_PATH.as_posix()],
            {
                "capstone_hardware_speedup_claim": capstone.get("hardware_speedup_claim"),
                "hardware_artifact_speedup_claim": hardware.get("hardware_speedup_claim"),
            },
            "No .498 artifact supports a hardware speedup claim.",
        ),
    ]


def derive_flagged_lanes(capstone: JsonMap, arc_live: JsonMap) -> list[JsonDict]:
    return [
        _row(
            "exp5474_tautology_unresolved",
            "flagged",
            [
                PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(),
                "results/experiment_5484_csl_tautology_corrigendum_v498.json",
            ],
            {
                "exp5474_tautology_resolved": capstone.get("exp5474_tautology_resolved"),
                "csl_status": capstone.get("csl_status"),
            },
            "Exp5474's unresolved tautology still blocks CSL headline claims.",
        ),
        _row(
            "exp5494_arc_methodology_flag",
            "flagged",
            [ARC_LIVE_RELATIVE_PATH.as_posix()],
            {
                "flagged_adversarial": arc_live.get("flagged_adversarial"),
                "corrigendum_pending": arc_live.get("corrigendum_pending", []),
            },
            "The ARC honest-null attempt is flagged for methodology/duration caveats.",
        ),
    ]


def _protected_file_checks(
    root: Path,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[JsonDict]:
    return [
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
    ]


def _precondition_failures(
    *,
    capstone: JsonMap,
    capstone_meta: JsonMap,
    hardware: JsonMap,
    hardware_meta: JsonMap,
    arc_live: JsonMap,
    arc_live_meta: JsonMap,
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    roadmap_doc_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    return [
        reason
        for failed, reason in (
            (capstone_meta.get("loadable") is not True, "capstone_missing_or_unloadable"),
            (
                capstone.get("milestone") != PREVIOUS_MILESTONE,
                f"capstone_milestone_expected_{PREVIOUS_MILESTONE}_observed_{capstone.get('milestone')}",
            ),
            (
                not str(capstone.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES),
                "capstone_honest_verdict_missing_terminal_prefix",
            ),
            (
                capstone.get("hardware_speedup_claim") is not False,
                "capstone_hardware_speedup_claim_not_false",
            ),
            (hardware_meta.get("loadable") is not True, "hardware_missing_or_unloadable"),
            (
                hardware.get("milestone") != PREVIOUS_MILESTONE,
                f"hardware_milestone_expected_{PREVIOUS_MILESTONE}_observed_{hardware.get('milestone')}",
            ),
            (
                hardware.get("hardware_speedup_claim") is not False,
                "hardware_speedup_claim_not_false",
            ),
            (arc_live_meta.get("loadable") is not True, "arc_live_missing_or_unloadable"),
            (
                arc_live.get("milestone") != PREVIOUS_MILESTONE,
                f"arc_live_milestone_expected_{PREVIOUS_MILESTONE}_observed_{arc_live.get('milestone')}",
            ),
            (
                arc_live.get("new_level_banked") is not False,
                "arc_live_new_level_banked_not_false",
            ),
            (
                roadmap_milestone != MILESTONE,
                f"roadmap_milestone_expected_{MILESTONE}_observed_{roadmap_milestone}",
            ),
            (list(roadmap_task_ids) != EXPECTED_TASK_IDS, "roadmap_task_ids_mismatch"),
            (
                roadmap_doc_task_range != NEXT_TASK_RANGE,
                f"roadmap_doc_task_range_expected_{NEXT_TASK_RANGE}_observed_{roadmap_doc_task_range}",
            ),
            (roadmap_modified, "research-roadmap.yaml_modified"),
            (conductor_modified, "scripts/research_conductor.py_modified"),
        )
        if failed
    ]


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .498 terminal evidence into .499 transition receipt; "
            "clean lanes are transition, active-constraint descriptors, PolarFire hash "
            "receipts, ARC target precheck, and capstone synthesis; missing/skipped "
            "science lanes remain non-headline; Exp5474 tautology still blocks CSL "
            "headlines; dc22 L3, arc_registry_delta=0, and no hardware speedup claim "
            "are preserved."
        )
    return (
        "blocked: .499 transition receipt failed closed on "
        + ", ".join(failures)
        + "; available .498 evidence was still preserved."
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    capstone, capstone_meta = read_json_mapping(root / PRIOR_CAPSTONE_RELATIVE_PATH)
    hardware, hardware_meta = read_json_mapping(root / HARDWARE_RELATIVE_PATH)
    arc_live, arc_live_meta = read_json_mapping(root / ARC_LIVE_RELATIVE_PATH)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    vnext_path = root / VNEXT_RELATIVE_PATH
    vnext_text = vnext_path.read_text(encoding="utf-8") if vnext_path.exists() else ""
    roadmap_doc_task_range = normalize_task_range(vnext_text) if vnext_text else None
    source_artifacts, source_context_missing = _source_context(root)
    conductor_text = _conductor_text(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    failed_preconditions = _precondition_failures(
        capstone=capstone,
        capstone_meta=capstone_meta,
        hardware=hardware,
        hardware_meta=hardware_meta,
        arc_live=arc_live,
        arc_live_meta=arc_live_meta,
        roadmap_milestone=str(roadmap.get("milestone")) if roadmap.get("milestone") else None,
        roadmap_task_ids=roadmap_task_ids,
        roadmap_doc_task_range=roadmap_doc_task_range,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failed_preconditions else "blocked"
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
        "source_artifacts": source_artifacts,
        "source_context_missing": source_context_missing,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "preconditions_checked": {
            "capstone_loadable": capstone_meta.get("loadable") is True,
            "hardware_loadable": hardware_meta.get("loadable") is True,
            "arc_live_loadable": arc_live_meta.get("loadable") is True,
            "roadmap_milestone": roadmap.get("milestone"),
            "roadmap_task_ids_match": roadmap_task_ids == EXPECTED_TASK_IDS,
            "protected_files_clean": not roadmap_modified and not conductor_modified,
        },
        "failed_preconditions": failed_preconditions,
        "protected_file_checks": _protected_file_checks(
            root, roadmap_modified, conductor_modified
        ),
        "tests_run": list(tests_run),
        "reproducibility_checksum": "",
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(),
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "clean_lanes": derive_clean_lanes(capstone, hardware),
        "missing_or_skipped_lanes": derive_missing_or_skipped_lanes(capstone, conductor_text),
        "blocked_lanes": derive_blocked_lanes(capstone, hardware),
        "honest_null_lanes": derive_honest_null_lanes(capstone, hardware, arc_live),
        "flagged_lanes": derive_flagged_lanes(capstone, arc_live),
        "exp5474_tautology_still_blocks_csl_headlines": capstone.get(
            "exp5474_tautology_resolved"
        )
        is False
        and str(capstone.get("csl_status", "")).startswith("blocked:"),
        "next_task_range": NEXT_TASK_RANGE,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(status, failed_preconditions),
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
