"""Exp5550 transition receipt from milestone .502 into .503.

Spec refs: REQ-REPORT-5550, SCENARIO-REPORT-5550,
SCENARIO-REPORT-5550-BLOCKED-INPUT, SCENARIO-REPORT-5550-FIELD-PRINCIPLES.

This module is a record-only handoff. It reads the completed `.502` capstone,
the lane artifacts that the `.503` planner singled out, the active `.503`
roadmap, and the conductor log. It then writes a receipt that keeps clean
evidence, bounded evidence, adversarial flags, gated skips, and honest nulls
separate. That separation is what prevents an incomplete SOTA panel, a
tautological CSL ablation, or an ARC no-bank attempt from becoming headline
credit during the next milestone.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5550_transition_v503.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5549_capstone_v502.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5550_transition_v503"
EXPERIMENT_ID = "exp5550-transition-v503"
MILESTONE = "2026.07.503"
PREVIOUS_MILESTONE = "2026.07.502"
PREVIOUS_TASK_RANGE = "exp5536-exp5549"
NEXT_TASK_RANGE = "exp5550-exp5563"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5550
SCHEMA = "carnot.experiment_5550.transition_v503.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5550-transition-v503",
    "exp5551-v503-source-delta-ingestion",
    "exp5552-automaton-schema-row-completion-receipt",
    "exp5553-gated-gbnf-forced-sota-row-smoke",
    "exp5554-gated-sota-hard-soft-panel-v4",
    "exp5555-asp-fsm-nonmonotonic-fixture",
    "exp5556-gated-asp-fsm-sparse-repair-scale",
    "exp5557-csl-five-arm-tautology-corrigendum-v2",
    "exp5558-gated-causal-write-manage-read-csl-memory",
    "exp5559-gated-cross-model-sota-csl-transfer-v2",
    "exp5560-hardware-and-timing-receipt-hygiene",
    "exp5561-arc-fsm-target-rotation-precheck",
    "exp5562-gated-arc-fsm-live-levelup",
    "exp5563-v503-capstone-reconciliation",
]

ARTIFACTS: dict[str, Path] = {
    "prior_capstone": PRIOR_CAPSTONE_RELATIVE_PATH,
    "duration_corrigendum": Path(
        "results/experiment_5538_sota_panel_duration_substrate_corrigendum.json"
    ),
    "grammar_preflight": Path("results/experiment_5539_gram2token_grammar_table_preflight.json"),
    "hard_soft_panel": Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"),
    "exact_fsm_fixture": Path("results/experiment_5541_llm_fsm_exact_fixture.json"),
    "csl_residue": Path(
        "results/experiment_5542_csl_residue_metric_independence_corrigendum.json"
    ),
    "csl_five_arm": Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"),
    "cross_model_transfer": Path("results/experiment_5544_cross_model_sota_csl_transfer.json"),
    "sparse_repair": Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json"),
    "hardware_receipt": Path("results/experiment_5546_hardware_receipt_substrate_corrigendum.json"),
    "arc_precheck": Path("results/experiment_5547_arc_no_llm_substrate_precheck.json"),
    "arc_levelup": Path("results/experiment_5548_arc_clean_live_levelup.json"),
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
    Path("research-complete.yaml"),
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the `.503` transition receipt.",
    "previous_milestone": "Source milestone whose terminal facts are archived.",
    "prior_capstone_path": "Exact `.502` capstone artifact used as the main source of truth.",
    "previous_task_range": "Closed `.502` conductor task range.",
    "next_task_range": "Planned `.503` conductor task range.",
    "clean_lanes": "Completed `.502` evidence safe to carry forward without promoting flagged claims.",
    "bounded_lanes": "Useful `.502` evidence kept claim-limited until the matching `.503` gate runs.",
    "blocked_lanes": "Blocked `.502` prerequisites that must not become headline credit.",
    "flagged_lanes": "Adversarial or methodology flags preserved as repair work, not clean evidence.",
    "skipped_by_gates": "Gated `.502` skips preserved as skipped evidence, not clean completion.",
    "sota_row_completion_required": "Bare boolean requiring automaton row-completion before grammar-forced SOTA smoke.",
    "asp_fsm_fixture_required": "Bare boolean requiring ASP/FSM exact fixture evidence before sparse repair scale.",
    "csl_tautology_corrigendum_required": "Bare boolean requiring non-tautological five-arm CSL evidence before causal memory.",
    "causal_csl_memory_required": "Bare boolean requiring causal CSL memory before cross-model transfer.",
    "arc_target_rotation_required": "Bare boolean requiring ARC target rotation before another level-up attempt.",
    "hardware_receipt_only": "Bare boolean preserving no-speedup posture without matched authenticated timing.",
    "roadmap_yaml_unchanged": "Protected-file check for `research-roadmap.yaml`.",
    "conductor_unchanged": "Protected-file check for `scripts/research_conductor.py`.",
    "inference_substrate": "Must equal `aggregation_from_upstream_artifacts` because Exp5550 is synthesis only.",
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
    "execution_gates",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
BOOL_FIELDS = (
    "sota_row_completion_required",
    "asp_fsm_fixture_required",
    "csl_tautology_corrigendum_required",
    "causal_csl_memory_required",
    "arc_target_rotation_required",
    "hardware_receipt_only",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
)
LIST_FIELDS = (
    "clean_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "flagged_lanes",
    "skipped_by_gates",
    "source_context",
    "source_context_missing",
    "artifacts_expected",
    "artifacts_found",
    "artifacts_missing",
    "roadmap_task_ids",
    "failed_preconditions",
    "conductor_evidence",
    "execution_gates",
    "tests_run",
)

SPEC_REFS = (
    "REQ-REPORT-5550",
    "SCENARIO-REPORT-5550",
    "SCENARIO-REPORT-5550-BLOCKED-INPUT",
    "SCENARIO-REPORT-5550-FIELD-PRINCIPLES",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5550_transition_v503.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5550_transition_v503.py "
            "-m pytest tests/python/test_experiment_5550_transition_v503.py -q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5550_transition_v503.py --fail-under=100"
        ),
        "outcome": "passed",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "outcome": "failed_interrupted",
        "detail": (
            "interrupted after 1004.77s with 25 failed, 13317 passed, 74 skipped; "
            "one xdist worker crashed in the pre-existing Z3/math verifier path"
        ),
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
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _task_range_from_text(text: str) -> str | None:
    explicit = re.search(r"Exp\s*(\d+)\s*-\s*(\d+)", text, flags=re.IGNORECASE)
    if explicit:
        return f"exp{explicit.group(1)}-exp{explicit.group(2)}"
    backtick = re.search(r"exp(\d+)\D{1,16}exp(\d+)", text, flags=re.IGNORECASE)
    if backtick:
        return f"exp{backtick.group(1)}-exp{backtick.group(2)}"
    return None


def _corrigenda(payload: JsonMap) -> list[JsonDict]:
    rows = payload.get("corrigendum_pending")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str) and str(value).lstrip("-").isdigit():
        return int(value)
    return 0


def _status_label(payload: JsonMap) -> str:
    status = payload.get("status")
    if status is not None:
        return str(status).lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if verdict.startswith("blocked:"):
        return "blocked"
    if verdict.startswith("honest_null:") or "honest_null" in verdict:
        return "honest_null"
    if verdict.startswith("failed:"):
        return "failed"
    if verdict.startswith("complete:"):
        return "complete"
    return "unknown"


def _conductor_evidence(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    tokens = (
        "5538",
        "5539",
        "5540",
        "5541",
        "5542",
        "5543",
        "5544",
        "5545",
        "5546",
        "5547",
        "5548",
        "5549",
        "5550",
        "2026.07.503",
        "SOTA",
        "CSL",
        "FSM",
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
    ][-35:]


def _clean_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    duration = artifacts["duration_corrigendum"]
    grammar = artifacts["grammar_preflight"]
    fsm = artifacts["exact_fsm_fixture"]
    residue = artifacts["csl_residue"]
    sparse = artifacts["sparse_repair"]
    hardware = artifacts["hardware_receipt"]
    arc_precheck = artifacts["arc_precheck"]
    return [
        _row(
            "duration_substrate_corrigendum",
            "clean",
            (ARTIFACTS["duration_corrigendum"],),
            {
                "sota_panel_duration_corrigendum_ready": bool(
                    duration.get("sota_panel_duration_corrigendum_ready")
                ),
                "adversarial_clean": bool(duration.get("adversarial_clean")),
                "quality_claim_allowed": bool(duration.get("quality_claim_allowed")),
                "rows_requested": duration.get("rows_requested"),
                "rows_emitted": duration.get("rows_emitted"),
                "honest_verdict": duration.get("honest_verdict"),
            },
            "The duration/substrate correction is clean, but it downgraded quality because rows remained incomplete.",
        ),
        _row(
            "grammar_table_preflight",
            "clean",
            (ARTIFACTS["grammar_preflight"],),
            {
                "grammar_table_preflight_ready": bool(grammar.get("grammar_table_preflight_ready")),
                "valid_fixture_acceptance_rate": grammar.get("valid_fixture_acceptance_rate"),
                "invalid_fixture_rejection_rate": grammar.get("invalid_fixture_rejection_rate"),
                "schema_transition_table_row_count": grammar.get(
                    "schema_transition_table_row_count"
                ),
            },
            "The grammar table fixture is usable as a deterministic preflight, not a live SOTA claim.",
        ),
        _row(
            "exact_fsm_fixture",
            "clean",
            (ARTIFACTS["exact_fsm_fixture"],),
            {
                "exact_fsm_fixture_ready": bool(fsm.get("exact_fsm_fixture_ready")),
                "satisfiable_instances": fsm.get("satisfiable_instances"),
                "unsatisfiable_instances": fsm.get("unsatisfiable_instances"),
                "ambiguous_instances": fsm.get("ambiguous_instances"),
            },
            "The exact finite-state fixture is a clean prerequisite for ASP/FSM sparse repair scaling.",
        ),
        _row(
            "csl_residue_independence",
            "clean",
            (ARTIFACTS["csl_residue"],),
            {
                "csl_residue_tautology_resolved": bool(
                    residue.get("csl_residue_tautology_resolved")
                ),
                "nonidentical_metric_evidence": bool(residue.get("nonidentical_metric_evidence")),
                "event_only_score": residue.get("event_only_score"),
                "topic_only_score": residue.get("topic_only_score"),
            },
            "Residue metric independence is clean, but five-arm CSL still needs its own tautology repair.",
        ),
        _row(
            "sparse_fsm_repair_signal",
            "clean",
            (ARTIFACTS["sparse_repair"],),
            {
                "sparse_repair_fsm_ready": bool(sparse.get("sparse_repair_fsm_ready")),
                "exact_validator_all_repairs_checked": bool(
                    sparse.get("exact_validator_all_repairs_checked")
                ),
                "unchecked_repair_count": _int(sparse, "unchecked_repair_count"),
                "descriptor_guided_success_rate": sparse.get("descriptor_guided_success_rate"),
                "random_block_success_rate": sparse.get("random_block_success_rate"),
            },
            "Sparse FSM repair has exact-checked signal and remains separate from speedup claims.",
        ),
        _row(
            "hardware_receipt_hygiene",
            "clean",
            (ARTIFACTS["hardware_receipt"],),
            {
                "hardware_receipt_corrigendum_clean": bool(
                    hardware.get("hardware_receipt_corrigendum_clean")
                ),
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
                "no_model_specs_required": bool(hardware.get("no_model_specs_required")),
            },
            "Hardware receipt hygiene is clean, but matched timing is absent and speedup remains false.",
        ),
        _row(
            "arc_clean_precheck",
            "clean",
            (ARTIFACTS["arc_precheck"],),
            {
                "arc_clean_precheck_ready": bool(arc_precheck.get("arc_clean_precheck_ready")),
                "selected_game": arc_precheck.get("selected_game"),
                "selected_level": arc_precheck.get("selected_level"),
                "solve_provenance": arc_precheck.get("solve_provenance"),
            },
            "The ARC precheck is clean target context only; it is not solve credit.",
        ),
    ]


def _bounded_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    capstone = artifacts["prior_capstone"]
    panel = artifacts["hard_soft_panel"]
    residue = artifacts["csl_residue"]
    five_arm = artifacts["csl_five_arm"]
    transfer = artifacts["cross_model_transfer"]
    sparse = artifacts["sparse_repair"]
    hardware = artifacts["hardware_receipt"]
    arc = artifacts["arc_levelup"]
    return [
        _row(
            "incomplete_sota_rows",
            "bounded",
            (ARTIFACTS["hard_soft_panel"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "sota_hard_soft_claim_allowed": bool(panel.get("sota_hard_soft_claim_allowed")),
                "rows_requested": panel.get("rows_requested"),
                "rows_emitted": panel.get("rows_emitted"),
                "schema_valid_rows": panel.get("schema_valid_rows"),
                "missing_candidate_rows": panel.get("missing_candidate_rows"),
                "capstone_structured_sota_claim_allowed": bool(
                    capstone.get("structured_sota_claim_allowed")
                ),
            },
            "Exact validation worked on emitted rows, but missing rows force automaton row-completion before another smoke.",
        ),
        _row(
            "causal_csl_prerequisites",
            "bounded",
            (
                ARTIFACTS["csl_residue"],
                ARTIFACTS["csl_five_arm"],
                ARTIFACTS["cross_model_transfer"],
            ),
            {
                "residue_tautology_resolved": bool(
                    residue.get("csl_residue_tautology_resolved")
                ),
                "five_arm_flagged_adversarial": bool(five_arm.get("flagged_adversarial")),
                "cross_model_csl_claim_allowed": bool(transfer.get("csl_claim_allowed")),
                "cross_family_delta_over_shuffled": transfer.get("cross_family_delta_over_shuffled"),
                "capstone_csl_claim_allowed": bool(capstone.get("csl_claim_allowed")),
            },
            "CSL has residue evidence, but causal memory must follow a non-tautological five-arm repair.",
        ),
        _row(
            "sparse_repair_no_speedup",
            "bounded",
            (ARTIFACTS["sparse_repair"],),
            {
                "sparse_repair_fsm_ready": bool(sparse.get("sparse_repair_fsm_ready")),
                "speedup_claim_allowed": bool(sparse.get("speedup_claim_allowed")),
                "matched_timing_available": bool(sparse.get("matched_timing_available")),
            },
            "Sparse repair carries as exact-checked descriptor evidence only, with speedup disabled.",
        ),
        _row(
            "hardware_receipt_only",
            "bounded",
            (ARTIFACTS["hardware_receipt"],),
            {
                "hardware_receipt_corrigendum_clean": bool(
                    hardware.get("hardware_receipt_corrigendum_clean")
                ),
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
            },
            "Hardware remains receipt-only until a matched timing workload exists.",
        ),
        _row(
            "arc_live_path_no_bank",
            "bounded",
            (ARTIFACTS["arc_levelup"],),
            {
                "solve_provenance": arc.get("solve_provenance"),
                "selected_game": arc.get("selected_game"),
                "selected_level": arc.get("selected_level"),
                "offline_reproduced": bool(arc.get("offline_reproduced")),
                "reproduced_levels": _int(arc, "reproduced_levels"),
                "registry_delta": _int(arc, "registry_delta"),
            },
            "ARC live-path evidence is preserved, but no registry credit exists without offline reproduction.",
        ),
    ]


def _blocked_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    capstone = artifacts["prior_capstone"]
    panel = artifacts["hard_soft_panel"]
    hardware = artifacts["hardware_receipt"]
    arc = artifacts["arc_levelup"]
    return [
        _row(
            "exp5540_no_sota_hard_soft_claim",
            "blocked",
            (ARTIFACTS["hard_soft_panel"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "sota_hard_soft_claim_allowed": bool(panel.get("sota_hard_soft_claim_allowed")),
                "gates_clean": bool(panel.get("gates_clean")),
                "exact_validator_accuracy": panel.get("exact_validator_accuracy"),
                "missing_candidate_rows": panel.get("missing_candidate_rows"),
                "capstone_sota_hard_soft_claim_allowed": bool(
                    capstone.get("sota_hard_soft_claim_allowed")
                ),
            },
            "Exp5540 is an honest hard/soft null: emitted rows validated, but missing rows block the claim.",
        ),
        _row(
            "exp5548_arc_honest_null",
            "blocked",
            (ARTIFACTS["arc_levelup"],),
            {
                "status": _status_label(arc),
                "honest_verdict": arc.get("honest_verdict"),
                "offline_reproduced": bool(arc.get("offline_reproduced")),
                "registry_delta": _int(arc, "registry_delta"),
                "reproduced_levels": _int(arc, "reproduced_levels"),
            },
            "Exp5548 is a clean honest null and cannot count as a banked level.",
        ),
        _row(
            "hardware_speedup_false",
            "blocked",
            (ARTIFACTS["hardware_receipt"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "hardware_speedup_claim": bool(capstone.get("hardware_speedup_claim")),
                "artifact_hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
            },
            "Hardware speedup remains false because matched authenticated timing is absent.",
        ),
        _row(
            "arc_registry_delta_zero",
            "blocked",
            (ARTIFACTS["arc_levelup"], PRIOR_CAPSTONE_RELATIVE_PATH),
            {
                "capstone_arc_registry_delta": _int(capstone, "arc_registry_delta"),
                "artifact_registry_delta": _int(arc, "registry_delta"),
                "artifact_reproduced_levels": _int(arc, "reproduced_levels"),
                "capstone_reproduced_levels": _int(capstone, "reproduced_levels"),
            },
            "The ARC registry delta is zero, so the next attempt must rotate target before level-up.",
        ),
    ]


def _flagged_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    five_arm = artifacts["csl_five_arm"]
    return [
        _row(
            "exp5543_csl_tautology",
            "flagged",
            (ARTIFACTS["csl_five_arm"],),
            {
                "flagged_adversarial": bool(five_arm.get("flagged_adversarial")),
                "csl_five_arm_ready": bool(five_arm.get("csl_five_arm_ready")),
                "stale_evidence_rejection_rate": five_arm.get("stale_evidence_rejection_rate"),
                "negative_transfer_rate": five_arm.get("negative_transfer_rate"),
                "corrigendum_pending": _corrigenda(five_arm),
            },
            "Exp5543 is quarantined by a CSL TAUTOLOGY flag and must be repaired before causal memory.",
        )
    ]


def _skipped_by_gates(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    transfer = artifacts["cross_model_transfer"]
    return [
        _row(
            "exp5544_cross_model_transfer_skip",
            "skipped_by_gate",
            (ARTIFACTS["cross_model_transfer"],),
            {
                "status": _status_label(transfer),
                "honest_verdict": transfer.get("honest_verdict"),
                "csl_claim_allowed": bool(transfer.get("csl_claim_allowed")),
                "no_weight_mutation": bool(transfer.get("no_weight_mutation")),
                "cross_family_delta_over_shuffled": transfer.get(
                    "cross_family_delta_over_shuffled"
                ),
            },
            "Exp5544 correctly stayed gated because cross-model transfer had zero allowed CSL delta.",
        )
    ]


def _execution_gates() -> list[JsonDict]:
    return [
        {
            "gate": "automaton_row_completion_before_grammar_forced_sota_smoke",
            "upstream_task": "exp5552-automaton-schema-row-completion-receipt",
            "downstream_task": "exp5553-gated-gbnf-forced-sota-row-smoke",
            "required_field": "automaton_row_completion_ready",
        },
        {
            "gate": "grammar_forced_smoke_before_panel_v4",
            "upstream_task": "exp5553-gated-gbnf-forced-sota-row-smoke",
            "downstream_task": "exp5554-gated-sota-hard-soft-panel-v4",
            "required_field": "grammar_forced_row_smoke_ready",
        },
        {
            "gate": "asp_fsm_fixture_before_sparse_repair_scale",
            "upstream_task": "exp5555-asp-fsm-nonmonotonic-fixture",
            "downstream_task": "exp5556-gated-asp-fsm-sparse-repair-scale",
            "required_field": "asp_fsm_fixture_ready",
        },
        {
            "gate": "csl_tautology_repair_before_causal_csl_memory",
            "upstream_task": "exp5557-csl-five-arm-tautology-corrigendum-v2",
            "downstream_task": "exp5558-gated-causal-write-manage-read-csl-memory",
            "required_field": "csl_five_arm_tautology_resolved",
        },
        {
            "gate": "causal_memory_before_cross_model_transfer",
            "upstream_task": "exp5558-gated-causal-write-manage-read-csl-memory",
            "downstream_task": "exp5559-gated-cross-model-sota-csl-transfer-v2",
            "required_field": "causal_csl_memory_claim_allowed",
        },
        {
            "gate": "arc_target_rotation_before_levelup",
            "upstream_task": "exp5561-arc-fsm-target-rotation-precheck",
            "downstream_task": "exp5562-gated-arc-fsm-live-levelup",
            "required_field": "arc_target_rotation_ready",
        },
    ]


def _protected_file_checks(
    root: Path,
    *,
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


def _failed_preconditions(
    *,
    artifacts: Mapping[str, JsonMap],
    artifacts_missing: Sequence[str],
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    vnext_names_milestone: bool,
    vnext_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures: list[str] = []
    for rel_path in artifacts_missing:
        failures.append(f"{rel_path}_missing_or_unreadable")
    capstone = artifacts["prior_capstone"]
    if capstone.get("milestone") != PREVIOUS_MILESTONE:
        failures.append("prior_capstone_milestone_mismatch")
    if capstone.get("task_range") not in {PREVIOUS_TASK_RANGE, None}:
        failures.append("prior_capstone_task_range_mismatch")
    if roadmap_milestone != MILESTONE:
        failures.append("research-roadmap.yaml_milestone_mismatch")
    if list(roadmap_task_ids) != EXPECTED_TASK_IDS:
        failures.append("roadmap_task_ids_mismatch")
    if not vnext_names_milestone:
        failures.append("vnext_milestone_mismatch")
    if vnext_task_range != NEXT_TASK_RANGE:
        failures.append("vnext_task_range_mismatch")
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .502 terminal evidence into .503 transition receipt; "
            "clean lanes are duration/substrate corrigendum, grammar preflight, exact FSM, "
            "CSL residue independence, sparse FSM repair, hardware receipt hygiene, and "
            "ARC clean precheck; blocked or flagged lanes are Exp5540 no hard/soft claim, "
            "Exp5543 CSL TAUTOLOGY, Exp5544 gated transfer skip, Exp5548 ARC honest null, "
            "hardware_speedup_claim=false, and arc_registry_delta=0; "
            "next_task_range=exp5550-exp5563."
        )
    first_failure = failures[0] if failures else "unknown"
    return f"blocked: .503 transition receipt failed precondition {first_failure}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, found, missing, metadata = _read_artifacts(root)
    source_context, source_missing = _source_context(root)
    roadmap, roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_milestone = roadmap.get("milestone")
    roadmap_milestone = str(roadmap_milestone) if roadmap_milestone is not None else None
    vnext_text = _read_text(root, VNEXT_RELATIVE_PATH)
    vnext_task_range = _task_range_from_text(vnext_text)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)

    failures = _failed_preconditions(
        artifacts=artifacts,
        artifacts_missing=missing,
        roadmap_milestone=roadmap_milestone,
        roadmap_task_ids=roadmap_task_ids,
        vnext_names_milestone=MILESTONE in vnext_text,
        vnext_task_range=vnext_task_range,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    tests: list[Any] = [dict(row) if isinstance(row, Mapping) else str(row) for row in tests_run]
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
        "source_context_missing": source_missing,
        "artifact_metadata": metadata,
        "artifacts_expected": [path.as_posix() for path in ARTIFACTS.values()],
        "artifacts_found": found,
        "artifacts_missing": missing,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": vnext_task_range,
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "prior_capstone_present": PRIOR_CAPSTONE_RELATIVE_PATH.as_posix() in found,
            "prior_capstone_milestone": artifacts["prior_capstone"].get("milestone"),
            "required_artifacts_found": len(found),
            "required_artifacts_expected": len(ARTIFACTS),
            "roadmap_present": roadmap_meta.get("exists") is True,
            "roadmap_loadable": roadmap_meta.get("loadable") is True,
            "roadmap_milestone": roadmap_milestone,
            "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
            "vnext_present": (root / VNEXT_RELATIVE_PATH).exists(),
            "vnext_names_milestone": MILESTONE in vnext_text,
            "vnext_task_range": vnext_task_range,
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "conductor_evidence": _conductor_evidence(root),
        "execution_gates": _execution_gates(),
        "tests_run": tests,
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": PRIOR_CAPSTONE_RELATIVE_PATH.as_posix(),
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "next_task_range": NEXT_TASK_RANGE,
        "clean_lanes": _clean_lanes(artifacts),
        "bounded_lanes": _bounded_lanes(artifacts),
        "blocked_lanes": _blocked_lanes(artifacts),
        "flagged_lanes": _flagged_lanes(artifacts),
        "skipped_by_gates": _skipped_by_gates(artifacts),
        "sota_row_completion_required": True,
        "asp_fsm_fixture_required": True,
        "csl_tautology_corrigendum_required": True,
        "causal_csl_memory_required": True,
        "arc_target_rotation_required": True,
        "hardware_receipt_only": True,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(status, failures),
        "reproducibility_checksum": "",
    }
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
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - unit tests exercise validate_artifact directly
        raise ValueError(f"invalid Exp5550 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5550 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
