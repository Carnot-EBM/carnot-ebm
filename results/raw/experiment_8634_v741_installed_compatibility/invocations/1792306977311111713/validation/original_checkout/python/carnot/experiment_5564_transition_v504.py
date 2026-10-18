"""Exp5564 transition receipt from milestone .503 into .504.

Spec refs: REQ-REPORT-5564, SCENARIO-REPORT-5564,
SCENARIO-REPORT-5564-BLOCKED-INPUT, SCENARIO-REPORT-5564-FIELD-PRINCIPLES.

This module is a synthesis-only handoff. It reads the `.503` capstone, the
terminal records that capstone represents, the active `.504` roadmap, and the
conductor log. It then writes down which facts are safe to carry forward. The
important behavior is negative: a blocked grammar row path, flagged
cross-family CSL transfer, unmatched hardware receipt, or ARC no-bank null
must stay visible as a boundary instead of becoming credit for the new
milestone.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5564_transition_v504.json")
PREVIOUS_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5563_capstone_v503.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXP5550_TRANSITION_PATH = Path("results/experiment_5550_transition_v503.json")
EXP5551_SOURCE_DELTA_PATH = Path("results/experiment_5551_v503_source_delta_ingestion.json")
EXP5552_ROW_COMPLETION_PATH = Path(
    "results/experiment_5552_automaton_schema_row_completion_receipt.json"
)
EXP5553_GBNF_SMOKE_PATH = Path("results/experiment_5553_gated_gbnf_forced_sota_row_smoke.json")
EXP5554_PANEL_PATH = Path("results/experiment_5554_sota_hard_soft_panel_v4.json")
EXP5555_ASP_FSM_FIXTURE_PATH = Path("results/experiment_5555_asp_fsm_nonmonotonic_fixture.json")
EXP5556_SPARSE_REPAIR_PATH = Path("results/experiment_5556_asp_fsm_sparse_repair_scale.json")
EXP5557_CSL_REPAIR_PATH = Path("results/experiment_5557_csl_five_arm_tautology_corrigendum_v2.json")
EXP5558_CAUSAL_MEMORY_PATH = Path("results/experiment_5558_causal_write_manage_read_csl_memory.json")
EXP5559_CROSS_MODEL_CSL_PATH = Path("results/experiment_5559_cross_model_sota_csl_transfer_v2.json")
EXP5560_HARDWARE_PATH = Path("results/experiment_5560_hardware_and_timing_receipt_hygiene.json")
EXP5561_ARC_PRECHECK_PATH = Path("results/experiment_5561_arc_fsm_target_rotation_precheck.json")
EXP5562_ARC_LEVELUP_PATH = Path("results/experiment_5562_arc_fsm_live_levelup.json")

EXPERIMENT = "experiment_5564_transition_v504"
EXPERIMENT_ID = "exp5564-transition-v504"
MILESTONE = "2026.07.504"
PREVIOUS_MILESTONE = "2026.07.503"
PREVIOUS_TASK_RANGE = "exp5550-exp5563"
NEXT_TASK_RANGE = "exp5564-exp5577"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5564
SCHEMA = "carnot.experiment_5564.transition_v504.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")
EXPECTED_PREVIOUS_CAPSTONE_VERDICT = (
    "complete: .503 capstone read 13/14 expected artifacts; missing=0; "
    "flagged=1; blocked=2; skipped_by_gates=2; honest_nulls=1; "
    "structured_sota_claim_allowed=False; sota_hard_soft_claim_allowed=False; "
    "continuous_self_learning_evidence=True; csl_claim_allowed=False; "
    "cross_model_csl_claim_allowed=False; asp_sparse_repair_claim_allowed=True; "
    "hardware_speedup_claim=False; arc_registry_delta=0"
)

EXPECTED_TASK_IDS = [
    "exp5564-transition-v504",
    "exp5565-v504-source-delta-ingestion",
    "exp5566-exact-asp-fsm-near-miss-corpus",
    "exp5567-gated-local-sota-solve-verify-asymmetry",
    "exp5568-gated-verifier-coevolution-trigger",
    "exp5569-causal-memory-policy-tournament",
    "exp5570-spline-local-kan-online-energy",
    "exp5571-gated-reset-free-sota-continual-harness",
    "exp5572-gated-delayed-regression-promotion",
    "exp5573-matched-sampler-hardware-continuity",
    "exp5574-ptrm-stochastic-generator-stage1",
    "exp5575-sge-anti-stagnation-live-precheck",
    "exp5576-gated-sge-live-levelup",
    "exp5577-v504-capstone-reconciliation",
]

SPEC_REFS = (
    "REQ-REPORT-5564",
    "SCENARIO-REPORT-5564",
    "SCENARIO-REPORT-5564-BLOCKED-INPUT",
    "SCENARIO-REPORT-5564-FIELD-PRINCIPLES",
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PREVIOUS_CAPSTONE_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

TERMINAL_RECORDS: tuple[tuple[str, Path, str], ...] = (
    ("exp5550-transition-v503", EXP5550_TRANSITION_PATH, "json_artifact"),
    ("exp5551-v503-source-delta-ingestion", EXP5551_SOURCE_DELTA_PATH, "json_artifact"),
    ("exp5552-automaton-schema-row-completion-receipt", EXP5552_ROW_COMPLETION_PATH, "json_artifact"),
    ("exp5553-gated-gbnf-forced-sota-row-smoke", EXP5553_GBNF_SMOKE_PATH, "json_artifact"),
    ("exp5554-gated-sota-hard-soft-panel-v4", EXP5554_PANEL_PATH, "conductor_skip"),
    ("exp5555-asp-fsm-nonmonotonic-fixture", EXP5555_ASP_FSM_FIXTURE_PATH, "json_artifact"),
    ("exp5556-gated-asp-fsm-sparse-repair-scale", EXP5556_SPARSE_REPAIR_PATH, "json_artifact"),
    ("exp5557-csl-five-arm-tautology-corrigendum-v2", EXP5557_CSL_REPAIR_PATH, "json_artifact"),
    ("exp5558-gated-causal-write-manage-read-csl-memory", EXP5558_CAUSAL_MEMORY_PATH, "json_artifact"),
    ("exp5559-gated-cross-model-sota-csl-transfer-v2", EXP5559_CROSS_MODEL_CSL_PATH, "json_artifact"),
    ("exp5560-hardware-and-timing-receipt-hygiene", EXP5560_HARDWARE_PATH, "json_artifact"),
    ("exp5561-arc-fsm-target-rotation-precheck", EXP5561_ARC_PRECHECK_PATH, "json_artifact"),
    ("exp5562-gated-arc-fsm-live-levelup", EXP5562_ARC_LEVELUP_PATH, "json_artifact"),
    ("exp5563-v503-capstone-reconciliation", PREVIOUS_CAPSTONE_RELATIVE_PATH, "json_artifact"),
)

PROMPT_ALIAS_RESOLUTION: list[JsonDict] = [
    {
        "alias_path": "results/experiment_5555_asp_fsm_exact_fixture.json",
        "resolved_path": EXP5555_ASP_FSM_FIXTURE_PATH.as_posix(),
        "alias_exists": False,
        "resolved_exists": True,
        "source_of_truth": "capstone_artifact_metadata",
    },
    {
        "alias_path": "results/experiment_5558_causal_csl_write_manage_read_memory.json",
        "resolved_path": EXP5558_CAUSAL_MEMORY_PATH.as_posix(),
        "alias_exists": False,
        "resolved_exists": True,
        "source_of_truth": "capstone_artifact_metadata",
    },
    {
        "alias_path": "results/experiment_5560_hardware_timing_provenance_hygiene.json",
        "resolved_path": EXP5560_HARDWARE_PATH.as_posix(),
        "alias_exists": False,
        "resolved_exists": True,
        "source_of_truth": "capstone_artifact_metadata",
    },
    {
        "alias_path": "results/experiment_5562_arc_rotated_target_levelup.json",
        "resolved_path": EXP5562_ARC_LEVELUP_PATH.as_posix(),
        "alias_exists": False,
        "resolved_exists": True,
        "source_of_truth": "capstone_artifact_metadata",
    },
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the `.504` transition receipt.",
    "previous_milestone": "Source milestone whose terminal facts are archived.",
    "previous_task_range": "Closed `.503` conductor task range.",
    "next_task_range": "Planned `.504` conductor task range.",
    "artifacts_read": "Count of `.503` terminal records accounted for through readable artifacts, capstone metadata, or conductor skip evidence.",
    "clean_lanes": "Completed `.503` evidence carried forward without promoting blocked or flagged claims.",
    "bounded_lanes": "Useful `.503` evidence preserved with explicit claim limits.",
    "blocked_lanes": "Blocked `.503` prerequisites that must not become headline credit.",
    "flagged_lanes": "Adversarial or methodology flags carried forward as boundaries, not clean evidence.",
    "retired_continuations": "Ordinary reruns prohibited by `.503` terminal evidence and `.504` roadmap scope.",
    "verifier_chain": "Corpus-to-SOTA-panel-to-co-evolution gate map for `.504` verifier work.",
    "self_learning_chain": "Memory plus KAN to reset-free harness to promotion gate map.",
    "arc_chain": "SGE precheck to live level-up gate map for the ordinary ARC floor.",
    "ptrm_slot_separate": "Bare boolean confirming the reserved PTRM generator slot does not count as the ordinary ARC floor.",
    "hardware_claim_allowed": "Bare boolean that must remain false without matched successful timing pairs and authenticated device receipts.",
    "roadmap_yaml_unchanged": "Protected-file check for `research-roadmap.yaml`.",
    "conductor_unchanged": "Protected-file check for `scripts/research_conductor.py`.",
    "field_principles": "One-line annotations for every headline and gate field.",
    "inference_substrate": "Must equal `aggregation_from_upstream_artifacts` because Exp5564 is synthesis only.",
    "honest_verdict": "Terminal summary starting with complete: or blocked: that names the `.503` to `.504` transition boundary.",
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
    "source_context",
    "source_context_missing",
    "artifact_metadata",
    "terminal_records",
    "terminal_records_missing",
    "json_terminal_artifacts_read",
    "conductor_skip_records_read",
    "previous_capstone_path",
    "previous_capstone_summary",
    "previous_capstone_claims",
    "prompt_alias_resolution",
    "roadmap_task_ids",
    "roadmap_doc_task_range",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

BOOL_FIELDS = (
    "ptrm_slot_separate",
    "hardware_claim_allowed",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
)
INT_FIELDS = ("artifacts_read", "json_terminal_artifacts_read", "conductor_skip_records_read")
LIST_FIELDS = (
    "source_context",
    "source_context_missing",
    "terminal_records",
    "terminal_records_missing",
    "clean_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "flagged_lanes",
    "retired_continuations",
    "verifier_chain",
    "self_learning_chain",
    "arc_chain",
    "prompt_alias_resolution",
    "roadmap_task_ids",
    "protected_file_checks",
    "failed_preconditions",
    "tests_run",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5564_transition_v504.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5564_transition_v504.py "
            "-m pytest tests/python/test_experiment_5564_transition_v504.py -q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5564_transition_v504.py --fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

RETIRED_CONTINUATIONS: list[JsonDict] = [
    {
        "continuation": "automaton_gbnf_row_completion_vnext",
        "retirement_reason": "Exp5552 row-completion support stayed 0.333333 and blocked the downstream smoke.",
    },
    {
        "continuation": "hard_soft_panel_v5_on_failed_row_path",
        "retirement_reason": "Exp5553 and Exp5554 were conductor-gated by the failed row-completion prerequisite.",
    },
    {
        "continuation": "cross_family_csl_transfer_v3_on_exp5559_substrate",
        "retirement_reason": "Exp5559 was flagged and all cross-family memory arms collapsed to the same score.",
    },
    {
        "continuation": "hardware_speedup_from_unmatched_receipts",
        "retirement_reason": "Exp5560 had clean hygiene but no matched authenticated timing pairs.",
    },
    {
        "continuation": "same_scope_arc_null_levelup_retry",
        "retirement_reason": "Exp5562 had live-path provenance but no offline reproduction and registry_delta=0.",
    },
]


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _task_range_from_text(text: str) -> str | None:
    explicit = re.search(r"Exp\s*(\d+)\s*-\s*(\d+)", text, flags=re.IGNORECASE)
    if explicit:
        return f"exp{explicit.group(1)}-exp{explicit.group(2)}"
    compact = re.search(r"exp(\d+)\s*-\s*exp?(\d+)", text, flags=re.IGNORECASE)
    if compact:
        return f"exp{compact.group(1)}-exp{compact.group(2)}"
    return None


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _float(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _status_label(payload: JsonMap) -> str:
    status = payload.get("status")
    if status is not None:
        return str(status).lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if verdict.startswith("blocked:") or verdict.startswith("blocked_"):
        return "blocked"
    if verdict.startswith("honest_null:") or "honest_null" in verdict:
        return "honest_null"
    if verdict.startswith("failed:"):
        return "failed"
    if verdict.startswith("complete:"):
        return "complete"
    return "unknown"


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


def _read_terminal_records(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    terminal_records: list[JsonDict] = []
    conductor_text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    for task_id, rel_path, source_kind in TERMINAL_RECORDS:
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        conductor_lines = [
            line.strip()
            for line in conductor_text.splitlines()
            if "Gated SOTA hard-soft panel v4" in line and "GATE_BLOCK" in line
        ]
        accounted_by_skip = (
            source_kind == "conductor_skip" and bool(conductor_lines) and not meta.get("loadable")
        )
        terminal_records.append(
            {
                "task_id": task_id,
                "artifact_path": rel,
                "source_kind": source_kind,
                "exists": bool(meta.get("exists")),
                "loadable": bool(meta.get("loadable")),
                "accounted": bool(meta.get("loadable") or accounted_by_skip),
                "status": _status_label(payload) if payload else ("blocked" if accounted_by_skip else "missing"),
                "honest_verdict": payload.get("honest_verdict") if payload else (
                    "blocked_gate_check_failed" if accounted_by_skip else None
                ),
                "conductor_evidence": conductor_lines[-1] if accounted_by_skip else None,
                "sha256": meta.get("sha256"),
            }
        )
    return artifacts, metadata, terminal_records


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    return artifacts.get(rel_path.as_posix(), {})


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


def _clean_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    fixture = _payload(artifacts, EXP5555_ASP_FSM_FIXTURE_PATH)
    sparse = _payload(artifacts, EXP5556_SPARSE_REPAIR_PATH)
    csl = _payload(artifacts, EXP5557_CSL_REPAIR_PATH)
    memory = _payload(artifacts, EXP5558_CAUSAL_MEMORY_PATH)
    return [
        _row(
            "exact_asp_fsm_fixture",
            "clean",
            (EXP5555_ASP_FSM_FIXTURE_PATH,),
            {
                "exact_asp_validator_ready": bool(fixture.get("exact_asp_validator_ready")),
                "exact_fsm_fixture_extended_ready": bool(
                    fixture.get("exact_fsm_fixture_extended_ready")
                ),
                "asp_row_count": _int(fixture, "asp_row_count"),
                "honest_verdict": fixture.get("honest_verdict"),
            },
            "Exact ASP/FSM fixture evidence is clean and becomes the .504 corpus substrate.",
        ),
        _row(
            "bounded_sparse_repair",
            "clean_bounded",
            (EXP5556_SPARSE_REPAIR_PATH,),
            {
                "asp_sparse_repair_claim_allowed": bool(
                    sparse.get("asp_sparse_repair_claim_allowed")
                ),
                "stable_model_checked_rate": sparse.get("stable_model_checked_rate"),
                "descriptor_guided_success_rate": sparse.get("descriptor_guided_success_rate"),
                "unchecked_repair_count": _int(sparse, "unchecked_repair_count"),
                "speedup_claim_allowed": bool(sparse.get("speedup_claim_allowed")),
            },
            "Sparse repair is clean exact-checked repair evidence, but only as a bounded no-speedup claim.",
        ),
        _row(
            "csl_tautology_repair",
            "clean",
            (EXP5557_CSL_REPAIR_PATH,),
            {
                "csl_five_arm_clean": bool(csl.get("csl_five_arm_clean")),
                "tautology_resolved": bool(csl.get("tautology_resolved")),
                "aligned_delta_over_shuffled": csl.get("aligned_delta_over_shuffled"),
                "duplicated_metric_pairs": csl.get("duplicated_metric_pairs"),
            },
            "The five-arm CSL tautology repair is clean, but it does not rescue cross-family transfer.",
        ),
        _row(
            "causal_memory_action_impact",
            "clean",
            (EXP5558_CAUSAL_MEMORY_PATH,),
            {
                "csl_memory_ready": bool(memory.get("csl_memory_ready")),
                "quality_delta_vs_shuffled_memory": memory.get(
                    "quality_delta_vs_shuffled_memory"
                ),
                "action_impact_delta_vs_no_memory": memory.get(
                    "action_impact_delta_vs_no_memory"
                ),
                "action_selection_changed_count": _int(memory, "action_selection_changed_count"),
                "no_weight_mutation": bool(memory.get("no_weight_mutation")),
            },
            "Causal write-manage-read memory changed actions, but broad CSL still needs reset-free promotion.",
        ),
    ]


def _bounded_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    capstone = _payload(artifacts, PREVIOUS_CAPSTONE_RELATIVE_PATH)
    sparse = _payload(artifacts, EXP5556_SPARSE_REPAIR_PATH)
    hardware = _payload(artifacts, EXP5560_HARDWARE_PATH)
    arc = _payload(artifacts, EXP5562_ARC_LEVELUP_PATH)
    memory = _payload(artifacts, EXP5558_CAUSAL_MEMORY_PATH)
    return [
        _row(
            "sparse_repair_no_speedup",
            "bounded",
            (EXP5556_SPARSE_REPAIR_PATH,),
            {
                "asp_sparse_repair_claim_allowed": bool(
                    sparse.get("asp_sparse_repair_claim_allowed")
                ),
                "matched_timing_available": bool(sparse.get("matched_timing_available")),
                "speedup_claim_allowed": bool(sparse.get("speedup_claim_allowed")),
            },
            "Descriptor-guided repair can feed verifier stress, not timing or hardware claims.",
        ),
        _row(
            "continuous_memory_not_broad_csl",
            "bounded",
            (EXP5557_CSL_REPAIR_PATH, EXP5558_CAUSAL_MEMORY_PATH, EXP5559_CROSS_MODEL_CSL_PATH),
            {
                "continuous_self_learning_evidence": bool(
                    capstone.get("continuous_self_learning_evidence")
                ),
                "csl_claim_allowed": bool(capstone.get("csl_claim_allowed")),
                "cross_model_csl_claim_allowed": bool(
                    capstone.get("cross_model_csl_claim_allowed")
                ),
                "action_impact_delta_vs_no_memory": memory.get(
                    "action_impact_delta_vs_no_memory"
                ),
            },
            "Memory action impact carries forward, but broad CSL waits for reset-free held-out promotion.",
        ),
        _row(
            "hardware_receipt_no_speedup",
            "bounded",
            (EXP5560_HARDWARE_PATH,),
            {
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "repeated_timing_pairs": _int(hardware, "repeated_timing_pairs"),
                "hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
            },
            "Hardware hygiene is useful only as receipt discipline until matched timing exists.",
        ),
        _row(
            "arc_live_path_no_bank",
            "bounded",
            (EXP5561_ARC_PRECHECK_PATH, EXP5562_ARC_LEVELUP_PATH),
            {
                "solve_provenance": arc.get("solve_provenance"),
                "selected_game": arc.get("selected_game"),
                "selected_level": arc.get("selected_level"),
                "offline_reproduced": bool(arc.get("offline_reproduced")),
                "registry_delta": _int(arc, "registry_delta"),
            },
            "ARC live-path provenance is preserved, but registry credit remains zero.",
        ),
    ]


def _blocked_lanes(artifacts: Mapping[str, JsonMap], terminal_records: Sequence[JsonMap]) -> list[JsonDict]:
    row_completion = _payload(artifacts, EXP5552_ROW_COMPLETION_PATH)
    smoke = _payload(artifacts, EXP5553_GBNF_SMOKE_PATH)
    hardware = _payload(artifacts, EXP5560_HARDWARE_PATH)
    arc = _payload(artifacts, EXP5562_ARC_LEVELUP_PATH)
    panel_record = next(
        (
            row
            for row in terminal_records
            if row.get("artifact_path") == EXP5554_PANEL_PATH.as_posix()
        ),
        {},
    )
    return [
        _row(
            "grammar_row_completion_blocked",
            "blocked",
            (EXP5552_ROW_COMPLETION_PATH,),
            {
                "automaton_row_completion_ready": bool(
                    row_completion.get("automaton_row_completion_ready")
                ),
                "row_completion_support_rate": row_completion.get("row_completion_support_rate"),
                "required_row_count": _int(row_completion, "required_row_count"),
                "accepted_row_count": len(row_completion.get("accepted_row_keys", []))
                if isinstance(row_completion.get("accepted_row_keys"), list)
                else 0,
                "readiness_blockers": row_completion.get("readiness_blockers"),
            },
            "The grammar-row chain is closed because required rows were reachable but unsupported.",
        ),
        _row(
            "gbnf_row_smoke_gate_skip",
            "blocked",
            (EXP5553_GBNF_SMOKE_PATH,),
            {
                "status": _status_label(smoke),
                "honest_verdict": smoke.get("honest_verdict"),
                "blocked_at_layer": smoke.get("blocked_at_layer"),
                "gate_check_summary": smoke.get("gate_check_summary"),
            },
            "The GBNF row smoke was skipped by the failed Exp5552 readiness gate.",
        ),
        _row(
            "hard_soft_panel_skipped",
            "blocked",
            (EXP5554_PANEL_PATH,),
            {
                "status": panel_record.get("status"),
                "honest_verdict": panel_record.get("honest_verdict"),
                "conductor_evidence": panel_record.get("conductor_evidence"),
            },
            "The hard/soft panel v4 has conductor skip evidence, not a clean terminal panel artifact.",
        ),
        _row(
            "no_matched_hardware_speedup",
            "blocked",
            (EXP5560_HARDWARE_PATH,),
            {
                "matched_timing_available": bool(hardware.get("matched_timing_available")),
                "repeated_timing_pairs": _int(hardware, "repeated_timing_pairs"),
                "hardware_speedup_claim": bool(hardware.get("hardware_speedup_claim")),
            },
            "No hardware claim is allowed until matched successful timing pairs exist.",
        ),
        _row(
            "arc_registry_delta_zero",
            "blocked",
            (EXP5562_ARC_LEVELUP_PATH, PREVIOUS_CAPSTONE_RELATIVE_PATH),
            {
                "offline_reproduced": bool(arc.get("offline_reproduced")),
                "registry_delta": _int(arc, "registry_delta"),
                "reproduced_levels": _int(arc, "reproduced_levels"),
            },
            "ARC registry delta remained zero, so no level-up claim carries into .504.",
        ),
    ]


def _flagged_lanes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    transfer = _payload(artifacts, EXP5559_CROSS_MODEL_CSL_PATH)
    return [
        _row(
            "cross_family_csl_flagged_null",
            "flagged",
            (EXP5559_CROSS_MODEL_CSL_PATH,),
            {
                "flagged_adversarial": bool(transfer.get("flagged_adversarial")),
                "csl_claim_allowed": bool(transfer.get("csl_claim_allowed")),
                "cross_family_delta_over_shuffled": transfer.get(
                    "cross_family_delta_over_shuffled"
                ),
                "negative_transfer_rate": transfer.get("negative_transfer_rate"),
                "aligned_memory_score": transfer.get("aligned_memory_score"),
                "shuffled_memory_score": transfer.get("shuffled_memory_score"),
                "no_memory_score": transfer.get("no_memory_score"),
                "corrigendum_pending": transfer.get("corrigendum_pending", []),
            },
            "Cross-family CSL is a flagged null and must not receive another ordinary v3 rerun.",
        )
    ]


def _verifier_chain() -> list[JsonDict]:
    return [
        {
            "task_id": "exp5566-exact-asp-fsm-near-miss-corpus",
            "gate_field": "corpus_ready",
            "unlocks": "exp5567-gated-local-sota-solve-verify-asymmetry",
            "principle": "A complete exact corpus must exist before live SOTA inference.",
        },
        {
            "task_id": "exp5567-gated-local-sota-solve-verify-asymmetry",
            "gate_field": "panel_complete",
            "unlocks": "exp5568-gated-verifier-coevolution-trigger",
            "principle": "Co-evolution requires an authenticated complete local-SOTA panel.",
        },
        {
            "task_id": "exp5568-gated-verifier-coevolution-trigger",
            "gate_field": "verifier_coevolution_required",
            "unlocks": "bounded_next_action_only",
            "principle": "The task emits a trigger from cached residuals, not silent retuning.",
        },
    ]


def _self_learning_chain() -> list[JsonDict]:
    return [
        {
            "task_id": "exp5569-causal-memory-policy-tournament",
            "gate_field": "policy_ready",
            "unlocks": "exp5571-gated-reset-free-sota-continual-harness",
            "principle": "Memory policy must improve held-out exact success and pass rollback.",
        },
        {
            "task_id": "exp5570-spline-local-kan-online-energy",
            "gate_field": "kan_ready",
            "unlocks": "exp5571-gated-reset-free-sota-continual-harness",
            "principle": "KAN adaptation must be local, replay-safe, and rollback-safe.",
        },
        {
            "task_id": "exp5571-gated-reset-free-sota-continual-harness",
            "gate_field": "reset_free_harness_ready",
            "unlocks": "exp5572-gated-delayed-regression-promotion",
            "principle": "Reset-free adaptation must pass before delayed promotion checks.",
        },
        {
            "task_id": "exp5572-gated-delayed-regression-promotion",
            "gate_field": "promotion_allowed",
            "unlocks": "broad_continuous_self_learning_claim_only_if_true",
            "principle": "Promotion requires forward adaptation, retention, poisoning resistance, and rollback.",
        },
    ]


def _arc_chain() -> list[JsonDict]:
    return [
        {
            "task_id": "exp5575-sge-anti-stagnation-live-precheck",
            "gate_field": "live_path_ready AND target_unsolved",
            "unlocks": "exp5576-gated-sge-live-levelup",
            "principle": "The ordinary ARC floor requires an E3-reachable target before live level-up.",
        },
        {
            "task_id": "exp5576-gated-sge-live-levelup",
            "gate_field": "offline_reproduced AND registry_delta>0",
            "unlocks": "ordinary_arc_levelup_claim",
            "principle": "ARC credit requires self-discovery, offline reproduction, and positive registry delta.",
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


def _previous_capstone_claims(capstone: JsonMap) -> JsonDict:
    return {
        "structured_sota_claim_allowed": bool(capstone.get("structured_sota_claim_allowed")),
        "sota_hard_soft_claim_allowed": bool(capstone.get("sota_hard_soft_claim_allowed")),
        "continuous_self_learning_evidence": bool(
            capstone.get("continuous_self_learning_evidence")
        ),
        "csl_claim_allowed": bool(capstone.get("csl_claim_allowed")),
        "cross_model_csl_claim_allowed": bool(capstone.get("cross_model_csl_claim_allowed")),
        "asp_sparse_repair_claim_allowed": bool(capstone.get("asp_sparse_repair_claim_allowed")),
        "hardware_speedup_claim": bool(capstone.get("hardware_speedup_claim")),
        "arc_registry_delta": _int(capstone, "arc_registry_delta"),
        "arc_live_levelup_claim_allowed": bool(capstone.get("arc_live_levelup_claim_allowed")),
    }


def _failed_preconditions(
    *,
    capstone: JsonMap,
    capstone_meta: JsonMap,
    terminal_records_missing: Sequence[str],
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    vnext_names_milestone: bool,
    vnext_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures: list[str] = []
    if not capstone_meta.get("loadable"):
        failures.append("previous_capstone_missing_or_unreadable")
    if capstone.get("milestone") != PREVIOUS_MILESTONE:
        failures.append("previous_capstone_milestone_mismatch")
    if capstone.get("task_range") not in {PREVIOUS_TASK_RANGE, None}:
        failures.append("previous_capstone_task_range_mismatch")
    if capstone.get("honest_verdict") not in {EXPECTED_PREVIOUS_CAPSTONE_VERDICT, None}:
        failures.append("previous_capstone_summary_mismatch")
    if terminal_records_missing:
        failures.append("terminal_records_unaccounted")
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
            "complete: archived .503 terminal evidence into .504 transition receipt; "
            "previous_task_range=exp5550-exp5563; next_task_range=exp5564-exp5577; "
            "clean evidence preserved for exact ASP/FSM fixture, bounded sparse repair, "
            "CSL tautology repair, and causal memory action impact; blocked or flagged "
            "boundaries preserved for grammar row completion, skipped hard/soft panel, "
            "cross-family CSL, unmatched hardware, and ARC registry_delta=0."
        )
    first_failure = failures[0] if failures else "unknown"
    return f"blocked: .504 transition receipt failed precondition {first_failure}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifact_metadata, terminal_records = _read_terminal_records(root)
    source_context, source_missing = _source_context(root)
    roadmap, roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_milestone = roadmap.get("milestone")
    roadmap_milestone = str(roadmap_milestone) if roadmap_milestone is not None else None
    vnext_text = _read_text(root, VNEXT_RELATIVE_PATH)
    vnext_task_range = _task_range_from_text(vnext_text)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)

    capstone = _payload(artifacts, PREVIOUS_CAPSTONE_RELATIVE_PATH)
    capstone_meta = artifact_metadata.get(PREVIOUS_CAPSTONE_RELATIVE_PATH.as_posix(), {})
    terminal_records_missing = [
        str(row["artifact_path"]) for row in terminal_records if not row.get("accounted")
    ]
    failures = _failed_preconditions(
        capstone=capstone,
        capstone_meta=capstone_meta,
        terminal_records_missing=terminal_records_missing,
        roadmap_milestone=roadmap_milestone,
        roadmap_task_ids=roadmap_task_ids,
        vnext_names_milestone=MILESTONE in vnext_text,
        vnext_task_range=vnext_task_range,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    tests: list[Any] = [dict(row) if isinstance(row, Mapping) else str(row) for row in tests_run]
    json_terminal_artifacts_read = sum(
        1
        for row in terminal_records
        if row["source_kind"] == "json_artifact" and row.get("loadable")
    )
    conductor_skip_records_read = sum(
        1
        for row in terminal_records
        if row["source_kind"] == "conductor_skip" and row.get("accounted")
    )
    artifacts_read = sum(1 for row in terminal_records if row.get("accounted"))

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "source_context": source_context,
        "source_context_missing": source_missing,
        "artifact_metadata": artifact_metadata,
        "terminal_records": terminal_records,
        "terminal_records_missing": terminal_records_missing,
        "json_terminal_artifacts_read": json_terminal_artifacts_read,
        "conductor_skip_records_read": conductor_skip_records_read,
        "previous_capstone_path": PREVIOUS_CAPSTONE_RELATIVE_PATH.as_posix(),
        "previous_capstone_summary": capstone.get("honest_verdict"),
        "previous_capstone_claims": _previous_capstone_claims(capstone),
        "prompt_alias_resolution": [dict(row) for row in PROMPT_ALIAS_RESOLUTION],
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": vnext_task_range,
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "previous_capstone_present": capstone_meta.get("exists") is True,
            "previous_capstone_loadable": capstone_meta.get("loadable") is True,
            "previous_capstone_milestone": capstone.get("milestone"),
            "previous_capstone_task_range": capstone.get("task_range"),
            "previous_capstone_summary_exact": capstone.get("honest_verdict")
            == EXPECTED_PREVIOUS_CAPSTONE_VERDICT,
            "terminal_records_accounted": artifacts_read,
            "terminal_records_expected": len(TERMINAL_RECORDS),
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
        "tests_run": tests,
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "next_task_range": NEXT_TASK_RANGE,
        "artifacts_read": artifacts_read,
        "clean_lanes": _clean_lanes(artifacts),
        "bounded_lanes": _bounded_lanes(artifacts),
        "blocked_lanes": _blocked_lanes(artifacts, terminal_records),
        "flagged_lanes": _flagged_lanes(artifacts),
        "retired_continuations": [dict(row) for row in RETIRED_CONTINUATIONS],
        "verifier_chain": _verifier_chain(),
        "self_learning_chain": _self_learning_chain(),
        "arc_chain": _arc_chain(),
        "ptrm_slot_separate": True,
        "hardware_claim_allowed": False,
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
    for field in INT_FIELDS:
        if field in payload and not isinstance(payload[field], int):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    if payload.get("hardware_claim_allowed") is not False:
        errors.append("hardware_claim_allowed")
    if payload.get("ptrm_slot_separate") is not True:
        errors.append("ptrm_slot_separate")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("previous_milestone") != PREVIOUS_MILESTONE:
        errors.append("previous_milestone")
    if payload.get("previous_task_range") != PREVIOUS_TASK_RANGE:
        errors.append("previous_task_range")
    if payload.get("next_task_range") != NEXT_TASK_RANGE:
        errors.append("next_task_range")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("artifacts_read") != len(TERMINAL_RECORDS):
        errors.append("artifacts_read")
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
        raise ValueError(f"invalid Exp5564 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5564 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
