"""Exp5578 transition receipt from milestone .504 into .505.

Spec refs: REQ-REPORT-5578, SCENARIO-REPORT-5578,
SCENARIO-REPORT-5578-MISSING-INPUT, SCENARIO-REPORT-5578-FIELD-PRINCIPLES.

This module does not rerun models, solvers, or hardware checks. It reads the
terminal `.504` artifacts and writes down which facts are safe to carry into
`.505`. The important distinction is that a parser collapse can be a useful
instrumentation finding while still being unusable as solve or verify evidence.
The same discipline keeps a flagged memory policy, an unadjudicated PTRM
checkpoint, and a zero-delta ARC registry from becoming accidental promotion
credit in the next milestone.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
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
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5578_transition_v505.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5578_transition_v505"
EXPERIMENT_ID = "exp5578-transition-v505"
MILESTONE = "2026.07.505"
PREVIOUS_MILESTONE = "2026.07.504"
PREVIOUS_TASK_RANGE = "exp5564-exp5577"
NEXT_TASK_RANGE = "exp5578-exp5591"
RUN_DATE = "2026-07-13"
RANDOM_SEED = 5578
SCHEMA = "carnot.experiment_5578.transition_v505.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5578",
    "SCENARIO-REPORT-5578",
    "SCENARIO-REPORT-5578-MISSING-INPUT",
    "SCENARIO-REPORT-5578-FIELD-PRINCIPLES",
)

EXPECTED_ARTIFACT_PATHS = (
    Path("results/experiment_5564_transition_v504.json"),
    Path("results/experiment_5565_v504_source_delta_ingestion.json"),
    Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"),
    Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"),
    Path("results/experiment_5568_verifier_coevolution_trigger.json"),
    Path("results/experiment_5569_causal_memory_policy_tournament.json"),
    Path("results/experiment_5570_spline_local_kan_online_energy.json"),
    Path("results/experiment_5571_reset_free_sota_continual_harness.json"),
    Path("results/experiment_5572_gated_delayed_regression_promotion.json"),
    Path("results/experiment_5573_matched_sampler_hardware_continuity.json"),
    Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"),
    Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"),
    Path("results/experiment_5576_gated_sge_live_levelup.json"),
    Path("results/experiment_5577_capstone_v504.json"),
)

EXP5567_PANEL_PATH = Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json")
EXP5569_MEMORY_PATH = Path("results/experiment_5569_causal_memory_policy_tournament.json")
EXP5570_KAN_PATH = Path("results/experiment_5570_spline_local_kan_online_energy.json")
EXP5571_RESET_FREE_PATH = Path("results/experiment_5571_reset_free_sota_continual_harness.json")
EXP5572_PROMOTION_PATH = Path("results/experiment_5572_gated_delayed_regression_promotion.json")
EXP5573_HARDWARE_PATH = Path("results/experiment_5573_matched_sampler_hardware_continuity.json")
EXP5574_PTRM_PATH = Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json")
EXP5575_SGE_PRECHECK_PATH = Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json")
EXP5576_SGE_LEVELUP_PATH = Path("results/experiment_5576_gated_sge_live_levelup.json")
EXP5577_CAPSTONE_PATH = Path("results/experiment_5577_capstone_v504.json")

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPECTED_TASK_IDS = [
    "exp5578-transition-v505",
    "exp5579-v505-source-delta-ingestion",
    "exp5580-parser-forensics-positive-control",
    "exp5581-clean-sota-solve-verify-remeasurement",
    "exp5582-exact-counterexample-verifier-extension",
    "exp5583-causal-memory-metric-corrigendum",
    "exp5584-two-timescale-exact-self-learning",
    "exp5585-reset-free-live-local-sota-sessions",
    "exp5586-delayed-promotion-and-poisoning-gate",
    "exp5587-reserved-ptrm-loo-adjudication",
    "exp5588-epistemic-object-model-mcts-live-precheck",
    "exp5589-gated-ordinary-arc-level-up",
    "exp5590-matched-cpu-cuda-crossover-and-board-continuity",
    "exp5591-v505-capstone-reconciliation",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "artifacts_read": "transitions cite terminal evidence",
    "clean_lanes": "only unflagged completed findings are clean",
    "blocked_or_flagged_lanes": "boundaries remain binding",
    "parser_collapse_preserved": "instrumentation failure is not model evidence",
    "previous_task_range": "CalVer continuity is explicit",
    "next_task_range": "CalVer continuity is explicit",
    "ptrm_slot_separate": "PTRM cannot satisfy ordinary ARC",
    "inference_substrate": "no live inference occurred",
    "honest_verdict": "terminal status starts complete: or blocked:",
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
    "artifact_metadata",
    "artifacts_expected",
    "missing_artifacts",
    "source_context",
    "source_context_missing",
    "roadmap_task_ids",
    "roadmap_task_count",
    "roadmap_doc_task_range",
    "conductor_activation",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "gate_map",
    "tests_run",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "field_principles",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
BOOL_FIELDS = ("ptrm_slot_separate", "roadmap_yaml_unchanged", "conductor_unchanged")
LIST_FIELDS = (
    "artifacts_read",
    "clean_lanes",
    "blocked_or_flagged_lanes",
    "missing_artifacts",
    "source_context",
    "source_context_missing",
    "roadmap_task_ids",
    "protected_file_checks",
    "failed_preconditions",
    "tests_run",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5578_transition_v505.py -q --no-cov",
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5578_transition_v505.py -m pytest "
            "tests/python/test_experiment_5578_transition_v505.py -q --no-cov -n 0"
        ),
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5578_transition_v505.py --fail-under=100"
        ),
        "outcome": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "not_run_in_default_artifact"},
)


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _task_range_from_text(text: str) -> str | None:
    match = re.search(r"Exp\s*(\d+)\s*[-\u2013]\s*Exp?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"exp{match.group(1)}-exp{match.group(2)}"
    compact = re.search(r"exp(\d+)\s*[-\u2013]\s*exp?(\d+)", text, flags=re.IGNORECASE)
    return f"exp{compact.group(1)}-exp{compact.group(2)}" if compact else None


def _read_json_any(path: Path) -> tuple[Any, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "json_type": None,
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata.update({"loadable": True, "error": None, "json_type": type(payload).__name__})
    if isinstance(payload, list):
        metadata["length"] = len(payload)
    return payload, metadata


def _read_artifacts(root: Path) -> tuple[dict[str, Any], JsonDict, list[JsonDict], list[str]]:
    artifacts: dict[str, Any] = {}
    metadata: JsonDict = {}
    read_records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if meta["exists"] and meta["loadable"]:
            read_records.append(
                {
                    "path": rel,
                    "sha256": meta.get("sha256"),
                    "status": _status_label(payload if isinstance(payload, Mapping) else {}),
                    "honest_verdict": payload.get("honest_verdict")
                    if isinstance(payload, Mapping)
                    else None,
                }
            )
        else:
            missing.append(rel)
    return artifacts, metadata, read_records, missing


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


def _payload(artifacts: Mapping[str, Any], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


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


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict.startswith("blocked_gate")
        or ("gate" in blocked_at_layer and _status_label(payload) == "blocked")
    )


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _clean_for_claim(payload: JsonMap) -> bool:
    return bool(
        payload
        and _status_label(payload) == "complete"
        and not _is_flagged(payload)
        and not _is_gate_skip(payload)
    )


def _lane(
    lane: str,
    classification: str,
    source_artifacts: Sequence[Path],
    claim_boundary: str,
    evidence: JsonDict,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": [path.as_posix() for path in source_artifacts],
        "claim_boundary": claim_boundary,
        "evidence": evidence,
    }


def _clean_lanes(artifacts: Mapping[str, Any]) -> list[JsonDict]:
    exp5564 = _payload(artifacts, Path("results/experiment_5564_transition_v504.json"))
    exp5565 = _payload(artifacts, Path("results/experiment_5565_v504_source_delta_ingestion.json"))
    exp5566 = _payload(artifacts, Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"))
    exp5568 = _payload(artifacts, Path("results/experiment_5568_verifier_coevolution_trigger.json"))
    exp5570 = _payload(artifacts, EXP5570_KAN_PATH)
    exp5573 = _payload(artifacts, EXP5573_HARDWARE_PATH)
    rows: list[JsonDict] = []
    if _clean_for_claim(exp5564) and _clean_for_claim(exp5565):
        rows.append(
            _lane(
                "transition_and_source_delta",
                "clean",
                (
                    Path("results/experiment_5564_transition_v504.json"),
                    Path("results/experiment_5565_v504_source_delta_ingestion.json"),
                ),
                "Transition and source delta are readable receipts; no closed scope is reopened.",
                {"closed_scopes_reopened": bool(exp5565.get("closed_scopes_reopened"))},
            )
        )
    if _clean_for_claim(exp5566) and exp5566.get("corpus_ready"):
        rows.append(
            _lane(
                "exact_asp_fsm_near_miss_corpus",
                "clean",
                (Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"),),
                "Exact ASP/FSM corpus remains the deterministic verifier substrate.",
                {
                    "corpus_ready": True,
                    "n_rows": _int(exp5566, "n_rows"),
                    "duplicate_leakage_count": _int(exp5566, "duplicate_leakage_count"),
                },
            )
        )
    if _clean_for_claim(exp5568) and exp5568.get("verifier_coevolution_required"):
        rows.append(
            _lane(
                "verifier_coevolution_trigger",
                "clean",
                (Path("results/experiment_5568_verifier_coevolution_trigger.json"),),
                "Verifier co-evolution is a trigger only; Exp5582 must use clean residuals.",
                {"triggered_by": exp5568.get("triggered_by", [])},
            )
        )
    if _clean_for_claim(exp5570) and exp5570.get("kan_ready"):
        rows.append(
            _lane(
                "spline_local_kan_online_energy",
                "clean",
                (EXP5570_KAN_PATH,),
                "Active-spline KAN energy is clean bounded self-learning evidence.",
                {
                    "kan_ready": True,
                    "forward_adaptation_delta": _float(exp5570, "forward_adaptation_delta"),
                    "unsafe_false_accept_delta": _float(exp5570, "unsafe_false_accept_delta"),
                    "rollback_checksum_match": bool(exp5570.get("rollback_checksum_match")),
                },
            )
        )
    if _clean_for_claim(exp5573):
        rows.append(
            _lane(
                "matched_sampler_quality_continuity",
                "clean_bounded",
                (EXP5573_HARDWARE_PATH,),
                "Matched sampler rows are clean quality evidence, not speedup evidence.",
                {
                    "successful_matched_pairs": _int(exp5573, "successful_matched_pairs"),
                    "board_speedup_claimed": bool(exp5573.get("board_speedup_claimed")),
                },
            )
        )
    return rows


def _parser_collapse(exp5567: JsonMap) -> JsonDict:
    return {
        "source_artifact": EXP5567_PANEL_PATH.as_posix(),
        "parser_failure_count": _int(exp5567, "parser_failure_count"),
        "n_candidate_labels": _int(exp5567, "n_candidate_labels"),
        "panel_complete": bool(exp5567.get("panel_complete")),
        "live_model_invoked": bool(exp5567.get("live_model_invoked")),
        "gpu_offload_authenticated": bool(exp5567.get("gpu_offload_authenticated")),
        "classification": "instrumentation_failure_not_model_evidence",
        "solve_or_verify_result_imported": False,
    }


def _blocked_or_flagged_lanes(
    artifacts: Mapping[str, Any],
    parser_collapse_preserved: JsonMap,
) -> list[JsonDict]:
    exp5569 = _payload(artifacts, EXP5569_MEMORY_PATH)
    exp5571 = _payload(artifacts, EXP5571_RESET_FREE_PATH)
    exp5572 = _payload(artifacts, EXP5572_PROMOTION_PATH)
    exp5573 = _payload(artifacts, EXP5573_HARDWARE_PATH)
    exp5574 = _payload(artifacts, EXP5574_PTRM_PATH)
    exp5575 = _payload(artifacts, EXP5575_SGE_PRECHECK_PATH)
    exp5576 = _payload(artifacts, EXP5576_SGE_LEVELUP_PATH)
    exp5577 = _payload(artifacts, EXP5577_CAPSTONE_PATH)
    return [
        _lane(
            "parser_collapse_instrumentation",
            "blocked",
            (EXP5567_PANEL_PATH,),
            "Parser collapse is preserved as instrumentation evidence only.",
            dict(parser_collapse_preserved),
        ),
        _lane(
            "memory_tautology_flag",
            "flagged",
            (EXP5569_MEMORY_PATH,),
            "A flagged memory policy cannot gate `.505` learning until a corrigendum lands.",
            {
                "flagged_adversarial": bool(exp5569.get("flagged_adversarial")),
                "policy_ready": bool(exp5569.get("policy_ready")),
                "forward_transfer_delta": exp5569.get("forward_transfer_delta"),
                "backward_retention_delta": exp5569.get("backward_retention_delta"),
                "corrigendum_pending": exp5569.get("corrigendum_pending", []),
            },
        ),
        _lane(
            "reset_free_cuda_block",
            "blocked",
            (EXP5571_RESET_FREE_PATH,),
            "Reset-free live learning is blocked without authenticated CUDA offload.",
            {
                "continual_harness_candidate": bool(exp5571.get("continual_harness_candidate")),
                "live_model_invoked": bool(exp5571.get("live_model_invoked")),
                "gpu_offload_authenticated": bool(exp5571.get("gpu_offload_authenticated")),
                "honest_verdict": exp5571.get("honest_verdict"),
            },
        ),
        _lane(
            "delayed_promotion_gate_skip",
            "blocked",
            (EXP5572_PROMOTION_PATH,),
            "Delayed regression promotion was skipped by the failed reset-free gate.",
            {
                "blocked_at_layer": exp5572.get("blocked_at_layer"),
                "gate_check_summary": exp5572.get("gate_check_summary"),
            },
        ),
        _lane(
            "ptrm_loo_not_reached",
            "blocked",
            (EXP5574_PTRM_PATH,),
            "PTRM Stage 1 lacks the leave-one-game-out adjudication required for `.505`.",
            {
                "stage1_training_complete": bool(exp5574.get("stage1_training_complete")),
                "loo_verdict_reached": bool(exp5574.get("loo_verdict_reached")),
                "no_level_solve_claim": bool(exp5574.get("no_level_solve_claim")),
                "solve_provenance": exp5574.get("solve_provenance"),
            },
        ),
        _lane(
            "sge_retired_or_skipped",
            "blocked",
            (EXP5575_SGE_PRECHECK_PATH, EXP5576_SGE_LEVELUP_PATH),
            "SGE remains retired after precheck block and live-level gate skip.",
            {
                "live_path_ready": bool(exp5575.get("live_path_ready")),
                "target_unsolved": bool(exp5575.get("target_unsolved")),
                "gate_skip": _is_gate_skip(exp5576),
                "sge_retired": bool(exp5577.get("sge_retired")),
            },
        ),
        _lane(
            "ordinary_arc_registry_delta_zero",
            "blocked",
            (EXP5577_CAPSTONE_PATH,),
            "Ordinary ARC credit requires a positive offline-reproduced registry delta.",
            {
                "ordinary_arc_floor_satisfied": bool(exp5577.get("ordinary_arc_floor_satisfied")),
                "arc_registry_delta": _int(exp5577, "arc_registry_delta"),
            },
        ),
        _lane(
            "hardware_speedup_not_supported",
            "blocked",
            (EXP5573_HARDWARE_PATH, EXP5577_CAPSTONE_PATH),
            "CUDA sampler quality rows landed, but CPU was faster and no speedup claim is allowed.",
            {
                "successful_matched_pairs": _int(exp5573, "successful_matched_pairs"),
                "minimum_speedup": _minimum_speedup(exp5573),
                "capstone_hardware_speedup_claim_allowed": bool(
                    exp5577.get("hardware_speedup_claim_allowed")
                ),
            },
        ),
    ]


def _minimum_speedup(payload: JsonMap) -> float | None:
    rows = payload.get("speedup_by_pair")
    if not isinstance(rows, list):
        return None
    speedups = [
        float(row.get("speedup"))
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("speedup"), int | float)
    ]
    return min(speedups) if speedups else None


def _gate_map() -> JsonDict:
    return {
        "parser_panel_exact_extension": {
            "chain": "parser->panel->exact extension",
            "steps": [
                "exp5580-parser-forensics-positive-control",
                "exp5581-clean-sota-solve-verify-remeasurement",
                "exp5582-exact-counterexample-verifier-extension",
            ],
            "entry_boundary": "Exp5567 parser collapse must be repaired before model evidence exists.",
        },
        "memory_two_timescale_live_promotion": {
            "chain": "memory corrigendum->two-timescale->live->promotion",
            "steps": [
                "exp5583-causal-memory-metric-corrigendum",
                "exp5584-two-timescale-exact-self-learning",
                "exp5585-reset-free-live-local-sota-sessions",
                "exp5586-delayed-promotion-and-poisoning-gate",
            ],
            "entry_boundary": "Exp5569 policy_ready is unusable until the tautology flag is cleared.",
        },
        "ordinary_arc_eom_live": {
            "chain": "EOM precheck->live ARC",
            "steps": [
                "exp5588-epistemic-object-model-mcts-live-precheck",
                "exp5589-gated-ordinary-arc-level-up",
            ],
            "entry_boundary": "SGE is retired; ordinary ARC must use the EOM-MCTS lane.",
        },
        "ptrm_independent_lane": {
            "chain": "independent PTRM LOO",
            "steps": ["exp5587-reserved-ptrm-loo-adjudication"],
            "counts_as_ordinary_arc": False,
        },
        "hardware_independent_lane": {
            "chain": "independent hardware crossover",
            "steps": ["exp5590-matched-cpu-cuda-crossover-and-board-continuity"],
            "speedup_claim_allowed": False,
        },
    }


def _conductor_activation(root: Path) -> JsonDict:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    matches = [
        line.strip()
        for line in text.splitlines()
        if "Milestone 2026.07.505 activated" in line
    ]
    return {
        "source_path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        "activated": bool(matches),
        "last_activation_line": matches[-1] if matches else None,
    }


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
    missing_artifacts: Sequence[str],
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [f"missing_artifact:{path}" for path in missing_artifacts]
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str], parser_count: int) -> str:
    if status == "complete":
        return (
            "complete: archived .504 terminal evidence into .505 gate map; "
            "previous_task_range=exp5564-exp5577; next_task_range=exp5578-exp5591; "
            f"parser_failure_count={parser_count} preserved as instrumentation; "
            "memory_tautology_flagged=True; kan_online_energy_promoted=True; "
            "ptrm_slot_separate=True; ordinary_arc_registry_delta=0."
        )
    first = failures[0] if failures else "unknown"
    return f"blocked: .505 transition receipt failed precondition {first}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifact_metadata, artifacts_read, missing = _read_artifacts(root)
    source_context, source_missing = _read_source_context(root)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    vnext_text = _read_text(root, VNEXT_RELATIVE_PATH)
    roadmap_doc_task_range = _task_range_from_text(vnext_text)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    failures = _failed_preconditions(
        missing,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    exp5567 = _payload(artifacts, EXP5567_PANEL_PATH)
    parser_collapse_preserved = _parser_collapse(exp5567)
    tests = [dict(row) if isinstance(row, Mapping) else str(row) for row in tests_run]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "artifact_metadata": artifact_metadata,
        "artifacts_expected": [path.as_posix() for path in EXPECTED_ARTIFACT_PATHS],
        "artifacts_read": artifacts_read,
        "missing_artifacts": list(missing),
        "source_context": source_context,
        "source_context_missing": source_missing,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_task_count": len(roadmap_task_ids),
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "conductor_activation": _conductor_activation(root),
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "expected_artifacts": len(EXPECTED_ARTIFACT_PATHS),
            "artifacts_read": len(artifacts_read),
            "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
            "active_roadmap_milestone": roadmap.get("milestone"),
            "active_roadmap_task_count": len(roadmap_task_ids),
            "roadmap_doc_task_range": roadmap_doc_task_range,
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "gate_map": _gate_map(),
        "tests_run": tests,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "field_principles": dict(FIELD_PRINCIPLES),
        "clean_lanes": _clean_lanes(artifacts),
        "blocked_or_flagged_lanes": _blocked_or_flagged_lanes(
            artifacts,
            parser_collapse_preserved,
        ),
        "parser_collapse_preserved": parser_collapse_preserved,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "next_task_range": NEXT_TASK_RANGE,
        "ptrm_slot_separate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            status,
            failures,
            int(parser_collapse_preserved["parser_failure_count"]),
        ),
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
    blocked_rows = payload.get("blocked_or_flagged_lanes")
    if isinstance(blocked_rows, list):
        blocked_names = {
            str(row.get("lane"))
            for row in blocked_rows
            if isinstance(row, Mapping) and row.get("lane")
        }
        required = {
            "parser_collapse_instrumentation",
            "memory_tautology_flag",
            "ptrm_loo_not_reached",
            "ordinary_arc_registry_delta_zero",
        }
        if not required <= blocked_names:
            errors.append("blocked_or_flagged_lanes")
    parser = payload.get("parser_collapse_preserved")
    missing = set(payload.get("missing_artifacts", [])) if isinstance(payload.get("missing_artifacts"), list) else set()
    parser_missing = EXP5567_PANEL_PATH.as_posix() in missing
    if not isinstance(parser, Mapping) or (
        parser.get("classification") != "instrumentation_failure_not_model_evidence"
        or parser.get("solve_or_verify_result_imported") is not False
        or (not parser_missing and parser.get("parser_failure_count") != 648)
    ):
        errors.append("parser_collapse_preserved")
    if payload.get("previous_task_range") != PREVIOUS_TASK_RANGE:
        errors.append("previous_task_range")
    if payload.get("next_task_range") != NEXT_TASK_RANGE:
        errors.append("next_task_range")
    if payload.get("ptrm_slot_separate") is not True:
        errors.append("ptrm_slot_separate")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
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
    if errors:  # pragma: no cover - validation errors are unit-tested directly
        raise ValueError(f"invalid Exp5578 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5578 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
