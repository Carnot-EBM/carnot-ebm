"""Exp 5168: archive .473 and activate the .474 frame.

Spec refs: REQ-REPORT-5168, SCENARIO-REPORT-5168,
SCENARIO-REPORT-5168-DIRTY-RUNTIME.

This module is record-only. It reads the completed .473 artifacts, corrects the
stale Exp 5161 capstone exclusion against the live artifact state, checks the
runtime handoff, and writes the .474 activation artifact. It does not edit the
roadmap or the research conductor.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.experiment_5150_archive_471_activate_472 import (
    CommandResult,
    RuntimeSnapshot,
    _bool,
    _dirty_paths,
    _list,
    _mapping,
    _process_row,
    capture_runtime_snapshot,
    file_sha256,
    payload_checksum,
    read_json_mapping,
    run_adversarial_verification,
    verification_payload,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
VerificationRunner = Callable[[Path], CommandResult]
RuntimeProbe = Callable[[Path], RuntimeSnapshot]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5168_archive_473_activate_474.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")

EXPERIMENT = "experiment_5168_archive_473_activate_474"
EXPERIMENT_ID = "exp5168-archive-473-activate-474"
ARCHIVED_MILESTONE = "2026.07.473"
MILESTONE = "2026.07.474"
SCHEMA = "carnot.experiment_5168_archive_473_activate_474.v1"
RANDOM_SEED = 5168
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_473_closed_474_active_runtime_clean_exp5161_unquarantined"
DIRTY_HANDOFF_VERDICT = "complete_archive_473_closed_474_activation_gated_dirty_handoff"
ACTIVATION_GATED_VERDICT = "complete_archive_473_closed_474_activation_gated_roadmap_or_ops_drift"
MISSING_INPUTS_VERDICT = "complete_archive_473_closed_474_activation_gated_missing_inputs"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5168, 5181))
SPEC_REFS = [
    "REQ-REPORT-5168",
    "SCENARIO-REPORT-5168",
    "SCENARIO-REPORT-5168-DIRTY-RUNTIME",
]

V473_RESULT_PATHS: dict[int, Path] = {
    5156: Path("results/experiment_5156_archive_472_activate_473.json"),
    5157: Path("results/experiment_5157_deepen_warmstart_replay_ablation_v473.json"),
    5158: Path("results/experiment_5158_deepen_goal_energy_ranker_replay_v473.json"),
    5159: Path("results/experiment_5159_deepen_live_levelup_attempt_v473.json"),
    5160: Path("results/experiment_5160_oracle_distinct_cross_corpus_closure_v473.json"),
    5161: Path("results/experiment_5161_gap4_protocol_execution_pilot_v473.json"),
    5162: Path("results/experiment_5162_sota_ingestion_multilevel_v473.json"),
    5163: Path("results/experiment_5163_mmlu_pro_verifier_rescale_v473.json"),
    5164: Path("results/experiment_5164_retro_timing_falsezero_fix_v473.json"),
    5165: Path("results/experiment_5165_generation_axis_retirement_hygiene_v473.json"),
    5166: Path("results/experiment_5166_hardware_continuity_board_timing_v473.json"),
    5167: Path("results/experiment_5167_capstone_v473.json"),
}

TASK_META: dict[int, JsonDict] = {
    5156: {
        "experiment_id": "exp5156-archive-472-activate-473",
        "axis": "transition",
        "label": "archive_472_activate_473",
    },
    5157: {
        "experiment_id": "exp5157-deepen-warmstart-replay-ablation-v473",
        "axis": "deepen_wall",
        "label": "warmstart_replay_ablation",
    },
    5158: {
        "experiment_id": "exp5158-deepen-goal-energy-ranker-replay-v473",
        "axis": "deepen_wall",
        "label": "goal_energy_ranker_replay",
    },
    5159: {
        "experiment_id": "exp5159-deepen-live-levelup-attempt-v473",
        "axis": "live_levelup",
        "label": "gated_live_levelup_attempt",
    },
    5160: {
        "experiment_id": "exp5160-oracle-distinct-cross-corpus-closure-v473",
        "axis": "oracle_distinct_cross_corpus",
        "label": "set_encoder_cross_corpus_closure",
    },
    5161: {
        "experiment_id": "exp5161-gap4-protocol-execution-pilot-v473",
        "axis": "gap4_protocol",
        "label": "gap4_forward_protocol_pilot",
    },
    5162: {
        "experiment_id": "exp5162-sota-ingestion-multilevel-v473",
        "axis": "sota_ingestion",
        "label": "multilevel_sota_ingestion",
    },
    5163: {
        "experiment_id": "exp5163-mmlu-pro-verifier-rescale-v473",
        "axis": "phase_d_off_arc",
        "label": "mmlu_pro_verifier_rescale",
    },
    5164: {
        "experiment_id": "exp5164-retro-timing-falsezero-fix-v473",
        "axis": "ops_timing",
        "label": "retro_timing_falsezero_fix",
    },
    5165: {
        "experiment_id": "exp5165-generation-axis-retirement-hygiene-v473",
        "axis": "retirement_hygiene",
        "label": "generation_axis_retirement_hygiene",
    },
    5166: {
        "experiment_id": "exp5166-hardware-continuity-board-timing-v473",
        "axis": "hardware_continuity",
        "label": "board_timing_continuity",
    },
    5167: {
        "experiment_id": "exp5167-capstone-v473",
        "axis": "capstone",
        "label": "capstone_v473",
    },
}

EXPECTED_TRANSITION_DIRTY_PATHS = {
    "openspec/capabilities/research-reporting/spec.md",
    "python/carnot/experiment_5168_archive_473_activate_474.py",
    "tests/python/test_experiment_5168_archive_473_activate_474.py",
    "results/experiment_5168_archive_473_activate_474.json",
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "task_verdicts",
    "milestone_archive_summary",
    "v473_runtime_clean",
    "runtime_clean_details",
    "exp5161_unquarantine_noted",
    "capstone_stale_exclusions_corrected",
    "arc_registry_reconciliation",
    "gap4_status_reconciliation",
    "active_roadmap_ready",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "archived_milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "adversarial_verification",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ per Verdict "
        "Terminal-Prefix Discipline."
    ),
    "inference_substrate": (
        "This task reads/reconciles upstream JSON; it does not invoke an LLM or run new compute."
    ),
    "duration_s": "timing accountability",
    "source_artifacts_read": "evidence provenance",
    "task_verdicts": ".473 task truth without rounding nulls, blocked gates, or corrected flags",
    "milestone_archive_summary": (
        "archive summary must preserve real wins, honest nulls, blocked gates, and stale capstone "
        "corrections precisely"
    ),
    "v473_runtime_clean": (
        "Downstream .474 tasks may gate on this; a dirty handoff should block, not silently proceed."
    ),
    "runtime_clean_details": "handoff diagnostics must explain the gate, not just emit a boolean",
    "exp5161_unquarantine_noted": (
        "The capstone's own summary is stale on this one point (recorded pre-fix); .474 must not "
        "inherit a fabrication exclusion that was already corrected same-day."
    ),
    "capstone_stale_exclusions_corrected": (
        "capstone exclusions must be reconciled against live artifact state before .474 planning "
        "inherits them"
    ),
    "arc_registry_reconciliation": (
        "ARC level totals must stay at the measured 69/24 unless a live level-up artifact proves "
        "otherwise"
    ),
    "gap4_status_reconciliation": (
        "GAP-4 remains open until the significance and decentralization bars are actually met"
    ),
    "active_roadmap_ready": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5168_archive_473_activate_474.py -q "
    "-o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5168_archive_473_activate_474.py' "
    "-m pytest tests/python/test_experiment_5168_archive_473_activate_474.py -q --no-cov "
    "-o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5168_archive_473_activate_474.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5168_archive_473_activate_474.py",
    ".venv/bin/pytest tests/python -q",
]


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return _unwrap(value.get("value"))
    return value


def _as_bool(value: Any) -> bool:
    return _unwrap(value) is True


def _as_int(value: Any, default: int = 0) -> int:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    raw = _unwrap(value)
    if isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    raw = _unwrap(value)
    return _list(raw)


def _as_str(value: Any) -> str:
    raw = _unwrap(value)
    return str(raw if raw is not None else "")


def _task_prefixes_present(task_ids: Sequence[str], prefixes: Sequence[str]) -> bool:
    return all(any(task_id.startswith(prefix) for task_id in task_ids) for prefix in prefixes)


def _roadmap_check(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(ACTIVE_ROADMAP_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "milestone": "missing",
        "task_ids": [],
        "required_task_ids_present": False,
        "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
        "ready": False,
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "milestone": "yaml_poison", "error": str(exc)}
    mapping = _mapping(loaded)
    task_ids = [
        str(_mapping(task).get("id", ""))
        for task in _list(mapping.get("tasks"))
        if isinstance(_mapping(task).get("id", ""), str)
    ]
    missing = [
        prefix
        for prefix in REQUIRED_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    milestone = str(mapping.get("milestone", "unknown"))
    return {
        **base,
        "exists": True,
        "parses": True,
        "milestone": milestone,
        "task_ids": task_ids,
        "required_task_ids_present": _task_prefixes_present(task_ids, REQUIRED_TASK_PREFIXES),
        "missing_required_task_prefixes": missing,
        "ready": milestone == MILESTONE and not missing,
    }


def _vnext_check(path: Path) -> JsonDict:
    base = {
        "path": str(VNEXT_RELATIVE_PATH),
        "exists": path.exists(),
        "milestone": "missing",
        "predecessor": "missing",
        "mentions_exp5161_correction": False,
        "ready": False,
    }
    if not path.exists():
        return base
    text = path.read_text(encoding="utf-8")
    milestone_ok = f"**Milestone:** `{MILESTONE}`" in text or MILESTONE in text
    predecessor_ok = f"**Predecessor:** `{ARCHIVED_MILESTONE}`" in text or ARCHIVED_MILESTONE in text
    correction = "exp5161" in text and "un-quarantined" in text
    return {
        **base,
        "exists": True,
        "milestone": MILESTONE if milestone_ok else "mismatch",
        "predecessor": ARCHIVED_MILESTONE if predecessor_ok else "mismatch",
        "mentions_exp5161_correction": correction,
        "ready": milestone_ok and predecessor_ok and correction,
    }


def _source_row(root: Path, *, kind: str, source_id: str, relative_path: Path) -> JsonDict:
    path = root / relative_path
    return {
        "kind": kind,
        "source_id": source_id,
        "path": str(relative_path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }


def load_v473_results(root: Path) -> tuple[dict[int, JsonDict], dict[int, JsonDict]]:
    payloads: dict[int, JsonDict] = {}
    statuses: dict[int, JsonDict] = {}
    for exp_id, relative_path in V473_RESULT_PATHS.items():
        payload, status = read_json_mapping(root / relative_path)
        statuses[exp_id] = status
        if status.get("loadable") is True:
            payloads[exp_id] = payload
    return payloads, statuses


def build_source_artifacts_read(root: Path) -> list[JsonDict]:
    rows = [
        _source_row(
            root,
            kind="v473_result_artifact",
            source_id=TASK_META[exp_id]["experiment_id"],
            relative_path=relative_path,
        )
        for exp_id, relative_path in V473_RESULT_PATHS.items()
    ]
    rows.extend(
        [
            _source_row(
                root,
                kind="roadmap_yaml",
                source_id="active_research_roadmap",
                relative_path=ACTIVE_ROADMAP_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="vnext_plan",
                source_id="research_roadmap_vnext",
                relative_path=VNEXT_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="ops_registry",
                source_id="arc_solve_registry",
                relative_path=ARC_REGISTRY_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="ops_gap_doc",
                source_id="verifier_gaps",
                relative_path=VERIFIER_GAPS_RELATIVE_PATH,
            ),
        ]
    )
    return rows


def _game_names(payload: JsonMap) -> list[str]:
    return [str(_mapping(row).get("game", "")) for row in _as_list(payload.get("games_tested"))]


def _capstone_excluded_task_ids(capstone: JsonMap) -> list[str]:
    return [str(item) for item in _as_list(capstone.get("flagged_adversarial_artifacts_excluded"))]


def _m450_validation(payload: JsonMap) -> JsonDict:
    for row in _as_list(payload.get("validated_milestones")):
        mapping = _mapping(row)
        if mapping.get("milestone") == "2026.06.450":
            return mapping
    return {}


def _summary_exp5156(payload: JsonMap) -> JsonDict:
    flags = _list(_mapping(payload.get("adversarial_verification")).get("flags"))
    return {
        "classification": "transition_archive_warn_flagged",
        "v472_runtime_clean": _as_bool(payload.get("v472_runtime_clean")),
        "flagged_adversarial": _as_bool(payload.get("flagged_adversarial")),
        "flag_severities": [str(_mapping(flag).get("severity", "")) for flag in flags],
        "archive_read": "Transition into .473 is recorded; upstream warn flag is not a fresh .473 win.",
    }


def _summary_exp5157(payload: JsonMap) -> JsonDict:
    return {
        "classification": "honest_null_warmstart_gate_failed",
        "gate_passed": _as_bool(payload.get("gate_passed")),
        "warmstart_vs_cold_delta_median": _as_float(
            payload.get("warmstart_vs_cold_delta_median")
        ),
        "transition_count": len(_as_list(payload.get("per_transition_breakdown"))),
        "game_count": len(_game_names(payload)),
        "archive_read": "Warm-start residual replay showed no median lift and failed its gate.",
    }


def _summary_exp5158(payload: JsonMap) -> JsonDict:
    return {
        "classification": "honest_null_goal_energy_ranker_gate_failed",
        "gate_passed": _as_bool(payload.get("gate_passed")),
        "target_games": _game_names(payload),
        "improved_games_count": _as_int(payload.get("games_improved_count")),
        "target_games_count": len(_game_names(payload)),
        "reciprocal_rank_cold": _mapping(payload.get("reciprocal_rank_cold")),
        "reciprocal_rank_warmstart": _mapping(payload.get("reciprocal_rank_warmstart")),
        "archive_read": "Goal-energy ranker replay improved only one of three target games.",
    }


def _summary_exp5159(payload: JsonMap) -> JsonDict:
    return {
        "classification": "blocked_upstream_gate_no_live_run",
        "blocked_at_layer": str(payload.get("blocked_at_layer", "")),
        "gate_check_summary": str(payload.get("gate_check_summary", "")),
        "new_levels_banked": 0,
        "archive_read": "Live level-up attempt did not run because Exp 5157 gate_passed was false.",
    }


def _summary_exp5160(payload: JsonMap) -> JsonDict:
    held_out_n = _as_int(payload.get("held_out_task_n"))
    return {
        "classification": "real_cross_corpus_win_below_clt_floor",
        "cross_corpus_delta": _as_float(payload.get("cross_corpus_delta")),
        "cross_corpus_delta_ci95": _as_list(payload.get("cross_corpus_delta_ci95")),
        "held_out_task_n": held_out_n,
        "meets_clt_floor_n30": held_out_n >= 30,
        "game_id_misnomer_confirmed": _as_bool(payload.get("game_id_misnomer_confirmed")),
        "second_pool_leak_audit_passed": _as_bool(
            payload.get("second_pool_leak_audit_passed")
        ),
        "diffusiongemma_gate_updated_recommendation": _as_str(
            payload.get("diffusiongemma_gate_updated_recommendation")
        ),
        "verifier_is_oracle": _as_bool(payload.get("verifier_is_oracle")),
        "archive_read": "Set-Encoder win replicated cross-corpus, with n=24 caveat carried forward.",
    }


def _summary_exp5161(payload: JsonMap) -> JsonDict:
    return {
        "classification": "directional_replication_unquarantined_not_significant",
        "flagged_adversarial": _as_bool(payload.get("flagged_adversarial")),
        "pilot_n_achieved": _as_int(payload.get("pilot_n_achieved")),
        "replicated_prior_direction": _as_bool(payload.get("replicated_prior_direction")),
        "exact_test_discordant_wins": _as_int(payload.get("exact_test_discordant_wins")),
        "exact_test_discordant_losses": _as_int(payload.get("exact_test_discordant_losses")),
        "exact_test_p_value_two_sided": _as_float(payload.get("exact_test_p_value_two_sided")),
        "exact_test_passes_min6_rule": _as_bool(payload.get("exact_test_passes_min6_rule")),
        "gap4_status_recommendation": _as_str(payload.get("gap4_status_recommendation")),
        "archive_read": "Exp 5161 is live-unquarantined but remains below the significance floor.",
    }


def _summary_exp5162(payload: JsonMap) -> JsonDict:
    return {
        "classification": "sota_ingestion_no_new_primary_findings",
        "archive_read": _as_str(payload.get("honest_verdict")),
    }


def _summary_exp5163(payload: JsonMap) -> JsonDict:
    return {
        "classification": "underpowered_tautology_flagged_not_headline_clean",
        "flagged_adversarial": _as_bool(payload.get("flagged_adversarial")),
        "verifier_vs_cheap_delta": _as_float(payload.get("verifier_vs_cheap_delta")),
        "verifier_vs_cheap_delta_ci95": _as_list(payload.get("verifier_vs_cheap_delta_ci95")),
        "still_underpowered": _as_bool(payload.get("still_underpowered")),
        "fewshot_oracle_at_k": _as_float(payload.get("fewshot_oracle_at_k")),
        "oracle_at_k_ceiling": _as_float(payload.get("oracle_at_k_ceiling")),
        "archive_read": "MMLU-Pro result is underpowered and tautology-flagged, not headline-clean.",
    }


def _summary_exp5164(payload: JsonMap) -> JsonDict:
    m450 = _m450_validation(payload)
    return {
        "classification": "retro_timing_fix_tested_not_wired",
        "m450_reconstruction_correct": _as_bool(payload.get("m450_reconstruction_correct")),
        "m450_wall_minutes": _as_float(m450.get("reconstructed_wall_time_minutes")),
        "m450_compute_bound_arms": _as_int(m450.get("reconstructed_compute_bound_count")),
        "validated_milestone_count": len(_as_list(payload.get("validated_milestones"))),
        "tests_added": _as_int(payload.get("tests_added")),
        "tests_passing": _as_bool(payload.get("tests_passing")),
        "research_conductor_py_modified": _as_bool(
            payload.get("research_conductor_py_modified")
        ),
        "archive_read": "Timing module is tested; conductor wiring remains a later operator action.",
    }


def _summary_exp5165(payload: JsonMap) -> JsonDict:
    return {
        "classification": "retirement_hygiene_load_bearing",
        "entry_id": str(payload.get("entry_id", "")),
        "exclusion_manifest_entry_added": _as_bool(
            payload.get("exclusion_manifest_entry_added")
        ),
        "synthetic_match_check_passed": _as_bool(payload.get("synthetic_match_check_passed")),
        "false_positive_check_against_this_milestone": _as_bool(
            payload.get("false_positive_check_against_this_milestone")
        ),
        "archive_read": "Generation-axis exploration-signal retirement is mechanically load-bearing.",
    }


def _summary_exp5166(payload: JsonMap) -> JsonDict:
    gatemate = _mapping(payload.get("gatemate_result"))
    return {
        "classification": "hardware_continuity_2_of_3",
        "boards_reachable_count": _as_int(payload.get("boards_reachable_count")),
        "kv260_reachable": _as_bool(_mapping(payload.get("kv260_result")).get("reachable")),
        "polarfire_reachable": _as_bool(_mapping(payload.get("polarfire_result")).get("reachable")),
        "gatemate_reachable": _as_bool(gatemate.get("reachable")),
        "gatemate_blocked_reason": str(gatemate.get("blocked_reason", "")),
        "gatemate_expected_idcode": str(
            _mapping(gatemate.get("timing_output")).get("expected_idcode", "")
        ),
        "no_speedup_claim": _as_bool(payload.get("no_speedup_claim")),
        "archive_read": "KV260 and PolarFire are reachable; GateMate remains IDCODE-blocked.",
    }


def _summary_exp5167(payload: JsonMap) -> JsonDict:
    return {
        "classification": "capstone_stale_exp5161_exclusion",
        "capstone_excluded_task_ids": _capstone_excluded_task_ids(payload),
        "registry_reconciliation": _mapping(payload.get("registry_reconciliation")),
        "gap4_status_reconciled": _as_str(payload.get("gap4_status_reconciled")),
        "reproducible_total_levels_delta": _as_int(
            payload.get("reproducible_total_levels_delta")
        ),
        "archive_read": "Capstone is stale only where it excludes Exp 5161 as flagged.",
    }


SUMMARY_BY_EXP = {
    5156: _summary_exp5156,
    5157: _summary_exp5157,
    5158: _summary_exp5158,
    5159: _summary_exp5159,
    5160: _summary_exp5160,
    5161: _summary_exp5161,
    5162: _summary_exp5162,
    5163: _summary_exp5163,
    5164: _summary_exp5164,
    5165: _summary_exp5165,
    5166: _summary_exp5166,
    5167: _summary_exp5167,
}


def _task_summary(exp_id: int, payload: JsonMap, status: JsonMap) -> JsonDict:
    meta = TASK_META[exp_id]
    if status.get("loadable") is not True:
        return {
            "experiment_id": meta["experiment_id"],
            "axis": meta["axis"],
            "label": meta["label"],
            "path": str(V473_RESULT_PATHS[exp_id]),
            "classification": "missing_input",
            "honest_verdict": "missing",
            "archive_read": "Required .473 source artifact was missing or unreadable.",
        }
    return {
        "experiment_id": meta["experiment_id"],
        "axis": meta["axis"],
        "label": meta["label"],
        "path": str(V473_RESULT_PATHS[exp_id]),
        "honest_verdict": _as_str(payload.get("honest_verdict")),
        **SUMMARY_BY_EXP[exp_id](payload),
    }


def build_task_verdicts(
    payloads: Mapping[int, JsonMap], statuses: Mapping[int, JsonMap]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for exp_id in V473_RESULT_PATHS:
        meta = TASK_META[exp_id]
        status = _mapping(statuses.get(exp_id))
        payload = _mapping(payloads.get(exp_id))
        rows.append(
            {
                "experiment_number": exp_id,
                "experiment_id": meta["experiment_id"],
                "axis": meta["axis"],
                "label": meta["label"],
                "path": str(V473_RESULT_PATHS[exp_id]),
                "exists": _bool(status.get("exists")),
                "loadable": _bool(status.get("loadable")),
                "honest_verdict": _as_str(payload.get("honest_verdict", "missing")),
                "flagged_adversarial": _as_bool(payload.get("flagged_adversarial")),
                "reproducibility_checksum": payload.get("reproducibility_checksum", ""),
            }
        )
    return rows


def build_milestone_archive_summary(
    payloads: Mapping[int, JsonMap], statuses: Mapping[int, JsonMap]
) -> list[JsonDict]:
    return [
        _task_summary(exp_id, _mapping(payloads.get(exp_id)), _mapping(statuses.get(exp_id)))
        for exp_id in V473_RESULT_PATHS
    ]


def analyze_runtime_snapshot(snapshot: RuntimeSnapshot) -> JsonDict:
    dirty_paths = _dirty_paths(snapshot.git_status_porcelain)
    ignored = [path for path in dirty_paths if path in EXPECTED_TRANSITION_DIRTY_PATHS]
    non_transition = [path for path in dirty_paths if path not in EXPECTED_TRANSITION_DIRTY_PATHS]
    process_rows = [
        row for line in snapshot.process_table.splitlines() if (row := _process_row(line)) is not None
    ]
    conductor_rows = [
        row
        for row in process_rows
        if "research_conductor.py" in str(row["command"]) or "carnot-conductor" in str(row["command"])
    ]
    active_task_rows = [row for row in process_rows if "codex exec" in str(row["command"])]
    orphaned = [row for row in conductor_rows if row["ppid"] == 1]
    return {
        "git_status_porcelain": snapshot.git_status_porcelain,
        "dirty_paths": dirty_paths,
        "ignored_transition_dirty_paths": ignored,
        "non_transition_dirty_paths": non_transition,
        "conductor_processes": conductor_rows,
        "active_task_processes": active_task_rows,
        "orphaned_conductor_processes": orphaned,
        "expected_transition_dirty_paths": sorted(EXPECTED_TRANSITION_DIRTY_PATHS),
        "touched_conductor_process": False,
        "runtime_clean": not non_transition and not orphaned,
    }


def _registry_reconciliation(path: Path, capstone: JsonMap) -> JsonDict:
    base: JsonDict = {
        "path": str(ARC_REGISTRY_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "reproducible_total_levels": None,
        "reproducible_total_games": None,
        "capstone_reproducible_total_levels": _mapping(
            capstone.get("registry_reconciliation")
        ).get("reproducible_total_levels"),
        "capstone_reproducible_total_games": _mapping(
            capstone.get("registry_reconciliation")
        ).get("reproducible_total_games"),
        "drift_detected": True,
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "error": str(exc)}
    mapping = _mapping(loaded)
    levels = _as_int(mapping.get("reproducible_total_levels"), default=-1)
    games = _as_int(mapping.get("reproducible_total_games"), default=-1)
    capstone_levels = _as_int(base["capstone_reproducible_total_levels"], default=-1)
    capstone_games = _as_int(base["capstone_reproducible_total_games"], default=-1)
    drift = levels != 69 or games != 24 or levels != capstone_levels or games != capstone_games
    return {
        **base,
        "exists": True,
        "parses": True,
        "reproducible_total_levels": levels,
        "reproducible_total_games": games,
        "capstone_reproducible_total_levels": capstone_levels,
        "capstone_reproducible_total_games": capstone_games,
        "expected_reproducible_total_levels": 69,
        "expected_reproducible_total_games": 24,
        "drift_detected": drift,
    }


def _gap4_status_reconciliation(path: Path, capstone: JsonMap) -> JsonDict:
    capstone_status = _as_str(capstone.get("gap4_status_reconciled"))
    base = {
        "path": str(VERIFIER_GAPS_RELATIVE_PATH),
        "exists": path.exists(),
        "gap4_exp5161_section_found": False,
        "status_line_matches_capstone": False,
        "capstone_status": capstone_status,
        "drift_detected": True,
    }
    if not path.exists():
        return base
    text = path.read_text(encoding="utf-8")
    section_found = "GAP-4: Exp 5161 .473 forward-protocol pilot" in text
    open_directional = (
        "status` stays **open" in text
        and "bounded-scale (n=60) directional replication" in text
        and "still short of the significance floor" in text
    )
    capstone_not_filled = "not_filled" in capstone_status or "scale_up_recommended" in capstone_status
    return {
        **base,
        "exists": True,
        "gap4_exp5161_section_found": section_found,
        "status_line_matches_capstone": section_found and open_directional and capstone_not_filled,
        "drift_detected": not (section_found and open_directional and capstone_not_filled),
    }


def capstone_stale_exclusions_corrected(payloads: Mapping[int, JsonMap]) -> JsonDict:
    exp5161 = _mapping(payloads.get(5161))
    exp5163 = _mapping(payloads.get(5163))
    capstone = _mapping(payloads.get(5167))
    capstone_excluded = _capstone_excluded_task_ids(capstone)
    exp5161_task_id = TASK_META[5161]["experiment_id"]
    not_headline_clean = []
    if _as_bool(exp5163.get("flagged_adversarial")):
        not_headline_clean.append(TASK_META[5163]["experiment_id"])
    return {
        "capstone_excluded_task_ids": capstone_excluded,
        "live_exp5161_flagged_adversarial": _as_bool(exp5161.get("flagged_adversarial")),
        "exp5161_removed_from_exclusion": (
            exp5161_task_id in capstone_excluded
            and not _as_bool(exp5161.get("flagged_adversarial"))
        ),
        "not_headline_clean_task_ids": not_headline_clean,
        "capstone_stale_note": (
            "Exp 5167 recorded Exp 5161 before same-day un-quarantine; Exp 5168 corrects that "
            "handoff premise."
        ),
    }


def build_preconditions(
    root: Path, payloads: Mapping[int, JsonMap], statuses: Mapping[int, JsonMap]
) -> JsonDict:
    result_statuses = {
        TASK_META[exp_id]["experiment_id"]: {
            "path": str(V473_RESULT_PATHS[exp_id]),
            "exists": _bool(_mapping(status).get("exists")),
            "loadable": _bool(_mapping(status).get("loadable")),
            "sha256": _mapping(status).get("sha256"),
        }
        for exp_id, status in statuses.items()
    }
    capstone = _mapping(payloads.get(5167))
    return {
        "v473_results": result_statuses,
        "v473_results_all_loadable": all(row["loadable"] for row in result_statuses.values()),
        "active_roadmap": _roadmap_check(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "vnext_plan": _vnext_check(root / VNEXT_RELATIVE_PATH),
        "arc_registry": _registry_reconciliation(root / ARC_REGISTRY_RELATIVE_PATH, capstone),
        "gap4_status": _gap4_status_reconciliation(root / VERIFIER_GAPS_RELATIVE_PATH, capstone),
    }


def _honest_verdict(preconditions: JsonMap, *, runtime_clean: bool) -> str:
    if preconditions.get("v473_results_all_loadable") is not True:
        return MISSING_INPUTS_VERDICT
    ready_checks = [
        _mapping(preconditions.get("active_roadmap")).get("ready") is True,
        _mapping(preconditions.get("vnext_plan")).get("ready") is True,
        _mapping(preconditions.get("arc_registry")).get("drift_detected") is False,
        _mapping(preconditions.get("gap4_status")).get("drift_detected") is False,
    ]
    if not all(ready_checks):
        return ACTIVATION_GATED_VERDICT
    if not runtime_clean:
        return DIRTY_HANDOFF_VERDICT
    return COMPLETE_VERDICT


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    runtime_snapshot: RuntimeSnapshot,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    payloads, statuses = load_v473_results(root)
    source_artifacts_read = build_source_artifacts_read(root)
    task_verdicts = build_task_verdicts(payloads, statuses)
    archive_summary = build_milestone_archive_summary(payloads, statuses)
    preconditions = build_preconditions(root, payloads, statuses)
    runtime_details = analyze_runtime_snapshot(runtime_snapshot)
    runtime_clean = _bool(runtime_details.get("runtime_clean"))
    correction = capstone_stale_exclusions_corrected(payloads)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(preconditions, runtime_clean=runtime_clean),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "source_artifacts_read": source_artifacts_read,
        "task_verdicts": task_verdicts,
        "milestone_archive_summary": archive_summary,
        "v473_runtime_clean": runtime_clean,
        "runtime_clean_details": runtime_details,
        "exp5161_unquarantine_noted": _bool(correction.get("exp5161_removed_from_exclusion")),
        "capstone_stale_exclusions_corrected": correction,
        "arc_registry_reconciliation": preconditions["arc_registry"],
        "gap4_status_reconciliation": preconditions["gap4_status"],
        "active_roadmap_check": preconditions["active_roadmap"],
        "active_roadmap_ready": _bool(
            _mapping(preconditions.get("active_roadmap")).get("ready")
        ),
        "vnext_check": preconditions["vnext_plan"],
        "vnext_ready": _bool(_mapping(preconditions.get("vnext_plan")).get("ready")),
        "roadmap_task_snapshot": _mapping(preconditions.get("active_roadmap")).get("task_ids", []),
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle.{field}")
    checks = [
        (artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id.invalid"),
        (artifact.get("milestone") != MILESTONE, "milestone.invalid"),
        (artifact.get("archived_milestone") != ARCHIVED_MILESTONE, "archived_milestone.invalid"),
        (
            not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES),
            "honest_verdict.not_terminal",
        ),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate.invalid"),
        (
            not isinstance(artifact.get("duration_s"), (int, float))
            or artifact.get("duration_s", 0) <= 0,
            "duration_s.invalid",
        ),
        (not _list(artifact.get("source_artifacts_read")), "source_artifacts_read.empty"),
        (len(_list(artifact.get("task_verdicts"))) != len(V473_RESULT_PATHS), "task_verdicts.invalid"),
        (
            len(_list(artifact.get("milestone_archive_summary"))) != len(V473_RESULT_PATHS),
            "milestone_archive_summary.invalid",
        ),
        (not isinstance(artifact.get("v473_runtime_clean"), bool), "v473_runtime_clean.invalid"),
        (
            not isinstance(artifact.get("runtime_clean_details"), Mapping),
            "runtime_clean_details.invalid",
        ),
        (
            not isinstance(artifact.get("exp5161_unquarantine_noted"), bool),
            "exp5161_unquarantine_noted.invalid",
        ),
        (
            not isinstance(artifact.get("capstone_stale_exclusions_corrected"), Mapping),
            "capstone_stale_exclusions_corrected.invalid",
        ),
        (
            not isinstance(artifact.get("arc_registry_reconciliation"), Mapping),
            "arc_registry_reconciliation.invalid",
        ),
        (
            not isinstance(artifact.get("gap4_status_reconciliation"), Mapping),
            "gap4_status_reconciliation.invalid",
        ),
        (not isinstance(artifact.get("active_roadmap_ready"), bool), "active_roadmap_ready.invalid"),
        (artifact.get("active_roadmap_modified") is not False, "active_roadmap_modified.invalid"),
        (artifact.get("conductor_modified") is not False, "conductor_modified.invalid"),
        (not isinstance(artifact.get("flagged_adversarial"), bool), "flagged_adversarial.invalid"),
        (not _list(artifact.get("tests_run")), "tests_run.empty"),
        (
            artifact.get("reproducibility_checksum") != payload_checksum(artifact),
            "reproducibility_checksum.invalid",
        ),
    ]
    errors.extend(error for invalid, error in checks if invalid)
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5168 archive artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260702",
    verification_runner: VerificationRunner | None = None,
    runtime_probe: RuntimeProbe | None = None,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    clock: Clock = time.perf_counter,
) -> Path:
    root = Path(root)
    output_path = artifact_path or root / RESULT_RELATIVE_PATH
    runner = verification_runner or (lambda path: run_adversarial_verification(root, path))
    probe = runtime_probe or capture_runtime_snapshot
    start = clock()
    active_before = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    runtime_snapshot = probe(root)
    placeholder = verification_payload(
        CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr="")
    )
    artifact = build_artifact(
        root=root,
        duration_s=max(clock() - start, 0.0001),
        run_date=run_date,
        verification=placeholder,
        runtime_snapshot=runtime_snapshot,
        tests_run=tests_run,
    )
    write_json(output_path, artifact)
    verification = verification_payload(runner(output_path))
    final_artifact = {
        **artifact,
        "active_roadmap_modified": active_before != file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "conductor_modified": conductor_before != file_sha256(root / CONDUCTOR_RELATIVE_PATH),
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    final_artifact["reproducibility_checksum"] = payload_checksum(final_artifact)
    validate_artifact(final_artifact)
    write_json(output_path, final_artifact)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the Exp 5168 archive .473 / activate .474 artifact."
    )
    parser.add_argument("--date", default="20260702", help="Run date label, e.g. 20260702.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to read.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(root=args.root, artifact_path=args.output, run_date=args.date)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(f"{EXPERIMENT}: wrote {output}")
    print(f"{EXPERIMENT}: honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
