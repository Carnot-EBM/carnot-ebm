"""Experiment 5042: archive .463, activate .464, and record the close-state.

Spec refs: REQ-CAPSTONE-5042, SCENARIO-CAPSTONE-5042,
SCENARIO-CAPSTONE-5042-BLOCKED-YAML.

This record-only transition reads roadmap YAML and landed upstream JSON
artifacts. It does not change the active conductor, does not submit to any
leaderboard, and treats absent source fields as blockers rather than inferred
facts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_5028_archive_462_activate_463 import (  # noqa: E402
    CommandResult,
    command_summary,
    duration_from,
    file_sha256,
    payload_checksum,
    run_command,
    write_payload,
    _float,
    _int,
    _is_sha256,
    _list,
    _mapping,
)


CommandRunner = Callable[[list[str], Path], CommandResult]
EXPERIMENT = "experiment_5042_archive_463_activate_464"
EXPERIMENT_ID = 5042
SCHEMA = "carnot.exp5042.archive_463_activate_464.v1"
RANDOM_SEED = 20260630
PRIOR_MILESTONE = "2026.06.463"
NEXT_MILESTONE = "2026.06.464"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

OUTPUT_REL_PATH = Path("results/experiment_5042_archive_463_activate_464.json")
ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5041_capstone_v463.json")
D1_REL_PATH = Path("results/experiment_5031_lora_ebm_scorer_musr_v3.json")
D2_REL_PATH = Path("results/experiment_5032_uprm_replication_v3.json")
D3_REL_PATH = Path("results/experiment_5033_ebrm_uncertainty_verifier_v3.json")
D6_REL_PATH = Path("results/experiment_5034_uncertainty_routed_cascade_v2.json")
D4_REL_PATH = Path("results/experiment_5035_moat_second_corpus_v3.json")
KV260_REL_PATH = Path("results/experiment_5037_kv260_continuity.json")
ARC_LEVEL_REL_PATH = Path("results/experiment_5040_levelup_attempt.json")

PRETEST_COMMAND = [
    ".venv/bin/pytest",
    "tests/python/test_experiment_5042_archive_463_activate_464.py",
    "-q",
    "--no-cov",
]

SPEC_REFS = [
    "REQ-CAPSTONE-5042",
    "SCENARIO-CAPSTONE-5042",
    "SCENARIO-CAPSTONE-5042-BLOCKED-YAML",
    "SCENARIO-CAPSTONE-5042-FIELD-PRINCIPLES",
]

TERMINAL_PREFIXES = (
    "complete_",
    "success_",
    "passed_",
    "shipped_",
    "blocked_",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; clean transition is "
            "complete_463_archived_464_activated_phase_d_power_confirmation."
        )
    },
    "d1_delta_vs_tuned_sc": {
        "principle": (
            "the real but underpowered LoRA-EBM MuSR delta (+0.080) from "
            "Exp5031/Exp5041; CI touches zero so no moat claim."
        )
    },
    "d1_ci_touches_zero": {
        "principle": (
            "true iff the paired CI lower bound is <=0<=upper bound; explains "
            "why the D1 signal is underpowered."
        )
    },
    "d6_blocked_reason": {
        "principle": (
            "blocked_judge_server from Exp5034, one of the two missing "
            "confirmation axes."
        )
    },
    "d4_blocked_reason": {
        "principle": (
            "blocked_second_corpus_unavailable from Exp5035, the second "
            "missing confirmation axis."
        )
    },
    "kv260_continuity_live": {
        "principle": (
            "true iff Exp5037 reports KV260 reachable with overlay/energy "
            "smoke OK."
        )
    },
    "no_arc_level_bank": {
        "principle": (
            "true iff Exp5040/capstone reports zero new ARC levels banked and "
            "total remains 69."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads roadmap + upstream "
            "JSON, no LLM; 0.0001s floor)."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "prior_milestone",
    "next_milestone",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "close_state_463",
    "d1_delta_vs_tuned_sc",
    "d1_ci_touches_zero",
    "d6_blocked_reason",
    "d4_blocked_reason",
    "lora_ebm_signal",
    "scalar_uprm_result",
    "ebrm_result",
    "blocked_confirmation_axes",
    "kv260_continuity_live",
    "no_arc_level_bank",
    "leaderboard_submission",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

SOURCE_PATHS: dict[str, Path] = {
    "CAPSTONE": CAPSTONE_REL_PATH,
    "D1_LORA_EBM": D1_REL_PATH,
    "D2_UPRM": D2_REL_PATH,
    "D3_EBRM": D3_REL_PATH,
    "D6_CASCADE": D6_REL_PATH,
    "D4_SECOND_CORPUS": D4_REL_PATH,
    "C_KV260": KV260_REL_PATH,
    "E3_ARC_LEVEL": ARC_LEVEL_REL_PATH,
}

REQUIRED_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "CAPSTONE": (
        "honest_verdict",
        "moat_verdict",
        "best_arm_and_delta",
        "hardware_rollup",
        "arc_opportunistic_rollup",
        "reproducible_total_levels",
    ),
    "D1_LORA_EBM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "n_questions",
        "genuine_tuned_sc_accuracy",
        "trained_scorer_accuracy",
        "scorer_trained",
        "headroom_present",
        "oracle_at_k",
        "verifier_is_oracle",
    ),
    "D2_UPRM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "n_questions",
        "genuine_tuned_sc_accuracy",
        "uprm_selection_accuracy",
        "headroom_present",
        "oracle_at_k",
        "verifier_is_oracle",
    ),
    "D3_EBRM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "n_questions",
        "genuine_tuned_sc_accuracy",
        "ebrm_selection_accuracy",
        "headroom_present",
        "oracle_at_k",
        "abstention_rate",
        "verifier_is_oracle",
    ),
    "D6_CASCADE": (
        "honest_verdict",
        "blocked_error",
        "preconditions_checked",
        "cascade_accuracy",
        "judge_call_fraction",
        "verifier_is_oracle",
    ),
    "D4_SECOND_CORPUS": (
        "honest_verdict",
        "blocked_error",
        "preconditions_checked",
        "delta_vs_tuned_sc_second",
        "paired_ci95_second",
        "second_corpus",
        "verifier_is_oracle",
    ),
    "C_KV260": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "energy_smoke",
    ),
    "E3_ARC_LEVEL": (
        "honest_verdict",
        "new_levels_banked",
        "reproducible_total_levels_after",
    ),
}


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _parse_yaml_status(root: Path, rel_path: Path, *, absent_status: str) -> tuple[JsonDict, JsonDict]:
    path = root / rel_path
    status: JsonDict = {"path": str(rel_path), "exists": path.exists()}
    if not path.exists():
        status.update({"parse_ok": None, "status": absent_status})
        return {}, status
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        status.update({"parse_ok": False, "error": str(exc)})
        return {}, status
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        status.update({"parse_ok": False, "error": "yaml_not_mapping"})
        return {}, status
    status.update(
        {
            "parse_ok": True,
            "status": "parsed",
            "milestone": str(payload.get("milestone", "")),
            "sha256": file_sha256(path),
        }
    )
    return dict(payload), status


def check_roadmaps(root: Path) -> tuple[JsonDict, JsonDict]:
    """Parse active and pre-staged roadmaps with explicit YAML failure status."""

    active, active_status = _parse_yaml_status(
        root,
        ROADMAP_ACTIVE_REL_PATH,
        absent_status="missing_active_roadmap",
    )
    staged, staged_status = _parse_yaml_status(
        root,
        ROADMAP_NEXT_REL_PATH,
        absent_status="absent_already_promoted",
    )
    return (
        {"active": active, "pre_staged": staged},
        {"active": active_status, "pre_staged": staged_status},
    )


def roadmap_blocker(roadmaps_checked: Mapping[str, Any]) -> str:
    active = _mapping(roadmaps_checked.get("active"))
    staged = _mapping(roadmaps_checked.get("pre_staged"))
    if active.get("parse_ok") is not True:
        return "blocked_yaml_parse"
    if staged.get("exists") is True and staged.get("parse_ok") is not True:
        return "blocked_yaml_parse"
    return ""


def load_sources(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    statuses: dict[str, JsonDict] = {}
    for source, rel_path in SOURCE_PATHS.items():
        artifact, status = _read_json_mapping(root / rel_path)
        artifacts[source] = artifact
        statuses[source] = {"path": str(rel_path), **status}
    return artifacts, statuses


def source_blocker(source_statuses: Mapping[str, Any]) -> str:
    for status in source_statuses.values():
        mapping = _mapping(status)
        if mapping.get("exists") is not True:
            return "blocked_missing_required_artifact"
        if mapping.get("loadable") is not True:
            return "blocked_unloadable_required_artifact"
    return ""


def missing_required_fields(artifacts: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    missing: list[JsonDict] = []
    for source, fields in REQUIRED_SOURCE_FIELDS.items():
        artifact = artifacts.get(source, {})
        path = SOURCE_PATHS[source]
        for field in fields:
            if field not in artifact:
                missing.append({"path": str(path), "field": field})
    return missing


def _ci_touches_zero(ci: Any) -> bool:
    if not isinstance(ci, list) or len(ci) != 2:
        return False
    low = _float(ci[0])
    high = _float(ci[1])
    return low is not None and high is not None and low <= 0.0 <= high


def _blocked_resource_detail(artifact: Mapping[str, Any]) -> str:
    for row in _list(artifact.get("preconditions_checked")):
        mapping = _mapping(row)
        if mapping.get("available") is False:
            return str(mapping.get("detail", mapping.get("resource", "")))
    return str(artifact.get("blocked_error", ""))


def build_close_state(artifacts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the true .463 close-state from explicit upstream fields."""

    capstone = _mapping(artifacts.get("CAPSTONE"))
    d1 = _mapping(artifacts.get("D1_LORA_EBM"))
    d2 = _mapping(artifacts.get("D2_UPRM"))
    d3 = _mapping(artifacts.get("D3_EBRM"))
    d6 = _mapping(artifacts.get("D6_CASCADE"))
    d4 = _mapping(artifacts.get("D4_SECOND_CORPUS"))
    kv260 = _mapping(artifacts.get("C_KV260"))
    arc = _mapping(artifacts.get("E3_ARC_LEVEL"))

    moat = _mapping(capstone.get("moat_verdict"))
    d1_delta = _float(d1.get("delta_vs_tuned_sc"))
    d2_delta = _float(d2.get("delta_vs_tuned_sc"))
    d3_delta = _float(d3.get("delta_vs_tuned_sc"))
    d1_ci = d1.get("paired_ci95")
    d3_ci = d3.get("paired_ci95")
    d1_accuracy = _float(d1.get("trained_scorer_accuracy"))
    d3_accuracy = _float(d3.get("ebrm_selection_accuracy"))
    d1_ci_touches_zero = _ci_touches_zero(d1_ci)
    d3_tied_d1 = d3_delta == d1_delta and d3_accuracy == d1_accuracy and d3_ci == d1_ci
    kv260_live = (
        kv260.get("kv260_ssh_reachable") is True
        and _mapping(kv260.get("energy_smoke")).get("success") is True
    )
    new_levels_banked = _int(arc.get("new_levels_banked"))
    total_after = _int(arc.get("reproducible_total_levels_after"))
    no_arc_level_bank = new_levels_banked == 0 and total_after == _int(
        capstone.get("reproducible_total_levels")
    )

    return {
        "summary": (
            "v463_real_underpowered_lora_ebm_signal_negative_uprm_ebrm_tie_"
            "d6_d4_blocked_kv260_live_arc_no_bank"
        ),
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "best_arm_and_delta": dict(_mapping(capstone.get("best_arm_and_delta"))),
        },
        "moat_verdict": {
            "decision": str(moat.get("decision", "")),
            "state": str(moat.get("state", "")),
            "moat_realized": moat.get("moat_realized") is True,
            "moat_retired_bounded": moat.get("moat_retired_bounded") is True,
            "efficiency_win": capstone.get("efficiency_win") is True,
            "execution_incomplete_arms": [
                dict(_mapping(row)) for row in _list(moat.get("execution_incomplete_arms"))
            ],
        },
        "d1_lora_ebm_signal": {
            "arm_id": "D1",
            "source_experiment_id": 5031,
            "honest_verdict": str(d1.get("honest_verdict", "")),
            "corpus": "MuSR",
            "delta_vs_tuned_sc": d1_delta,
            "paired_ci95": d1_ci,
            "ci_touches_zero": d1_ci_touches_zero,
            "mcnemar_p": _float(d1.get("mcnemar_p")),
            "n_questions": _int(d1.get("n_questions")),
            "genuine_tuned_sc_accuracy": _float(d1.get("genuine_tuned_sc_accuracy")),
            "trained_scorer_accuracy": d1_accuracy,
            "scorer_trained": d1.get("scorer_trained") is True,
            "headroom_present": d1.get("headroom_present") is True,
            "oracle_at_k": _float(d1.get("oracle_at_k")),
            "verifier_is_oracle": d1.get("verifier_is_oracle") is True,
            "real_signal": d1.get("scorer_trained") is True and (d1_delta or 0.0) > 0.0,
            "underpowered": d1_ci_touches_zero,
            "moat_claim": False,
        },
        "d2_scalar_uprm_result": {
            "arm_id": "D2",
            "source_experiment_id": 5032,
            "honest_verdict": str(d2.get("honest_verdict", "")),
            "corpus": "MuSR",
            "delta_vs_tuned_sc": d2_delta,
            "paired_ci95": d2.get("paired_ci95"),
            "mcnemar_p": _float(d2.get("mcnemar_p")),
            "n_questions": _int(d2.get("n_questions")),
            "selection_accuracy": _float(d2.get("uprm_selection_accuracy")),
            "negative_result": d2_delta is not None and d2_delta < 0.0,
            "verifier_is_oracle": d2.get("verifier_is_oracle") is True,
        },
        "d3_ebrm_result": {
            "arm_id": "D3",
            "source_experiment_id": 5033,
            "honest_verdict": str(d3.get("honest_verdict", "")),
            "corpus": "MuSR",
            "delta_vs_tuned_sc": d3_delta,
            "paired_ci95": d3_ci,
            "mcnemar_p": _float(d3.get("mcnemar_p")),
            "n_questions": _int(d3.get("n_questions")),
            "selection_accuracy": d3_accuracy,
            "abstention_rate": _float(d3.get("abstention_rate")),
            "tied_d1": d3_tied_d1,
            "improved_over_d1": False,
            "verifier_is_oracle": d3.get("verifier_is_oracle") is True,
        },
        "blocked_confirmation_axes": [
            {
                "axis": "D6_judge_cascade",
                "source_experiment_id": 5034,
                "blocked_reason": str(d6.get("honest_verdict", "")),
                "blocked_error": str(d6.get("blocked_error", "")),
                "resource_detail": _blocked_resource_detail(d6),
            },
            {
                "axis": "D4_second_corpus",
                "source_experiment_id": 5035,
                "blocked_reason": str(d4.get("honest_verdict", "")),
                "blocked_error": str(d4.get("blocked_error", "")),
                "resource_detail": _blocked_resource_detail(d4),
            },
        ],
        "kv260_continuity": {
            "source_experiment_id": 5037,
            "honest_verdict": str(kv260.get("honest_verdict", "")),
            "live": kv260_live,
            "kv260_ssh_reachable": kv260.get("kv260_ssh_reachable") is True,
            "loaded_overlay": str(kv260.get("loaded_overlay", "")),
            "energy_smoke": dict(_mapping(kv260.get("energy_smoke"))),
        },
        "arc_level_bank": {
            "source_experiment_id": 5040,
            "honest_verdict": str(arc.get("honest_verdict", "")),
            "target_game": str(arc.get("target_game", "")),
            "target_level": _int(arc.get("target_level")),
            "new_levels_banked": new_levels_banked,
            "reproducible_total_levels_after": total_after,
            "no_arc_level_bank": no_arc_level_bank,
        },
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            rows.append({"path": str(rel_path), "sha256": file_sha256(path)})
    for source, rel_path in SOURCE_PATHS.items():
        path = root / rel_path
        if path.exists():
            rows.append({"source": source, "path": str(rel_path), "sha256": file_sha256(path)})
    return rows


def _pretest_gate_from_result(result: CommandResult) -> JsonDict:
    summary = command_summary(result)
    return {"ran": True, "green": result.exit_code == 0, **summary}


def _empty_pretest(reason: str) -> JsonDict:
    return {"ran": False, "green": False, "reason": reason}


def _transition_state(roadmaps: Mapping[str, Any], roadmaps_checked: Mapping[str, Any]) -> JsonDict:
    active = _mapping(roadmaps.get("active"))
    staged = _mapping(roadmaps.get("pre_staged"))
    active_status = _mapping(roadmaps_checked.get("active"))
    staged_status = _mapping(roadmaps_checked.get("pre_staged"))
    active_milestone = str(active.get("milestone", active_status.get("milestone", "")))
    staged_milestone = str(staged.get("milestone", staged_status.get("milestone", "")))
    return {
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "active_milestone_confirmed": active_milestone,
        "pre_staged_milestone_confirmed": staged_milestone,
        "active_roadmap_path": str(ROADMAP_ACTIVE_REL_PATH),
        "pre_staged_roadmap_path": str(ROADMAP_NEXT_REL_PATH)
        if staged_status.get("exists") is True
        else "",
        "pre_staged_roadmap_status": str(
            staged_status.get("status", "present_parsed" if staged_status.get("exists") else "")
        ),
        "activation_state": "already_active_or_activated_464"
        if active_milestone == NEXT_MILESTONE
        else "pre_staged_464_not_promoted_by_script",
        "active_conductor_changed": False,
    }


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    roadmaps: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    close_state: Mapping[str, Any],
    missing_fields: list[JsonDict] | None = None,
    duration_s: float,
) -> JsonDict:
    """Build the Exp 5042 transition artifact."""

    close = _mapping(close_state)
    d1_signal = _mapping(close.get("d1_lora_ebm_signal"))
    d2_result = _mapping(close.get("d2_scalar_uprm_result"))
    d3_result = _mapping(close.get("d3_ebrm_result"))
    axes = _list(close.get("blocked_confirmation_axes"))
    kv260 = _mapping(close.get("kv260_continuity"))
    arc = _mapping(close.get("arc_level_bank"))

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "transition": {
            **_transition_state(roadmaps, _mapping(preconditions_checked.get("roadmaps"))),
            "transition_performed": transition_performed,
        },
        "transition_performed": transition_performed,
        "close_state_463": dict(close_state),
        "d1_delta_vs_tuned_sc": d1_signal.get("delta_vs_tuned_sc"),
        "d1_ci_touches_zero": d1_signal.get("ci_touches_zero") is True,
        "d6_blocked_reason": str(_mapping(axes[0] if axes else {}).get("blocked_reason", "")),
        "d4_blocked_reason": str(_mapping(axes[1] if len(axes) > 1 else {}).get("blocked_reason", "")),
        "lora_ebm_signal": "real_but_underpowered"
        if d1_signal.get("real_signal") is True and d1_signal.get("underpowered") is True
        else "unknown",
        "scalar_uprm_result": "negative"
        if d2_result.get("negative_result") is True
        else "unknown",
        "ebrm_result": "tie_with_d1" if d3_result.get("tied_d1") is True else "unknown",
        "blocked_confirmation_axes": [
            str(_mapping(row).get("axis", "")) for row in axes if _mapping(row).get("axis")
        ],
        "kv260_continuity_live": kv260.get("live") is True,
        "no_arc_level_bank": arc.get("no_arc_level_bank") is True,
        "leaderboard_submission": False,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": max(0.0001, round(float(duration_s), 6)),
    }
    if missing_fields:
        payload["missing_required_fields"] = missing_fields
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run the record-only .463/.464 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    roadmaps, roadmaps_checked = check_roadmaps(root)
    preconditions: JsonDict = {"roadmaps": roadmaps_checked}

    blocker = roadmap_blocker(roadmaps_checked)
    if blocker:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker,
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=_empty_pretest("skipped_after_yaml_parse_failure"),
            transition_performed=False,
            close_state={},
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifacts, source_statuses = load_sources(root)
    preconditions["source_artifacts"] = source_statuses
    blocker = source_blocker(source_statuses)
    missing_fields = missing_required_fields(artifacts)
    if blocker or missing_fields:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker or "blocked_missing_required_field",
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=_empty_pretest("skipped_after_required_source_failure"),
            transition_performed=False,
            close_state={},
            missing_fields=missing_fields,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    pretest_result = command_runner(PRETEST_COMMAND, root)
    pretest_gate = _pretest_gate_from_result(pretest_result)
    if pretest_result.exit_code != 0:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_pretest_gate_failed",
            roadmaps=roadmaps,
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            close_state={},
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    close_state = build_close_state(artifacts)
    artifact = build_artifact(
        root=root,
        honest_verdict="complete_463_archived_464_activated_phase_d_power_confirmation",
        roadmaps=roadmaps,
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        close_state=close_state,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 5042 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")  # pragma: no cover - defensive validator
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")  # pragma: no cover
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")  # pragma: no cover
    if payload.get("prior_milestone") != PRIOR_MILESTONE:
        errors.append("invalid_prior_milestone")  # pragma: no cover
    if payload.get("next_milestone") != NEXT_MILESTONE:
        errors.append("invalid_next_milestone")  # pragma: no cover
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")  # pragma: no cover
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not blocked:
        if _float(payload.get("d1_delta_vs_tuned_sc")) != 0.08:
            errors.append("invalid_d1_delta_vs_tuned_sc")  # pragma: no cover
        if payload.get("d1_ci_touches_zero") is not True:
            errors.append("invalid_d1_ci_touches_zero")  # pragma: no cover
        if payload.get("d6_blocked_reason") != "blocked_judge_server":
            errors.append("invalid_d6_blocked_reason")  # pragma: no cover
        if payload.get("d4_blocked_reason") != "blocked_second_corpus_unavailable":
            errors.append("invalid_d4_blocked_reason")  # pragma: no cover
        if payload.get("lora_ebm_signal") != "real_but_underpowered":
            errors.append("invalid_lora_ebm_signal")  # pragma: no cover
        if payload.get("scalar_uprm_result") != "negative":
            errors.append("invalid_scalar_uprm_result")  # pragma: no cover
        if payload.get("ebrm_result") != "tie_with_d1":
            errors.append("invalid_ebrm_result")  # pragma: no cover
        if payload.get("kv260_continuity_live") is not True:
            errors.append("invalid_kv260_continuity_live")  # pragma: no cover
        if payload.get("no_arc_level_bank") is not True:
            errors.append("invalid_no_arc_level_bank")  # pragma: no cover
        close = _mapping(payload.get("close_state_463"))
        moat = _mapping(close.get("moat_verdict"))
        if moat.get("decision") != "EXECUTION-INCOMPLETE":
            errors.append("invalid_moat_decision")  # pragma: no cover
        if moat.get("moat_realized") is not False:
            errors.append("invalid_moat_realized")  # pragma: no cover
        if moat.get("moat_retired_bounded") is not False:
            errors.append("invalid_moat_retired_bounded")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 5042 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 5042 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
