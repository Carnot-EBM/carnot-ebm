"""Experiment 5014: archive .461, activate .462, and record the close-state.

Spec refs: REQ-CAPSTONE-5014, SCENARIO-CAPSTONE-5014.

This record-only transition runs the active-roadmap and offline-arcade
preconditions first, uses an own-file ``--no-cov`` pre-test gate, and records
the true .461 PHASE D close-state for the .462 re-execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4990_archive_459_activate_460 import (  # noqa: E402
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
    _json_resource_status,
    _list,
    _mapping,
    _read_json_object_safe,
    _read_yaml_object_safe,
    _yaml_resource_status,
)


CommandRunner = Callable[[list[str], Path], CommandResult]
EXPERIMENT = "experiment_5014_archive_461_activate_462"
EXPERIMENT_ID = 5014
SCHEMA = "carnot.exp5014.archive_461_activate_462.v1"
RANDOM_SEED = 20260630
ARCHIVED_MILESTONE = "2026.06.461"
ACTIVATED_MILESTONE = "2026.06.462"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_5014_archive_461_activate_462.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5013_capstone_v461.json")
D1_REL_PATH = Path("results/experiment_5003_lora_ebm_scorer_musr.json")
D2_REL_PATH = Path("results/experiment_5004_uprm_replication.json")
D3_REL_PATH = Path("results/experiment_5005_ebrm_uncertainty_verifier.json")
D4_REL_PATH = Path("results/experiment_5006_moat_second_corpus.json")
PRETEST_COMMAND = [
    ".venv/bin/pytest",
    "tests/python/test_experiment_5014_archive_461_activate_462.py",
    "-q",
    "--no-cov",
]
SPEC_REFS = [
    "REQ-CAPSTONE-5014",
    "SCENARIO-CAPSTONE-5014",
    "SCENARIO-CAPSTONE-5014-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-5014-FIELD-PRINCIPLES",
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
            "complete_461_archived_462_activated_phase_d_continues."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; "
            "0.0001s floor)."
        )
    },
    "phase_d_first_real_test": {
        "principle": (
            "true -- .462 is the FIRST milestone that actually measures the moat "
            "(.461 was all execution failures)."
        )
    },
    "prior_milestone_moat_verdict": {
        "principle": (
            "MIXED-SCOPED carried from exp5013 -- no realized win, no bounded "
            "retirement (D1/D2 nulls were not clean)."
        )
    },
    "execution_defects_to_fix": {
        "principle": (
            "the three .461 PHASE D execution defects .462 addresses (D1 "
            "skeleton-bail, D2 logprob-cache block, D3 degenerate-always-abstain)."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (69; "
            "ARC LOCKED + opportunistic)."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "archived_milestone",
    "activated_milestone",
    "honest_verdict",
    "inference_substrate",
    "phase_d_first_real_test",
    "prior_milestone_moat_verdict",
    "execution_defects_to_fix",
    "reproducible_total_levels",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "poison_test_resolved",
    "close_state_461",
    "diffusiongemma_gate_status",
    "phase_d_majority_lever",
    "arc_locked",
    "leaderboard_submission",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def _active_milestone(root: Path) -> tuple[str, str]:
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        payload = _read_yaml_object_safe(root / rel_path)
        milestone = payload.get("milestone")
        if milestone:
            return str(milestone), str(rel_path)
    return "unknown", str(ROADMAP_ACTIVE_REL_PATH)


def _roadmap_text(root: Path) -> str:
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - resource status already records this
                return ""
    return ""


def _flagged_from_capstone(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows = [dict(_mapping(item)) for item in _list(capstone.get("flagged_artifacts_skipped"))]
    return [
        row
        for row in rows
        if _int(row.get("experiment_id")) in {5003, 5004, 5006}
        or str(row.get("source", "")).startswith(("D1", "D2", "D4"))
    ]


def _arm_flag_row(artifact: Mapping[str, Any], arm_id: str, experiment_id: int, path: Path) -> JsonDict:
    return {
        "arm_id": arm_id,
        "experiment_id": experiment_id,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "path": str(path),
        "reason": "flagged_adversarial" if artifact.get("flagged_adversarial") is True else "",
    }


def _failed_resource_detail(artifact: Mapping[str, Any], resource: str) -> str:
    for row in _list(artifact.get("preconditions_checked")):
        mapping = _mapping(row)
        if mapping.get("resource") == resource:
            return str(mapping.get("detail", ""))
    return ""


def _max_abstention_rate(d3: Mapping[str, Any]) -> float:
    calibration = _mapping(d3.get("uncertainty_calibration"))
    rates = [
        _float(_mapping(row).get("abstain_rate"))
        for row in _list(calibration.get("calibration_curve"))
        if "abstain_rate" in _mapping(row)
    ]
    return max(rates) if rates else 0.0


def _tuned_sc_k(d3: Mapping[str, Any]) -> int:
    evaluation = _mapping(d3.get("evaluation"))
    tuned = _mapping(evaluation.get("tuned_self_consistency"))
    config = _mapping(tuned.get("config"))
    model_specs = _mapping(d3.get("model_specs"))
    spec_config = _mapping(model_specs.get("tuned_self_consistency_config"))
    return _int(config.get("k"), _int(spec_config.get("k")))


def _best_clean_arm(capstone: Mapping[str, Any], d3: Mapping[str, Any]) -> JsonDict:
    best = dict(_mapping(capstone.get("best_arm_and_delta")))
    if not best:
        per_arm = [dict(_mapping(item)) for item in _list(capstone.get("per_arm_table"))]
        best = per_arm[0] if per_arm else {}
    evaluation = _mapping(d3.get("evaluation"))
    return {
        "arm": str(best.get("arm", "EBRM")),
        "arm_id": str(best.get("arm_id", "D3")),
        "corpus": str(best.get("corpus", "MuSR")),
        "delta_vs_tuned_sc": _float(
            best.get("delta_vs_tuned_sc"), _float(d3.get("delta_vs_tuned_sc"))
        ),
        "paired_ci95": [
            _float(item) for item in _list(best.get("paired_ci95") or d3.get("paired_ci95"))
        ],
        "mcnemar_p": _float(best.get("mcnemar_p"), _float(d3.get("mcnemar_p"))),
        "selection_accuracy": _float(
            best.get("selection_accuracy"), _float(d3.get("ebrm_selection_accuracy"))
        ),
        "tuned_sc_accuracy": _float(
            best.get("tuned_sc_accuracy"), _float(d3.get("tuned_sc_accuracy"))
        ),
        "point_estimate_accuracy": _float(evaluation.get("point_estimate_accuracy")),
        "headroom_present": best.get("headroom_present") is True
        or d3.get("headroom_present") is True,
        "verifier_is_oracle": best.get("verifier_is_oracle") is True
        or d3.get("verifier_is_oracle") is True,
        "win_vs_tuned_sc": best.get("win_vs_tuned_sc") is True,
        "source_experiment_id": _int(best.get("source_experiment_id"), 5005),
    }


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the mandatory active-roadmap and offline-arcade checks first."""

    roadmap_command = [
        ".venv/bin/python",
        "-c",
        (
            "import yaml,os; p="
            + repr(str(root / ROADMAP_ACTIVE_REL_PATH))
            + "; q="
            + repr(str(root / ROADMAP_NEXT_REL_PATH))
            + "; f=p if os.path.exists(p) else q; yaml.safe_load(open(f)); "
            "print('ok',f)"
        ),
    ]
    arcade_command = [
        ".venv/bin/python",
        "-c",
        "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()",
    ]
    roadmap_result = command_runner(roadmap_command, root)
    arcade_result = command_runner(arcade_command, root)
    active_exists = (root / ROADMAP_ACTIVE_REL_PATH).exists()
    next_exists = (root / ROADMAP_NEXT_REL_PATH).exists()
    selected = ROADMAP_ACTIVE_REL_PATH if active_exists else ROADMAP_NEXT_REL_PATH
    return {
        "active_roadmap_yaml": {
            **command_summary(roadmap_result),
            "primary_path": str(ROADMAP_ACTIVE_REL_PATH),
            "fallback_path": str(ROADMAP_NEXT_REL_PATH),
            "selected_path": str(selected),
            "active_exists": active_exists,
            "next_exists": next_exists,
        },
        "offline_arcade": command_summary(arcade_result),
        "registry": _yaml_resource_status(root / REGISTRY_REL_PATH),
        "capstone_v461": _json_resource_status(root / CAPSTONE_REL_PATH),
    }


def precondition_blocker(preconditions_checked: Mapping[str, Any]) -> str:
    """Return the blocked verdict for failed preconditions, or an empty string."""

    roadmap = _mapping(preconditions_checked.get("active_roadmap_yaml"))
    arcade = _mapping(preconditions_checked.get("offline_arcade"))
    registry = _mapping(preconditions_checked.get("registry"))
    capstone = _mapping(preconditions_checked.get("capstone_v461"))
    if roadmap.get("passed") is not True:
        if not roadmap.get("active_exists") and not roadmap.get("next_exists"):
            return "blocked_roadmap_yaml_missing"
        return "blocked_roadmap_yaml_unparseable"
    if arcade.get("passed") is not True:
        return "blocked_offline_arcade_unavailable"
    if registry.get("exists") is not True:
        return "blocked_arc_solve_registry_missing"
    if registry.get("loadable") is not True:
        return "blocked_arc_solve_registry_unloadable"
    if capstone.get("exists") is not True:
        return "blocked_capstone_v461_missing"
    if capstone.get("loadable") is not True:
        return "blocked_capstone_v461_unloadable"
    return ""


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the true .461 close-state from registry, roadmap, and D arms."""

    registry = _read_yaml_object_safe(root / REGISTRY_REL_PATH)
    capstone = _read_json_object_safe(root / CAPSTONE_REL_PATH)
    d1 = _read_json_object_safe(root / D1_REL_PATH)
    d2 = _read_json_object_safe(root / D2_REL_PATH)
    d3 = _read_json_object_safe(root / D3_REL_PATH)
    d4 = _read_json_object_safe(root / D4_REL_PATH)
    roadmap_text = _roadmap_text(root)

    registry_total = _int(
        registry.get("reproducible_total_levels"),
        _int(capstone.get("reproducible_total_levels")),
    )
    moat = dict(_mapping(capstone.get("moat_verdict")))
    clean_arm = _best_clean_arm(capstone, d3)
    diffusion = _mapping(capstone.get("diffusiongemma_gate_status"))
    if not diffusion and isinstance(capstone.get("diffusiongemma_gate_status"), str):
        diffusion = {"status": str(capstone.get("diffusiongemma_gate_status"))}
    diffusion_status = {
        "status": str(diffusion.get("status", "STILL-PENDING" if capstone else "")),
        "activation": str(diffusion.get("activation", "not_activated" if capstone else "")),
        "autonomously_flipped_to_met": diffusion.get("autonomously_flipped_to_met") is True,
        "conditions_satisfied_off_arc": diffusion.get("conditions_satisfied_off_arc") is True,
        "operator_gated": diffusion.get("operator_gated") is True or bool(capstone),
    }
    phase_d_majority = "PHASE D" in roadmap_text and "majority" in roadmap_text.lower()
    active_462 = _active_milestone(root)[0] == ACTIVATED_MILESTONE
    capstone_ready = capstone.get("capstone_ready") is True
    arc_locked = registry_total == 69 and (
        capstone.get("arc_deliverable_locked") is True or "ARC LOCKED" in roadmap_text
    )
    flagged = _flagged_from_capstone(capstone)
    if not flagged:
        flagged = [
            _arm_flag_row(d1, "D1", 5003, D1_REL_PATH),
            _arm_flag_row(d2, "D2", 5004, D2_REL_PATH),
            _arm_flag_row(d4, "D4", 5006, D4_REL_PATH),
        ]

    return {
        "summary": "v461_mixed_scoped_phase_d_execution_failures_recorded_for_v462",
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "capstone_ready": capstone_ready,
        },
        "moat_verdict": {
            "decision": str(moat.get("decision", "unknown")),
            "state": str(moat.get("state", "")),
            "moat_realized": moat.get("moat_realized") is True,
            "moat_retired_bounded": moat.get("moat_retired_bounded") is True,
            "summary": str(moat.get("summary", "")),
        },
        "clean_arm": clean_arm,
        "d1_skeleton_bail": {
            "arm_id": "D1",
            "source_experiment_id": 5003,
            "honest_verdict": str(d1.get("honest_verdict", "")),
            "flagged_adversarial": d1.get("flagged_adversarial") is True,
            "n_pairs": _int(d1.get("n_pairs")),
            "train_loss": d1.get("train_loss"),
            "all_preconditions_available": all(
                _mapping(row).get("available") is True
                for row in _list(d1.get("preconditions_checked"))
            ),
            "scientific_null": False,
            "defect": "skeleton_bail_never_trained",
        },
        "d2_logprob_cache_block": {
            "arm_id": "D2",
            "source_experiment_id": 5004,
            "honest_verdict": str(d2.get("honest_verdict", "")),
            "flagged_adversarial": d2.get("flagged_adversarial") is True,
            "blocked_on": "uprm_logprob_candidate_cache",
            "resource_detail": _failed_resource_detail(d2, "uprm_logprob_candidate_cache"),
            "fresh_generation_disabled": "CARNOT_UPRM_ENABLE_FRESH_GENERATION=''"
            in _failed_resource_detail(d2, "uprm_logprob_candidate_cache"),
            "scientific_null": False,
            "defect": "logprob_candidate_cache_block",
        },
        "d3_degeneracy": {
            "arm_id": "D3",
            "source_experiment_id": 5005,
            "honest_verdict": str(d3.get("honest_verdict", "")),
            "flagged_adversarial": d3.get("flagged_adversarial") is True,
            "abstention_rate": _max_abstention_rate(d3),
            "tuned_sc_k": _tuned_sc_k(d3),
            "weak_base_scorer": str(_mapping(d3.get("model_specs")).get("base_scorer", "")),
            "delta_vs_tuned_sc": clean_arm["delta_vs_tuned_sc"],
            "paired_ci95": clean_arm["paired_ci95"],
            "mcnemar_p": clean_arm["mcnemar_p"],
            "scientific_null": False,
            "defect": "degenerate_always_abstain_to_strawman_sc",
        },
        "d4_cross_corpus_skeleton_bail": {
            "arm_id": "D4",
            "source_experiment_id": 5006,
            "honest_verdict": str(d4.get("honest_verdict", "")),
            "flagged_adversarial": d4.get("flagged_adversarial") is True,
            "usable_best_verifier_missing": _failed_resource_detail(d4, "d1_verifier") != ""
            or _failed_resource_detail(d4, "d2_verifier") != "",
            "scientific_null": False,
            "defect": "cross_corpus_skeleton_bail",
        },
        "flagged_adversarial_skipped": flagged,
        "diffusiongemma_gate_status": diffusion_status,
        "phase_d": {
            "majority_lever_for_462": phase_d_majority and active_462,
            "first_real_test_in_462": (
                phase_d_majority
                and active_462
                and str(moat.get("decision", "")) == "MIXED-SCOPED"
                and not bool(moat.get("moat_realized"))
                and not bool(moat.get("moat_retired_bounded"))
            ),
            "arc_sprint_retired": "sprint" in roadmap_text.lower()
            and "retired" in roadmap_text.lower(),
        },
        "arc": {
            "locked": arc_locked,
            "mode": "opportunistic_only" if arc_locked else "",
        },
        "do_not_queue": ["energy-as-ARC", "verifier-as-reward"],
        "prior_milestone_moat_verdict": str(moat.get("decision", "unknown")),
        "reproducible_total_levels": registry_total,
    }


def build_execution_defects_to_fix(close_state: Mapping[str, Any]) -> list[JsonDict]:
    """Return the three .462 fixes requested by the transition contract."""

    d1 = _mapping(close_state.get("d1_skeleton_bail"))
    d2 = _mapping(close_state.get("d2_logprob_cache_block"))
    d3 = _mapping(close_state.get("d3_degeneracy"))
    return [
        {
            "arm_id": "D1",
            "source_experiment_id": 5003,
            "defect": "skeleton_bail_never_trained",
            "evidence": f"n_pairs={_int(d1.get('n_pairs'))}, train_loss={d1.get('train_loss')}",
            "scientific_null": False,
            "addressed_by_462": "train_first_lora_ebm_with_smaller_qwen_base",
        },
        {
            "arm_id": "D2",
            "source_experiment_id": 5004,
            "defect": "logprob_cache_block_fresh_generation_disabled",
            "evidence": str(d2.get("resource_detail", "")),
            "scientific_null": False,
            "addressed_by_462": "shared_logprob_enriched_candidate_cache",
        },
        {
            "arm_id": "D3",
            "source_experiment_id": 5005,
            "defect": "degenerate_always_abstain_to_k1_tuned_sc",
            "evidence": (
                f"abstention_rate={_float(d3.get('abstention_rate'))}, "
                f"tuned_sc_k={_int(d3.get('tuned_sc_k'))}"
            ),
            "scientific_null": False,
            "addressed_by_462": "genuine_k_way_sc_baseline_and_abstention_guard",
        },
    ]


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [
        REGISTRY_REL_PATH,
        ROADMAP_ACTIVE_REL_PATH,
        CAPSTONE_REL_PATH,
        D1_REL_PATH,
        D2_REL_PATH,
        D3_REL_PATH,
        D4_REL_PATH,
    ]
    if (root / ROADMAP_NEXT_REL_PATH).exists():
        rel_paths.insert(2, ROADMAP_NEXT_REL_PATH)
    return [
        {"path": str(rel_path), "sha256": file_sha256(root / rel_path)}
        for rel_path in rel_paths
        if (root / rel_path).exists()
    ]


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    activation_state: str,
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 5014 transition artifact."""

    close_state = build_close_state(root)
    blocked = honest_verdict.startswith("blocked_")
    active_milestone, active_roadmap_path = _active_milestone(root)
    execution_defects = build_execution_defects_to_fix(close_state)
    phase_d_first_real = _mapping(close_state.get("phase_d")).get("first_real_test_in_462") is True
    phase_d_majority = _mapping(close_state.get("phase_d")).get("majority_lever_for_462") is True
    arc_locked = _mapping(close_state.get("arc")).get("locked") is True
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "phase_d_first_real_test": phase_d_first_real and not blocked,
        "prior_milestone_moat_verdict": close_state["prior_milestone_moat_verdict"],
        "execution_defects_to_fix": execution_defects,
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "transition": {
            "archived_milestone": ARCHIVED_MILESTONE,
            "activated_milestone": ACTIVATED_MILESTONE,
            "active_milestone_confirmed": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "activation_state": activation_state,
            "transition_performed": transition_performed,
        },
        "transition_performed": transition_performed,
        "poison_test_resolved": dict(poison_test_resolved),
        "close_state_461": close_state,
        "diffusiongemma_gate_status": close_state["diffusiongemma_gate_status"],
        "phase_d_majority_lever": phase_d_majority and not blocked,
        "arc_locked": arc_locked,
        "leaderboard_submission": False,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": max(0.0001, round(float(duration_s), 6)),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _pretest_gate_from_result(result: CommandResult) -> JsonDict:
    summary = command_summary(result)
    return {"ran": True, "green": result.exit_code == 0, **summary}


def run(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run the record-only .461/.462 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(root, command_runner)
    blocker = precondition_blocker(preconditions)
    no_poison = {"quarantined": False, "test": "", "reason": ""}
    if blocker:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            pretest_gate={
                "ran": False,
                "green": False,
                "reason": "skipped_after_precondition_failure",
            },
            transition_performed=False,
            activation_state="blocked_missing_or_failed_precondition",
            poison_test_resolved=no_poison,
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
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            activation_state="blocked_pretest_gate_failed",
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        root=root,
        honest_verdict="complete_461_archived_462_activated_phase_d_continues",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        activation_state="already_active_or_activated_462",
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 5014 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")  # pragma: no cover - defensive validator
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")  # pragma: no cover
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")  # pragma: no cover
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")  # pragma: no cover
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")  # pragma: no cover
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    if not isinstance(payload.get("close_state_461"), Mapping):
        errors.append("invalid_close_state_461")  # pragma: no cover
    if not blocked:
        if payload.get("phase_d_first_real_test") is not True:
            errors.append("invalid_phase_d_first_real_test")  # pragma: no cover
        if payload.get("prior_milestone_moat_verdict") != "MIXED-SCOPED":
            errors.append("invalid_prior_milestone_moat_verdict")  # pragma: no cover
        defect_ids = [
            str(_mapping(row).get("arm_id")) for row in _list(payload.get("execution_defects_to_fix"))
        ]
        if defect_ids != ["D1", "D2", "D3"]:
            errors.append("invalid_execution_defects_to_fix")  # pragma: no cover
        if _mapping(payload.get("diffusiongemma_gate_status")).get("status") != "STILL-PENDING":
            errors.append("invalid_diffusiongemma_gate_status")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 5014 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 5014 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
