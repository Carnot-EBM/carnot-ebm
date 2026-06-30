"""Experiment 5028: archive .462, activate .463, and record the close-state.

Spec refs: REQ-CAPSTONE-5028, SCENARIO-CAPSTONE-5028.

This record-only transition runs the active-roadmap and offline-arcade
preconditions first, uses an own-file ``--no-cov`` pre-test gate, and records
the true .462 PHASE D close-state for the third execution attempt in .463.
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

from carnot.experiment_5014_archive_461_activate_462 import (  # noqa: E402
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
EXPERIMENT = "experiment_5028_archive_462_activate_463"
EXPERIMENT_ID = 5028
SCHEMA = "carnot.exp5028.archive_462_activate_463.v1"
RANDOM_SEED = 20260630
ARCHIVED_MILESTONE = "2026.06.462"
ACTIVATED_MILESTONE = "2026.06.463"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_5028_archive_462_activate_463.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5027_capstone_v462.json")
B1_REL_PATH = Path("results/experiment_5015_genuine_sc_baseline_fix.json")
B2_REL_PATH = Path("results/experiment_5016_shared_logprob_candidate_cache.json")
D1_REL_PATH = Path("results/experiment_5017_lora_ebm_scorer_musr_v2.json")
D2_REL_PATH = Path("results/experiment_5018_uprm_replication_v2.json")
D3_REL_PATH = Path("results/experiment_5019_d3.json")
D6_REL_PATH = Path("results/experiment_5020_uncertainty_routed_cascade.json")
D4_REL_PATH = Path("results/experiment_5021_moat_second_corpus_v2.json")
HARNESS_REL_PATH = Path("python/carnot/moat_benchmark_harness.py")
PRETEST_COMMAND = [
    ".venv/bin/pytest",
    "tests/python/test_experiment_5028_archive_462_activate_463.py",
    "-q",
    "--no-cov",
]
SPEC_REFS = [
    "REQ-CAPSTONE-5028",
    "SCENARIO-CAPSTONE-5028",
    "SCENARIO-CAPSTONE-5028-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-5028-FIELD-PRINCIPLES",
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
            "complete_462_archived_463_activated_phase_d_third_attempt."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; "
            "0.0001s floor)."
        )
    },
    "phase_d_third_execution_attempt": {
        "principle": (
            "true -- .463 is the THIRD PHASE D milestone; the first two both "
            "went EXECUTION-INCOMPLETE on infra."
        )
    },
    "prior_milestone_root_causes": {
        "principle": (
            "the two .462 infra root causes .463 fixes (D1 404 base "
            "`Qwen/Qwen3.5-1.7B`, B2 0-rows generator)."
        )
    },
    "reusable_b1_baseline": {
        "principle": (
            "the .462 B1 genuine tuned-SC 0.585 / oracle@K 0.865 / degeneracy "
            "guard (moat_benchmark_harness.py) -- reused, not rebuilt."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry "
            "(69; ARC LOCKED + opportunistic)."
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
    "phase_d_third_execution_attempt",
    "prior_milestone_root_causes",
    "reusable_b1_baseline",
    "reproducible_total_levels",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "poison_test_resolved",
    "close_state_462",
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
            except OSError:  # pragma: no cover - resource status records the failure
                return ""
    return ""


def _resource_detail(artifact: Mapping[str, Any], resource: str) -> str:
    for row in _list(artifact.get("preconditions_checked")):
        mapping = _mapping(row)
        if mapping.get("resource") == resource:
            return str(mapping.get("detail", ""))
    return ""


def _first_failed_resource_detail(artifact: Mapping[str, Any]) -> str:
    for row in _list(artifact.get("preconditions_checked")):
        mapping = _mapping(row)
        if mapping.get("available") is False:
            return str(mapping.get("detail", ""))
    return ""


def _flagged_from_capstone(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows = [dict(_mapping(item)) for item in _list(capstone.get("flagged_artifacts_skipped"))]
    return [
        row
        for row in rows
        if _int(row.get("experiment_id")) in {5017, 5018, 5020, 5021}
        or str(row.get("source", "")).startswith(("D1", "D2", "D6", "D4"))
    ]


def _arm_flag_row(artifact: Mapping[str, Any], arm_id: str, experiment_id: int, path: Path) -> JsonDict:
    return {  # pragma: no cover - fallback for absent capstone flagged rows
        "arm_id": arm_id,
        "experiment_id": experiment_id,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "path": str(path),
        "reason": "flagged_adversarial" if artifact.get("flagged_adversarial") is True else "",
    }


def _diffusion_status(capstone: Mapping[str, Any]) -> JsonDict:
    diffusion = _mapping(capstone.get("diffusiongemma_gate_status"))
    return {
        "status": str(diffusion.get("status", "STILL-PENDING" if capstone else "")),
        "activation": str(diffusion.get("activation", "not_activated" if capstone else "")),
        "autonomously_flipped_to_met": diffusion.get("autonomously_flipped_to_met") is True,
        "conditions_satisfied_off_arc": diffusion.get("conditions_satisfied_off_arc") is True,
        "operator_gated": diffusion.get("operator_gated") is True or bool(capstone),
    }


def _d3_gated_on(d3: Mapping[str, Any]) -> str:
    for row in _list(d3.get("gates_evaluated")):
        mapping = _mapping(row)
        if str(mapping.get("upstream")) == "exp5017-d1":
            return "D1." + str(mapping.get("artifact_field", ""))
    return ""


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
        "capstone_v462": _json_resource_status(root / CAPSTONE_REL_PATH),
        "b1_genuine_sc_baseline": {
            **_json_resource_status(root / B1_REL_PATH),
            "optional": True,
        },
        "moat_benchmark_harness": {
            "path": str(root / HARNESS_REL_PATH),
            "exists": (root / HARNESS_REL_PATH).exists(),
            "optional": True,
        },
    }


def precondition_blocker(preconditions_checked: Mapping[str, Any]) -> str:
    """Return the blocked verdict for failed preconditions, or an empty string."""

    roadmap = _mapping(preconditions_checked.get("active_roadmap_yaml"))
    arcade = _mapping(preconditions_checked.get("offline_arcade"))
    registry = _mapping(preconditions_checked.get("registry"))
    capstone = _mapping(preconditions_checked.get("capstone_v462"))
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
        return "blocked_capstone_v462_missing"
    if capstone.get("loadable") is not True:
        return "blocked_capstone_v462_unloadable"
    return ""


def build_reusable_b1_baseline(root: Path, capstone: Mapping[str, Any]) -> JsonDict:
    """Build the B1 reuse row from Exp5015, falling back to capstone rollup."""

    b1 = _read_json_object_safe(root / B1_REL_PATH)
    rollup = _mapping(_mapping(capstone.get("infra_rollup")).get("b1_genuine_sc_baseline"))
    accuracy = _float(
        b1.get("genuine_tuned_sc_accuracy"),
        _float(rollup.get("genuine_tuned_sc_accuracy")),
    )
    oracle = _float(b1.get("oracle_at_k"), _float(rollup.get("oracle_at_k")))
    return {
        "source_experiment_id": 5015,
        "source_path": str(B1_REL_PATH),
        "honest_verdict": str(
            b1.get("honest_verdict", rollup.get("honest_verdict", ""))
        ),
        "genuine_tuned_sc_accuracy": accuracy,
        "oracle_at_k": oracle,
        "headroom_delta": round(oracle - accuracy, 6),
        "headroom_present": b1.get("genuine_headroom_present") is True
        or rollup.get("genuine_headroom_present") is True,
        "degeneracy_guard_fires": b1.get("degeneracy_guard_fires") is True
        or rollup.get("degeneracy_guard_fires") is True,
        "harness_module_path": str(b1.get("harness_module_path", HARNESS_REL_PATH)),
        "harness_module_present": (root / HARNESS_REL_PATH).exists(),
        "no_new_llm_generation": b1.get("no_new_llm_generation") is True,
        "reuse_action": "reuse_not_rebuild",
    }


def build_prior_milestone_root_causes(root: Path) -> list[JsonDict]:
    """Return the two pure-infra .462 root causes that .463 fixes."""

    d1 = _read_json_object_safe(root / D1_REL_PATH)
    b2 = _read_json_object_safe(root / B2_REL_PATH)
    d1_detail = _resource_detail(d1, "trainable_qwen_base")
    b2_detail = _first_failed_resource_detail(b2)
    return [
        {
            "source": "D1_LORA_EBM",
            "source_experiment_id": 5017,
            "defect": "invalid_hf_base_repository_404",
            "base_model": str(_mapping(d1.get("model_specs")).get("base_model", "Qwen/Qwen3.5-1.7B")),
            "honest_verdict": str(d1.get("honest_verdict", "")),
            "evidence": d1_detail,
            "scientific_null": False,
            "addressed_by_463": "real_cached_base_resolver_and_moat_trainer_smoke",
        },
        {
            "source": "B2_LOGPROB_CACHE",
            "source_experiment_id": 5016,
            "defect": "zero_row_logprob_cache_generator",
            "honest_verdict": str(b2.get("honest_verdict", "")),
            "duration_s": _float(b2.get("duration_s")),
            "n_cached_rows": _int(b2.get("n_cached_rows")),
            "candidate_cache_built": b2.get("candidate_cache_built") is True,
            "evidence": b2_detail,
            "cascaded_to": ["D2", "D3"],
            "scientific_null": False,
            "addressed_by_463": "incremental_rescore_existing_candidates_not_regenerate",
        },
    ]


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the true .462 close-state from registry, roadmap, and D arms."""

    registry = _read_yaml_object_safe(root / REGISTRY_REL_PATH)
    capstone = _read_json_object_safe(root / CAPSTONE_REL_PATH)
    d1 = _read_json_object_safe(root / D1_REL_PATH)
    b2 = _read_json_object_safe(root / B2_REL_PATH)
    d2 = _read_json_object_safe(root / D2_REL_PATH)
    d3 = _read_json_object_safe(root / D3_REL_PATH)
    d6 = _read_json_object_safe(root / D6_REL_PATH)
    d4 = _read_json_object_safe(root / D4_REL_PATH)
    roadmap_text = _roadmap_text(root)

    registry_total = _int(
        registry.get("reproducible_total_levels"),
        _int(capstone.get("reproducible_total_levels")),
    )
    moat = _mapping(capstone.get("moat_verdict"))
    pointer = dict(_mapping(capstone.get("next_milestone_pointer")))
    pointer_plan = str(pointer.get("plan", ""))
    pointer["route_off_codex_if_same_arm_bails_twice"] = (
        "route off codex" in pointer_plan.lower() and "same arm bails twice" in pointer_plan.lower()
    )
    active_463 = _active_milestone(root)[0] == ACTIVATED_MILESTONE
    phase_d_majority = "PHASE D" in roadmap_text and "majority" in roadmap_text.lower()
    arc_locked = registry_total == 69 and (
        _mapping(capstone.get("arc_deliverable_locked")).get("locked") is True
        or "ARC LOCKED" in roadmap_text
    )
    flagged = _flagged_from_capstone(capstone)
    if not flagged:
        flagged = [  # pragma: no cover - capstone normally carries the skipped rows
            _arm_flag_row(d1, "D1", 5017, D1_REL_PATH),
            _arm_flag_row(d2, "D2", 5018, D2_REL_PATH),
            _arm_flag_row(d6, "D6", 5020, D6_REL_PATH),
            _arm_flag_row(d4, "D4", 5021, D4_REL_PATH),
        ]

    root_causes = build_prior_milestone_root_causes(root)
    reusable_b1 = build_reusable_b1_baseline(root, capstone)
    return {
        "summary": "v462_execution_incomplete_phase_d_infra_failures_recorded_for_v463",
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "capstone_ready": capstone.get("capstone_ready") is True,
        },
        "moat_verdict": {
            "decision": str(moat.get("decision", "unknown")),
            "state": str(moat.get("state", "")),
            "moat_realized": moat.get("moat_realized") is True,
            "moat_retired_bounded": moat.get("moat_retired_bounded") is True,
            "execution_incomplete_arms": [
                dict(_mapping(item)) for item in _list(moat.get("execution_incomplete_arms"))
            ],
            "summary": str(moat.get("summary", "")),
        },
        "next_milestone_pointer": pointer,
        "prior_milestone_root_causes": root_causes,
        "reusable_b1_baseline": reusable_b1,
        "d1_404_base": {
            "arm_id": "D1",
            "source_experiment_id": 5017,
            "honest_verdict": str(d1.get("honest_verdict", "")),
            "flagged_adversarial": d1.get("flagged_adversarial") is True,
            "base_model": str(_mapping(d1.get("model_specs")).get("base_model", "")),
            "resource_detail": _resource_detail(d1, "trainable_qwen_base"),
            "scorer_trained": d1.get("scorer_trained") is True,
            "train_loss": d1.get("train_loss"),
            "scientific_null": False,
        },
        "b2_zero_row_logprob_cache": {
            "arm_id": "B2",
            "source_experiment_id": 5016,
            "honest_verdict": str(b2.get("honest_verdict", "")),
            "duration_s": _float(b2.get("duration_s")),
            "n_cached_rows": _int(b2.get("n_cached_rows")),
            "candidate_cache_built": b2.get("candidate_cache_built") is True,
            "resource_detail": _first_failed_resource_detail(b2),
            "scientific_null": False,
        },
        "d2_cascade_block": {
            "arm_id": "D2",
            "source_experiment_id": 5018,
            "honest_verdict": str(d2.get("honest_verdict", "")),
            "flagged_adversarial": d2.get("flagged_adversarial") is True,
            "blocked_on": "b2_logprob_cache",
            "resource_detail": _resource_detail(d2, "b2_logprob_cache"),
            "scientific_null": False,
        },
        "d3_cascade_block": {
            "arm_id": "D3",
            "source_experiment_id": 5019,
            "honest_verdict": str(d3.get("honest_verdict", "")),
            "blocked_at_layer": str(d3.get("blocked_at_layer", "")),
            "gate_check_summary": str(d3.get("gate_check_summary", "")),
            "gated_on": _d3_gated_on(d3),
            "scientific_null": False,
        },
        "d6_flagged_execution_failure": {
            "arm_id": "D6",
            "source_experiment_id": 5020,
            "honest_verdict": str(d6.get("honest_verdict", "")),
            "flagged_adversarial": d6.get("flagged_adversarial") is True,
            "scientific_null": False,
        },
        "d4_flagged_execution_failure": {
            "arm_id": "D4",
            "source_experiment_id": 5021,
            "honest_verdict": str(d4.get("honest_verdict", "")),
            "flagged_adversarial": d4.get("flagged_adversarial") is True,
            "scientific_null": False,
        },
        "flagged_adversarial_skipped": flagged,
        "diffusiongemma_gate_status": _diffusion_status(capstone),
        "phase_d": {
            "majority_lever_for_463": phase_d_majority and active_463,
            "third_execution_attempt": active_463
            and str(moat.get("decision", "")) == "EXECUTION-INCOMPLETE",
            "prior_execution_incomplete_milestones": ["2026.06.461", "2026.06.462"],
        },
        "arc": {
            "locked": arc_locked,
            "mode": "opportunistic_only" if arc_locked else "",
        },
        "do_not_queue": ["energy-as-ARC", "verifier-as-reward"],
        "reproducible_total_levels": registry_total,
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [
        REGISTRY_REL_PATH,
        ROADMAP_ACTIVE_REL_PATH,
        CAPSTONE_REL_PATH,
        B1_REL_PATH,
        B2_REL_PATH,
        D1_REL_PATH,
        D2_REL_PATH,
        D3_REL_PATH,
        D6_REL_PATH,
        D4_REL_PATH,
        HARNESS_REL_PATH,
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
    """Build the Exp 5028 transition artifact."""

    close_state = build_close_state(root)
    blocked = honest_verdict.startswith("blocked_")
    active_milestone, active_roadmap_path = _active_milestone(root)
    phase_d = _mapping(close_state.get("phase_d"))
    arc = _mapping(close_state.get("arc"))
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
        "phase_d_third_execution_attempt": phase_d.get("third_execution_attempt") is True
        and not blocked,
        "prior_milestone_root_causes": close_state["prior_milestone_root_causes"],
        "reusable_b1_baseline": close_state["reusable_b1_baseline"],
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
        "close_state_462": close_state,
        "diffusiongemma_gate_status": close_state["diffusiongemma_gate_status"],
        "phase_d_majority_lever": phase_d.get("majority_lever_for_463") is True and not blocked,
        "arc_locked": arc.get("locked") is True,
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
    """Run the record-only .462/.463 transition workflow."""

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
        honest_verdict="complete_462_archived_463_activated_phase_d_third_attempt",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        activation_state="already_active_or_activated_463",
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 5028 artifact."""

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
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    if not isinstance(payload.get("close_state_462"), Mapping):
        errors.append("invalid_close_state_462")  # pragma: no cover
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not blocked:
        if payload.get("phase_d_third_execution_attempt") is not True:
            errors.append("invalid_phase_d_third_execution_attempt")  # pragma: no cover
        cause_sources = [
            str(_mapping(row).get("source"))
            for row in _list(payload.get("prior_milestone_root_causes"))
        ]
        if cause_sources != ["D1_LORA_EBM", "B2_LOGPROB_CACHE"]:
            errors.append("invalid_prior_milestone_root_causes")  # pragma: no cover
        b1 = _mapping(payload.get("reusable_b1_baseline"))
        if _float(b1.get("genuine_tuned_sc_accuracy")) != 0.585:
            errors.append("invalid_reusable_b1_tuned_sc")  # pragma: no cover
        if _float(b1.get("oracle_at_k")) != 0.865:
            errors.append("invalid_reusable_b1_oracle_at_k")  # pragma: no cover
        if _mapping(payload.get("diffusiongemma_gate_status")).get("status") != "STILL-PENDING":
            errors.append("invalid_diffusiongemma_gate_status")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 5028 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 5028 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))
