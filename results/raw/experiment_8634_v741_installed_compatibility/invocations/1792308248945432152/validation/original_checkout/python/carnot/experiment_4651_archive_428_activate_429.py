"""Experiment 4651: archive `.428`, activate `.429`, and record `.428` honestly.

Spec refs: REQ-CAPSTONE-4651, SCENARIO-CAPSTONE-4651,
SCENARIO-CAPSTONE-4651-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4651-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.archive_v391_activate_v392_4230 import (  # noqa: E402
    CommandResult,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    run_smart_subset,
)


JsonDict = dict[str, Any]
OfflineArcadeChecker = Callable[[], bool]
SmartSubsetChecker = Callable[[Path], CommandResult]

EXPERIMENT = "experiment_4651_archive_428_activate_429"
EXPERIMENT_ID = 4651
SCHEMA = "carnot.archive_activation.v428_to_v429_4651.v1"
RESULT_RELATIVE_PATH = "results/experiment_4651_archive_428_activate_429.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4650_capstone_v428.json")
A1_REL_PATH = Path("results/experiment_4640_goal_energy_generation_live.json")
A2_REL_PATH = Path("results/experiment_4641_action_effect_expansion_prior_live.json")
A3_REL_PATH = Path("results/experiment_4642_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4643_refresh_submission_package.json")
A5_REL_PATH = Path("results/experiment_4644_primitive_persist_transfer.json")
A6_REL_PATH = Path("results/experiment_4645_integration_gate.json")
B1_REL_PATH = Path("results/experiment_4646_live_multi_level_solve_rate_metric.json")
B2_REL_PATH = Path("results/experiment_4647_adversarial_verify_hardening.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.428"
ACTIVATED_MILESTONE = "2026.06.429"
RANDOM_SEED = 4651
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 56
LIVE_SUBMITTABLE_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4651",
    "SCENARIO-CAPSTONE-4651",
    "SCENARIO-CAPSTONE-4651-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4651-FIELD-PRINCIPLES",
]

FIELD_PROVENANCE = {
    "honest_verdict": {
        "principle": (
            "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/"
            "passed_/shipped:/shipped_ so the reconciler classifies it terminal."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts so adversarial_verify applies the 100us "
            "floor, not the 60s live-model floor."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
        )
    },
    "close_state_428": {
        "principle": (
            "the honest .428 numbers (A3 56->57; A1/A2 both energy-generation NULLS; "
            "A4 57>33) carried forward so the record does not drift."
        )
    },
    "v429_pivot": {
        "principle": (
            "the .429 headline rationale (PIVOT to GENERATION GUIDANCE: A1 productionize "
            "the compute-cost value-routing fix; A2 energy-as-fitness QD evolution) "
            "recorded so the milestone intent is traceable."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_428",
    "v429_pivot",
    "cited_upstream_artifacts",
    "field_provenance",
    "leaderboard_submission",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _yaml_info(path: Path) -> JsonDict:
    text = _read_text(path)
    if text is None:
        return {"path": str(path), "available": False, "parses": False, "milestone": None}
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {
            "path": str(path),
            "available": True,
            "parses": False,
            "milestone": None,
            "error": str(exc),
        }
    milestone = loaded.get("milestone") if isinstance(loaded, Mapping) else None
    return {
        "path": str(path),
        "available": True,
        "parses": True,
        "milestone": str(milestone) if milestone is not None else None,
    }


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _registry_total_levels(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, Mapping):
        return None
    value = loaded.get("reproducible_total_levels")
    return None if isinstance(value, bool) or not isinstance(value, int | float) else int(value)


def _command_check(result: CommandResult | None) -> JsonDict:
    if result is None:
        return {
            "command": [],
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "passed": None,
            "not_run_reason": "blocked_before_smart_subset_gate",
        }
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout_tail": result.stdout[-500:],
        "stderr_tail": result.stderr[-500:],
        "passed": result.exit_code == 0,
    }


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - integration smoke wrapper
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def _default_smart_subset_checker(root: Path) -> CommandResult:  # pragma: no cover - subprocess wrapper
    return run_smart_subset(root)


def _next_roadmap_ready(next_info: Mapping[str, Any]) -> bool:
    return next_info.get("available") is True and next_info.get("parses") is True and next_info.get(
        "milestone"
    ) == ACTIVATED_MILESTONE


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    active_path = root / RESEARCH_ROADMAP_REL_PATH
    try:
        active_path.write_text(next_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem defensive path
        return False, str(exc)
    return True, ""


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)

    activation_attempted = False
    activation_error = ""
    if offline_ok:
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH)
    complete_info = _yaml_info(root / RESEARCH_COMPLETE_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)

    should_run_smart_subset = (
        _next_roadmap_ready(next_info)
        and active_info["parses"] is True
        and active_info["milestone"] == ACTIVATED_MILESTONE
        and offline_ok
        and activation_error == ""
    )
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None

    return {
        "research_roadmap_next_yaml": {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "available": next_info["available"],
            "parses": next_info["parses"],
            "milestone": next_info["milestone"],
            "literal_precondition_command": (
                ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
                "'research-roadmap-next.yaml')); print('yaml_ok')\""
            ),
            "activation_attempted": activation_attempted,
            "activation_error": activation_error,
        },
        "active_research_roadmap_yaml": {
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "available": active_info["available"],
            "parses": active_info["parses"],
            "milestone": active_info["milestone"],
        },
        "research_complete_yaml": {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "available": complete_text is not None,
            "parses": complete_info["parses"],
            "contains_2026_06_428": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
        },
        "offline_arcade": {
            "available": offline_ok,
            "command": (
                ".venv/bin/python -c \"from carnot.agentic import arc_solver_kit as k; "
                "k.offline_arcade()\""
            ),
            "error": offline_error,
        },
        "smart_subset_pretest_gate": _command_check(smart_subset),
        "registry": {
            "path": str(REGISTRY_REL_PATH),
            "available": registry_levels is not None,
            "reproducible_total_levels": registry_levels,
        },
        "capstone_4650": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_exp4640": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_exp4641": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_exp4642": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_exp4643": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "a5_exp4644": {"path": str(A5_REL_PATH), "available": (root / A5_REL_PATH).exists()},
        "a6_exp4645": {"path": str(A6_REL_PATH), "available": (root / A6_REL_PATH).exists()},
        "b1_exp4646": {"path": str(B1_REL_PATH), "available": (root / B1_REL_PATH).exists()},
        "b2_exp4647": {"path": str(B2_REL_PATH), "available": (root / B2_REL_PATH).exists()},
        "vnext_design": {
            "path": str(VNEXT_DESIGN_REL_PATH),
            "available": (root / VNEXT_DESIGN_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone = _mapping(preconditions.get("capstone_4650"))

    if not _next_roadmap_ready(next_info):
        return "research_roadmap_next_yaml"
    if not (active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE):
        return "research_roadmap_429_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4650_capstone_v428"
    for key, reason in (
        ("a1_exp4640", "missing_experiment_4640_goal_energy_generation_live"),
        ("a2_exp4641", "missing_experiment_4641_action_effect_expansion_prior_live"),
        ("a3_exp4642", "missing_experiment_4642_levelup_selfplay"),
        ("a4_exp4643", "missing_experiment_4643_refresh_submission_package"),
        ("a5_exp4644", "missing_experiment_4644_primitive_persist_transfer"),
        ("a6_exp4645", "missing_experiment_4645_integration_gate"),
        ("b1_exp4646", "missing_experiment_4646_live_multi_level_solve_rate_metric"),
        ("b2_exp4647", "missing_experiment_4647_adversarial_verify_hardening"),
        ("vnext_design", "missing_research_roadmap_vnext_design"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    research_complete = _mapping(preconditions.get("research_complete_yaml"))
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(
            complete
            and active.get("parses") is True
            and active.get("milestone") == ACTIVATED_MILESTONE
        ),
        "activation_state": (
            "activated_from_research_roadmap_next"
            if complete
            else "blocked_missing_or_failed_precondition"
        ),
        "archive_state": (
            "research_complete_contains_2026.06.428"
            if research_complete.get("contains_2026_06_428") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "active_research_roadmap_yaml",
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4650_capstone_v428",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "scorecard",
                "live_solve_rate_delta",
                "live_multi_level_solve_rate",
                "live_submittable_level_count",
                "reproducible_total_levels",
                "ready_for_operator_submit",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4640_goal_energy_generation_live",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "uniform_energy_ablation_passed",
                "live_solve_rate_goal_energy",
                "live_solve_rate_baseline",
                "solve_rate_delta",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4641_action_effect_expansion_prior_live",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_measurement",
                "depth_of_live_solve_delta",
                "first_win_rate_delta",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4642_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "target_game",
                "prior_reproduced_level",
                "reached_level",
                "reproducible_total_levels_before",
                "reproducible_total_levels_after",
                "offline_reproduced",
            ],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4643_refresh_submission_package",
            "path": str(A4_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_submittable_level_count",
                "ready_for_operator_submit",
            ],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "experiment_4644_primitive_persist_transfer",
            "path": str(A5_REL_PATH),
            "fields_imported": ["honest_verdict"],
            "sha256": file_sha256(root / A5_REL_PATH),
        },
        {
            "source": "experiment_4645_integration_gate",
            "path": str(A6_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_multi_level_solve_rate_integrated",
                "live_submittable_level_count_integrated",
            ],
            "sha256": file_sha256(root / A6_REL_PATH),
        },
        {
            "source": "experiment_4646_live_multi_level_solve_rate_metric",
            "path": str(B1_REL_PATH),
            "fields_imported": ["honest_verdict", "live_multi_level_solve_rate"],
            "sha256": file_sha256(root / B1_REL_PATH),
        },
        {
            "source": "experiment_4647_adversarial_verify_hardening",
            "path": str(B2_REL_PATH),
            "fields_imported": ["honest_verdict", "goal_energy_ablation_guard_added"],
            "sha256": file_sha256(root / B2_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.429 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _close_state_428(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    a5: Mapping[str, Any],
    a6: Mapping[str, Any],
    b1: Mapping[str, Any],
    b2: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    capstone_a1 = _mapping(scorecard.get("A1"))
    capstone_a2 = _mapping(scorecard.get("A2"))
    capstone_a4 = _mapping(scorecard.get("A4"))
    a1_live = _mapping(capstone.get("live_solve_rate_delta")) or _mapping(
        capstone_a1.get("live_solve_rate_delta")
    )
    a2_multi = _mapping(capstone.get("live_multi_level_solve_rate")) or _mapping(
        capstone_a2.get("live_multi_level_solve_rate")
    )
    first_win = _mapping(capstone.get("first_win_rate_scored")) or _mapping(
        capstone_a2.get("first_win_rate_scored")
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank_ft09": {
            "honest_verdict": a3.get("honest_verdict"),
            "target_game": str(a3.get("target_game") or "ft09"),
            "prior_reproduced_level": _int(a3.get("prior_reproduced_level"), 2),
            "target_level": _int(a3.get("reached_level"), 3),
            "reproducible_total_before": _int(
                a3.get("reproducible_total_levels_before"), BASELINE_REPRODUCIBLE_TOTAL_LEVELS
            ),
            "reproducible_total_after": _int(
                a3.get("reproducible_total_levels_after"), registry_total_levels
            ),
            "reproducible_total_delta": _int(a3.get("reproduced_levels"), 1),
            "offline_reproduced": a3.get("offline_reproduced") is True,
        },
        "a1_goal_energy_generation": {
            "honest_verdict": a1.get("honest_verdict"),
            "included_in_headline": capstone_a1.get("included_in_headline") is True,
            "null_reason": str(a1_live.get("reason") or capstone_a1.get("reason") or "uniform_energy_ablation_failed"),
            "uniform_energy_ablation_passed": a1.get("uniform_energy_ablation_passed") is True,
            "live_solve_rate_goal_energy": _float(
                a1.get("live_solve_rate_goal_energy"),
                _float(a1_live.get("live_solve_rate_goal_energy"), 0.04),
            ),
            "live_solve_rate_baseline": _float(
                a1.get("live_solve_rate_baseline"),
                _float(a1_live.get("live_solve_rate_baseline"), 0.04),
            ),
            "solve_rate_delta": _float(
                a1.get("solve_rate_delta"), _float(a1_live.get("solve_rate_delta"), 0.0)
            ),
            "first_win_rate_delta": _float(
                a1.get("first_win_rate_delta"), _float(a1_live.get("first_win_rate_delta"), 0.0)
            ),
        },
        "a2_action_effect_expansion_prior": {
            "honest_verdict": a2.get("honest_verdict"),
            "null_reason": (
                "no_deeper_solve"
                if "no_deeper_solve" in str(a2.get("honest_verdict", ""))
                else str(a2_multi.get("reason") or "no_deeper_solve")
            ),
            "depth_of_live_solve_delta": _float(
                a2.get("depth_of_live_solve_delta"),
                _float(a2_multi.get("depth_of_live_solve_delta"), 0.0),
            ),
            "live_multi_level_solve_rate": _float(
                a2.get("live_multi_level_solve_rate"),
                _float(a2_multi.get("live_multi_level_solve_rate"), 0.0),
            ),
            "ranker_baseline_multi_level_rate": _float(
                a2_multi.get("ranker_baseline_multi_level_rate"), 0.0
            ),
            "first_win_rate_expansion": _float(a2_multi.get("first_win_rate_expansion"), 1.0),
            "first_win_held_at_or_above_427": first_win.get("regressed_vs_427_baseline") is False,
        },
        "a4_submission_package": {
            "honest_verdict": a4.get("honest_verdict"),
            "live_submittable_level_count": _int(
                a4.get("live_submittable_level_count"),
                _int(capstone_a4.get("live_submittable_level_count"), 57),
            ),
            "beats_submission_baseline": LIVE_SUBMITTABLE_BASELINE,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True
            or capstone.get("ready_for_operator_submit") is True,
        },
        "a5_a6_b1_b2_shipped": {
            "a5_honest_verdict": a5.get("honest_verdict"),
            "a6_honest_verdict": a6.get("honest_verdict"),
            "b1_honest_verdict": b1.get("honest_verdict"),
            "b2_honest_verdict": b2.get("honest_verdict"),
            "goal_energy_ablation_guard_active": b2.get("goal_energy_ablation_guard_added") is True
            or _mapping(scorecard.get("B2")).get("goal_energy_ablation_guard_active") is True,
        },
        "registry_total_levels": registry_total_levels,
        "energy_generation_lesson": "generation_guidance_needed_after_energy_levers_nulled",
    }


def _v429_pivot() -> JsonDict:
    return {
        "headline_rationale": "GENERATION GUIDANCE",
        "builds_on": "complete: capability_grew_56_to_57",
        "a1": {
            "lever": "productionize_compute_cost_value_routing_fix",
            "fix": "scipy.ndimage.label_connected_components",
            "timing_before_ms": 13.0,
            "timing_after_ms": 0.64,
            "identical_output": True,
            "auroc": 0.725,
            "value_weight_target": "raise_off_0.0",
            "purpose": "discriminator_guides_live_without_timeout",
        },
        "a2": {
            "lever": "energy_as_fitness_qd_evolution",
            "operator_menu": "#2",
            "role": "next_sequenced_generation_lever",
            "gate": "winner_generated",
        },
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    cited_upstream_artifacts: list[JsonDict],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_428": {},
        "v429_pivot": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_provenance": FIELD_PROVENANCE,
        "leaderboard_submission": False,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    offline_arcade_checker: OfflineArcadeChecker = _default_offline_arcade_checker,
    smart_subset_checker: SmartSubsetChecker = _default_smart_subset_checker,
) -> JsonDict:
    root_path = Path(root)
    duration_s = duration_from(started_s, now_s)
    preconditions = _preconditions(
        root_path,
        offline_arcade_checker=offline_arcade_checker,
        smart_subset_checker=smart_subset_checker,
    )
    cited = _cited_upstream(root_path)
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            duration_s=duration_s,
            cited_upstream_artifacts=cited,
        )
        validate_artifact(artifact)
        return artifact

    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_428(
        capstone=_read_json(root_path / CAPSTONE_REL_PATH),
        a1=_read_json(root_path / A1_REL_PATH),
        a2=_read_json(root_path / A2_REL_PATH),
        a3=_read_json(root_path / A3_REL_PATH),
        a4=_read_json(root_path / A4_REL_PATH),
        a5=_read_json(root_path / A5_REL_PATH),
        a6=_read_json(root_path / A6_REL_PATH),
        b1=_read_json(root_path / B1_REL_PATH),
        b2=_read_json(root_path / B2_REL_PATH),
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_428_activate_429_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_428": close_state,
        "v429_pivot": _v429_pivot(),
        "cited_upstream_artifacts": cited,
        "field_provenance": FIELD_PROVENANCE,
        "leaderboard_submission": False,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_checksum(artifact: Mapping[str, Any]) -> None:
    checksum = artifact.get("reproducibility_checksum")
    _require(
        isinstance(checksum, str)
        and checksum.startswith("sha256:")
        and is_sha256(checksum.removeprefix("sha256:")),
        "reproducibility_checksum must be sha256-prefixed",
    )
    expected = "sha256:" + payload_checksum(artifact)
    _require(checksum == expected, "reproducibility_checksum does not match payload")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")

    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str), "honest_verdict must be a string")
    blocked = verdict.startswith("blocked_")
    if not blocked:
        _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")

    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation")
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_provenance") == FIELD_PROVENANCE, "field_provenance principles drifted")

    close = _mapping(artifact.get("close_state_428"))
    pivot = _mapping(artifact.get("v429_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .429 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank_ft09"))
    _require(
        a3.get("target_game") == "ft09"
        and a3.get("target_level") == 3
        and a3.get("reproducible_total_before") == 56
        and a3.get("reproducible_total_after") == 57
        and a3.get("reproducible_total_delta") == 1
        and a3.get("offline_reproduced") is True,
        "A3 ft09 L3 bank 56->57 must be recorded",
    )

    a1 = _mapping(close.get("a1_goal_energy_generation"))
    _require(
        a1.get("included_in_headline") is False
        and a1.get("null_reason") == "uniform_energy_ablation_failed"
        and a1.get("uniform_energy_ablation_passed") is False
        and a1.get("live_solve_rate_goal_energy") == 0.04
        and a1.get("live_solve_rate_baseline") == 0.04
        and a1.get("solve_rate_delta") == 0.0,
        "A1 goal-energy null and ablation failure must be recorded",
    )

    a2 = _mapping(close.get("a2_action_effect_expansion_prior"))
    _require(
        a2.get("null_reason") == "no_deeper_solve"
        and a2.get("depth_of_live_solve_delta") == 0.0
        and a2.get("live_multi_level_solve_rate") == 0.0
        and a2.get("ranker_baseline_multi_level_rate") == 0.0
        and a2.get("first_win_held_at_or_above_427") is True,
        "A2 expansion-prior null must be recorded",
    )

    a4 = _mapping(close.get("a4_submission_package"))
    _require(
        a4.get("live_submittable_level_count") == 57
        and a4.get("beats_submission_baseline") == 33
        and a4.get("ready_for_operator_submit") is True,
        "A4 57>33 package readiness must be recorded",
    )

    shipped = _mapping(close.get("a5_a6_b1_b2_shipped"))
    _require(
        str(shipped.get("a5_honest_verdict", "")).startswith("complete:")
        and str(shipped.get("a6_honest_verdict", "")).startswith("success:")
        and str(shipped.get("b1_honest_verdict", "")).startswith("success:")
        and str(shipped.get("b2_honest_verdict", "")).startswith("success:")
        and shipped.get("goal_energy_ablation_guard_active") is True,
        "A5/A6/B1/B2 shipped state must be recorded",
    )

    _require(
        close.get("registry_total_levels") == 57
        and close.get("energy_generation_lesson") == "generation_guidance_needed_after_energy_levers_nulled",
        "registry total and .428 energy-generation lesson must be recorded",
    )
    _require(
        pivot.get("headline_rationale") == "GENERATION GUIDANCE"
        and _mapping(pivot.get("a1")).get("fix") == "scipy.ndimage.label_connected_components"
        and _mapping(pivot.get("a1")).get("timing_after_ms") == 0.64
        and _mapping(pivot.get("a1")).get("auroc") == 0.725
        and _mapping(pivot.get("a1")).get("value_weight_target") == "raise_off_0.0"
        and _mapping(pivot.get("a2")).get("gate") == "winner_generated",
        "v429 pivot must record generation-guidance rationale",
    )
    _validate_checksum(artifact)


def run(
    root: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
    offline_arcade_checker: OfflineArcadeChecker = _default_offline_arcade_checker,
    smart_subset_checker: SmartSubsetChecker = _default_smart_subset_checker,
) -> JsonDict:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        offline_arcade_checker=offline_arcade_checker,
        smart_subset_checker=smart_subset_checker,
    )
    if write:
        output_path = root_path / OUTPUT_REL_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    result = run()
    print(
        json.dumps(
            {
                "honest_verdict": result["honest_verdict"],
                "result_path": result["result_path"],
            },
            sort_keys=True,
        )
    )
