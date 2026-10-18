"""Experiment 4521: .417 action-efficiency capstone aggregation.

Spec refs: REQ-CAPSTONE-4521, SCENARIO-CAPSTONE-4521.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot.reporting import capstone_v400_4335 as base


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4521_capstone_v417.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4521
RANDOM_SEED = 4521
SCHEMA = "carnot.capstone_v417_4521.v1"
SPEC_REFS = ["REQ-CAPSTONE-4521", "SCENARIO-CAPSTONE-4521"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
BASELINE_MEDIAN_ACTIONS = 7760.0
PRIOR_SUBMITTED_BASELINE_LEVELS = 13
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; e.g. complete: v417_<lever>_median_actions_<n>_vs_7760_heldout_<r>."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- reads upstream JSON, no compute (100us floor)."
    ),
    "median_actions_best_lever": (
        "the headline -- the lowest median achieved + which lever, vs the 7760 baseline."
    ),
    "median_actions_baseline": "7760, so the delta is auditable.",
    "heldout_solve_rate": "the real transfer signal (was 0.143); the moat-track number.",
    "submission_package_ready": (
        "True if the package > 13 levels is ready for the OPERATOR to submit; the capstone never submits."
    ),
    "flagged_artifacts_excluded": (
        "the list of flagged_adversarial artifacts skipped (never aggregate a fabricated number)."
    ),
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "variant_transfer_rate": "bare float from the clean submitted-agent scoreboard context.",
    "reproducible_total_levels": (
        "bare int from ops/arc_solve_registry.yaml for the submission gate vs 13."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; upstream claims declare circularity separately."
    ),
    "reproducibility_checksum": "content hash for reproducibility",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "median_actions_best_lever",
    "median_actions_baseline",
    "heldout_solve_rate",
    "submission_package_ready",
    "flagged_artifacts_excluded",
    "preconditions_checked",
    "per_lever_scorecard",
    "integrated_scorecard",
    "action_efficiency_decision",
    "variant_transfer_rate",
    "reproducible_total_levels",
    "verifier_is_oracle",
    "verifier_claims",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "A1_prune": Upstream(4511, Path("results/experiment_4511_frame_change_prune_predictor.json")),
    "A2_imitation": Upstream(4512, Path("results/experiment_4512_imitation_action_prior.json")),
    "A3_adaptive_budget": Upstream(
        4513,
        Path("results/experiment_4513_adaptive_per_step_budget.json"),
    ),
    "A4_lazy_best_first": Upstream(
        4514,
        Path("results/experiment_4514_lazy_best_first_value_weight.json"),
    ),
    "A5_level_up": Upstream(4515, Path("results/experiment_4515_deepen_graph_explore_l2.json")),
    "A6_integration": Upstream(4516, Path("results/experiment_4516_integration_8game_gate.json")),
    "scoreboard_context": Upstream(
        4505,
        Path("results/experiment_4505_submitted_agent_scoreboard.json"),
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "A1_prune": [
        "honest_verdict",
        "median_actions_baseline",
        "median_actions_with_prune",
        "solve_rate_baseline",
        "solve_rate_with_prune",
        "solve_rate_denominator",
        "verifier_is_oracle",
    ],
    "A2_imitation": [
        "honest_verdict",
        "median_actions_baseline",
        "median_actions_with_prior",
        "solve_rate_baseline",
        "solve_rate_with_prior",
        "solve_rate_denominator",
        "verifier_is_oracle",
    ],
    "A3_adaptive_budget": [
        "honest_verdict",
        "median_actions_baseline",
        "median_actions_with_adaptive",
        "solve_rate_baseline",
        "solve_rate_with_adaptive",
        "solve_rate_denominator",
        "verifier_is_oracle",
    ],
    "A4_lazy_best_first": [
        "honest_verdict",
        "per_weight_results",
        "control_value_weight_0",
        "chosen_submitted_value_weight",
        "decision",
        "verifier_is_oracle",
    ],
    "A5_level_up": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "verifier_is_oracle",
    ],
    "A6_integration": [
        "honest_verdict",
        "median_actions_baseline",
        "median_actions_integrated",
        "solve_rate_integrated",
        "heldout_solve_rate",
        "verifier_is_oracle",
    ],
    "scoreboard_context": [
        "honest_verdict",
        "headline_metrics",
        "scoreboard_row",
        "leaderboard_submission",
        "verifier_is_oracle",
    ],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _flagged_payload(payload: JsonDict) -> JsonDict:
    flagged = dict(payload)
    flagged["flagged_adversarial"] = True
    return flagged


def _number(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int_or_none(payload: Mapping[str, Any] | None, field: str) -> int | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _rate(count: float | None, denominator: float | None) -> float | None:
    if count is None or denominator is None or denominator == 0:
        return None
    return float(count) / float(denominator)


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict]]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    exclusions: list[JsonDict] = []

    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        if not path.exists():
            raw_artifacts[key] = None
            continue

        sha = base.sha256_file(path)
        summarize_exit_code, summarize_error = base._safe_summarize(  # noqa: SLF001
            path,
            root,
            summarize_runner,
        )
        live_flags = base._safe_live_flags(path, live_flag_runner)  # noqa: SLF001
        critical = base.live_has_critical(live_flags)
        parse_error = ""
        payload: JsonDict | None = None
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover
            parse_error = f"{type(exc).__name__}: {exc}"

        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        skipped = stamped or critical or payload is None
        raw_artifacts[key] = _flagged_payload(payload) if payload is not None and skipped else payload
        row = {
            "artifact_key": key,
            "experiment_id": upstream.experiment_id,
            "path": str(upstream.path),
            "sha256": sha,
            "payload_reproducibility_checksum": base.sha_from_payload_checksum(payload or {}),
            "summarize_exit_code": summarize_exit_code,
            "summarize_error": summarize_error,
            "live_adversarial_flags": live_flags,
            "stamped_flagged_adversarial": stamped,
            "live_critical": critical,
            "parse_error": parse_error,
            "skipped": skipped,
            "fields_imported": _fields_for_payload(key, skipped),
        }
        provenance.append(row)
        if skipped:
            exclusions.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "parse_error": parse_error,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": base._exclusion_reason(stamped, critical, parse_error),  # noqa: SLF001
                }
            )
    return raw_artifacts, provenance, exclusions


def load_registry_totals(root: Path | str = REPO_ROOT) -> JsonDict:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {
            "registry_path": str(REGISTRY_RELATIVE_PATH),
            "registry_present": False,
            "reproducible_total_levels": 0,
            "prior_submitted_baseline_levels": PRIOR_SUBMITTED_BASELINE_LEVELS,
        }
    text = path.read_text(encoding="utf-8")

    def _first_int(key: str, default: int) -> int:
        match = re.search(rf"(?m)^{re.escape(key)}:\s*(\d+)\b", text)
        return int(match.group(1)) if match else int(default)

    return {
        "registry_path": str(REGISTRY_RELATIVE_PATH),
        "registry_present": True,
        "reproducible_total_levels": _first_int("reproducible_total_levels", 0),
        "prior_submitted_baseline_levels": _first_int(
            "prior_submitted_baseline_levels",
            PRIOR_SUBMITTED_BASELINE_LEVELS,
        ),
    }


def _clean_payloads(raw_artifacts: Mapping[str, Any], skipped: Mapping[str, bool]) -> dict[str, JsonDict | None]:
    return {
        key: base.clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }


def _direct_lever_row(
    *,
    lever: str,
    payload: JsonDict | None,
    skipped: bool,
    median_field: str,
    solve_field: str,
) -> JsonDict:
    if skipped:
        return {
            "lever": lever,
            "status": "excluded_flagged_adversarial",
            "median_actions": None,
            "heldout_solve_rate": None,
            "baseline_solve_rate": None,
            "equal_or_better_solve_rate": False,
            "action_efficiency_win": False,
        }
    if payload is None:
        return {
            "lever": lever,
            "status": "missing_or_excluded",
            "median_actions": None,
            "heldout_solve_rate": None,
            "baseline_solve_rate": None,
            "equal_or_better_solve_rate": False,
            "action_efficiency_win": False,
        }

    baseline_median = _number(payload, "median_actions_baseline") or BASELINE_MEDIAN_ACTIONS
    median_actions = _number(payload, median_field)
    denominator = _number(payload, "solve_rate_denominator")
    baseline_rate = _rate(_number(payload, "solve_rate_baseline"), denominator)
    treatment_rate = _rate(_number(payload, solve_field), denominator)
    equal_solve = (
        treatment_rate is not None and baseline_rate is not None and treatment_rate >= baseline_rate
    )
    action_win = (
        median_actions is not None
        and median_actions < BASELINE_MEDIAN_ACTIONS
        and equal_solve
    )
    return {
        "lever": lever,
        "status": "action_efficiency_win" if action_win else "no_clean_equal_solve_win",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "median_actions": median_actions,
        "median_actions_delta_vs_baseline": (
            None if median_actions is None else float(median_actions - baseline_median)
        ),
        "action_reduction_vs_baseline": (
            None if median_actions is None else float(baseline_median - median_actions)
        ),
        "heldout_solve_rate": treatment_rate,
        "baseline_solve_rate": baseline_rate,
        "solve_rate_count": _number(payload, solve_field),
        "baseline_solve_rate_count": _number(payload, "solve_rate_baseline"),
        "solve_rate_denominator": denominator,
        "equal_or_better_solve_rate": bool(equal_solve),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
        "action_efficiency_win": bool(action_win),
    }


def _weight_key(weight: float) -> str:
    return f"{weight:g}" if weight else "0.0"


def _a4_scorecard(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "lever": "A4_lazy_best_first",
            "status": "excluded_flagged_adversarial",
            "median_actions": None,
            "heldout_solve_rate": None,
            "baseline_solve_rate": None,
            "equal_or_better_solve_rate": False,
            "action_efficiency_win": False,
        }
    if payload is None:
        return {
            "lever": "A4_lazy_best_first",
            "status": "missing_or_excluded",
            "median_actions": None,
            "heldout_solve_rate": None,
            "baseline_solve_rate": None,
            "equal_or_better_solve_rate": False,
            "action_efficiency_win": False,
        }

    decision = _mapping(payload, "decision")
    selected_weight = _number(payload, "chosen_submitted_value_weight")
    if selected_weight is None:
        selected_weight = _number(decision, "selected_value_weight") or 0.0
    per_weight = _mapping(payload, "per_weight_results")
    control = _mapping(payload, "control_value_weight_0") or _mapping(per_weight, "0.0")
    selected = _mapping(per_weight, _weight_key(selected_weight)) or control
    baseline_actions = _number(control, "median_actions_on_core") or BASELINE_MEDIAN_ACTIONS
    median_actions = _number(selected, "median_actions_on_core")
    baseline_rate = _number(control, "heldout_solve_rate")
    selected_rate = _number(selected, "heldout_solve_rate")
    core_preserved = base.bool_metric(selected, "core_solves_preserved") is True
    equal_solve = selected_rate is not None and baseline_rate is not None and selected_rate >= baseline_rate
    action_win = (
        selected_weight > 0.0
        and median_actions is not None
        and median_actions < BASELINE_MEDIAN_ACTIONS
        and equal_solve
        and core_preserved
    )
    return {
        "lever": "A4_lazy_best_first",
        "status": "action_efficiency_win" if action_win else "no_clean_equal_solve_win",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "selected_value_weight": float(selected_weight),
        "median_actions": median_actions,
        "median_actions_delta_vs_baseline": (
            None if median_actions is None else float(median_actions - BASELINE_MEDIAN_ACTIONS)
        ),
        "action_reduction_vs_baseline": (
            None if median_actions is None else float(BASELINE_MEDIAN_ACTIONS - median_actions)
        ),
        "control_median_actions": float(baseline_actions),
        "heldout_solve_rate": selected_rate,
        "baseline_solve_rate": baseline_rate,
        "core_solves_preserved": core_preserved,
        "equal_or_better_solve_rate": bool(equal_solve and core_preserved),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
        "action_efficiency_win": bool(action_win),
    }


def _integrated_scorecard(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "lever": "A6_integration",
            "status": "excluded_flagged_adversarial",
            "median_actions": None,
            "heldout_solve_rate": None,
            "action_efficiency_win": False,
        }
    if payload is None:
        return {
            "lever": "A6_integration",
            "status": "missing_or_excluded",
            "median_actions": None,
            "heldout_solve_rate": None,
            "action_efficiency_win": False,
        }
    median_actions = _number(payload, "median_actions_integrated")
    solve_rate = _number(payload, "solve_rate_integrated")
    heldout = _number(payload, "heldout_solve_rate") or solve_rate
    action_win = median_actions is not None and median_actions < BASELINE_MEDIAN_ACTIONS
    return {
        "lever": "A6_integration",
        "status": "integrated_action_efficiency_win" if action_win else "integrated_honest_null",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "median_actions": median_actions,
        "median_actions_delta_vs_baseline": (
            None if median_actions is None else float(median_actions - BASELINE_MEDIAN_ACTIONS)
        ),
        "action_reduction_vs_baseline": (
            None if median_actions is None else float(BASELINE_MEDIAN_ACTIONS - median_actions)
        ),
        "solve_rate_integrated": solve_rate,
        "heldout_solve_rate": heldout,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
        "action_efficiency_win": bool(action_win),
    }


def _level_up_context(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"status": "excluded_flagged_adversarial", "level_up_banked": False}
    if payload is None:
        return {"status": "missing_or_excluded", "level_up_banked": False}
    reproduced = base.bool_metric(payload, "offline_reproduced") is True
    levels = _int_or_none(payload, "reproduced_levels") or 0
    return {
        "status": "level_up_context",
        "target_game": base.str_metric(payload, "target_game"),
        "offline_reproduced": reproduced,
        "reproduced_levels": levels,
        "level_up_banked": bool(reproduced and levels >= 2),
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
    }


def _scoreboard_context(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped or payload is None:
        return {
            "status": "missing_or_excluded",
            "heldout_solve_rate": 0.0,
            "variant_transfer_rate": 0.0,
            "leaderboard_submission": False,
        }
    metrics = _mapping(payload, "headline_metrics")
    return {
        "status": "scoreboard_context",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "heldout_solve_rate": _number(metrics, "submitted_default_heldout_generic_solve_rate") or 0.0,
        "heldout_solved": _int_or_none(metrics, "submitted_default_heldout_generic_solved") or 0,
        "heldout_attempted": _int_or_none(metrics, "submitted_default_heldout_generic_attempted") or 0,
        "variant_transfer_rate": _number(metrics, "variant_transfer_rate") or 0.0,
        "variant_transfer_solved": _int_or_none(metrics, "variant_transfer_solved") or 0,
        "variant_transfer_attempted": _int_or_none(metrics, "variant_transfer_attempted") or 0,
        "leaderboard_submission": base.bool_metric(payload, "leaderboard_submission") is True,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
    }


def _best_lever(per_lever_scorecard: list[JsonDict], heldout_solve_rate: float) -> JsonDict:
    winners = [row for row in per_lever_scorecard if row.get("action_efficiency_win") is True]
    if not winners:
        return {
            "lever": "none_clean_equal_solve_rate",
            "median_actions": BASELINE_MEDIAN_ACTIONS,
            "median_actions_delta_vs_baseline": 0.0,
            "action_reduction_vs_baseline": 0.0,
            "heldout_solve_rate": heldout_solve_rate,
            "reason": "no_clean_lever_beat_7760_at_equal_or_better_solve_rate",
        }
    best = min(winners, key=lambda row: float(row["median_actions"]))
    return {
        "lever": best["lever"],
        "median_actions": float(best["median_actions"]),
        "median_actions_delta_vs_baseline": float(best["median_actions_delta_vs_baseline"]),
        "action_reduction_vs_baseline": float(best["action_reduction_vs_baseline"]),
        "heldout_solve_rate": best.get("heldout_solve_rate"),
        "reason": "clean_lever_beats_7760_at_equal_or_better_solve_rate",
    }


def _action_efficiency_decision(best: Mapping[str, Any], integrated: Mapping[str, Any]) -> JsonDict:
    winning_lever = base.str_metric(best, "lever")
    beats = winning_lever not in {"", "none_clean_equal_solve_rate", "blocked"}
    return {
        "beats_7760_at_equal_solve_rate": bool(beats),
        "winning_lever": winning_lever if beats else None,
        "decision": (
            f"{winning_lever}_beat_7760_at_equal_or_better_solve_rate"
            if beats
            else "no_clean_lever_beat_7760_at_equal_or_better_solve_rate"
        ),
        "integrated_metric_used": integrated.get("status") != "excluded_flagged_adversarial",
        "integrated_status": integrated.get("status"),
        "baseline_median_actions": BASELINE_MEDIAN_ACTIONS,
    }


def _honest_verdict(best: Mapping[str, Any], heldout_solve_rate: float) -> str:
    lever = base.str_metric(best, "lever")
    median_actions = int(float(best.get("median_actions") or BASELINE_MEDIAN_ACTIONS))
    prefix = "success:" if lever not in {"", "none_clean_equal_solve_rate", "blocked"} else "complete:"
    return (
        f"{prefix} v417_{lever}_median_actions_{median_actions}_vs_7760_"
        f"heldout_{heldout_solve_rate:.3f}"
    )


def _cited_upstream_artifacts(provenance: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "artifact_key": row["artifact_key"],
            "experiment_id": row["experiment_id"],
            "path": row["path"],
            "sha256": row["sha256"],
            "fields_imported": row["fields_imported"],
        }
        for row in provenance
    ]


def _verifier_claims(raw_artifacts: Mapping[str, Any], skipped: Mapping[str, bool]) -> list[JsonDict]:
    claims: list[JsonDict] = []
    for key in DEFAULT_UPSTREAMS:
        payload = raw_artifacts.get(key)
        if not isinstance(payload, Mapping):
            continue
        claims.append(
            {
                "source": key,
                "experiment_id": DEFAULT_UPSTREAMS[key].experiment_id,
                "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle") is True,
                "skipped": bool(skipped.get(key, False)),
            }
        )
    return claims


def _preconditions_checked(
    root: Path,
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
) -> JsonDict:
    provenance_by_key = {row["artifact_key"]: row for row in provenance}
    upstreams: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        row = provenance_by_key.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": DEFAULT_UPSTREAMS[key].experiment_id,
                "path": str(DEFAULT_UPSTREAMS[key].path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
            }
        )
    a6 = provenance_by_key.get("A6_integration")
    return {
        "upstream_artifacts": upstreams,
        "summarize_artifact_required": "scripts/summarize_artifact.py",
        "reading_results_discipline": True,
        "a6_integration_artifact_present": (root / DEFAULT_UPSTREAMS["A6_integration"].path).exists(),
        "a6_summarize_exit_code": a6.get("summarize_exit_code") if a6 else None,
        "a6_clean_for_aggregation": bool(a6 and not a6.get("skipped")),
        "registry_present": bool(registry.get("registry_present")),
        "registry_path": str(REGISTRY_RELATIVE_PATH),
        "leaderboard_submission": False,
    }


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    *,
    median_actions_best_lever: Mapping[str, Any],
    per_lever_scorecard: list[Mapping[str, Any]],
    integrated_scorecard: Mapping[str, Any],
    action_efficiency_decision: Mapping[str, Any],
    heldout_solve_rate: float,
    variant_transfer_rate: float,
    reproducible_total_levels: int,
    submission_package_ready: bool,
) -> str:
    payload = {
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "median_actions_best_lever": median_actions_best_lever,
        "per_lever_scorecard": per_lever_scorecard,
        "integrated_scorecard": integrated_scorecard,
        "action_efficiency_decision": action_efficiency_decision,
        "heldout_solve_rate": float(heldout_solve_rate),
        "variant_transfer_rate": float(variant_transfer_rate),
        "reproducible_total_levels": int(reproducible_total_levels),
        "submission_package_ready": bool(submission_package_ready),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _blocked_artifact(
    *,
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
) -> JsonDict:
    levels = int(registry.get("reproducible_total_levels") or 0)
    ready = False
    best = {
        "lever": "blocked",
        "median_actions": None,
        "median_actions_delta_vs_baseline": None,
        "action_reduction_vs_baseline": None,
        "heldout_solve_rate": 0.0,
        "reason": "a6_integration_artifact_missing",
    }
    integrated = {"lever": "A6_integration", "status": "missing_required_resource"}
    decision = _action_efficiency_decision(best, integrated)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": "blocked_a6_integration_artifact_missing",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "median_actions_best_lever": best,
        "median_actions_baseline": BASELINE_MEDIAN_ACTIONS,
        "heldout_solve_rate": 0.0,
        "submission_package_ready": ready,
        "flagged_artifacts_excluded": [],
        "preconditions_checked": dict(preconditions_checked),
        "per_lever_scorecard": [],
        "integrated_scorecard": integrated,
        "action_efficiency_decision": decision,
        "variant_transfer_rate": 0.0,
        "reproducible_total_levels": levels,
        "prior_submitted_baseline_levels": int(
            registry.get("prior_submitted_baseline_levels") or PRIOR_SUBMITTED_BASELINE_LEVELS
        ),
        "verifier_is_oracle": False,
        "verifier_claims": [],
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "upstream_provenance": provenance,
        "random_seed": RANDOM_SEED,
        "leaderboard_submission": False,
        "submitted_to_leaderboard": False,
        "duration_s": float(duration_s),
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        median_actions_best_lever=best,
        per_lever_scorecard=[],
        integrated_scorecard=integrated,
        action_efficiency_decision=decision,
        heldout_solve_rate=0.0,
        variant_transfer_rate=0.0,
        reproducible_total_levels=levels,
        submission_package_ready=ready,
    )
    return artifact


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> JsonDict:
    root_path = Path(root)
    start = time.time() if started_s is None else started_s
    raw_artifacts, provenance, exclusions = _read_inputs(root_path, live_flag_runner, summarize_runner)
    registry = load_registry_totals(root_path)
    preconditions = _preconditions_checked(root_path, provenance, registry)
    end = time.time() if now_s is None else now_s
    duration_s = round(float(end - start), 6)

    if not preconditions["a6_integration_artifact_present"]:
        artifact = _blocked_artifact(
            duration_s=duration_s,
            preconditions_checked=preconditions,
            provenance=provenance,
            registry=registry,
        )
        validate_artifact(artifact)
        return artifact

    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = _clean_payloads(raw_artifacts, skipped)
    per_lever = [
        _direct_lever_row(
            lever="A1_prune",
            payload=clean["A1_prune"],
            skipped=skipped.get("A1_prune", False),
            median_field="median_actions_with_prune",
            solve_field="solve_rate_with_prune",
        ),
        _direct_lever_row(
            lever="A2_imitation",
            payload=clean["A2_imitation"],
            skipped=skipped.get("A2_imitation", False),
            median_field="median_actions_with_prior",
            solve_field="solve_rate_with_prior",
        ),
        _direct_lever_row(
            lever="A3_adaptive_budget",
            payload=clean["A3_adaptive_budget"],
            skipped=skipped.get("A3_adaptive_budget", False),
            median_field="median_actions_with_adaptive",
            solve_field="solve_rate_with_adaptive",
        ),
        _a4_scorecard(clean["A4_lazy_best_first"], skipped.get("A4_lazy_best_first", False)),
    ]
    integrated = _integrated_scorecard(clean["A6_integration"], skipped.get("A6_integration", False))
    scoreboard = _scoreboard_context(
        clean["scoreboard_context"],
        skipped.get("scoreboard_context", False),
    )
    heldout = float(scoreboard["heldout_solve_rate"])
    variant = float(scoreboard["variant_transfer_rate"])
    best = _best_lever(per_lever, heldout)
    decision = _action_efficiency_decision(best, integrated)
    levels = int(registry.get("reproducible_total_levels") or 0)
    prior = int(registry.get("prior_submitted_baseline_levels") or PRIOR_SUBMITTED_BASELINE_LEVELS)
    ready = levels > prior

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": _honest_verdict(best, heldout),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "median_actions_best_lever": best,
        "median_actions_baseline": BASELINE_MEDIAN_ACTIONS,
        "heldout_solve_rate": heldout,
        "submission_package_ready": ready,
        "flagged_artifacts_excluded": exclusions,
        "preconditions_checked": preconditions,
        "per_lever_scorecard": per_lever,
        "integrated_scorecard": integrated,
        "action_efficiency_decision": decision,
        "variant_transfer_rate": variant,
        "variant_transfer_context": scoreboard,
        "reproducible_total_levels": levels,
        "prior_submitted_baseline_levels": prior,
        "submission_readiness_decision": {
            "submission_package_ready": ready,
            "operator_only": True,
            "submitted_to_leaderboard": False,
            "decision": "ready_for_operator_submit" if ready else "not_ready_for_operator_submit",
        },
        "level_up_context": _level_up_context(
            clean["A5_level_up"],
            skipped.get("A5_level_up", False),
        ),
        "verifier_is_oracle": False,
        "verifier_claims": _verifier_claims(raw_artifacts, skipped),
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "random_seed": RANDOM_SEED,
        "leaderboard_submission": False,
        "submitted_to_leaderboard": False,
        "duration_s": duration_s,
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        median_actions_best_lever=best,
        per_lever_scorecard=per_lever,
        integrated_scorecard=integrated,
        action_efficiency_decision=decision,
        heldout_solve_rate=heldout,
        variant_transfer_rate=variant,
        reproducible_total_levels=levels,
        submission_package_ready=ready,
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if not isinstance(artifact.get("median_actions_best_lever"), Mapping):
        raise ValueError("median_actions_best_lever must be an object")
    if float(artifact.get("median_actions_baseline") or 0.0) != BASELINE_MEDIAN_ACTIONS:
        raise ValueError("median_actions_baseline must equal 7760")
    for field in ("heldout_solve_rate", "variant_transfer_rate"):
        value = artifact.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact.get("submission_package_ready"), bool):
        raise ValueError("submission_package_ready must be a bare bool")
    for field in ("flagged_artifacts_excluded", "per_lever_scorecard", "verifier_claims", "cited_upstream_artifacts"):
        if not isinstance(artifact.get(field), list):
            raise ValueError(f"{field} must be a list")
    for field in ("preconditions_checked", "integrated_scorecard", "action_efficiency_decision"):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be an object")
    levels = artifact.get("reproducible_total_levels")
    if not isinstance(levels, int) or isinstance(levels, bool):
        raise ValueError("reproducible_total_levels must be a bare int")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    for claim in artifact["verifier_claims"]:
        if not isinstance(claim, Mapping) or not isinstance(claim.get("verifier_is_oracle"), bool):
            raise ValueError("verifier_claims must declare verifier_is_oracle as bool")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")
    if artifact.get("leaderboard_submission") is not False or artifact.get("submitted_to_leaderboard") is not False:
        raise ValueError("leaderboard_submission must be false")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream provenance row must be an object")
        if not base.is_sha256(row.get("sha256")):
            raise ValueError("upstream provenance row has invalid sha256")
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must not import fields")
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    expected = checksum_from_inputs(
        provenance,
        median_actions_best_lever=artifact["median_actions_best_lever"],
        per_lever_scorecard=artifact["per_lever_scorecard"],
        integrated_scorecard=artifact["integrated_scorecard"],
        action_efficiency_decision=artifact["action_efficiency_decision"],
        heldout_solve_rate=float(artifact["heldout_solve_rate"]),
        variant_transfer_rate=float(artifact["variant_transfer_rate"]),
        reproducible_total_levels=int(artifact["reproducible_total_levels"]),
        submission_package_ready=bool(artifact["submission_package_ready"]),
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match inputs")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )
    artifact["result_path"] = str(output_path)
    validate_artifact(artifact)
    path = Path(root) / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> JsonDict:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )
    if write:
        out = Path(root) / OUTPUT_REL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
