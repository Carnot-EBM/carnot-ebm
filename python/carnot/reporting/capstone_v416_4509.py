"""Build the Exp 4509 .416 ARC affordance/value capstone.

Spec refs: REQ-CAPSTONE-4509, SCENARIO-CAPSTONE-4509.

This aggregation reads the .416 upstream result artifacts through the
Reading-Results discipline, skips adversarial-stamped or live-critical inputs
before importing metrics, and reports the requested A1-A5 plus submitted-agent
headline signals without promoting reproducible total levels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import capstone_v400_4335 as base


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4509_capstone_v416.json")
EXPERIMENT_ID = 4509
RANDOM_SEED = 4509
SCHEMA = "carnot.capstone_v416_4509.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-CAPSTONE-4509", "SCENARIO-CAPSTONE-4509"]
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


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "4500_value_weight": Upstream(
        4500,
        Path("results/experiment_4500_value_weight_remeasure.json"),
    ),
    "4501_frame_change": Upstream(
        4501,
        Path("results/experiment_4501_frame_change_predictor_rerun.json"),
    ),
    "4502_energy_ranking": Upstream(
        4502,
        Path("results/experiment_4502_energy_augmented_ranking.json"),
    ),
    "4503_hud_l2": Upstream(
        4503,
        Path("results/experiment_4503_hud_register_deepen_l2.json"),
    ),
    "4504_adapter_l2": Upstream(
        4504,
        Path("results/experiment_4504_adapter_deepen_l2.json"),
    ),
    "4505_scoreboard": Upstream(
        4505,
        Path("results/experiment_4505_submitted_agent_scoreboard.json"),
    ),
    "4506_lazy_value": Upstream(
        4506,
        Path("results/experiment_4506_lazy_value_eval_prototype.json"),
    ),
    "4507_hardware": Upstream(
        4507,
        Path("results/experiment_4507_hardware_continuity_audit.json"),
    ),
    "4508_sota": Upstream(
        4508,
        Path("results/experiment_4508_arc_affordance_sota_416.json"),
    ),
}

ARTIFACT_EXPERIMENT_IDS = {
    key: upstream.experiment_id for key, upstream in DEFAULT_UPSTREAMS.items()
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "a1_value_weight_verdict",
    "a2_frame_change_predictor_efficiency_delta",
    "a3_energy_augmented_ranking",
    "a4_a5_l2_banked",
    "submitted_agent_heldout_solve_rate",
    "variant_transfer_rate",
    "verifier_is_oracle",
    "verifier_claims",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_."
    ),
    "inference_substrate": (
        "explicit substrate so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource "
        "fabrication."
    ),
    "a1_value_weight_verdict": (
        "reports whether a positive v3-head value_weight beats zero within budget, "
        "but imports no metric from an adversarial-stamped source."
    ),
    "a2_frame_change_predictor_efficiency_delta": (
        "reports held-out median actions before/after, efficiency delta, and "
        "solve-rate guard for the frame-change predictor."
    ),
    "a3_energy_augmented_ranking": (
        "reports energy-augmented ranking solve-rate and efficiency deltas without "
        "turning a null delta into a win."
    ),
    "a4_a5_l2_banked": (
        "bare bool from clean offline reproduction gates: true only when any A4/A5 "
        "L2 reproduction is banked."
    ),
    "submitted_agent_heldout_solve_rate": (
        "bare float from the submitted-agent scoreboard, with nested source "
        "provenance kept explicit."
    ),
    "variant_transfer_rate": (
        "bare float from the submitted-agent scoreboard variant-transfer headline."
    ),
    "verifier_is_oracle": (
        "BARE bool=false for the capstone itself; verifier-derived upstream claims "
        "declare their own circularity."
    ),
    "verifier_claims": (
        "each verifier claim declares verifier_is_oracle so circularity is explicit."
    ),
    "flagged_artifacts_skipped": "stamped or live-critical artifacts import no fields.",
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256}; skipped flagged artifacts "
        "import no fields."
    ),
    "reproducibility_checksum": "content hash for reproducibility",
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "4500_value_weight": [
        "honest_verdict",
        "selected_value_weight",
        "submitted_value_weight_after",
        "action_budget",
        "eval_budget_median_wall_s",
        "selection",
        "per_weight",
        "preconditions_checked",
    ],
    "4501_frame_change": [
        "honest_verdict",
        "heldout_median_actions_before",
        "heldout_median_actions_after",
        "implied_efficiency_delta",
        "solve_rate_before",
        "solve_rate_after",
        "solve_rate_dropped",
        "positive_control",
        "preconditions_checked",
    ],
    "4502_energy_ranking": [
        "honest_verdict",
        "predictor_only_median_actions",
        "energy_augmented_median_actions",
        "efficiency_delta_vs_predictor_only",
        "predictor_only_solve_rate",
        "energy_augmented_solve_rate",
        "solve_rate_delta_vs_predictor_only",
        "solve_rate_dropped",
        "energy_term_added_value",
        "ranking_formula",
        "gate_artifact_summary",
        "preconditions_checked",
    ],
    "4503_hud_l2": [
        "honest_verdict",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "residual_blockers",
        "preconditions_checked",
    ],
    "4504_adapter_l2": [
        "honest_verdict",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "residual_blockers",
        "preconditions_checked",
    ],
    "4505_scoreboard": [
        "honest_verdict",
        "a1_value_weight_verdict",
        "headline_metrics",
        "scoreboard_row",
        "parity_gate",
        "leaderboard_submission",
        "preconditions_checked",
    ],
    "4506_lazy_value": [
        "honest_verdict",
        "speedup_factor",
        "routing_quality_preserved",
        "routing_quality_match_rate",
        "value_head_call_reduction_factor",
        "preconditions_checked",
    ],
    "4507_hardware": [
        "honest_verdict",
        "per_board_reachability",
        "per_board_status",
        "preconditions_checked",
    ],
    "4508_sota": [
        "honest_verdict",
        "strongest_for_v417",
        "source_ids",
        "preconditions_checked",
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


def _number_or_none(payload: Mapping[str, Any] | None, field: str) -> float | None:
    return base.float_metric(payload, field)


def _mapping_metric(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


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
        except (OSError, json.JSONDecodeError, ValueError) as exc:
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


def _scoreboard_a1(scoreboard_payload: JsonDict | None, scoreboard_skipped: bool) -> Mapping[str, Any]:
    if scoreboard_skipped or not isinstance(scoreboard_payload, Mapping):
        return {}
    value = scoreboard_payload.get("a1_value_weight_verdict")
    return value if isinstance(value, Mapping) else {}


def a1_read(
    payload: JsonDict | None,
    skipped: bool,
    scoreboard_payload: JsonDict | None,
    scoreboard_skipped: bool,
) -> JsonDict:
    scoreboard = _scoreboard_a1(scoreboard_payload, scoreboard_skipped)
    scoreboard_after = _number_or_none(scoreboard, "submitted_value_weight_after")
    scoreboard_selected = _number_or_none(scoreboard, "selected_value_weight")
    scoreboard_source_flagged = base.bool_metric(scoreboard, "source_flagged_adversarial") is True
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "headline_eligible": False,
            "positive_weight_beats_zero_within_budget": False,
            "selected_value_weight": scoreboard_selected,
            "submitted_value_weight_after": scoreboard_after,
            "scoreboard_state": base.str_metric(scoreboard, "state"),
            "scoreboard_source_flagged_adversarial": scoreboard_source_flagged,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "headline_eligible": False,
            "positive_weight_beats_zero_within_budget": False,
            "selected_value_weight": scoreboard_selected,
            "submitted_value_weight_after": scoreboard_after,
            "scoreboard_state": base.str_metric(scoreboard, "state"),
            "scoreboard_source_flagged_adversarial": scoreboard_source_flagged,
        }

    selection = _mapping_metric(payload, "selection")
    selected = _number_or_none(selection, "selected_value_weight")
    if selected is None:
        selected = _number_or_none(payload, "selected_value_weight")
    within_budget = base.bool_metric(selection, "within_wall_budget") is True
    beats_control = base.bool_metric(selection, "beats_control") is True
    positive_selected = selected is not None and selected > 0.0
    positive_win = positive_selected and beats_control and within_budget
    return {
        "state": "positive_weight_beats_zero" if positive_win else "keep_zero_value_weight",
        "headline_eligible": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "positive_weight_beats_zero_within_budget": positive_win,
        "selected_value_weight": selected,
        "submitted_value_weight_after": _number_or_none(payload, "submitted_value_weight_after"),
        "control_solve_rate": _number_or_none(selection, "control_solve_rate"),
        "selected_solve_rate": _number_or_none(selection, "selected_solve_rate"),
        "within_wall_budget": within_budget,
        "action_budget": base.int_metric(payload, "action_budget"),
        "eval_budget_median_wall_s": _number_or_none(payload, "eval_budget_median_wall_s"),
        "per_weight_count": len(base.list_metric(payload, "per_weight")),
        "scoreboard_state": base.str_metric(scoreboard, "state"),
        "scoreboard_source_flagged_adversarial": scoreboard_source_flagged,
    }


def a2_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "efficiency_delta": 0.0,
            "median_actions_delta": 0.0,
            "efficiency_win": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "efficiency_delta": 0.0,
            "median_actions_delta": 0.0,
            "efficiency_win": False,
        }

    before = _number_or_none(payload, "heldout_median_actions_before")
    after = _number_or_none(payload, "heldout_median_actions_after")
    actions_delta = before - after if before is not None and after is not None else 0.0
    efficiency_delta = _number_or_none(payload, "implied_efficiency_delta") or 0.0
    solve_rate_dropped = base.bool_metric(payload, "solve_rate_dropped") is True
    efficiency_win = (efficiency_delta > 0.0 or actions_delta > 0.0) and not solve_rate_dropped
    positive_control = _mapping_metric(payload, "positive_control")
    return {
        "state": "efficiency_win" if efficiency_win else "efficiency_null",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "heldout_median_actions_before": before,
        "heldout_median_actions_after": after,
        "median_actions_delta": actions_delta,
        "efficiency_delta": efficiency_delta,
        "solve_rate_before": _number_or_none(payload, "solve_rate_before"),
        "solve_rate_after": _number_or_none(payload, "solve_rate_after"),
        "solve_rate_dropped": solve_rate_dropped,
        "positive_control_actions_reduced": base.bool_metric(positive_control, "actions_reduced") is True,
        "efficiency_win": efficiency_win,
    }


def a3_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "energy_augmented_solve_rate": 0.0,
            "solve_rate_delta": 0.0,
            "efficiency_delta": 0.0,
            "efficiency_win": False,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "energy_augmented_solve_rate": 0.0,
            "solve_rate_delta": 0.0,
            "efficiency_delta": 0.0,
            "efficiency_win": False,
        }

    efficiency_delta = _number_or_none(payload, "efficiency_delta_vs_predictor_only") or 0.0
    solve_rate_delta = _number_or_none(payload, "solve_rate_delta_vs_predictor_only") or 0.0
    solve_rate_dropped = base.bool_metric(payload, "solve_rate_dropped") is True
    efficiency_win = (efficiency_delta > 0.0 or solve_rate_delta > 0.0) and not solve_rate_dropped
    return {
        "state": "energy_augmented_efficiency_win" if efficiency_win else "energy_augmented_null",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "predictor_only_median_actions": _number_or_none(payload, "predictor_only_median_actions"),
        "energy_augmented_median_actions": _number_or_none(
            payload,
            "energy_augmented_median_actions",
        ),
        "efficiency_delta": efficiency_delta,
        "predictor_only_solve_rate": _number_or_none(payload, "predictor_only_solve_rate") or 0.0,
        "energy_augmented_solve_rate": _number_or_none(payload, "energy_augmented_solve_rate")
        or 0.0,
        "solve_rate_delta": solve_rate_delta,
        "solve_rate_dropped": solve_rate_dropped,
        "energy_term_added_value": base.bool_metric(payload, "energy_term_added_value") is True,
        "ranking_formula": base.str_metric(payload, "ranking_formula"),
        "gate_artifact_summary": dict(_mapping_metric(payload, "gate_artifact_summary")),
        "efficiency_win": efficiency_win,
    }


def _gate_l2_banked(gate: Mapping[str, Any]) -> bool:
    claimed = base.int_metric(gate, "claimed_level")
    reached = base.int_metric(gate, "reached_level")
    return base.bool_metric(gate, "reproduced") is True and claimed >= 2 and reached >= 2


def _l2_detail(payload: JsonDict | None, skipped: bool, fallback_game: str) -> JsonDict:
    if skipped:
        return {"skipped": True, "state": "excluded_flagged_adversarial", "l2_banked": False}
    if payload is None:
        return {"skipped": False, "state": "missing_or_excluded", "l2_banked": False}
    gate = _mapping_metric(payload, "reproduction_gate")
    l2_banked = _gate_l2_banked(gate)
    return {
        "skipped": False,
        "state": "l2_banked" if l2_banked else "l2_not_banked",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "game": base.str_metric(gate, "game") or base.str_metric(payload, "target_game") or fallback_game,
        "offline_reproduced": base.bool_metric(payload, "offline_reproduced") is True,
        "reproduced_levels": base.int_metric(payload, "reproduced_levels"),
        "reproduction_gate": dict(gate),
        "residual_blockers": base.list_metric(payload, "residual_blockers"),
        "l2_banked": l2_banked,
    }


def a4_a5_l2_read(
    hud_payload: JsonDict | None,
    adapter_payload: JsonDict | None,
    hud_skipped: bool,
    adapter_skipped: bool,
) -> JsonDict:
    if hud_skipped and adapter_skipped:
        return {"state": "excluded_flagged_adversarial", "any_l2_banked": False}
    if hud_payload is None and adapter_payload is None:
        return {"state": "missing_or_excluded", "any_l2_banked": False}

    hud = _l2_detail(hud_payload, hud_skipped, "ka59")
    adapter = _l2_detail(adapter_payload, adapter_skipped, "cd82")
    any_l2 = bool(hud.get("l2_banked")) or bool(adapter.get("l2_banked"))
    return {
        "state": "l2_banked" if any_l2 else "l2_not_banked",
        "any_l2_banked": any_l2,
        "hud_register": hud,
        "adapter_l2": adapter,
    }


def scoreboard_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {
            "state": "excluded_flagged_adversarial",
            "heldout_solve_rate": 0.0,
            "variant_transfer_rate": 0.0,
        }
    if payload is None:
        return {
            "state": "missing_or_excluded",
            "heldout_solve_rate": 0.0,
            "variant_transfer_rate": 0.0,
        }
    metrics = _mapping_metric(payload, "headline_metrics")
    row = _mapping_metric(payload, "scoreboard_row")
    heldout = _mapping_metric(row, "heldout_generic_measurement")
    variant = _mapping_metric(row, "variant_transfer_measurement")
    source_flagged = base.bool_metric(heldout, "source_flagged_adversarial") is True
    if not heldout:
        source_flagged = (
            base.bool_metric(_mapping_metric(payload, "a1_value_weight_verdict"), "source_flagged_adversarial")
            is True
        )
    return {
        "state": "submitted_agent_scoreboard_measured",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "heldout_solve_rate": _number_or_none(
            metrics,
            "submitted_default_heldout_generic_solve_rate",
        )
        or 0.0,
        "heldout_solved": base.int_metric(metrics, "submitted_default_heldout_generic_solved"),
        "heldout_attempted": base.int_metric(metrics, "submitted_default_heldout_generic_attempted"),
        "heldout_source_flagged_adversarial": source_flagged,
        "heldout_median_actions_to_first_levelup": _number_or_none(
            heldout,
            "median_actions_to_first_levelup",
        ),
        "variant_transfer_rate": _number_or_none(metrics, "variant_transfer_rate") or 0.0,
        "variant_transfer_solved": base.int_metric(metrics, "variant_transfer_solved"),
        "variant_transfer_attempted": base.int_metric(metrics, "variant_transfer_attempted"),
        "variant_transfer_source_artifact": base.str_metric(variant, "source_artifact"),
        "leaderboard_submission": base.bool_metric(payload, "leaderboard_submission") is True,
        "parity_gate": dict(_mapping_metric(payload, "parity_gate")),
    }


def operational_context_read(payload: JsonDict | None, skipped: bool) -> JsonDict:
    if skipped:
        return {"state": "excluded_flagged_adversarial"}
    if payload is None:
        return {"state": "missing_or_excluded"}
    return {
        "state": "available",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "speedup_factor": _number_or_none(payload, "speedup_factor"),
        "routing_quality_preserved": base.bool_metric(payload, "routing_quality_preserved"),
        "routing_quality_match_rate": _number_or_none(payload, "routing_quality_match_rate"),
        "value_head_call_reduction_factor": _number_or_none(
            payload,
            "value_head_call_reduction_factor",
        ),
        "per_board_reachability": dict(_mapping_metric(payload, "per_board_reachability")),
        "strongest_for_v417": base.str_metric(payload, "strongest_for_v417"),
        "source_ids": base.list_metric(payload, "source_ids"),
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="a1_value_weight_verdict",
            required_keys=("4500_value_weight",),
            verdict_fn=lambda present: a1_read(
                present.get("4500_value_weight"),
                False,
                None,
                False,
            )["positive_weight_beats_zero_within_budget"],
        ),
        aggregate.AxisSpec(
            name="a2_frame_change_predictor_efficiency_delta",
            required_keys=("4501_frame_change",),
            verdict_fn=lambda present: a2_read(
                present.get("4501_frame_change"),
                False,
            )["efficiency_win"],
        ),
        aggregate.AxisSpec(
            name="a3_energy_augmented_ranking",
            required_keys=("4502_energy_ranking",),
            verdict_fn=lambda present: a3_read(
                present.get("4502_energy_ranking"),
                False,
            )["efficiency_win"],
        ),
        aggregate.AxisSpec(
            name="a4_a5_l2_banked",
            required_keys=("4503_hud_l2", "4504_adapter_l2"),
            verdict_fn=lambda present: a4_a5_l2_read(
                present.get("4503_hud_l2"),
                present.get("4504_adapter_l2"),
                False,
                False,
            )["any_l2_banked"],
        ),
        aggregate.AxisSpec(
            name="submitted_agent_scoreboard",
            required_keys=("4505_scoreboard",),
            verdict_fn=lambda present: scoreboard_read(
                present.get("4505_scoreboard"),
                False,
            )["heldout_solve_rate"],
        ),
        aggregate.AxisSpec(
            name="operational_context",
            required_keys=("4506_lazy_value", "4507_hardware", "4508_sota"),
            verdict_fn=lambda present: sorted(present),
        ),
    ]


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


def _claim(name: str, verifier_is_oracle: bool, source: str, skipped: bool = False) -> JsonDict:
    return {
        "claim": name,
        "source": source,
        "verifier_is_oracle": bool(verifier_is_oracle),
        "skipped": bool(skipped),
    }


def _verifier_claims(skipped: Mapping[str, bool]) -> list[JsonDict]:
    return [
        _claim(
            "a1_value_weight_verdict",
            False,
            "exp4500",
            skipped.get("4500_value_weight", False),
        ),
        _claim(
            "a2_frame_change_predictor_efficiency_delta",
            False,
            "exp4501",
            skipped.get("4501_frame_change", False),
        ),
        _claim(
            "a3_energy_augmented_ranking",
            False,
            "exp4502",
            skipped.get("4502_energy_ranking", False),
        ),
        _claim(
            "a4_a5_l2_banked",
            False,
            "exp4503+exp4504",
            skipped.get("4503_hud_l2", False) or skipped.get("4504_adapter_l2", False),
        ),
        _claim(
            "submitted_agent_scoreboard",
            False,
            "exp4505",
            skipped.get("4505_scoreboard", False),
        ),
    ]


def _preconditions_checked(root: Path, provenance: list[JsonDict], clean: Mapping[str, Any]) -> JsonDict:
    provenance_by_key = {row["artifact_key"]: row for row in provenance}
    upstreams: list[JsonDict] = []
    for key, path in _selected_paths(root).items():
        upstream = DEFAULT_UPSTREAMS[key]
        row = provenance_by_key.get(key)
        payload = clean.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
                "upstream_preconditions": payload.get("preconditions_checked")
                if isinstance(payload, Mapping)
                else None,
            }
        )
    return {
        "upstream_artifacts": upstreams,
        "summarize_artifact_required": "scripts/summarize_artifact.py",
        "reading_results_discipline": True,
        "leaderboard_submission": False,
    }


def _capstone_recheck_status(flags: list[dict[str, Any]]) -> JsonDict:
    circular = any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    critical = base.live_has_critical(flags)
    return {
        "status": "critical_flags" if critical else "clean",
        "flags": flags,
        "circular_moat_overclaim": circular,
    }


def _honest_verdict(
    *,
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    l2_banked: bool,
    heldout_rate: float,
    variant_rate: float,
) -> str:
    a1_state = (
        "a1_positive_weight_win"
        if a1.get("positive_weight_beats_zero_within_budget") is True
        else "a1_no_clean_positive_weight_win"
    )
    a2_state = "a2_efficiency_win" if a2.get("efficiency_win") is True else "a2_null_delta"
    a3_state = "a3_efficiency_win" if a3.get("efficiency_win") is True else "a3_null_delta"
    l2_state = "l2_banked" if l2_banked else "l2_not_banked"
    return (
        "complete: v416_"
        f"{a1_state}_{a2_state}_{a3_state}_{l2_state}_"
        f"heldout_{heldout_rate:.3f}_variant_{variant_rate:.2f}"
    )


def checksum_from_inputs(
    provenance: list[Mapping[str, Any]],
    *,
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4_a5: Mapping[str, Any],
    heldout_rate: float,
    variant_rate: float,
) -> str:
    payload = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4_a5": a4_a5,
        "submitted_agent_heldout_solve_rate": heldout_rate,
        "variant_transfer_rate": variant_rate,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> JsonDict:
    start = time.time() if started_s is None else started_s
    raw_artifacts, provenance, exclusions = _read_inputs(root, live_flag_runner, summarize_runner)
    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    skipped = {row["artifact_key"]: bool(row["skipped"]) for row in provenance}
    clean = {
        key: base.clean_payload(
            raw_artifacts.get(key) if isinstance(raw_artifacts.get(key), dict) else None,
            skipped.get(key, False),
        )
        for key in DEFAULT_UPSTREAMS
    }

    a1 = a1_read(
        clean["4500_value_weight"],
        skipped.get("4500_value_weight", False),
        clean["4505_scoreboard"],
        skipped.get("4505_scoreboard", False),
    )
    a2 = a2_read(clean["4501_frame_change"], skipped.get("4501_frame_change", False))
    a3 = a3_read(clean["4502_energy_ranking"], skipped.get("4502_energy_ranking", False))
    a4_a5 = a4_a5_l2_read(
        clean["4503_hud_l2"],
        clean["4504_adapter_l2"],
        skipped.get("4503_hud_l2", False),
        skipped.get("4504_adapter_l2", False),
    )
    scoreboard = scoreboard_read(clean["4505_scoreboard"], skipped.get("4505_scoreboard", False))
    heldout_rate = float(scoreboard.get("heldout_solve_rate") or 0.0)
    variant_rate = float(scoreboard.get("variant_transfer_rate") or 0.0)
    end = time.time() if now_s is None else now_s

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "random_seed": RANDOM_SEED,
        "duration_s": round(end - start, 6),
        "honest_verdict": _honest_verdict(
            a1=a1,
            a2=a2,
            a3=a3,
            l2_banked=bool(a4_a5.get("any_l2_banked")),
            heldout_rate=heldout_rate,
            variant_rate=variant_rate,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _preconditions_checked(root, provenance, clean),
        "a1_value_weight_verdict": a1,
        "a2_frame_change_predictor_efficiency_delta": a2,
        "a3_energy_augmented_ranking": a3,
        "a4_a5_l2_banked": bool(a4_a5.get("any_l2_banked")),
        "a4_a5_l2_details": a4_a5,
        "submitted_agent_heldout_solve_rate": heldout_rate,
        "variant_transfer_rate": variant_rate,
        "submitted_agent_scoreboard": scoreboard,
        "operational_context": {
            "exp4506_lazy_value": operational_context_read(
                clean["4506_lazy_value"],
                skipped.get("4506_lazy_value", False),
            ),
            "exp4507_hardware": operational_context_read(
                clean["4507_hardware"],
                skipped.get("4507_hardware", False),
            ),
            "exp4508_sota": operational_context_read(
                clean["4508_sota"],
                skipped.get("4508_sota", False),
            ),
        },
        "verifier_is_oracle": False,
        "verifier_claims": _verifier_claims(skipped),
        "flagged_artifacts_skipped": exclusions,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "per_axis_gaps": list(availability_report.get("missing_upstream_artifacts", []))
        + list(availability_report.get("flagged_artifacts_excluded", [])),
        "availability_report": availability_report,
        "upstream_provenance": provenance,
        "upstream_sha256_set": sorted(str(row["sha256"]) for row in provenance),
        "submitted_to_leaderboard": False,
        "capstone_live_adversarial_recheck": {"status": "not_run_until_write"},
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = checksum_from_inputs(
        provenance,
        a1=a1,
        a2=a2,
        a3=a3,
        a4_a5=a4_a5,
        heldout_rate=heldout_rate,
        variant_rate=variant_rate,
    )
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
    for field in (
        "preconditions_checked",
        "a1_value_weight_verdict",
        "a2_frame_change_predictor_efficiency_delta",
        "a3_energy_augmented_ranking",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be an object")
    if not isinstance(artifact.get("a4_a5_l2_banked"), bool):
        raise ValueError("a4_a5_l2_banked must be a bare bool")
    for field in ("submitted_agent_heldout_solve_rate", "variant_transfer_rate"):
        value = artifact.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be a bare float")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    claims = artifact.get("verifier_claims")
    if not isinstance(claims, list):
        raise ValueError("verifier_claims must be a list")
    for claim in claims:
        if not isinstance(claim, Mapping) or not isinstance(claim.get("verifier_is_oracle"), bool):
            raise ValueError("verifier_claims must declare verifier_is_oracle as bool")
    for field in ("flagged_artifacts_skipped", "cited_upstream_artifacts"):
        if not isinstance(artifact.get(field), list):
            raise ValueError(f"{field} must be a list")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed does not match experiment")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")
    if "reproducible_total_levels" in artifact:
        raise ValueError("reproducible_total_levels is not a .416 headline field")
    checksum = str(artifact.get("reproducibility_checksum", "")).removeprefix("sha256:")
    if not base.is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match")
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
    expected = checksum_from_inputs(
        provenance,
        a1=artifact["a1_value_weight_verdict"],
        a2=artifact["a2_frame_change_predictor_efficiency_delta"],
        a3=artifact["a3_energy_augmented_ranking"],
        a4_a5=artifact["a4_a5_l2_details"],
        heldout_rate=float(artifact["submitted_agent_heldout_solve_rate"]),
        variant_rate=float(artifact["variant_transfer_rate"]),
    )
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum does not match inputs")


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
    capstone_live_flag_runner: LiveFlagRunner = base.run_live_flags,
) -> Path:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )
    validate_artifact(artifact)
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact["capstone_live_adversarial_recheck"] = _capstone_recheck_status(
        capstone_live_flag_runner(path)
    )
    validate_artifact(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args() -> JsonDict:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_REL_PATH)
    args = parser.parse_args()
    return {"output": args.output}


def main() -> int:  # pragma: no cover
    args = _parse_args()
    output = write_artifact(REPO_ROOT, output_path=args["output"])
    print(output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
