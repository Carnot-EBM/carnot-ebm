"""Experiment 4542: .419 capstone aggregation.

Spec refs: REQ-CAPSTONE-4542, SCENARIO-CAPSTONE-4542,
SCENARIO-CAPSTONE-4542-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script import guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot.reporting import capstone_v400_4335 as base  # noqa: E402
from scripts import summarize_artifact as summary_reader  # noqa: E402


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]

RESULT_RELATIVE_PATH = "results/experiment_4542_capstone_v419.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4542
RANDOM_SEED = 4542
SCHEMA = "carnot.capstone_v419_4542.v1"
SPEC_REFS = ["REQ-CAPSTONE-4542", "SCENARIO-CAPSTONE-4542"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORE_EFFICIENCY_BASELINE = 2.0074
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
        "terminal prefix; success: reinduction_core_efficiency_<n>_above_2.0074 OR "
        "complete: reinduction_null_efficiency_unmoved_barrier_refined."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    ),
    "efficiency_moved": (
        "the bottom line -- did core_efficiency rise STRICTLY above 2.0074 (a CORE game "
        "reached a deeper level) at preserved solve-rate (the thing .418 did not achieve)."
    ),
    "deepest_level_gains_per_core_game": (
        "per-CORE-game deepest level before/after re-induction -- the direct score-lever evidence."
    ),
    "reproducible_total_levels_delta": (
        "did solve CAPABILITY grow this milestone (A3/A5 level-up)."
    ),
    "flagged_artifacts_handled": (
        "names any flagged_adversarial artifact excluded AND any null-delta-carve-out artifact "
        "whose diagnosis was read (B2) -- the fabrication-gate compliance + the no-lost-diagnosis "
        "guard."
    ),
    "cited_upstream_artifacts": (
        "every headline number traces to a real upstream measurement (the audit trail)."
    ),
    "ready_for_operator_submit": (
        "True only if the integrated config is a CORE-preserved core_efficiency improvement worth "
        "a 1/day slot; never submits."
    ),
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "efficiency_moved",
    "deepest_level_gains_per_core_game",
    "reproducible_total_levels_delta",
    "flagged_artifacts_handled",
    "cited_upstream_artifacts",
    "ready_for_operator_submit",
    "preconditions_checked",
    "scorecard",
    "energy_routing_generalization",
    "primitive_transfer_generalization",
    "operator_resubmission_verdict",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "A1_goal_reinduction": Upstream(
        4533, Path("results/experiment_4533_per_level_goal_reinduction.json")
    ),
    "A2_energy_routing": Upstream(
        4534, Path("results/experiment_4534_energy_trust_next_level_routing.json")
    ),
    "A3_levelup_attempt": Upstream(
        4535, Path("results/experiment_4535_levelup_attempt.json")
    ),
    "A4_integration": Upstream(
        4536, Path("results/experiment_4536_integration_8game_gate.json")
    ),
    "A5_primitive_transfer": Upstream(
        4537, Path("results/experiment_4537_reinduction_primitive_persist_transfer.json")
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "A1_goal_reinduction": [
        "core_efficiency_baseline",
        "core_efficiency_best",
        "efficiency_delta",
        "core_solves_preserved",
        "deepest_level_reached_per_core_game",
    ],
    "A2_energy_routing": [
        "core_efficiency_baseline",
        "core_efficiency_energy_routed",
        "energy_separation_auroc",
        "deepest_level_reached_per_core_game",
        "verifier_is_oracle",
    ],
    "A3_levelup_attempt": [
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "target_level",
        "reproduction_gate",
        "registry_update",
    ],
    "A4_integration": [
        "core_efficiency_baseline",
        "core_efficiency_integrated",
        "core_solves_preserved",
        "gate_result",
        "ready_for_operator_submit",
    ],
    "A5_primitive_transfer": [
        "primitive_persisted",
        "representation_transfer",
        "transfer_deepest_level_per_game",
        "new_levels_banked",
        "offline_reproduced",
    ],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _number(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int_or_none(payload: Mapping[str, Any] | None, field: str) -> int | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _list(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def _bool(payload: Mapping[str, Any] | None, field: str) -> bool:
    return base.bool_metric(payload, field) is True


def _gate_value_failed(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, Mapping):
        decision_keys = ("pass", "passed", "ok", "verdict_pass", "reproduced")
        decisions = [
            value[key]
            for key in decision_keys
            if isinstance(value.get(key), bool)
        ]
        if decisions:
            return any(decision is False for decision in decisions)
        return any(_gate_value_failed(item) for item in value.values())
    if isinstance(value, list):
        return any(_gate_value_failed(item) for item in value)
    return False


def _acceptance_gate_failed(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for key, value in payload.items():
        lower = key.lower()
        if "acceptance_gate" in lower or lower.startswith("gate_") or lower.endswith("_gate"):
            if _gate_value_failed(value):
                return True
    return False


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _skip_reason(
    *,
    stamped: bool,
    critical: bool,
    parse_error: str,
    acceptance_gate_failed: bool,
    diagnosis_context_read: bool,
) -> str:
    if diagnosis_context_read:
        return "null_delta_carve_out_diagnosis_only"
    if stamped or critical or parse_error:
        return base._exclusion_reason(stamped, critical, parse_error)  # noqa: SLF001
    if acceptance_gate_failed:
        return "failed_acceptance_gate"
    return ""


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], JsonDict]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    handled: JsonDict = {"excluded": [], "null_delta_carve_out_diagnosis_read": []}

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
        payload: JsonDict | None = None
        parse_error = ""
        try:
            payload = base.read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover
            parse_error = f"{type(exc).__name__}: {exc}"

        critical = base.live_has_critical(live_flags)
        diagnosis_context = (
            summary_reader.readable_diagnosis_context(payload, live_flags)
            if payload is not None
            else None
        )
        diagnosis_context_read = diagnosis_context is not None
        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        gate_failed = _acceptance_gate_failed(payload)
        skipped = stamped or critical or payload is None or bool(parse_error) or gate_failed
        reason = _skip_reason(
            stamped=stamped,
            critical=critical,
            parse_error=parse_error,
            acceptance_gate_failed=gate_failed,
            diagnosis_context_read=diagnosis_context_read,
        )
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
            "acceptance_gate_failed": gate_failed,
            "parse_error": parse_error,
            "skipped": skipped,
            "skip_reason": reason,
            "diagnosis_context_read": diagnosis_context_read,
            "diagnosis_context": diagnosis_context or {},
            "fields_imported": _fields_for_payload(key, skipped),
        }
        raw_artifacts[key] = payload
        provenance.append(row)

        if stamped or critical:
            excluded = {
                "artifact_key": key,
                "experiment_id": upstream.experiment_id,
                "path": str(upstream.path),
                "sha256": sha,
                "stamped_flagged_adversarial": stamped,
                "live_critical": critical,
                "reason": reason,
            }
            handled["excluded"].append(excluded)
        if diagnosis_context_read:
            handled["null_delta_carve_out_diagnosis_read"].append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "corrigendum": diagnosis_context["corrigendum"],
                    "diagnosis_context_fields_read": [
                        field for field in ("barrier_diagnosis", "levers_tried", "barrier_refinement")
                        if field in diagnosis_context
                    ],
                }
            )
    return raw_artifacts, provenance, handled


def _provenance_by_key(provenance: list[JsonDict]) -> dict[str, JsonDict]:
    return {str(row["artifact_key"]): row for row in provenance}


def _clean_payload(raw_artifacts: Mapping[str, Any], provenance: Mapping[str, JsonDict], key: str) -> JsonDict | None:
    payload = raw_artifacts.get(key)
    row = provenance.get(key, {})
    return payload if isinstance(payload, dict) and not row.get("skipped") else None


def load_registry_totals(root: Path | str = REPO_ROOT) -> JsonDict:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {
            "registry_path": str(REGISTRY_RELATIVE_PATH),
            "registry_present": False,
            "reproducible_total_levels": 0,
        }
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^reproducible_total_levels:\s*(\d+)\b", text)
    return {
        "registry_path": str(REGISTRY_RELATIVE_PATH),
        "registry_present": True,
        "reproducible_total_levels": int(match.group(1)) if match else 0,
    }


def _payload_status(row: Mapping[str, Any]) -> str:
    if row.get("acceptance_gate_failed"):
        return "failed_acceptance_gate"
    if row.get("skipped"):
        return "excluded_flagged_adversarial_or_live_critical"
    return "missing_or_excluded"


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _deepest_pair(payload: Mapping[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    levels = _mapping(payload, "deepest_level_reached_per_core_game")
    if "baseline" in levels or "best" in levels:
        return _int_mapping(levels.get("baseline")), _int_mapping(levels.get("best"))
    numeric_keys = sorted(
        (int(key), key) for key in levels if isinstance(key, str) and key.isdigit()
    )
    if numeric_keys:
        return _int_mapping(levels[numeric_keys[0][1]]), _int_mapping(levels[numeric_keys[-1][1]])
    return {}, {}


def _level_gains(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    games = sorted(set(before) | set(after))
    return {game: int(after.get(game, 0) - before.get(game, 0)) for game in games}


def _diagnosis_from_row(row: Mapping[str, Any]) -> JsonDict:
    context = row.get("diagnosis_context")
    if not isinstance(context, Mapping):
        return {}
    return {
        field: context[field]
        for field in ("barrier_diagnosis", "levers_tried", "barrier_refinement", "corrigendum")
        if field in context
    }


def _a1_reinduction(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        if row.get("diagnosis_context_read"):
            return {
                "status": "diagnosis_only_null_delta_carve_out",
                "headline_numbers_aggregated": False,
                "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
                "core_efficiency_best": None,
                "core_efficiency_delta": None,
                "core_solves_preserved": None,
                "moved": False,
                "diagnosis": _diagnosis_from_row(row),
            }
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_best": None,
            "core_efficiency_delta": None,
            "core_solves_preserved": False,
            "moved": False,
            "diagnosis": {},
        }

    baseline = _number(payload, "core_efficiency_baseline") or CORE_EFFICIENCY_BASELINE
    best = _number(payload, "core_efficiency_best")
    before, after = _deepest_pair(payload)
    gains = _level_gains(before, after)
    preserved = _bool(payload, "core_solves_preserved")
    moved = best is not None and best > CORE_EFFICIENCY_BASELINE and preserved and any(
        gain > 0 for gain in gains.values()
    )
    return {
        "status": "clean_reinduction_efficiency_improved" if moved else "clean_reinduction_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "core_efficiency_baseline": baseline,
        "core_efficiency_best": best,
        "core_efficiency_delta": None if best is None else round(best - baseline, 10),
        "core_solves_preserved": preserved,
        "deepest_level_before": before,
        "deepest_level_after": after,
        "deepest_level_gains": gains,
        "moved": bool(moved),
        "diagnosis": {},
    }


def _deepest_level_gains_per_core_game(a1: Mapping[str, Any]) -> JsonDict:
    if a1.get("headline_numbers_aggregated") is not True:
        return {
            "status": "no_clean_score_lever_evidence",
            "headline_numbers_aggregated": False,
            "clean_before_after": {},
            "gains": {},
            "any_core_game_deeper_clean": False,
        }
    before = a1.get("deepest_level_before") if isinstance(a1.get("deepest_level_before"), Mapping) else {}
    after = a1.get("deepest_level_after") if isinstance(a1.get("deepest_level_after"), Mapping) else {}
    gains = a1.get("deepest_level_gains") if isinstance(a1.get("deepest_level_gains"), Mapping) else {}
    return {
        "status": "clean_score_lever_evidence",
        "headline_numbers_aggregated": True,
        "clean_before_after": {"baseline": dict(before), "best": dict(after)},
        "gains": dict(gains),
        "any_core_game_deeper_clean": any(
            isinstance(gain, int) and gain > 0 for gain in gains.values()
        ),
    }


def _a2_energy_routing(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "generalized": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_energy_routed": None,
            "core_efficiency_delta": None,
            "energy_separation_auroc": None,
            "verifier_is_oracle": None,
            "reason": "headline numbers quarantined" if row.get("skipped") else "missing_clean_artifact",
        }
    baseline = _number(payload, "core_efficiency_baseline") or CORE_EFFICIENCY_BASELINE
    routed = _number(payload, "core_efficiency_energy_routed")
    auroc = _number(payload, "energy_separation_auroc")
    generalized = (
        routed is not None
        and routed > CORE_EFFICIENCY_BASELINE
        and _bool(payload, "core_solves_preserved")
    )
    return {
        "status": "clean_energy_routing_generalized" if generalized else "clean_energy_routing_null",
        "generalized": bool(generalized),
        "core_efficiency_baseline": baseline,
        "core_efficiency_energy_routed": routed,
        "core_efficiency_delta": None if routed is None else round(routed - baseline, 10),
        "energy_separation_auroc": auroc,
        "verifier_is_oracle": base.bool_metric(payload, "verifier_is_oracle"),
        "reason": "clean_measurement",
    }


def _a3_levelup(payload: JsonDict | None, row: Mapping[str, Any], registry: Mapping[str, Any]) -> JsonDict:
    registry_current = int(registry.get("reproducible_total_levels") or 0)
    if row.get("acceptance_gate_failed"):
        update = _mapping(payload, "registry_update")
        return {
            "status": "failed_acceptance_gate",
            "level_up_banked": False,
            "target_game": base.str_metric(payload, "target_game") if payload else "",
            "target_level": _int_or_none(payload, "target_level") if payload else None,
            "banked_levels": 0,
            "prior_total": _int_or_none(update, "prior_total_declared"),
            "current_total": registry_current,
            "delta": 0,
        }
    if payload is None:
        return {
            "status": "missing_or_excluded",
            "level_up_banked": False,
            "target_game": "",
            "target_level": None,
            "banked_levels": 0,
            "prior_total": None,
            "current_total": registry_current,
            "delta": 0,
        }
    update = _mapping(payload, "registry_update")
    prior = _int_or_none(update, "prior_total_declared")
    fallback_current = _int_or_none(update, "new_total_declared") or 0
    current = registry_current or fallback_current
    banked_levels = _int_or_none(update, "banked_levels") or 0
    gate = _mapping(payload, "reproduction_gate")
    level_up_banked = (
        _bool(payload, "offline_reproduced")
        and _bool(gate, "reproduced")
        and banked_levels > 0
    )
    delta = max(0, current - prior) if prior is not None else 0
    return {
        "status": "level_up_banked" if level_up_banked else "no_clean_level_growth",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "level_up_banked": bool(level_up_banked),
        "target_game": base.str_metric(payload, "target_game"),
        "target_level": _int_or_none(payload, "target_level"),
        "banked_levels": banked_levels,
        "prior_total": prior,
        "current_total": current,
        "delta": delta if level_up_banked else 0,
    }


def _a4_integration(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if row.get("acceptance_gate_failed"):
        return {
            "status": "failed_acceptance_gate",
            "submitted_config_improved": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_integrated": None,
            "core_efficiency_delta": None,
            "core_solves_preserved": False,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
        }
    if payload is None:
        return {
            "status": _payload_status(row),
            "submitted_config_improved": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_integrated": None,
            "core_efficiency_delta": None,
            "core_solves_preserved": False,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
        }
    gate = _mapping(payload, "gate_result")
    current = _mapping(gate, "current")
    baseline = _number(payload, "core_efficiency_baseline") or CORE_EFFICIENCY_BASELINE
    integrated = _number(payload, "core_efficiency_integrated") or _number(current, "core_efficiency")
    preserved = _bool(payload, "core_solves_preserved")
    improved = integrated is not None and integrated > CORE_EFFICIENCY_BASELINE and preserved
    return {
        "status": "clean_integrated_core_efficiency_improved" if improved else "clean_integrated_null",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "submitted_config_improved": bool(improved),
        "core_efficiency_baseline": baseline,
        "core_efficiency_integrated": integrated,
        "core_efficiency_delta": None if integrated is None else round(integrated - baseline, 10),
        "core_solves_preserved": preserved,
        "ready_for_operator_submit": _bool(payload, "ready_for_operator_submit"),
        "operator_submission_performed": _bool(payload, "operator_submission_performed"),
    }


def _a5_primitive_transfer(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "representation_generalized": False,
            "new_levels_banked": 0,
            "offline_reproduced_new_level": False,
            "transfer_deepest_level_per_game": {},
            "transfer_games": [],
            "primitive_persisted": {},
        }
    representation = _mapping(payload, "representation_transfer")
    new_levels = _int_or_none(payload, "new_levels_banked") or 0
    generalized = bool(representation) and all(value is True for value in representation.values())
    if generalized and new_levels == 0:
        status = "representation_generalized_no_reproducible_level_bank"
    elif generalized:
        status = "representation_generalized_and_level_banked"
    else:
        status = "transfer_null"
    return {
        "status": status,
        "representation_generalized": generalized,
        "new_levels_banked": new_levels,
        "offline_reproduced_new_level": _bool(payload, "offline_reproduced") and new_levels > 0,
        "transfer_deepest_level_per_game": dict(_mapping(payload, "transfer_deepest_level_per_game")),
        "transfer_games": _list(payload, "transfer_games"),
        "primitive_persisted": dict(_mapping(payload, "primitive_persisted")),
    }


def _reproducible_total_levels_delta(a3: Mapping[str, Any]) -> JsonDict:
    prior = a3.get("prior_total")
    current = a3.get("current_total")
    delta = a3.get("delta")
    return {
        "prior_total": prior if isinstance(prior, int) else None,
        "current_total": current if isinstance(current, int) else 0,
        "delta": delta if isinstance(delta, int) else 0,
        "banked_levels": int(a3.get("banked_levels") or 0),
        "source": "A3_levelup_attempt+ops/arc_solve_registry.yaml",
        "capability_grew": bool(a3.get("level_up_banked") and isinstance(delta, int) and delta > 0),
    }


def _operator_resubmission_verdict(
    *, ready: bool, score_gate_failed: bool
) -> JsonDict:
    if score_gate_failed:
        reason = "failed_acceptance_gate"
    elif ready:
        reason = "clean_integrated_core_efficiency_improvement"
    else:
        reason = "no_clean_integrated_core_efficiency_improvement"
    return {
        "resubmission_warranted": ready,
        "reason": reason,
        "operator_only": True,
    }


def _preconditions_checked(
    root: Path,
    provenance: list[JsonDict],
    registry: Mapping[str, Any],
) -> JsonDict:
    rows = _provenance_by_key(provenance)
    upstreams = []
    for key, path in _selected_paths(root).items():
        row = rows.get(key)
        upstreams.append(
            {
                "artifact_key": key,
                "experiment_id": DEFAULT_UPSTREAMS[key].experiment_id,
                "path": str(DEFAULT_UPSTREAMS[key].path),
                "exists": path.exists(),
                "summarize_exit_code": row.get("summarize_exit_code") if row else None,
                "skipped": row.get("skipped") if row else None,
                "skip_reason": row.get("skip_reason") if row else "missing",
            }
        )
    return {
        "upstream_artifacts": upstreams,
        "summarize_artifact_required": "scripts/summarize_artifact.py",
        "reading_results_discipline": True,
        "registry": {
            "path": str(REGISTRY_RELATIVE_PATH),
            "present": bool(registry.get("registry_present")),
            "reproducible_total_levels": int(registry.get("reproducible_total_levels") or 0),
        },
        "leaderboard_submission": False,
    }


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


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "efficiency_moved": artifact.get("efficiency_moved"),
        "deepest_level_gains_per_core_game": artifact.get("deepest_level_gains_per_core_game"),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "flagged_artifacts_handled": artifact.get("flagged_artifacts_handled"),
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "scorecard": artifact.get("scorecard"),
        "energy_routing_generalization": artifact.get("energy_routing_generalization"),
        "primitive_transfer_generalization": artifact.get("primitive_transfer_generalization"),
        "operator_resubmission_verdict": artifact.get("operator_resubmission_verdict"),
        "upstream_sha256_set": sorted(
            str(row.get("sha256", "")) for row in artifact.get("upstream_provenance", [])
        ),
    }


def checksum_from_artifact(artifact: Mapping[str, Any]) -> str:
    blob = json.dumps(_checksum_payload(artifact), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _honest_verdict(efficiency_moved: bool, a1: Mapping[str, Any], integration: Mapping[str, Any]) -> str:
    if not efficiency_moved:
        return "complete: reinduction_null_efficiency_unmoved_barrier_refined"
    value = integration.get("core_efficiency_integrated")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        value = a1.get("core_efficiency_best")
    return f"success: reinduction_core_efficiency_{float(value):.4f}_above_2.0074"


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
    raw_artifacts, provenance, handled = _read_inputs(
        root_path,
        live_flag_runner,
        summarize_runner,
    )
    rows = _provenance_by_key(provenance)
    clean = {
        key: _clean_payload(raw_artifacts, rows, key)
        for key in DEFAULT_UPSTREAMS
    }
    registry = load_registry_totals(root_path)

    a1_payload = clean["A1_goal_reinduction"]
    a1 = _a1_reinduction(a1_payload, rows.get("A1_goal_reinduction", {}))
    a2 = _a2_energy_routing(clean["A2_energy_routing"], rows.get("A2_energy_routing", {}))
    a3 = _a3_levelup(
        raw_artifacts.get("A3_levelup_attempt")
        if isinstance(raw_artifacts.get("A3_levelup_attempt"), dict)
        else None,
        rows.get("A3_levelup_attempt", {}),
        registry,
    )
    a4 = _a4_integration(
        raw_artifacts.get("A4_integration")
        if isinstance(raw_artifacts.get("A4_integration"), dict)
        and rows.get("A4_integration", {}).get("acceptance_gate_failed")
        else clean["A4_integration"],
        rows.get("A4_integration", {}),
    )
    a5 = _a5_primitive_transfer(clean["A5_primitive_transfer"], rows.get("A5_primitive_transfer", {}))
    deepest_gains = _deepest_level_gains_per_core_game(a1)
    score_gate_failed = bool(
        rows.get("A1_goal_reinduction", {}).get("acceptance_gate_failed")
        or rows.get("A4_integration", {}).get("acceptance_gate_failed")
    )
    efficiency_moved = bool(
        not score_gate_failed
        and (a1.get("moved") is True or a4.get("submitted_config_improved") is True)
    )
    ready = bool(
        efficiency_moved
        and a4.get("submitted_config_improved") is True
        and a4.get("ready_for_operator_submit") is True
        and a4.get("operator_submission_performed") is False
    )
    scorecard = {
        "a1_reinduction": a1,
        "a2_energy_routing": a2,
        "a3_levelup": a3,
        "a4_integration": a4,
        "a5_primitive_transfer": a5,
        "baseline_core_efficiency": CORE_EFFICIENCY_BASELINE,
    }
    duration_s = round(float((time.time() if now_s is None else now_s) - start), 6)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(efficiency_moved, a1, a4),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "efficiency_moved": efficiency_moved,
        "deepest_level_gains_per_core_game": deepest_gains,
        "reproducible_total_levels_delta": _reproducible_total_levels_delta(a3),
        "flagged_artifacts_handled": handled,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "ready_for_operator_submit": ready,
        "preconditions_checked": _preconditions_checked(root_path, provenance, registry),
        "scorecard": scorecard,
        "energy_routing_generalization": a2,
        "primitive_transfer_generalization": a5,
        "operator_resubmission_verdict": _operator_resubmission_verdict(
            ready=ready,
            score_gate_failed=score_gate_failed,
        ),
        "upstream_provenance": provenance,
        "leaderboard_submission": False,
        "operator_submission_performed": False,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = checksum_from_artifact(artifact)
    validate_artifact(artifact)
    return artifact


def _is_sha256_prefixed(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    return base.is_sha256(value.removeprefix("sha256:"))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("efficiency_moved"), bool):
        raise ValueError("efficiency_moved must be bool")
    for field in (
        "deepest_level_gains_per_core_game",
        "reproducible_total_levels_delta",
        "flagged_artifacts_handled",
        "preconditions_checked",
        "scorecard",
        "energy_routing_generalization",
        "primitive_transfer_generalization",
        "operator_resubmission_verdict",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be object")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be list")
    if not isinstance(artifact.get("ready_for_operator_submit"), bool):
        raise ValueError("ready_for_operator_submit must be bool")
    if not isinstance(artifact.get("duration_s"), (int, float)) or isinstance(
        artifact.get("duration_s"), bool
    ):
        raise ValueError("duration_s must be numeric")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed mismatch")
    if artifact.get("leaderboard_submission") is not False:
        raise ValueError("leaderboard_submission must remain false")
    if "gated_on" in artifact:
        raise ValueError("gated_on is forbidden")
    if not isinstance(artifact.get("upstream_provenance"), list):
        raise ValueError("upstream_provenance must be list")
    for row in artifact.get("upstream_provenance", []):
        if not isinstance(row, Mapping):
            raise ValueError("upstream provenance row must be object")
        if row.get("skipped") and row.get("fields_imported"):
            raise ValueError("skipped upstreams must import no fields")
        sha = row.get("sha256")
        if not isinstance(sha, str) or not base.is_sha256(sha):
            raise ValueError("invalid sha256 in upstream provenance")
    checksum = artifact.get("reproducibility_checksum")
    if not _is_sha256_prefixed(checksum):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if checksum != checksum_from_artifact(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("ready_for_operator_submit") is True and artifact.get("efficiency_moved") is not True:
        raise ValueError("ready_for_operator_submit requires efficiency_moved")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> Path:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )
    rel_output = Path(output_path)
    artifact["result_path"] = str(rel_output)
    validate_artifact(artifact)
    out_path = root_path / rel_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run(
    root: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
    live_flag_runner: LiveFlagRunner = base.run_live_flags,
    summarize_runner: SummarizeRunner = base.run_summarize_artifact,
) -> JsonDict:
    if write:
        path = write_artifact(
            root,
            output_path=OUTPUT_REL_PATH,
            started_s=started_s,
            now_s=now_s,
            live_flag_runner=live_flag_runner,
            summarize_runner=summarize_runner,
        )
        return json.loads(path.read_text(encoding="utf-8"))
    return build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        live_flag_runner=live_flag_runner,
        summarize_runner=summarize_runner,
    )


def main() -> int:  # pragma: no cover - thin CLI wrapper
    path = write_artifact(REPO_ROOT)
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by requested command
    raise SystemExit(main())
