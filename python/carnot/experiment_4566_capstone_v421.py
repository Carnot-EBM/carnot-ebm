"""Experiment 4566: .421 capstone aggregation.

Spec refs: REQ-CAPSTONE-4566, SCENARIO-CAPSTONE-4566,
SCENARIO-CAPSTONE-4566-FIELD-PRINCIPLES.
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
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot.reporting import capstone_v400_4335 as base  # noqa: E402
from scripts import summarize_artifact as summary_reader  # noqa: E402


JsonDict = dict[str, Any]
LiveFlagRunner = Callable[[Path], list[dict[str, Any]]]
SummarizeRunner = Callable[[Path, Path], int]

RESULT_RELATIVE_PATH = "results/experiment_4566_capstone_v421.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4566
RANDOM_SEED = 4566
SCHEMA = "carnot.capstone_v421_4566.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORE_EFFICIENCY_BASELINE = 2.0074
GENERIC_TRANSFER_BASELINE = 0.04
CHANCE_AUROC = 0.5
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "shipped:", "shipped_")
SPEC_REFS = [
    "REQ-CAPSTONE-4566",
    "SCENARIO-CAPSTONE-4566",
    "SCENARIO-CAPSTONE-4566-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: verifier_router_generic_transfer_<n>_above_0.04 OR "
            "complete: verifier_router_null_reinduction_retired_or_refined."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "generic_transfer_moved": {
        "principle": (
            "the bottom line -- did operationalizing the verifier (A1) raise "
            "generic_transfer_rate_over_variants STRICTLY above 0.04 with CI (the "
            "leaderboard-honest first-contact signal)."
        )
    },
    "verifier_router_value_added": {
        "principle": (
            "did the oracle-distinct verifier (verifier_is_oracle:false) add live value as "
            "a router (A1) -- the verifier-moat operationalized, the milestone's central claim."
        )
    },
    "executable_proposer_positive_control": {
        "principle": (
            "did the Family-B executable proposer (A2) PASS its positive control (vs .420's "
            "count=0) -- the precondition for any efficiency claim; informs the "
            "retire_if_same_verdict decision."
        )
    },
    "efficiency_moved": {
        "principle": (
            "did core_efficiency rise STRICTLY above 2.0074 (A2/A5) -- the thing "
            ".418/.419/.420 did not achieve."
        )
    },
    "reinduction_retired": {
        "principle": (
            "True if A2 reproduced the same no-deeper-level/control-failed verdict -> the "
            "re-induction lever is retired for the sprint (retire_if_same_verdict)."
        )
    },
    "reproducible_total_levels_delta": {
        "principle": "did solve CAPABILITY grow this milestone (A3 + A4 level-ups)."
    },
    "generic_transfer_rate_over_variants": {
        "principle": (
            "the honest held-out-proxy signal (B1) reported WITH a CI alongside the bank count "
            "-- the co-headline metric."
        )
    },
    "cross_game_discrimination_above_chance": {
        "principle": (
            "carried: the A2 .420 LOO-AUROC 0.674 win now operationalized -- the "
            "verifier-moat status (north-star section 5)."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial artifact excluded AND any null-delta-carve-out / "
            "positive-control-failed artifact handled (B2) -- fabrication-gate + "
            "false-negative-risk compliance."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream measurement (the audit trail)."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only if the integrated config is a CORE-preserved improvement on a real metric "
            "worth a 1/day slot (gate: beat 13 levels on the hidden eval); never submits."
        )
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "A1_verifier_router": Upstream(4556, Path("results/experiment_4556_verifier_router_generic_transfer.json")),
    "A2_executable_proposer": Upstream(4557, Path("results/experiment_4557_executable_world_model_proposer.json")),
    "A3_levelup_attempt": Upstream(4558, Path("results/experiment_4558_levelup_attempt.json")),
    "A4_hidden_state_probe": Upstream(4559, Path("results/experiment_4559_hidden_field_state_probe.json")),
    "A5_integration": Upstream(4560, Path("results/experiment_4560_integration_8game_gate.json")),
    "A6_transfer": Upstream(4561, Path("results/experiment_4561_primitive_persist_transfer.json")),
    "B1_generic_transfer_coheadline": Upstream(4562, Path("results/experiment_4562_generic_transfer_coheadline.json")),
    "P0_prior_420_transition": Upstream(4555, Path("results/experiment_4555_archive_420_activate_421.json")),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "A1_verifier_router": [
        "verifier_is_oracle",
        "generic_transfer_rate_with_verifier",
        "generic_transfer_rate_baseline",
        "generic_transfer_delta",
        "generic_transfer_ci",
        "solve_rate_preserved",
        "random_router_control_passed",
    ],
    "A2_executable_proposer": [
        "positive_control_passed",
        "core_efficiency_baseline",
        "core_efficiency_best",
        "llm_proposer_value",
        "core_solves_preserved",
    ],
    "A3_levelup_attempt": [
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "target_level",
        "registry_update",
    ],
    "A4_hidden_state_probe": ["offline_reproduced", "reproduced_levels", "registry_update"],
    "A5_integration": [
        "core_efficiency_integrated",
        "generic_transfer_rate_integrated",
        "core_solves_preserved",
        "levers_integrated",
        "ready_for_operator_submit",
    ],
    "A6_transfer": [
        "primitive_persisted",
        "transfer_games",
        "transfer_value_per_game",
        "new_levels_banked",
    ],
    "B1_generic_transfer_coheadline": [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "variant_attempts_count",
        "variant_solved_count",
    ],
    "P0_prior_420_transition": ["close_state_420.a2_cross_game_discrimination"],
}


def _selected_paths(root: Path) -> dict[str, Path]:
    return {key: root / upstream.path for key, upstream in DEFAULT_UPSTREAMS.items()}


def _number(payload: Mapping[str, Any] | None, field: str) -> float | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int_or_none(payload: Mapping[str, Any] | None, field: str) -> int | None:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _bool(payload: Mapping[str, Any] | None, field: str) -> bool:
    return base.bool_metric(payload, field) is True


def _mapping(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _list_value(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def _gate_value_failed(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, Mapping):
        decision_keys = ("pass", "passed", "ok", "verdict_pass", "reproduced")
        decisions = [value[key] for key in decision_keys if isinstance(value.get(key), bool)]
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


def _flags_false_negative_risk_open(flags: list[dict[str, Any]]) -> bool:
    return any(
        flag.get("kind") == "FALSE_NEGATIVE_RISK"
        and "false_negative_risk_open" in str(flag.get("detail", ""))
        for flag in flags
    )


def _payload_false_negative_risk_open(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("positive_control_passed") is False:
        return True
    verdict = str(payload.get("honest_verdict", ""))
    nullish = any(token in verdict for token in ("null", "no_value", "no_deeper", "failed"))
    return payload.get("false_negative_risk_checked") is False and nullish


def _fields_for_payload(key: str, skipped: bool) -> list[str]:
    return [] if skipped else list(IMPORTED_FIELDS[key])


def _skip_reason(
    *,
    stamped: bool,
    critical: bool,
    parse_error: str,
    acceptance_gate_failed: bool,
    diagnosis_context_read: bool,
    false_negative_risk_open: bool,
) -> str:
    if acceptance_gate_failed:
        return "failed_acceptance_gate"
    if false_negative_risk_open:
        return "false_negative_risk_open"
    if diagnosis_context_read:
        return "null_delta_carve_out_diagnosis_only"
    if stamped or critical or parse_error:
        return base._exclusion_reason(stamped, critical, parse_error)  # noqa: SLF001
    return ""


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], JsonDict]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    handled: JsonDict = {
        "excluded": [],
        "null_delta_carve_out_diagnosis_read": [],
        "positive_control_failed_or_false_negative_risk_open": [],
    }
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
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        diagnosis_context = (
            summary_reader.readable_diagnosis_context(payload, live_flags)
            if payload is not None
            else None
        )
        diagnosis_context_read = diagnosis_context is not None
        stamped = payload.get("flagged_adversarial") is True if payload is not None else False
        critical = base.live_has_critical(live_flags)
        gate_failed = _acceptance_gate_failed(payload)
        fnr_open = _flags_false_negative_risk_open(live_flags) or _payload_false_negative_risk_open(payload)
        skipped = bool(
            payload is None
            or parse_error
            or stamped
            or critical
            or gate_failed
            or fnr_open
        )
        reason = _skip_reason(
            stamped=stamped,
            critical=critical,
            parse_error=parse_error,
            acceptance_gate_failed=gate_failed,
            diagnosis_context_read=diagnosis_context_read,
            false_negative_risk_open=fnr_open,
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
            "false_negative_risk_open": fnr_open,
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
        if stamped or critical or parse_error:
            handled["excluded"].append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "reason": reason,
                }
            )
        if diagnosis_context_read:
            handled["null_delta_carve_out_diagnosis_read"].append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "corrigendum": diagnosis_context["corrigendum"],
                    "diagnosis_context_fields_read": [
                        field
                        for field in ("barrier_diagnosis", "levers_tried", "barrier_refinement")
                        if field in diagnosis_context
                    ],
                }
            )
        if fnr_open:
            handled["positive_control_failed_or_false_negative_risk_open"].append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "positive_control_passed": payload.get("positive_control_passed")
                    if payload is not None
                    else None,
                    "false_negative_risk_checked": payload.get("false_negative_risk_checked")
                    if payload is not None
                    else None,
                    "reason": "false_negative_risk_open",
                }
            )
    return raw_artifacts, provenance, handled


def _provenance_by_key(provenance: list[JsonDict]) -> dict[str, JsonDict]:
    return {str(row["artifact_key"]): row for row in provenance}


def _clean_payload(
    raw_artifacts: Mapping[str, Any],
    provenance: Mapping[str, JsonDict],
    key: str,
) -> JsonDict | None:
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
    if row.get("false_negative_risk_open"):
        return "false_negative_risk_open"
    if row.get("diagnosis_context_read"):
        return "diagnosis_only_null_delta_carve_out"
    if row.get("skipped"):
        return "excluded_flagged_adversarial_or_live_critical"
    return "missing_or_excluded"


def _verifier_router_value(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "value_added": False,
            "verifier_is_oracle": None,
            "generic_transfer_rate_with_verifier": None,
            "generic_transfer_rate_baseline": GENERIC_TRANSFER_BASELINE,
            "generic_transfer_delta": None,
            "generic_transfer_ci": [None, None],
            "solve_rate_preserved": False,
            "random_router_control_passed": False,
        }
    rate = _number(payload, "generic_transfer_rate_with_verifier")
    baseline = _number(payload, "generic_transfer_rate_baseline") or GENERIC_TRANSFER_BASELINE
    delta = _number(payload, "generic_transfer_delta")
    ci = _list_value(payload, "generic_transfer_ci")
    ci_pair = ci if len(ci) == 2 else [None, None]
    oracle = base.bool_metric(payload, "verifier_is_oracle")
    ci_lower = ci_pair[0]
    value_added = bool(
        rate is not None
        and delta is not None
        and rate > GENERIC_TRANSFER_BASELINE
        and delta > 0.0
        and isinstance(ci_lower, (int, float))
        and not isinstance(ci_lower, bool)
        and float(ci_lower) > 0.0
        and oracle is False
        and _bool(payload, "solve_rate_preserved")
        and _bool(payload, "random_router_control_passed")
        and _bool(payload, "false_negative_risk_checked")
    )
    return {
        "status": "clean_verifier_router_value_added" if value_added else "clean_verifier_router_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "value_added": value_added,
        "verifier_is_oracle": oracle,
        "generic_transfer_rate_with_verifier": rate,
        "generic_transfer_rate_baseline": baseline,
        "generic_transfer_delta": delta,
        "generic_transfer_ci": ci_pair,
        "solve_rate_preserved": _bool(payload, "solve_rate_preserved"),
        "random_router_control_passed": _bool(payload, "random_router_control_passed"),
        "false_negative_risk_checked": _bool(payload, "false_negative_risk_checked"),
        "chosen_submitted_config": base.str_metric(payload, "chosen_submitted_config"),
    }


def _b1_metric(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "reproducible_total_levels": None,
            "generic_transfer_rate_over_variants": 0.0,
            "generic_transfer_ci": [None, None],
            "variant_attempts_count": 0,
            "variant_solved_count": 0,
        }
    ci = _list_value(payload, "generic_transfer_ci")
    return {
        "status": "clean_generic_transfer_coheadline",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "reproducible_total_levels": _int_or_none(payload, "reproducible_total_levels"),
        "generic_transfer_rate_over_variants": _number(payload, "generic_transfer_rate_over_variants")
        or 0.0,
        "generic_transfer_ci": ci if len(ci) == 2 else [None, None],
        "variant_attempts_count": _int_or_none(payload, "variant_attempts_count") or 0,
        "variant_solved_count": _int_or_none(payload, "variant_solved_count") or 0,
        "metric_wired_into_capstone": dict(_mapping(payload, "metric_wired_into_capstone")),
    }


def _generic_transfer_moved(router: Mapping[str, Any], b1: Mapping[str, Any]) -> JsonDict:
    moved = bool(router.get("value_added") is True)
    return {
        "moved": moved,
        "reason": (
            "clean_verifier_router_generic_transfer_above_baseline"
            if moved
            else "no_clean_verifier_router_value_added_above_0.04"
        ),
        "baseline": GENERIC_TRANSFER_BASELINE,
        "coheadline_rate": float(b1.get("generic_transfer_rate_over_variants") or 0.0),
        "generic_transfer_ci": list(b1.get("generic_transfer_ci") or [None, None]),
        "verifier_router": {
            "headline_numbers_aggregated": router.get("headline_numbers_aggregated") is True,
            "rate_with_verifier": router.get("generic_transfer_rate_with_verifier"),
            "delta": router.get("generic_transfer_delta"),
            "delta_ci": router.get("generic_transfer_ci"),
        },
    }


def _executable_proposer_positive_control(
    payload: JsonDict | None,
    row: Mapping[str, Any],
) -> JsonDict:
    if payload is not None and row.get("skipped"):
        diagnosis = row.get("diagnosis_context") if isinstance(row.get("diagnosis_context"), Mapping) else {}
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "positive_control_passed": _bool(payload, "positive_control_passed"),
            "false_negative_risk_open": row.get("false_negative_risk_open") is True,
            "false_negative_risk_checked": _bool(payload, "false_negative_risk_checked"),
            "efficiency_claim_valid": False,
            "core_efficiency_baseline": _number(payload, "core_efficiency_baseline")
            or CORE_EFFICIENCY_BASELINE,
            "core_efficiency_best": None,
            "barrier_refinement": base.str_metric(payload, "barrier_refinement")
            or str(diagnosis.get("barrier_refinement", "")),
        }
    if payload is None:
        diagnosis = row.get("diagnosis_context") if isinstance(row.get("diagnosis_context"), Mapping) else {}
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "positive_control_passed": False,
            "false_negative_risk_open": row.get("false_negative_risk_open") is True,
            "false_negative_risk_checked": False,
            "efficiency_claim_valid": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_best": None,
            "barrier_refinement": diagnosis.get("barrier_refinement", ""),
        }
    best = _number(payload, "core_efficiency_best")
    positive = _bool(payload, "positive_control_passed")
    valid = bool(
        positive
        and best is not None
        and best > CORE_EFFICIENCY_BASELINE
        and _bool(payload, "core_solves_preserved")
        and _bool(payload, "offline_reproduced")
        and base.bool_metric(payload, "verifier_is_oracle") is False
    )
    return {
        "status": "clean_executable_proposer_positive_control_passed" if positive else "clean_positive_control_failed",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "positive_control_passed": positive,
        "false_negative_risk_open": False,
        "false_negative_risk_checked": _bool(payload, "false_negative_risk_checked"),
        "efficiency_claim_valid": valid,
        "core_efficiency_baseline": _number(payload, "core_efficiency_baseline")
        or CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "core_efficiency_delta": None
        if best is None
        else round(best - CORE_EFFICIENCY_BASELINE, 10),
        "core_solves_preserved": _bool(payload, "core_solves_preserved"),
        "offline_reproduced": _bool(payload, "offline_reproduced"),
        "barrier_refinement": base.str_metric(payload, "barrier_refinement"),
    }


def _integration(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "integrated_metric_improved": False,
            "core_efficiency_integrated": None,
            "generic_transfer_rate_integrated": None,
            "core_solves_preserved": False,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
            "levers_integrated": [],
        }
    core = _number(payload, "core_efficiency_integrated")
    rate = _number(payload, "generic_transfer_rate_integrated")
    preserved = _bool(payload, "core_solves_preserved")
    improved = bool(
        preserved
        and (
            (core is not None and core > CORE_EFFICIENCY_BASELINE)
            or (rate is not None and rate > GENERIC_TRANSFER_BASELINE)
        )
    )
    return {
        "status": "clean_integrated_metric_improved" if improved else "clean_integrated_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "integrated_metric_improved": improved,
        "core_efficiency_integrated": core,
        "generic_transfer_rate_integrated": rate,
        "core_solves_preserved": preserved,
        "levers_integrated": _list_value(payload, "levers_integrated"),
        "heldout_solve_rate": _number(payload, "heldout_solve_rate"),
        "ready_for_operator_submit": _bool(payload, "ready_for_operator_submit"),
        "operator_submission_performed": _bool(payload, "operator_submission_performed"),
    }


def _banked_levels(payload: JsonDict | None) -> int:
    if payload is None:
        return 0
    update = _mapping(payload, "registry_update")
    return (
        _int_or_none(update, "banked_levels")
        or _int_or_none(payload, "new_levels_banked")
        or _int_or_none(payload, "reproduced_levels")
        or 0
    )


def _level_context(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    banked = _banked_levels(payload) if payload is not None and not row.get("skipped") else 0
    return {
        "status": "level_banked" if banked > 0 else _payload_status(row) if payload is None else "no_new_level_banked",
        "banked_levels": banked,
        "offline_reproduced": _bool(payload, "offline_reproduced") if payload is not None else False,
        "target_game": base.str_metric(payload, "target_game") if payload is not None else "",
        "target_level": _int_or_none(payload, "target_level") if payload is not None else None,
    }


def _a6_transfer(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "primitive_persisted": {},
            "transfer_games": [],
            "transfer_value_per_game": {},
            "new_levels_banked": 0,
            "any_transfer_value_added": False,
        }
    values = _mapping(payload, "transfer_value_per_game")
    any_value = any(
        isinstance(value, Mapping) and value.get("value_added") is True for value in values.values()
    )
    new_levels = _int_or_none(payload, "new_levels_banked") or 0
    return {
        "status": "transfer_value_added" if any_value else "transfer_null",
        "primitive_persisted": dict(_mapping(payload, "primitive_persisted")),
        "transfer_games": _list_value(payload, "transfer_games"),
        "transfer_value_per_game": dict(values),
        "new_levels_banked": new_levels,
        "any_transfer_value_added": any_value,
        "offline_reproduced_new_level": _bool(payload, "offline_reproduced") and new_levels > 0,
        "registry_updated": _bool(payload, "registry_updated"),
    }


def _prior_total(prior: JsonDict | None, a3: Mapping[str, Any], registry_current: int) -> int:
    close = _mapping(prior, "close_state_420") if prior is not None else {}
    carried = _int_or_none(close, "reproducible_total_levels")
    if carried is not None:
        return carried
    update_prior = a3.get("prior_total")
    if isinstance(update_prior, int):
        return update_prior
    return registry_current


def _level_delta(
    prior_payload: JsonDict | None,
    a3_payload: JsonDict | None,
    a4_payload: JsonDict | None,
    a6: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> JsonDict:
    registry_current = int(registry.get("reproducible_total_levels") or 0)
    a3_update = _mapping(a3_payload, "registry_update")
    a3_context = {"prior_total": _int_or_none(a3_update, "prior_total_declared")}
    prior = _prior_total(prior_payload, a3_context, registry_current)
    a3_banked = _banked_levels(a3_payload)
    a4_banked = _banked_levels(a4_payload)
    a6_banked = int(a6.get("new_levels_banked") or 0)
    current = registry_current or prior + a3_banked + a4_banked + a6_banked
    delta = max(0, current - prior)
    return {
        "prior_total": prior,
        "current_total": current,
        "delta": delta,
        "a3_new_levels_banked": a3_banked,
        "a4_new_levels_banked": a4_banked,
        "a6_new_levels_banked": a6_banked,
        "capability_grew": delta > 0,
        "source": "A3_levelup_attempt+A4_hidden_state_probe+A6_transfer+ops/arc_solve_registry.yaml",
    }


def _cross_game_discrimination(prior: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    close = _mapping(prior, "close_state_420") if prior is not None else {}
    a2 = _mapping(close, "a2_cross_game_discrimination")
    if not a2:
        return {
            "status": _payload_status(row),
            "above_chance": False,
            "loo_auroc_mean": None,
            "loo_auroc_display": None,
            "loo_auroc_ci": [None, None],
            "chance_auroc": CHANCE_AUROC,
            "verifier_is_oracle": None,
        }
    return {
        "status": str(a2.get("status", "carried_prior_cross_game_discrimination")),
        "above_chance": a2.get("above_chance") is True,
        "loo_auroc_mean": _number(a2, "loo_auroc_mean"),
        "loo_auroc_display": _number(a2, "loo_auroc_display"),
        "loo_auroc_ci": _list_value(a2, "loo_auroc_ci"),
        "ci_excludes_chance": a2.get("ci_excludes_chance") is True,
        "chance_auroc": CHANCE_AUROC,
        "verifier_is_oracle": base.bool_metric(a2, "verifier_is_oracle"),
        "positive_control_passed": a2.get("positive_control_passed") is True,
        "source": "results/experiment_4555_archive_420_activate_421.json",
    }


def _operator_resubmission_verdict(*, ready: bool, score_gate_failed: bool) -> JsonDict:
    if score_gate_failed:
        reason = "failed_acceptance_gate"
    elif ready:
        reason = "clean_integrated_metric_improvement"
    else:
        reason = "no_clean_integrated_metric_improvement"
    return {
        "resubmission_warranted": ready,
        "reason": reason,
        "operator_only": True,
        "hidden_eval_gate": "beat_13_levels",
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


def _preconditions_checked(root: Path, provenance: list[JsonDict], registry: Mapping[str, Any]) -> JsonDict:
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


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "generic_transfer_moved": artifact.get("generic_transfer_moved"),
        "verifier_router_value_added": artifact.get("verifier_router_value_added"),
        "executable_proposer_positive_control": artifact.get("executable_proposer_positive_control"),
        "efficiency_moved": artifact.get("efficiency_moved"),
        "reinduction_retired": artifact.get("reinduction_retired"),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "generic_transfer_rate_over_variants": artifact.get("generic_transfer_rate_over_variants"),
        "cross_game_discrimination_above_chance": artifact.get("cross_game_discrimination_above_chance"),
        "flagged_artifacts_handled": artifact.get("flagged_artifacts_handled"),
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "scorecard": artifact.get("scorecard"),
        "upstream_sha256_set": sorted(
            str(row.get("sha256", "")) for row in artifact.get("upstream_provenance", [])
        ),
    }


def checksum_from_artifact(artifact: Mapping[str, Any]) -> str:
    blob = json.dumps(_checksum_payload(artifact), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _honest_verdict(generic_moved: Mapping[str, Any]) -> str:
    if generic_moved.get("moved") is True:
        rate = float(generic_moved.get("coheadline_rate") or 0.0)
        return f"success: verifier_router_generic_transfer_{rate:.4f}_above_0.04"
    return "complete: verifier_router_null_reinduction_retired_or_refined"


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
    raw_artifacts, provenance, handled = _read_inputs(root_path, live_flag_runner, summarize_runner)
    rows = _provenance_by_key(provenance)
    clean = {key: _clean_payload(raw_artifacts, rows, key) for key in DEFAULT_UPSTREAMS}
    registry = load_registry_totals(root_path)

    router = _verifier_router_value(clean["A1_verifier_router"], rows.get("A1_verifier_router", {}))
    b1 = _b1_metric(clean["B1_generic_transfer_coheadline"], rows.get("B1_generic_transfer_coheadline", {}))
    generic_moved = _generic_transfer_moved(router, b1)
    a2 = _executable_proposer_positive_control(
        raw_artifacts.get("A2_executable_proposer")
        if isinstance(raw_artifacts.get("A2_executable_proposer"), dict)
        else None,
        rows.get("A2_executable_proposer", {}),
    )
    integration = _integration(clean["A5_integration"], rows.get("A5_integration", {}))
    a3 = _level_context(clean["A3_levelup_attempt"], rows.get("A3_levelup_attempt", {}))
    a4 = _level_context(clean["A4_hidden_state_probe"], rows.get("A4_hidden_state_probe", {}))
    a6 = _a6_transfer(clean["A6_transfer"], rows.get("A6_transfer", {}))
    prior = clean["P0_prior_420_transition"]
    cross_game = _cross_game_discrimination(prior, rows.get("P0_prior_420_transition", {}))

    score_gate_failed = any(row.get("acceptance_gate_failed") for row in rows.values())
    efficiency_moved = bool(
        not score_gate_failed
        and a2.get("efficiency_claim_valid") is True
        and integration.get("integrated_metric_improved") is True
    )
    reinduction_retired = bool(
        a2.get("positive_control_passed") is False
        and a2.get("false_negative_risk_open") is True
    )
    ready = bool(
        not score_gate_failed
        and integration.get("ready_for_operator_submit") is True
        and integration.get("operator_submission_performed") is False
        and (generic_moved.get("moved") is True or efficiency_moved)
    )
    level_delta = _level_delta(
        prior,
        clean["A3_levelup_attempt"],
        clean["A4_hidden_state_probe"],
        a6,
        registry,
    )
    scorecard = {
        "a1_verifier_router": router,
        "a2_executable_proposer": a2,
        "a3_levelup_attempt": a3,
        "a4_hidden_state_probe": a4,
        "a5_integration": integration,
        "a6_transfer": a6,
        "b1_generic_transfer_coheadline": b1,
        "prior_420_cross_game_discrimination": cross_game,
        "baseline_core_efficiency": CORE_EFFICIENCY_BASELINE,
        "baseline_generic_transfer": GENERIC_TRANSFER_BASELINE,
    }
    duration_s = round(float((time.time() if now_s is None else now_s) - start), 6)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(generic_moved),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "generic_transfer_moved": generic_moved,
        "verifier_router_value_added": router,
        "executable_proposer_positive_control": a2,
        "efficiency_moved": efficiency_moved,
        "reinduction_retired": reinduction_retired,
        "reproducible_total_levels_delta": level_delta,
        "generic_transfer_rate_over_variants": float(
            b1.get("generic_transfer_rate_over_variants") or 0.0
        ),
        "generic_transfer_ci": list(b1.get("generic_transfer_ci") or [None, None]),
        "reproducible_total_levels": int(
            b1.get("reproducible_total_levels")
            or registry.get("reproducible_total_levels")
            or 0
        ),
        "cross_game_discrimination_above_chance": cross_game,
        "flagged_artifacts_handled": handled,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "ready_for_operator_submit": ready,
        "preconditions_checked": _preconditions_checked(root_path, provenance, registry),
        "scorecard": scorecard,
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
    for field in (
        "generic_transfer_moved",
        "verifier_router_value_added",
        "executable_proposer_positive_control",
        "reproducible_total_levels_delta",
        "cross_game_discrimination_above_chance",
        "flagged_artifacts_handled",
        "preconditions_checked",
        "scorecard",
        "operator_resubmission_verdict",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be object")
    if not isinstance(artifact.get("efficiency_moved"), bool):
        raise ValueError("efficiency_moved must be bool")
    if not isinstance(artifact.get("reinduction_retired"), bool):
        raise ValueError("reinduction_retired must be bool")
    generic_rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(generic_rate, float) or not 0.0 <= generic_rate <= 1.0:
        raise ValueError("generic_transfer_rate_over_variants must be float in [0,1]")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be list")
    if not isinstance(artifact.get("ready_for_operator_submit"), bool):
        raise ValueError("ready_for_operator_submit must be bool")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed mismatch")
    if artifact.get("leaderboard_submission") is not False:
        raise ValueError("leaderboard_submission must remain false")
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
    if not _is_sha256_prefixed(checksum) or checksum != checksum_from_artifact(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("ready_for_operator_submit") is True:
        moved = artifact.get("generic_transfer_moved")
        if not isinstance(moved, Mapping) or moved.get("moved") is not True:
            raise ValueError("ready_for_operator_submit requires generic_transfer_moved")


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
