"""Experiment 4578: .422 capstone aggregation.

Spec refs: REQ-CAPSTONE-4578, SCENARIO-CAPSTONE-4578,
SCENARIO-CAPSTONE-4578-FIELD-PRINCIPLES.
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

RESULT_RELATIVE_PATH = "results/experiment_4578_capstone_v422.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4578
RANDOM_SEED = 4578
SCHEMA = "carnot.capstone_v422_4578.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
GENERIC_TRANSFER_BASELINE = 0.04
LAST_SUBMITTED_LEVELS = 33
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
SPEC_REFS = [
    "REQ-CAPSTONE-4578",
    "SCENARIO-CAPSTONE-4578",
    "SCENARIO-CAPSTONE-4578-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: clickability_predictor_actions_below_blind_or_"
            "expansion_generic_transfer_above_0.04 OR complete: "
            "action_efficiency_null_gaps_sharpened."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    },
    "action_efficiency_moved": {
        "principle": (
            "the bottom line -- did the clickability predictor (A1) reduce median "
            "actions-to-first-levelup below blind BFS with CI (the leaderboard scoring lever)."
        )
    },
    "generic_transfer_moved": {
        "principle": (
            "did verifier-guided expansion (A2) raise generic_transfer_rate_over_variants "
            "STRICTLY above 0.04 with CI (the winner-not-in-pool fix)."
        )
    },
    "winner_generated_root_cause_addressed": {
        "principle": (
            "did A2 GENERATE the winning candidate (vs the .421 A6 ordering_gain=0 "
            "winner-never-in-pool) -- the central diagnostic this milestone tests."
        )
    },
    "reproducible_total_levels_delta": {
        "principle": "did solve CAPABILITY grow this milestone (A3 + A4 level-ups)."
    },
    "action_efficiency_score": {
        "principle": (
            "min(human/agent,1)^2 with CI (B1) -- the third co-headline metric "
            "reported alongside bank count + generic transfer."
        )
    },
    "generic_transfer_rate_over_variants": {
        "principle": (
            "the held-out-proxy first-contact signal (B1) reported WITH a CI -- "
            "a co-headline metric."
        )
    },
    "verifier_is_oracle_distinct_levers": {
        "principle": (
            "A1 (learned action-model) and A2 (learned verifier search-guide) are BOTH "
            "verifier_is_oracle:false -- a circular win would not count (Oracle-Distinctness)."
        )
    },
    "flagged_artifacts_handled": {
        "principle": (
            "names any flagged_adversarial artifact excluded AND any null-delta-carve-out / "
            "learned-CNN-substrate / positive-control-failed artifact handled (B2) -- "
            "fabrication-gate + false-negative-risk compliance."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "every headline number traces to a real upstream measurement (the audit trail)."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only if the integrated config is a CORE-preserved improvement on a real "
            "metric worth a 1/day slot (gate: beat the last 33-level submitted scorecard); "
            "never submits."
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
    "A1_clickability_predictor": Upstream(
        4568,
        Path("results/experiment_4568_clickability_action_effect_predictor.json"),
    ),
    "A2_verifier_guided_expansion": Upstream(
        4569,
        Path("results/experiment_4569_verifier_guided_expansion.json"),
    ),
    "A3_levelup_attempt": Upstream(4570, Path("results/experiment_4570_levelup_attempt.json")),
    "A4_hidden_state_probe_ka59": Upstream(
        4571,
        Path("results/experiment_4571_hidden_field_state_probe_ka59.json"),
    ),
    "A5_integration": Upstream(4572, Path("results/experiment_4572_integration_gate.json")),
    "A6_primitive_persist_transfer": Upstream(
        4573,
        Path("results/experiment_4573_primitive_persist_transfer.json"),
    ),
    "B1_action_efficiency_coheadline": Upstream(
        4574,
        Path("results/experiment_4574_action_efficiency_coheadline.json"),
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "A1_clickability_predictor": [
        "median_actions_to_first_levelup_with_predictor",
        "median_actions_to_first_levelup_baseline",
        "actions_delta",
        "actions_delta_ci",
        "efficiency_score_min_human_agent_sq",
        "generic_transfer_rate_with_predictor",
        "solve_rate_preserved",
        "positive_control_passed",
        "false_negative_risk_checked",
        "verifier_is_oracle",
    ],
    "A2_verifier_guided_expansion": [
        "generic_transfer_rate_with_expansion",
        "generic_transfer_rate_baseline",
        "transfer_delta",
        "transfer_ci",
        "expanded_states_to_goal_with_vs_without",
        "winner_generated",
        "random_priority_control_passed",
        "false_negative_risk_checked",
        "solve_rate_preserved",
        "verifier_is_oracle",
    ],
    "A3_levelup_attempt": [
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "target_level",
        "registry_update",
    ],
    "A4_hidden_state_probe_ka59": [
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "target_level",
        "registry_update",
    ],
    "A5_integration": [
        "median_actions_to_first_levelup_integrated",
        "generic_transfer_rate_integrated",
        "levers_integrated",
        "additivity_checked",
        "core_solves_preserved",
        "heldout_solve_rate",
        "ready_for_operator_submit",
    ],
    "A6_primitive_persist_transfer": [
        "primitive_persisted",
        "transfer_games",
        "transfer_value_per_game",
        "offline_reproduced",
        "registry_updated",
        "new_levels_banked",
    ],
    "B1_action_efficiency_coheadline": [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "median_actions_to_first_levelup",
        "human_baseline_actions",
        "action_efficiency_score",
        "action_efficiency_ci",
        "variant_attempts_count",
        "variant_solved_count",
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
    nullish = any(token in verdict for token in ("null", "no_value", "no_efficiency", "failed"))
    return payload.get("false_negative_risk_checked") is False and nullish


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


def _is_learned_cnn_substrate(key: str, payload: Mapping[str, Any] | None) -> bool:
    if key != "A1_clickability_predictor" or not isinstance(payload, Mapping):
        return False
    substrate = str(payload.get("inference_substrate", "")).lower()
    return (
        "verifier_ensemble_against_cached_candidates" in substrate
        and ("cnn" in substrate or "torch" in substrate or "cpu forward" in substrate)
        and "live_llm_inference" not in substrate
    )


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
        "learned_cnn_substrate_guard_honored": [],
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
            "learned_cnn_substrate_guard_honored": _is_learned_cnn_substrate(key, payload)
            and not critical,
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
        if row["learned_cnn_substrate_guard_honored"]:
            handled["learned_cnn_substrate_guard_honored"].append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "reason": "offline_learned_cnn_substrate_not_live_llm_fabrication",
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


def _a1_action_efficiency(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "moved": False,
            "verifier_is_oracle": None,
            "median_actions_to_first_levelup_with_predictor": None,
            "median_actions_to_first_levelup_baseline": None,
            "actions_delta": None,
            "actions_delta_ci": [None, None],
            "solve_rate_preserved": False,
            "positive_control_passed": False,
            "false_negative_risk_checked": False,
        }
    baseline = _number(payload, "median_actions_to_first_levelup_baseline")
    with_predictor = _number(payload, "median_actions_to_first_levelup_with_predictor")
    delta = _number(payload, "actions_delta")
    ci = _list_value(payload, "actions_delta_ci")
    ci_pair = ci if len(ci) == 2 else [None, None]
    ci_lower = ci_pair[0]
    oracle = base.bool_metric(payload, "verifier_is_oracle")
    moved = bool(
        baseline is not None
        and with_predictor is not None
        and delta is not None
        and with_predictor < baseline
        and delta > 0.0
        and isinstance(ci_lower, (int, float))
        and not isinstance(ci_lower, bool)
        and float(ci_lower) > 0.0
        and _bool(payload, "solve_rate_preserved")
        and _bool(payload, "positive_control_passed")
        and _bool(payload, "false_negative_risk_checked")
        and oracle is False
    )
    return {
        "status": "clean_action_efficiency_moved" if moved else "clean_action_efficiency_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "moved": moved,
        "verifier_is_oracle": oracle,
        "median_actions_to_first_levelup_with_predictor": with_predictor,
        "median_actions_to_first_levelup_baseline": baseline,
        "actions_delta": delta,
        "actions_delta_ci": ci_pair,
        "efficiency_score_min_human_agent_sq": _number(
            payload,
            "efficiency_score_min_human_agent_sq",
        ),
        "generic_transfer_rate_with_predictor": _number(payload, "generic_transfer_rate_with_predictor"),
        "solve_rate_preserved": _bool(payload, "solve_rate_preserved"),
        "positive_control_passed": _bool(payload, "positive_control_passed"),
        "false_negative_risk_checked": _bool(payload, "false_negative_risk_checked"),
        "chosen_submitted_config": base.str_metric(payload, "chosen_submitted_config"),
    }


def _a2_transfer(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "moved": False,
            "verifier_is_oracle": None,
            "generic_transfer_rate_with_expansion": None,
            "generic_transfer_rate_baseline": GENERIC_TRANSFER_BASELINE,
            "transfer_delta": None,
            "transfer_ci": [None, None],
            "random_priority_control_passed": False,
            "false_negative_risk_checked": False,
            "solve_rate_preserved": False,
            "winner_generated": {},
        }
    rate = _number(payload, "generic_transfer_rate_with_expansion")
    baseline = _number(payload, "generic_transfer_rate_baseline") or GENERIC_TRANSFER_BASELINE
    delta = _number(payload, "transfer_delta")
    ci = _list_value(payload, "transfer_ci")
    ci_pair = ci if len(ci) == 2 else [None, None]
    ci_lower = ci_pair[0]
    oracle = base.bool_metric(payload, "verifier_is_oracle")
    winner_generated = dict(_mapping(payload, "winner_generated"))
    winner = winner_generated.get("with_expansion") is True or int(
        winner_generated.get("generated_count") or 0
    ) > 0
    moved = bool(
        rate is not None
        and delta is not None
        and rate > GENERIC_TRANSFER_BASELINE
        and delta > 0.0
        and isinstance(ci_lower, (int, float))
        and not isinstance(ci_lower, bool)
        and float(ci_lower) > 0.0
        and winner
        and _bool(payload, "random_priority_control_passed")
        and _bool(payload, "false_negative_risk_checked")
        and _bool(payload, "solve_rate_preserved")
        and oracle is False
    )
    return {
        "status": "clean_generic_transfer_moved" if moved else "clean_generic_transfer_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "moved": moved,
        "verifier_is_oracle": oracle,
        "generic_transfer_rate_with_expansion": rate,
        "generic_transfer_rate_baseline": baseline,
        "transfer_delta": delta,
        "transfer_ci": ci_pair,
        "expanded_states_to_goal_with_vs_without": dict(
            _mapping(payload, "expanded_states_to_goal_with_vs_without")
        ),
        "winner_generated": winner_generated,
        "random_priority_control_passed": _bool(payload, "random_priority_control_passed"),
        "false_negative_risk_checked": _bool(payload, "false_negative_risk_checked"),
        "solve_rate_preserved": _bool(payload, "solve_rate_preserved"),
        "chosen_submitted_config": base.str_metric(payload, "chosen_submitted_config"),
        "missing_verifier_gaps": _list_value(payload, "missing_verifier_gaps"),
    }


def _a2_diagnostic(raw_payload: JsonDict | None, a2: Mapping[str, Any]) -> JsonDict:
    raw_winner = dict(_mapping(raw_payload, "winner_generated")) if raw_payload is not None else {}
    raw_status = str(a2.get("status", "missing_or_excluded"))
    evidence_valid = a2.get("headline_numbers_aggregated") is True
    addressed = bool(
        evidence_valid
        and (
            a2.get("winner_generated", {}).get("with_expansion") is True
            or int(a2.get("winner_generated", {}).get("generated_count") or 0) > 0
        )
    )
    return {
        "addressed": addressed,
        "evidence_status": "clean_evidence" if evidence_valid else raw_status,
        "headline_numbers_aggregated": evidence_valid,
        "winner_generated_with_expansion": (
            a2.get("winner_generated", {}).get("with_expansion")
            if evidence_valid
            else raw_winner.get("with_expansion")
        ),
        "generated_count": (
            a2.get("winner_generated", {}).get("generated_count")
            if evidence_valid
            else raw_winner.get("generated_count")
        ),
        "diagnosis_read_as_broken_test_signal": not evidence_valid and bool(raw_winner),
        "prior_root_cause": "winner_not_in_pool",
    }


def _b1_metric(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "reproducible_total_levels": None,
            "generic_transfer_rate_over_variants": 0.0,
            "generic_transfer_ci": [None, None],
            "median_actions_to_first_levelup": None,
            "human_baseline_actions": None,
            "action_efficiency_score": 0.0,
            "action_efficiency_ci": [None, None],
            "variant_attempts_count": 0,
            "variant_solved_count": 0,
        }
    generic_ci = _list_value(payload, "generic_transfer_ci")
    action_ci = _list_value(payload, "action_efficiency_ci")
    return {
        "status": "clean_action_efficiency_coheadline",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "reproducible_total_levels": _int_or_none(payload, "reproducible_total_levels"),
        "generic_transfer_rate_over_variants": _number(payload, "generic_transfer_rate_over_variants")
        or 0.0,
        "generic_transfer_ci": generic_ci if len(generic_ci) == 2 else [None, None],
        "median_actions_to_first_levelup": _number(payload, "median_actions_to_first_levelup"),
        "human_baseline_actions": _number(payload, "human_baseline_actions"),
        "action_efficiency_score": _number(payload, "action_efficiency_score") or 0.0,
        "action_efficiency_ci": action_ci if len(action_ci) == 2 else [None, None],
        "variant_attempts_count": _int_or_none(payload, "variant_attempts_count") or 0,
        "variant_solved_count": _int_or_none(payload, "variant_solved_count") or 0,
        "human_baseline_sample_count": _int_or_none(payload, "human_baseline_sample_count") or 0,
    }


def _generic_transfer_moved(a2: Mapping[str, Any], b1: Mapping[str, Any]) -> JsonDict:
    moved = bool(a2.get("moved") is True)
    return {
        "moved": moved,
        "status": str(a2.get("status", "missing_or_excluded")),
        "reason": (
            "clean_verifier_guided_expansion_transfer_above_0.04"
            if moved
            else "no_clean_verifier_guided_expansion_transfer_above_0.04"
        ),
        "baseline": GENERIC_TRANSFER_BASELINE,
        "coheadline_rate": float(b1.get("generic_transfer_rate_over_variants") or 0.0),
        "generic_transfer_ci": list(b1.get("generic_transfer_ci") or [None, None]),
        "verifier_guided_expansion": {
            "headline_numbers_aggregated": a2.get("headline_numbers_aggregated") is True,
            "rate_with_expansion": a2.get("generic_transfer_rate_with_expansion"),
            "delta": a2.get("transfer_delta"),
            "delta_ci": a2.get("transfer_ci"),
        },
    }


def _banked_levels(payload: Mapping[str, Any] | None) -> int:
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


def _level_delta(
    a3_payload: JsonDict | None,
    a4_payload: JsonDict | None,
    registry: Mapping[str, Any],
    b1: Mapping[str, Any],
) -> JsonDict:
    registry_current = int(registry.get("reproducible_total_levels") or 0)
    b1_total = int(b1.get("reproducible_total_levels") or 0)
    a3_update = _mapping(a3_payload, "registry_update")
    prior = _int_or_none(a3_update, "prior_total_declared") or b1_total or registry_current
    a3_banked = _banked_levels(a3_payload)
    a4_banked = _banked_levels(a4_payload)
    current = registry_current or b1_total or prior + a3_banked + a4_banked
    delta = max(0, current - prior)
    return {
        "prior_total": prior,
        "current_total": current,
        "delta": delta,
        "a3_new_levels_banked": a3_banked,
        "a4_new_levels_banked": a4_banked,
        "capability_grew": delta > 0,
        "source": "A3_levelup_attempt+A4_hidden_state_probe_ka59+ops/arc_solve_registry.yaml",
    }


def _a5_integration(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": _payload_status(row),
            "headline_numbers_aggregated": False,
            "integrated_metric_improved": False,
            "median_actions_to_first_levelup_integrated": None,
            "generic_transfer_rate_integrated": None,
            "core_solves_preserved": False,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
            "levers_integrated": [],
        }
    additivity = dict(_mapping(payload, "additivity_checked"))
    action_delta = _number(additivity, "integrated_actions_delta")
    generic_delta = _number(additivity, "integrated_generic_transfer_delta")
    rate = _number(payload, "generic_transfer_rate_integrated")
    preserved = _bool(payload, "core_solves_preserved")
    improved = bool(
        preserved
        and (
            (isinstance(action_delta, (int, float)) and action_delta > 0.0)
            or (isinstance(generic_delta, (int, float)) and generic_delta > 0.0)
            or (rate is not None and rate > GENERIC_TRANSFER_BASELINE)
        )
    )
    return {
        "status": "clean_integrated_metric_improved" if improved else "clean_integrated_null",
        "headline_numbers_aggregated": True,
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "integrated_metric_improved": improved,
        "median_actions_to_first_levelup_integrated": _number(
            payload,
            "median_actions_to_first_levelup_integrated",
        ),
        "generic_transfer_rate_integrated": rate,
        "additivity_checked": additivity,
        "core_solves_preserved": preserved,
        "levers_integrated": _list_value(payload, "levers_integrated"),
        "heldout_solve_rate": _number(payload, "heldout_solve_rate"),
        "ready_for_operator_submit": _bool(payload, "ready_for_operator_submit"),
        "operator_submission_performed": _bool(payload, "operator_submission_performed"),
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


def _oracle_distinct(a1: Mapping[str, Any], a2: Mapping[str, Any], raw_a2: JsonDict | None) -> JsonDict:
    a1_oracle = a1.get("verifier_is_oracle")
    a2_oracle = a2.get("verifier_is_oracle")
    if a2_oracle is None and raw_a2 is not None:
        a2_oracle = base.bool_metric(raw_a2, "verifier_is_oracle")
    return {
        "oracle_distinct": a1_oracle is False and a2_oracle is False,
        "a1_verifier_is_oracle": a1_oracle,
        "a2_verifier_is_oracle": a2_oracle,
        "circular_win_detected": a1_oracle is True or a2_oracle is True,
    }


def _operator_resubmission_verdict(
    *,
    ready: bool,
    level_delta: Mapping[str, Any],
    integration: Mapping[str, Any],
    score_gate_failed: bool,
) -> JsonDict:
    current_total = int(level_delta.get("current_total") or 0)
    if score_gate_failed:
        reason = "failed_acceptance_gate"
    elif ready and current_total > LAST_SUBMITTED_LEVELS and level_delta.get("capability_grew") is True:
        reason = f"bank_count_{current_total}_beats_last_submitted_{LAST_SUBMITTED_LEVELS}"
    elif ready and integration.get("integrated_metric_improved") is True:
        reason = "clean_integrated_metric_improvement"
    else:
        reason = "no_clean_resubmission_metric_improvement"
    return {
        "resubmission_warranted": ready,
        "reason": reason,
        "operator_only": True,
        "hidden_eval_gate": f"beat_{LAST_SUBMITTED_LEVELS}_levels",
        "last_submitted_levels": LAST_SUBMITTED_LEVELS,
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
        "last_submitted_levels": LAST_SUBMITTED_LEVELS,
        "leaderboard_submission": False,
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "action_efficiency_moved": artifact.get("action_efficiency_moved"),
        "generic_transfer_moved": artifact.get("generic_transfer_moved"),
        "winner_generated_root_cause_addressed": artifact.get(
            "winner_generated_root_cause_addressed"
        ),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "action_efficiency_score": artifact.get("action_efficiency_score"),
        "action_efficiency_ci": artifact.get("action_efficiency_ci"),
        "generic_transfer_rate_over_variants": artifact.get("generic_transfer_rate_over_variants"),
        "generic_transfer_ci": artifact.get("generic_transfer_ci"),
        "verifier_is_oracle_distinct_levers": artifact.get("verifier_is_oracle_distinct_levers"),
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


def _honest_verdict(action_moved: Mapping[str, Any], generic_moved: Mapping[str, Any]) -> str:
    if action_moved.get("moved") is True or generic_moved.get("moved") is True:
        return "success: clickability_predictor_actions_below_blind_or_expansion_generic_transfer_above_0.04"
    return "complete: action_efficiency_null_gaps_sharpened"


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

    a1 = _a1_action_efficiency(clean["A1_clickability_predictor"], rows.get("A1_clickability_predictor", {}))
    a2 = _a2_transfer(clean["A2_verifier_guided_expansion"], rows.get("A2_verifier_guided_expansion", {}))
    b1 = _b1_metric(clean["B1_action_efficiency_coheadline"], rows.get("B1_action_efficiency_coheadline", {}))
    generic_moved = _generic_transfer_moved(a2, b1)
    a3 = _level_context(clean["A3_levelup_attempt"], rows.get("A3_levelup_attempt", {}))
    a4 = _level_context(clean["A4_hidden_state_probe_ka59"], rows.get("A4_hidden_state_probe_ka59", {}))
    level_delta = _level_delta(
        clean["A3_levelup_attempt"],
        clean["A4_hidden_state_probe_ka59"],
        registry,
        b1,
    )
    integration = _a5_integration(clean["A5_integration"], rows.get("A5_integration", {}))
    a6 = _a6_transfer(clean["A6_primitive_persist_transfer"], rows.get("A6_primitive_persist_transfer", {}))
    winner_generated = _a2_diagnostic(
        raw_artifacts.get("A2_verifier_guided_expansion")
        if isinstance(raw_artifacts.get("A2_verifier_guided_expansion"), dict)
        else None,
        a2,
    )
    oracle_distinct = _oracle_distinct(
        a1,
        a2,
        raw_artifacts.get("A2_verifier_guided_expansion")
        if isinstance(raw_artifacts.get("A2_verifier_guided_expansion"), dict)
        else None,
    )

    score_gate_failed = any(row.get("acceptance_gate_failed") for row in rows.values())
    real_metric_improved = bool(
        a1.get("moved") is True
        or generic_moved.get("moved") is True
        or level_delta.get("capability_grew") is True
        or integration.get("integrated_metric_improved") is True
    )
    current_total = int(level_delta.get("current_total") or 0)
    ready = bool(
        not score_gate_failed
        and real_metric_improved
        and (
            current_total > LAST_SUBMITTED_LEVELS
            or integration.get("ready_for_operator_submit") is True
        )
        and integration.get("operator_submission_performed") is not True
    )
    scorecard = {
        "a1_clickability_predictor": a1,
        "a2_verifier_guided_expansion": a2,
        "a3_levelup_attempt": a3,
        "a4_hidden_state_probe_ka59": a4,
        "a5_integration": integration,
        "a6_transfer": a6,
        "b1_action_efficiency_coheadline": b1,
        "baseline_generic_transfer": GENERIC_TRANSFER_BASELINE,
        "last_submitted_levels": LAST_SUBMITTED_LEVELS,
    }
    duration_s = round(float((time.time() if now_s is None else now_s) - start), 6)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(a1, generic_moved),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "action_efficiency_moved": a1,
        "generic_transfer_moved": generic_moved,
        "winner_generated_root_cause_addressed": winner_generated,
        "reproducible_total_levels_delta": level_delta,
        "action_efficiency_score": float(b1.get("action_efficiency_score") or 0.0),
        "action_efficiency_ci": list(b1.get("action_efficiency_ci") or [None, None]),
        "median_actions_to_first_levelup": b1.get("median_actions_to_first_levelup"),
        "human_baseline_actions": b1.get("human_baseline_actions"),
        "generic_transfer_rate_over_variants": float(
            b1.get("generic_transfer_rate_over_variants") or 0.0
        ),
        "generic_transfer_ci": list(b1.get("generic_transfer_ci") or [None, None]),
        "reproducible_total_levels": int(
            b1.get("reproducible_total_levels")
            or registry.get("reproducible_total_levels")
            or 0
        ),
        "verifier_is_oracle_distinct_levers": oracle_distinct,
        "flagged_artifacts_handled": handled,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "ready_for_operator_submit": ready,
        "preconditions_checked": _preconditions_checked(root_path, provenance, registry),
        "scorecard": scorecard,
        "operator_resubmission_verdict": _operator_resubmission_verdict(
            ready=ready,
            level_delta=level_delta,
            integration=integration,
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
        "action_efficiency_moved",
        "generic_transfer_moved",
        "winner_generated_root_cause_addressed",
        "reproducible_total_levels_delta",
        "verifier_is_oracle_distinct_levers",
        "flagged_artifacts_handled",
        "preconditions_checked",
        "scorecard",
        "operator_resubmission_verdict",
    ):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be object")
    action_score = artifact.get("action_efficiency_score")
    if not isinstance(action_score, float) or not 0.0 <= action_score <= 1.0:
        raise ValueError("action_efficiency_score must be float in [0,1]")
    generic_rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(generic_rate, float) or not 0.0 <= generic_rate <= 1.0:
        raise ValueError("generic_transfer_rate_over_variants must be float in [0,1]")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be list")
    if not isinstance(artifact.get("ready_for_operator_submit"), bool):
        raise ValueError("ready_for_operator_submit must be bool")
    if artifact.get("leaderboard_submission") is not False:
        raise ValueError("leaderboard_submission must remain false")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed mismatch")
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
        operator = artifact.get("operator_resubmission_verdict")
        if not isinstance(operator, Mapping) or operator.get("resubmission_warranted") is not True:
            raise ValueError("ready_for_operator_submit requires a resubmission verdict")


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


def main() -> int:  # pragma: no cover - requested command boundary
    path = write_artifact(REPO_ROOT)
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
