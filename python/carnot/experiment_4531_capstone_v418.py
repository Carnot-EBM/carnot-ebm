"""Experiment 4531: .418 ARC capstone aggregation.

Spec refs: REQ-CAPSTONE-4531, SCENARIO-CAPSTONE-4531,
SCENARIO-CAPSTONE-4531-FIELD-PRINCIPLES.
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
RESULT_RELATIVE_PATH = "results/experiment_4531_capstone_v418.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = 4531
RANDOM_SEED = 4531
SCHEMA = "carnot.capstone_v418_4531.v1"
SPEC_REFS = ["REQ-CAPSTONE-4531", "SCENARIO-CAPSTONE-4531"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORE_EFFICIENCY_BASELINE = 2.0074
MEDIAN_ACTIONS_IMPROVED_THRESHOLD = 7760.0
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
        "terminal prefix; success: nav_fix_core_actions_<n>_below_7760 OR "
        "complete: nav_fix_null_efficiency_unmoved."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- reads upstream, no model load (100us floor)."
    ),
    "efficiency_moved": (
        "the bottom line -- did CORE median actions drop STRICTLY below 7760 (the IMPROVED tag, "
        "not merely non-inferior within the 10% slack) at preserved solve-rate (the thing .417 "
        "did not achieve)."
    ),
    "reproducible_total_levels_delta": "did solve CAPABILITY grow this milestone (A3 level-up).",
    "flagged_artifacts_skipped": (
        "names any flagged_adversarial artifact excluded from aggregation -- the fabrication-gate "
        "compliance."
    ),
    "cited_upstream_artifacts": (
        "every headline number traces to a real upstream measurement (the audit trail)."
    ),
    "ready_for_operator_submit": (
        "True only if the integrated config is a CORE-preserved efficiency improvement worth a "
        "1/day slot; never submits."
    ),
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "efficiency_moved",
    "reproducible_total_levels_delta",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "ready_for_operator_submit",
    "preconditions_checked",
    "scorecard",
    "a2_l1_l2_barrier_diagnosis",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class Upstream:
    experiment_id: int
    path: Path


DEFAULT_UPSTREAMS: Mapping[str, Upstream] = {
    "A1_forward_walk_navigation": Upstream(
        4523, Path("results/experiment_4523_forward_walk_navigation.json")
    ),
    "A2_reach_deeper_levels": Upstream(
        4524, Path("results/experiment_4524_reach_deeper_levels.json")
    ),
    "A2_stop_after_levelup": Upstream(
        4524, Path("results/experiment_4524_stop_after_levelup.json")
    ),
    "A3_levelup_attempt": Upstream(
        4525, Path("results/experiment_4525_levelup_attempt.json")
    ),
    "A4_integration": Upstream(
        4526, Path("results/experiment_4526_integration_8game_gate.json")
    ),
}

IMPORTED_FIELDS: Mapping[str, list[str]] = {
    "A1_forward_walk_navigation": [
        "honest_verdict",
        "median_actions_on_core_control",
        "median_actions_on_core_best",
        "core_solves_preserved",
        "nav_diagnostics_before_after",
        "chosen_submitted_config",
    ],
    "A2_reach_deeper_levels": [
        "honest_verdict",
        "core_efficiency_baseline",
        "core_efficiency_best",
        "barrier_diagnosis",
        "offline_reproduced",
    ],
    "A2_stop_after_levelup": [
        "honest_verdict",
        "median_actions_on_core_control",
        "median_actions_on_core_best",
        "core_solves_preserved",
        "levels_per_game_preserved",
    ],
    "A3_levelup_attempt": [
        "honest_verdict",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "target_level",
        "reproduction_gate",
        "registry_update",
    ],
    "A4_integration": [
        "honest_verdict",
        "core_efficiency_baseline",
        "core_efficiency_integrated",
        "core_solves_preserved",
        "ready_for_operator_submit",
        "gate_result",
        "nav_diagnostics",
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


def _bool(payload: Mapping[str, Any] | None, field: str) -> bool:
    return base.bool_metric(payload, field) is True


def _gate_value_failed(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, Mapping):
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
    *, stamped: bool, critical: bool, parse_error: str, acceptance_gate_failed: bool
) -> str:
    if stamped or critical or parse_error:
        return base._exclusion_reason(stamped, critical, parse_error)  # noqa: SLF001
    if acceptance_gate_failed:
        return "failed_acceptance_gate"
    return ""


def _read_inputs(
    root: Path,
    live_flag_runner: LiveFlagRunner,
    summarize_runner: SummarizeRunner,
) -> tuple[dict[str, Any], list[JsonDict], list[JsonDict]]:
    raw_artifacts: dict[str, Any] = {}
    provenance: list[JsonDict] = []
    flagged_skipped: list[JsonDict] = []

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
        gate_failed = _acceptance_gate_failed(payload)
        skipped = stamped or critical or payload is None or bool(parse_error) or gate_failed
        reason = _skip_reason(
            stamped=stamped,
            critical=critical,
            parse_error=parse_error,
            acceptance_gate_failed=gate_failed,
        )
        raw_artifacts[key] = payload
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
            "fields_imported": _fields_for_payload(key, skipped),
        }
        provenance.append(row)
        if stamped or critical:
            flagged_skipped.append(
                {
                    "artifact_key": key,
                    "experiment_id": upstream.experiment_id,
                    "path": str(upstream.path),
                    "sha256": sha,
                    "stamped_flagged_adversarial": stamped,
                    "live_critical": critical,
                    "live_critical_flags": [
                        flag
                        for flag in live_flags
                        if str(flag.get("severity", "")).lower() == "critical"
                    ],
                    "reason": reason,
                }
            )
    return raw_artifacts, provenance, flagged_skipped


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


def _nav_fix_delta(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": "excluded_flagged_adversarial"
            if row.get("skipped")
            else "missing_or_excluded",
            "median_actions_control": None,
            "median_actions_best": None,
            "median_actions_delta": None,
            "reset_replay_steps_before": None,
            "reset_replay_steps_after": None,
            "reset_replay_steps_delta": None,
            "score_lever_retired": True,
        }
    control = _number(payload, "median_actions_on_core_control")
    best = _number(payload, "median_actions_on_core_best")
    diagnostics = _mapping(payload, "nav_diagnostics_before_after")
    before = diagnostics.get("before") if isinstance(diagnostics.get("before"), Mapping) else {}
    after = diagnostics.get("after") if isinstance(diagnostics.get("after"), Mapping) else {}
    before_steps = _number(before, "reset_replay_steps")
    after_steps = _number(after, "reset_replay_steps")
    return {
        "status": "nav_fix_context",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "median_actions_control": control,
        "median_actions_best": best,
        "median_actions_delta": None if control is None or best is None else float(best - control),
        "reset_replay_steps_before": before_steps,
        "reset_replay_steps_after": after_steps,
        "reset_replay_steps_delta": (
            None if before_steps is None or after_steps is None else float(after_steps - before_steps)
        ),
        "forward_walk_hits_before": _int_or_none(before, "forward_walk_hits"),
        "forward_walk_hits_after": _int_or_none(after, "forward_walk_hits"),
        "chosen_submitted_config": base.str_metric(payload, "chosen_submitted_config"),
        "core_solves_preserved": _bool(payload, "core_solves_preserved"),
        "score_lever_retired": True,
    }


def _stop_after_levelup_delta(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        return {
            "status": "excluded_flagged_adversarial"
            if row.get("skipped")
            else "missing_or_excluded",
            "median_actions_control": None,
            "median_actions_best": None,
            "median_actions_delta": None,
            "moves_score": False,
            "score_lever_retired": True,
        }
    control = _number(payload, "median_actions_on_core_control")
    best = _number(payload, "median_actions_on_core_best")
    preserved = _mapping(payload, "levels_per_game_preserved")
    return {
        "status": "retired_action_trimming_context",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "median_actions_control": control,
        "median_actions_best": best,
        "median_actions_delta": None if control is None or best is None else float(best - control),
        "core_solves_preserved": _bool(payload, "core_solves_preserved"),
        "levels_per_game_preserved": _bool(preserved, "passed"),
        "moves_score": False,
        "score_lever_retired": True,
        "reason": "median_actions_retired_as_score_lever",
    }


def _a2_l1_l2_barrier_diagnosis(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        status = "excluded_flagged_adversarial" if row.get("skipped") else "missing_or_excluded"
        return {
            "status": status,
            "cleanly_reportable": False,
            "what_blocks_deeper_levels": None,
            "what_to_build_next": "not_cleanly_reportable_from_flagged_artifact"
            if status == "excluded_flagged_adversarial"
            else "missing_clean_barrier_artifact",
        }
    diagnosis = _mapping(payload, "barrier_diagnosis")
    return {
        "status": "clean_barrier_diagnosis",
        "cleanly_reportable": True,
        "what_blocks_deeper_levels": base.str_metric(diagnosis, "root_cause"),
        "what_to_build_next": base.str_metric(diagnosis, "actionable_next_step"),
        "new_win_condition_likely": _bool(diagnosis, "new_win_condition_likely"),
        "induction_not_engaged": _bool(diagnosis, "induction_not_engaged"),
    }


def _a3_levelup(payload: JsonDict | None, row: Mapping[str, Any], registry: Mapping[str, Any]) -> JsonDict:
    if row.get("acceptance_gate_failed"):
        return {
            "status": "failed_acceptance_gate",
            "level_up_banked": False,
            "target_game": base.str_metric(payload, "target_game") if payload else "",
            "target_level": _int_or_none(payload, "target_level") if payload else None,
            "banked_levels": 0,
            "prior_total": _int_or_none(_mapping(payload, "registry_update"), "prior_total_declared")
            if payload
            else None,
            "current_total": int(registry.get("reproducible_total_levels") or 0),
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
            "current_total": int(registry.get("reproducible_total_levels") or 0),
            "delta": 0,
        }
    registry_update = _mapping(payload, "registry_update")
    prior = _int_or_none(registry_update, "prior_total_declared")
    current = int(registry.get("reproducible_total_levels") or 0)
    if current == 0:
        current = _int_or_none(registry_update, "new_total_declared") or 0
    banked_levels = _int_or_none(registry_update, "banked_levels") or 0
    gate = _mapping(payload, "reproduction_gate")
    gate_reproduced = _bool(gate, "reproduced")
    offline_reproduced = _bool(payload, "offline_reproduced")
    level_up_banked = offline_reproduced and gate_reproduced and banked_levels > 0
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


def _integration_headline(payload: JsonDict | None, row: Mapping[str, Any]) -> JsonDict:
    if payload is None:
        status = "excluded_flagged_adversarial" if row.get("skipped") else "missing_or_excluded"
        return {
            "status": status,
            "submitted_config_improved": False,
            "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
            "core_efficiency_integrated": None,
            "core_efficiency_delta": None,
            "median_actions_on_core": None,
            "core_solves_preserved": False,
            "ready_for_operator_submit": False,
            "operator_submission_performed": False,
        }
    gate_result = _mapping(payload, "gate_result")
    current = _mapping(gate_result, "current")
    baseline = _number(payload, "core_efficiency_baseline") or CORE_EFFICIENCY_BASELINE
    integrated = _number(payload, "core_efficiency_integrated") or _number(current, "core_efficiency")
    median = _number(current, "median_actions_on_core")
    preserved = _bool(payload, "core_solves_preserved")
    improved = (
        integrated is not None
        and integrated > CORE_EFFICIENCY_BASELINE
        and median is not None
        and median < MEDIAN_ACTIONS_IMPROVED_THRESHOLD
        and preserved
    )
    return {
        "status": "integrated_efficiency_improved" if improved else "integrated_efficiency_unmoved",
        "honest_verdict": base.str_metric(payload, "honest_verdict"),
        "submitted_config_improved": bool(improved),
        "core_efficiency_baseline": baseline,
        "core_efficiency_integrated": integrated,
        "core_efficiency_delta": None if integrated is None else round(integrated - baseline, 10),
        "median_actions_on_core": median,
        "core_solves_preserved": preserved,
        "ready_for_operator_submit": _bool(payload, "ready_for_operator_submit"),
        "operator_submission_performed": _bool(payload, "operator_submission_performed"),
    }


def _core_efficiency_summary(integration: Mapping[str, Any]) -> JsonDict:
    integrated = integration.get("core_efficiency_integrated")
    if not isinstance(integrated, (int, float)) or isinstance(integrated, bool):
        return {
            "baseline": CORE_EFFICIENCY_BASELINE,
            "integrated": None,
            "delta": None,
            "moved": False,
            "reason": "integration_excluded_flagged_or_live_critical",
        }
    delta = round(float(integrated) - CORE_EFFICIENCY_BASELINE, 10)
    moved = bool(integration.get("submitted_config_improved"))
    return {
        "baseline": CORE_EFFICIENCY_BASELINE,
        "integrated": float(integrated),
        "delta": delta,
        "moved": moved,
        "reason": "core_efficiency_rose" if moved else "core_efficiency_not_above_baseline",
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
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "scorecard": artifact.get("scorecard"),
        "a2_l1_l2_barrier_diagnosis": artifact.get("a2_l1_l2_barrier_diagnosis"),
        "upstream_sha256_set": sorted(
            str(row.get("sha256", "")) for row in artifact.get("upstream_provenance", [])
        ),
    }


def checksum_from_artifact(artifact: Mapping[str, Any]) -> str:
    blob = json.dumps(_checksum_payload(artifact), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _honest_verdict(efficiency_moved: bool, integration: Mapping[str, Any]) -> str:
    if efficiency_moved:
        median = int(float(integration.get("median_actions_on_core") or 0))
        return f"success: nav_fix_core_actions_{median}_below_7760"
    return "complete: nav_fix_null_efficiency_unmoved"


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
    raw_artifacts, provenance, flagged_skipped = _read_inputs(
        root_path,
        live_flag_runner,
        summarize_runner,
    )
    rows = _provenance_by_key(provenance)
    registry = load_registry_totals(root_path)
    clean = {
        key: _clean_payload(raw_artifacts, rows, key)
        for key in DEFAULT_UPSTREAMS
    }

    nav_fix = _nav_fix_delta(clean["A1_forward_walk_navigation"], rows.get("A1_forward_walk_navigation", {}))
    stop_after_levelup = _stop_after_levelup_delta(
        clean["A2_stop_after_levelup"],
        rows.get("A2_stop_after_levelup", {}),
    )
    a2_diagnosis = _a2_l1_l2_barrier_diagnosis(
        clean["A2_reach_deeper_levels"],
        rows.get("A2_reach_deeper_levels", {}),
    )
    a3 = _a3_levelup(
        raw_artifacts.get("A3_levelup_attempt")
        if isinstance(raw_artifacts.get("A3_levelup_attempt"), dict)
        else None,
        rows.get("A3_levelup_attempt", {}),
        registry,
    )
    integration = _integration_headline(clean["A4_integration"], rows.get("A4_integration", {}))
    core_efficiency = _core_efficiency_summary(integration)
    efficiency_moved = bool(core_efficiency["moved"])
    ready_for_operator_submit = bool(
        efficiency_moved
        and integration.get("ready_for_operator_submit") is True
        and integration.get("operator_submission_performed") is False
    )
    scorecard = {
        "core_efficiency": core_efficiency,
        "nav_fix_delta": nav_fix,
        "stop_after_levelup_delta": stop_after_levelup,
        "a3_levelup": a3,
        "integration_headline": integration,
    }
    duration_s = round(float((time.time() if now_s is None else now_s) - start), 6)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(efficiency_moved, integration),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "efficiency_moved": efficiency_moved,
        "reproducible_total_levels_delta": _reproducible_total_levels_delta(a3),
        "flagged_artifacts_skipped": flagged_skipped,
        "cited_upstream_artifacts": _cited_upstream_artifacts(provenance),
        "ready_for_operator_submit": ready_for_operator_submit,
        "preconditions_checked": _preconditions_checked(root_path, provenance, registry),
        "scorecard": scorecard,
        "a2_l1_l2_barrier_diagnosis": a2_diagnosis,
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
    if not isinstance(artifact.get("reproducible_total_levels_delta"), Mapping):
        raise ValueError("reproducible_total_levels_delta must be object")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be list")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be list")
    if not isinstance(artifact.get("ready_for_operator_submit"), bool):
        raise ValueError("ready_for_operator_submit must be bool")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        raise ValueError("preconditions_checked must be object")
    if not isinstance(artifact.get("scorecard"), Mapping):
        raise ValueError("scorecard must be object")
    if not isinstance(artifact.get("a2_l1_l2_barrier_diagnosis"), Mapping):
        raise ValueError("a2_l1_l2_barrier_diagnosis must be object")
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
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
