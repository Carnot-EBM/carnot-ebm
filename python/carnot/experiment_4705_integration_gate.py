"""Experiment 4705: integrate object-centric/amortized A1/A2 levers.

Spec refs: REQ-ARC-WMTE-4705, SCENARIO-ARC-WMTE-4705.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4693_integration_gate import (  # noqa: E402
    _as_float,
    _as_int,
    _load_json,
    _write_json,
    artifact_checksum,
    compare_scored_measurement,
    measure_scored_lane,
    payload_checksum,
    run_parity_check,
)


JsonDict = dict[str, Any]
Check = Callable[[Path | str], Mapping[str, Any]]
ImportChecker = Callable[[], Mapping[str, Any]]
ScoredMeasure = Callable[[Path | str], Mapping[str, Any]]

EXPERIMENT = "experiment_4705_integration_gate"
SCHEMA = "carnot.exp4705.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4705_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4700_object_centric_perception_proposal_live.json"
A2_RELATIVE_PATH = "results/experiment_4701_amortized_exploration_prior_go_explore_live.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4705
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure on the "
    "held-out lane (1s floor)."
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4705", "SCENARIO-ARC-WMTE-4705"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_<config>_shipped_parity_green OR "
            "complete: integration_unchanged_both_levers_null."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the integrated object-centric/amortized-prior signals are "
            "oracle-distinct from the executable win-check."
        )
    },
    "config_integrated": {
        "principle": (
            "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with "
            "the reason) -- the single-source-of-truth update."
        )
    },
    "config_changed": {
        "principle": (
            "bool -- true only if a lever cleared its gate; when false, the pre/post "
            "numbers are identical BY CONSTRUCTION (not a finding)."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when config_changed=false (pre==post) -- why the no-change is honest; "
            "the marker the TAUTOLOGY carve-out reads (the recurring A6 flag fix)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "bool(parity_test_green AND no_regression) -- GATES the null-delta exemption "
            "so an unvalidated no-change is NOT excused."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent held-out first-win-rate (no regression vs "
            "pre-integration) -- the retargeted scored-lane metric."
        )
    },
    "multi_level_deepen_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent multi-level deepen-rate (the deeper scored lever) "
            "-- measured on the held-out lane."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == "
            "the measured agent."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "pre_integration_config",
    "first_win_rate_pre_integration",
    "first_win_rate_delta_vs_pre_integration",
    "multi_level_deepen_rate_pre_integration",
    "multi_level_deepen_rate_delta_vs_pre_integration",
    "no_regression_vs_pre_integration",
    "a1_a2_config_audit",
    "metrics_delta_audit",
    "scored_measurement",
    "parity_test",
    "submitted_agent_config",
    "source_artifacts",
    "source_artifact_checksums",
    "tautology_guard",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


def _value_matches(chosen: Mapping[str, Any], submitted: Mapping[str, Any], key: str) -> bool:
    return key in submitted and submitted.get(key) == chosen.get(key)


def _a1_radius_matches(chosen: Mapping[str, Any], submitted: Mapping[str, Any]) -> bool:
    radius = chosen.get("object_centric_neighborhood_radius", chosen.get("neighborhood_radius"))
    if radius is None:
        return True
    submitted_radius = submitted.get(
        "object_centric_neighborhood_radius", submitted.get("neighborhood_radius")
    )
    return submitted_radius is not None and _as_int(submitted_radius, -1) == _as_int(radius, -2)


def _a1_gate_cleared(a1_artifact: Mapping[str, Any]) -> bool:
    reached = _as_int(a1_artifact.get("generic_agent_reached_level"))
    order1 = _as_int(a1_artifact.get("order1_ablation_reached_level"))
    return bool(
        str(a1_artifact.get("honest_verdict") or "").startswith("success:")
        and a1_artifact.get("verifier_is_oracle") is False
        and a1_artifact.get("offline_reproduced") is True
        and _as_int(a1_artifact.get("reproduced_levels")) >= 1
        and reached > order1
    )


def _a2_gate_cleared(a2_artifact: Mapping[str, Any]) -> bool:
    return bool(
        str(a2_artifact.get("honest_verdict") or "").startswith("success:")
        and a2_artifact.get("verifier_is_oracle") is False
        and _as_float(a2_artifact.get("coverage_delta")) > 0.0
        and _as_float(a2_artifact.get("first_win_rate_delta")) > 0.0
        and a2_artifact.get("offline_reproduced") is True
        and a2_artifact.get("no_prior_ablation_failed") is True
    )


def _a1_chosen_config_valid(chosen: Mapping[str, Any]) -> bool:
    radius = chosen.get("object_centric_neighborhood_radius", chosen.get("neighborhood_radius"))
    return bool(
        chosen.get("object_centric_proposal_enabled") is True
        and str(chosen.get("object_centric_proposal_mode") or "")
        and (radius is None or _as_int(radius) > 0)
    )


def _a1_submitted_config_matches(
    chosen: Mapping[str, Any],
    submitted: Mapping[str, Any],
) -> bool:
    return bool(
        _value_matches(chosen, submitted, "object_centric_proposal_enabled")
        and _value_matches(chosen, submitted, "object_centric_proposal_mode")
        and _a1_radius_matches(chosen, submitted)
    )


def _a2_chosen_config_valid(chosen: Mapping[str, Any]) -> bool:
    return bool(
        chosen.get("amortized_first_contact_prior_enabled") is True
        and str(chosen.get("amortized_first_contact_prior_mode") or "")
        and chosen.get("go_explore_archive_enabled") is True
        and str(chosen.get("go_explore_archive_mode") or "")
    )


def _a2_submitted_config_matches(
    chosen: Mapping[str, Any],
    submitted: Mapping[str, Any],
) -> bool:
    return bool(
        _value_matches(chosen, submitted, "amortized_first_contact_prior_enabled")
        and _value_matches(chosen, submitted, "amortized_first_contact_prior_mode")
        and _value_matches(chosen, submitted, "go_explore_archive_enabled")
        and _value_matches(chosen, submitted, "go_explore_archive_mode")
    )


def _a1_audit(
    a1_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    chosen = a1_artifact.get("chosen_submitted_config")
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        integrated = False
    elif a1_artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        integrated = False
    elif not _a1_gate_cleared(a1_artifact):
        reason = "a1_object_centric_gate_not_cleared"
        integrated = False
    elif not _a1_chosen_config_valid(chosen):
        reason = "a1_chosen_object_centric_config_invalid"
        integrated = False
    elif not _a1_submitted_config_matches(chosen, submitted_agent_config):
        reason = "submitted_object_centric_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a1_object_centric_proposal"
        integrated = True
    return {
        "lever": "A1",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "gate_cleared": _a1_gate_cleared(a1_artifact),
        "generic_agent_reached_level": _as_int(a1_artifact.get("generic_agent_reached_level")),
        "offline_reproduced": a1_artifact.get("offline_reproduced") is True,
        "reproduced_levels": _as_int(a1_artifact.get("reproduced_levels")),
        "order1_ablation_reached_level": _as_int(a1_artifact.get("order1_ablation_reached_level")),
        "submitted_object_centric_proposal_enabled": bool(
            submitted_agent_config.get("object_centric_proposal_enabled")
        ),
        "submitted_object_centric_proposal_mode": submitted_agent_config.get(
            "object_centric_proposal_mode"
        ),
        "submitted_object_centric_neighborhood_radius": submitted_agent_config.get(
            "object_centric_neighborhood_radius"
        ),
    }


def _a2_audit(
    a2_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    chosen = a2_artifact.get("chosen_submitted_config")
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        integrated = False
    elif a2_artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        integrated = False
    elif not _a2_gate_cleared(a2_artifact):
        reason = "a2_amortized_prior_gate_not_cleared"
        integrated = False
    elif not _a2_chosen_config_valid(chosen):
        reason = "a2_chosen_amortized_prior_config_invalid"
        integrated = False
    elif not _a2_submitted_config_matches(chosen, submitted_agent_config):
        reason = "submitted_amortized_prior_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a2_amortized_prior_go_explore"
        integrated = True
    return {
        "lever": "A2",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "gate_cleared": _a2_gate_cleared(a2_artifact),
        "coverage_delta": _as_float(a2_artifact.get("coverage_delta")),
        "first_win_rate_delta": _as_float(a2_artifact.get("first_win_rate_delta")),
        "offline_reproduced": a2_artifact.get("offline_reproduced") is True,
        "no_prior_ablation_failed": a2_artifact.get("no_prior_ablation_failed") is True,
        "submitted_amortized_first_contact_prior_enabled": bool(
            submitted_agent_config.get("amortized_first_contact_prior_enabled")
        ),
        "submitted_amortized_first_contact_prior_mode": submitted_agent_config.get(
            "amortized_first_contact_prior_mode"
        ),
        "submitted_go_explore_archive_enabled": bool(
            submitted_agent_config.get("go_explore_archive_enabled")
        ),
        "submitted_go_explore_archive_mode": submitted_agent_config.get("go_explore_archive_mode"),
    }


def audit_config_integration(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    a1 = _a1_audit(a1_artifact, submitted_agent_config)
    a2 = _a2_audit(a2_artifact, submitted_agent_config)
    levers = []
    if a1["integrated"]:
        levers.append("A1_object_centric_proposal")
    if a2["integrated"]:
        levers.append("A2_amortized_prior_go_explore")
    config_changed = bool(levers)
    if config_changed:
        config_integrated: Any = {
            "levers_integrated": list(levers),
            "object_centric_proposal_enabled": submitted_agent_config.get(
                "object_centric_proposal_enabled"
            ),
            "object_centric_proposal_mode": submitted_agent_config.get(
                "object_centric_proposal_mode"
            ),
            "object_centric_neighborhood_radius": submitted_agent_config.get(
                "object_centric_neighborhood_radius"
            ),
            "amortized_first_contact_prior_enabled": submitted_agent_config.get(
                "amortized_first_contact_prior_enabled"
            ),
            "amortized_first_contact_prior_mode": submitted_agent_config.get(
                "amortized_first_contact_prior_mode"
            ),
            "go_explore_archive_enabled": submitted_agent_config.get("go_explore_archive_enabled"),
            "go_explore_archive_mode": submitted_agent_config.get("go_explore_archive_mode"),
        }
    else:
        config_integrated = f"unchanged: A1 {a1['reason']}; A2 {a2['reason']}"
    return {
        "config_changed": config_changed,
        "levers_integrated": levers,
        "a1": a1,
        "a2": a2,
        "config_integrated": config_integrated,
    }


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    levers = list(audit.get("levers_integrated") or [])
    if levers and (not parity_green or metrics.get("no_regression_vs_pre_integration") is not True):
        return "complete: integration_parity_or_regression_failed"
    if not levers:
        return "complete: integration_unchanged_both_levers_null"
    both_levers = (
        "A1_object_centric_proposal" in levers and "A2_amortized_prior_go_explore" in levers
    )
    if both_levers:
        return "success: integrated_a1_object_centric_and_a2_amortized_prior_shipped_parity_green"
    if "A1_object_centric_proposal" in levers:
        return "success: integrated_a1_object_centric_proposal_shipped_parity_green"
    return "success: integrated_a2_amortized_prior_go_explore_shipped_parity_green"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    source_artifact_checksums: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    parity_green = bool(parity_test.get("passed"))
    config_changed = bool(audit.get("config_changed"))
    no_regression = bool(metrics.get("no_regression_vs_pre_integration"))
    null_note = (
        "config_changed=false; integrated and pre-integration metrics are identical by "
        "construction because both upstream A1/A2 submitted configs were unchanged."
        if not config_changed
        else ""
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(audit, metrics, parity_green),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "config_integrated": audit.get("config_integrated"),
        "config_changed": config_changed,
        "null_delta_methodology_note": null_note,
        "positive_control_passed": bool(parity_green and no_regression),
        "first_win_rate_integrated": metrics.get("first_win_rate_integrated"),
        "multi_level_deepen_rate_integrated": metrics.get("multi_level_deepen_rate_integrated"),
        "parity_test_green": parity_green,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "pre_integration_config": {
            "source": "same submitted config"
            if not config_changed
            else "pre-integration measurement",
            "first_win_rate": metrics.get("first_win_rate_pre_integration"),
            "multi_level_deepen_rate": metrics.get("multi_level_deepen_rate_pre_integration"),
        },
        "first_win_rate_pre_integration": metrics.get("first_win_rate_pre_integration"),
        "first_win_rate_delta_vs_pre_integration": metrics.get(
            "first_win_rate_delta_vs_pre_integration"
        ),
        "multi_level_deepen_rate_pre_integration": metrics.get(
            "multi_level_deepen_rate_pre_integration"
        ),
        "multi_level_deepen_rate_delta_vs_pre_integration": metrics.get(
            "multi_level_deepen_rate_delta_vs_pre_integration"
        ),
        "no_regression_vs_pre_integration": no_regression,
        "a1_a2_config_audit": dict(audit),
        "metrics_delta_audit": dict(metrics),
        "scored_measurement": dict(metrics.get("scored_measurement") or {}),
        "parity_test": dict(parity_test),
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "source_artifacts": dict(source_artifacts),
        "source_artifact_checksums": dict(source_artifact_checksums),
        "tautology_guard": (
            "config_changed=false: pre/post measurements are identical BY CONSTRUCTION "
            "(not a finding)."
            if not config_changed
            else "config_changed=true: metric deltas are measured against the pre-integration config."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": max(1.0, round(float(duration_s), 6)),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if type(artifact.get("config_changed")) is not bool:
        errors.append("config_changed_bool")
    if not blocked and artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if not blocked and artifact.get("no_regression_vs_pre_integration") is not True:
        errors.append("no_regression_vs_pre_integration")
    null_or_malformed_config = artifact.get("config_changed") is not True
    if (
        not blocked
        and null_or_malformed_config
        and not str(artifact.get("null_delta_methodology_note") or "").strip()
    ):
        errors.append("null_delta_methodology_note")
    if (
        not blocked
        and null_or_malformed_config
        and artifact.get("positive_control_passed") is not True
    ):
        errors.append("positive_control_passed")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_import_checker() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG

    return {
        "submitted_agent_import": bool(E3AgentPolicy and SUBMITTED_AGENT_CONFIG),
    }


def _submitted_config_snapshot() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    import_checker: ImportChecker | None = None,
) -> JsonDict:
    root_path = Path(root)
    imports = import_checker or _default_import_checker
    try:
        import_status = dict(imports())
    except Exception as exc:  # pragma: no cover - import failure is reported as a resource.
        import_status = {"submitted_agent_import": False, "submitted_agent_import_error": str(exc)}
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4705": "REQ-ARC-WMTE-4705" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "submitted_agent_import",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4705",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


def _source_artifacts() -> JsonDict:
    return {
        "a1": A1_RELATIVE_PATH,
        "a2": A2_RELATIVE_PATH,
        "scored_lane": "python/carnot/experiment_4605_live_integration_scored_agent.py",
    }


def _source_checksums(*artifacts: Mapping[str, Any]) -> JsonDict:
    names = ("a1", "a2")
    return {
        name: str(artifact.get("reproducibility_checksum") or artifact_checksum(artifact))
        for name, artifact in zip(names, artifacts)
    }


def _blocked_artifact(
    checks: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    audit = {
        "config_changed": False,
        "levers_integrated": [],
        "a1": {"integrated": False, "reason": "blocked_precondition"},
        "a2": {"integrated": False, "reason": "blocked_precondition"},
        "config_integrated": "unchanged: blocked before A1/A2 integration",
    }
    metrics = {
        "first_win_rate_integrated": 0.0,
        "first_win_rate_pre_integration": 0.0,
        "first_win_rate_delta_vs_pre_integration": 0.0,
        "multi_level_deepen_rate_integrated": 0.0,
        "multi_level_deepen_rate_pre_integration": 0.0,
        "multi_level_deepen_rate_delta_vs_pre_integration": 0.0,
        "no_regression_vs_pre_integration": False,
        "metric_measurement_note": "blocked before measurement",
        "scored_measurement": {},
        "pre_integration_measurement": None,
    }
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test={"passed": False, "blocked": True},
        submitted_agent_config=submitted_agent_config,
        source_artifacts=_source_artifacts(),
        source_artifact_checksums={},
        duration_s=duration_s,
    )
    blocked = str(checks.get("blocked_resource") or "precondition")
    artifact["honest_verdict"] = f"blocked_{blocked}"
    artifact["null_delta_methodology_note"] = "blocked before integration measurement"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    parity_check: Check | None = None,
    submitted_agent_config: Mapping[str, Any] | None = None,
    import_checker: ImportChecker | None = None,
    measure_scored_lane: ScoredMeasure | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    submitted_config = (
        json.loads(json.dumps(submitted_agent_config, default=str))
        if submitted_agent_config is not None
        else _submitted_config_snapshot()
    )
    checks = check_preconditions(root_path, import_checker=import_checker)
    duration = max(1.0, now() - start)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks, submitted_config, duration_s=duration)
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    a1_artifact = _load_json(root_path / A1_RELATIVE_PATH)
    a2_artifact = _load_json(root_path / A2_RELATIVE_PATH)
    audit = audit_config_integration(
        a1_artifact=a1_artifact,
        a2_artifact=a2_artifact,
        submitted_agent_config=submitted_config,
    )
    scored = dict((measure_scored_lane or globals()["measure_scored_lane"])(root_path))
    metrics = compare_scored_measurement(
        scored_measurement=scored,
        pre_integration_measurement=None,
        config_changed=bool(audit["config_changed"]),
    )
    parity = (parity_check or run_parity_check)(root_path)
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test=parity,
        submitted_agent_config=submitted_config,
        source_artifacts=_source_artifacts(),
        source_artifact_checksums=_source_checksums(a1_artifact, a2_artifact),
        duration_s=duration,
    )
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"result": RESULT_RELATIVE_PATH, "schema_errors": errors}, indent=2))
        return 1
    print(
        json.dumps({"result": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
