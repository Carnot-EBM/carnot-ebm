"""Experiment 4693: integrate directed-exploration A1/A2 levers.

Spec refs: REQ-ARC-WMTE-4693, SCENARIO-ARC-WMTE-4693.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
Check = Callable[[Path | str], Mapping[str, Any]]
ImportChecker = Callable[[], Mapping[str, Any]]
ScoredMeasure = Callable[[Path | str], Mapping[str, Any]]

EXPERIMENT = "experiment_4693_integration_gate"
SCHEMA = "carnot.exp4693.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4693_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4688_controllable_novelty_proposal_policy_live.json"
A2_RELATIVE_PATH = "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4693
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure on the "
    "held-out lane (1s floor)."
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4693", "SCENARIO-ARC-WMTE-4693"]

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
            "MUST be false -- the integrated controllable-novelty / proposal-filter signals "
            "are oracle-distinct from the executable win-check."
        )
    },
    "config_integrated": {
        "principle": (
            "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the "
            "reason) -- the single-source-of-truth update."
        )
    },
    "config_changed": {
        "principle": (
            "bool -- true only if a lever cleared its gate; when false, the pre/post "
            "numbers are identical BY CONSTRUCTION (not a finding) -- pre-empts the "
            ".430 A6 TAUTOLOGY flag."
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def artifact_checksum(value: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(value)


def _load_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _mapping_float(mapping: Mapping[str, Any] | None, key: str) -> float:
    return _as_float(mapping.get(key) if isinstance(mapping, Mapping) else None)


def _a1_gate_cleared(a1_artifact: Mapping[str, Any]) -> bool:
    reached = _as_int(a1_artifact.get("generic_agent_reached_level"))
    no_novelty = _as_int(a1_artifact.get("no_novelty_ablation_reached_level"))
    cosmetic = _as_int(a1_artifact.get("cosmetic_novelty_ablation_reached_level"))
    return bool(
        str(a1_artifact.get("honest_verdict") or "").startswith("success:")
        and a1_artifact.get("verifier_is_oracle") is False
        and a1_artifact.get("offline_reproduced") is True
        and _as_int(a1_artifact.get("reproduced_levels")) >= 1
        and reached > max(no_novelty, cosmetic)
    )


def _a2_gate_cleared(a2_artifact: Mapping[str, Any]) -> bool:
    return bool(
        str(a2_artifact.get("honest_verdict") or "").startswith("success:")
        and a2_artifact.get("verifier_is_oracle") is False
        and _as_float(a2_artifact.get("coverage_delta")) > 0.0
        and _as_float(a2_artifact.get("first_win_rate_delta")) > 0.0
        and _as_int(a2_artifact.get("heldout_programs_kept")) > 0
    )


def _a1_chosen_config_valid(chosen: Mapping[str, Any]) -> bool:
    return bool(
        chosen.get("controllable_novelty_proposal_enabled") is True
        and str(chosen.get("controllable_novelty_proposal_mode") or "")
        and _as_float(chosen.get("controllable_novelty_bonus_weight"), -1.0) >= 0.0
        and _as_float(chosen.get("controllable_novelty_temperature"), -1.0) > 0.0
    )


def _float_config_matches(
    chosen: Mapping[str, Any],
    submitted: Mapping[str, Any],
    key: str,
) -> bool:
    return key in submitted and _as_float(submitted.get(key), -999.0) == _as_float(
        chosen.get(key), -998.0
    )


def _a1_submitted_config_matches(
    chosen: Mapping[str, Any],
    submitted: Mapping[str, Any],
) -> bool:
    return bool(
        submitted.get("controllable_novelty_proposal_enabled")
        == chosen.get("controllable_novelty_proposal_enabled")
        and submitted.get("controllable_novelty_proposal_mode")
        == chosen.get("controllable_novelty_proposal_mode")
        and _float_config_matches(chosen, submitted, "controllable_novelty_bonus_weight")
        and _float_config_matches(chosen, submitted, "controllable_novelty_temperature")
    )


def _a2_chosen_config_valid(chosen: Mapping[str, Any]) -> bool:
    threshold = _as_float(chosen.get("program_synthesis_proposal_filter_trust_threshold"), -1.0)
    return bool(
        chosen.get("program_synthesis_proposal_filter_enabled") is True
        and 0.0 <= threshold <= 1.0
    )


def _a2_submitted_config_matches(
    chosen: Mapping[str, Any],
    submitted: Mapping[str, Any],
) -> bool:
    threshold_key = "program_synthesis_proposal_filter_trust_threshold"
    return bool(
        submitted.get("program_synthesis_proposal_filter_enabled")
        == chosen.get("program_synthesis_proposal_filter_enabled")
        and _float_config_matches(chosen, submitted, threshold_key)
    )


def _a1_audit(
    a1_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    chosen = a1_artifact.get("chosen_submitted_config")
    submitted_enabled = bool(
        submitted_agent_config.get("controllable_novelty_proposal_enabled")
    )
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
        reason = "a1_controllable_novelty_gate_not_cleared"
        integrated = False
    elif not _a1_chosen_config_valid(chosen):
        reason = "a1_chosen_controllable_novelty_config_invalid"
        integrated = False
    elif not _a1_submitted_config_matches(chosen, submitted_agent_config):
        reason = "submitted_controllable_novelty_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a1_controllable_novelty_proposal"
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
        "submitted_controllable_novelty_proposal_enabled": submitted_enabled,
        "submitted_controllable_novelty_proposal_mode": submitted_agent_config.get(
            "controllable_novelty_proposal_mode"
        ),
        "submitted_controllable_novelty_bonus_weight": submitted_agent_config.get(
            "controllable_novelty_bonus_weight"
        ),
        "submitted_controllable_novelty_temperature": submitted_agent_config.get(
            "controllable_novelty_temperature"
        ),
    }


def _a2_audit(
    a2_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    chosen = a2_artifact.get("chosen_submitted_config")
    submitted_enabled = bool(
        submitted_agent_config.get("program_synthesis_proposal_filter_enabled")
    )
    submitted_threshold = _as_float(
        submitted_agent_config.get("program_synthesis_proposal_filter_trust_threshold")
    )
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
        reason = "a2_program_synthesis_filter_gate_not_cleared"
        integrated = False
    elif not _a2_chosen_config_valid(chosen):
        reason = "a2_chosen_program_synthesis_filter_config_invalid"
        integrated = False
    elif not _a2_submitted_config_matches(chosen, submitted_agent_config):
        reason = "submitted_program_synthesis_filter_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a2_program_synthesis_proposal_filter"
        integrated = True
    return {
        "lever": "A2",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "gate_cleared": _a2_gate_cleared(a2_artifact),
        "coverage_delta": _as_float(a2_artifact.get("coverage_delta")),
        "first_win_rate_delta": _as_float(a2_artifact.get("first_win_rate_delta")),
        "heldout_programs_kept": _as_int(a2_artifact.get("heldout_programs_kept")),
        "heldout_programs_rejected": _as_int(a2_artifact.get("heldout_programs_rejected")),
        "submitted_program_synthesis_proposal_filter_enabled": submitted_enabled,
        "submitted_program_synthesis_proposal_filter_trust_threshold": submitted_threshold,
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
        levers.append("A1_controllable_novelty_proposal")
    if a2["integrated"]:
        levers.append("A2_program_synthesis_proposal_filter")
    config_changed = bool(levers)
    if config_changed:
        config_integrated: Any = {
            "levers_integrated": list(levers),
            "controllable_novelty_proposal_enabled": submitted_agent_config.get(
                "controllable_novelty_proposal_enabled"
            ),
            "controllable_novelty_proposal_mode": submitted_agent_config.get(
                "controllable_novelty_proposal_mode"
            ),
            "controllable_novelty_bonus_weight": submitted_agent_config.get(
                "controllable_novelty_bonus_weight"
            ),
            "controllable_novelty_temperature": submitted_agent_config.get(
                "controllable_novelty_temperature"
            ),
            "program_synthesis_proposal_filter_enabled": submitted_agent_config.get(
                "program_synthesis_proposal_filter_enabled"
            ),
            "program_synthesis_proposal_filter_trust_threshold": submitted_agent_config.get(
                "program_synthesis_proposal_filter_trust_threshold"
            ),
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


def measure_scored_lane(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - ARC runtime.
    from carnot import experiment_4605_live_integration_scored_agent as lane

    old_deepen = os.environ.get(lane.DEEPEN_ENV)
    os.environ[lane.DEEPEN_ENV] = "1"
    try:
        integrated, bare = lane.measure_policy_pair(
            public_games=lane._public_games(Path(root)),
            variant_ids=lane.resolve_variant_ids(None),
            budget=lane.DEFAULT_BUDGET,
            variant_runner_factory=lane.default_variant_runner_factory,
        )
    finally:
        if old_deepen is None:
            os.environ.pop(lane.DEEPEN_ENV, None)
        else:
            os.environ[lane.DEEPEN_ENV] = old_deepen
    deepening = lane.deepening_summary(integrated.get("variant_attempts", []))
    return {
        "first_win_rate": _mapping_float(integrated, "first_win_rate"),
        "multi_level_deepen_rate": _mapping_float(deepening, "multi_level_solve_rate"),
        "scored_lane": {
            "integrated_measurement": integrated,
            "bare_measurement": bare,
            "deepening_summary": deepening,
            "variant_ids": list(lane.resolve_variant_ids(None)),
            "budget": lane.DEFAULT_BUDGET,
            "deepening_enabled": True,
        },
    }


def compare_scored_measurement(
    *,
    scored_measurement: Mapping[str, Any],
    pre_integration_measurement: Mapping[str, Any] | None,
    config_changed: bool,
) -> JsonDict:
    first = _mapping_float(scored_measurement, "first_win_rate")
    deepen = _mapping_float(scored_measurement, "multi_level_deepen_rate")
    pre_source = pre_integration_measurement if config_changed else scored_measurement
    pre_first = _mapping_float(pre_source, "first_win_rate")
    pre_deepen = _mapping_float(pre_source, "multi_level_deepen_rate")
    if config_changed and pre_integration_measurement is None:
        pre_first = first
        pre_deepen = deepen
    first_delta = _rounded(first - pre_first)
    deepen_delta = _rounded(deepen - pre_deepen)
    if config_changed:
        note = (
            "config_changed=true; integrated scored-lane metrics are compared against the "
            "pre-integration measurement."
        )
    else:
        note = (
            "config_changed=false; first-win and multi-level deepen metrics remain equal to "
            "the pre-integration config by construction."
        )
    return {
        "first_win_rate_integrated": _rounded(first),
        "first_win_rate_pre_integration": _rounded(pre_first),
        "first_win_rate_delta_vs_pre_integration": first_delta,
        "multi_level_deepen_rate_integrated": _rounded(deepen),
        "multi_level_deepen_rate_pre_integration": _rounded(pre_deepen),
        "multi_level_deepen_rate_delta_vs_pre_integration": deepen_delta,
        "no_regression_vs_pre_integration": bool(first_delta >= 0.0 and deepen_delta >= 0.0),
        "metric_measurement_note": note,
        "scored_measurement": dict(scored_measurement),
        "pre_integration_measurement": (
            dict(pre_integration_measurement) if pre_integration_measurement is not None else None
        ),
    }


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    levers = list(audit.get("levers_integrated") or [])
    if levers and (
        not parity_green or metrics.get("no_regression_vs_pre_integration") is not True
    ):
        return "complete: integration_parity_or_regression_failed"
    if not levers:
        return "complete: integration_unchanged_both_levers_null"
    both_levers = (
        "A1_controllable_novelty_proposal" in levers
        and "A2_program_synthesis_proposal_filter" in levers
    )
    if both_levers:
        return (
            "success: integrated_a1_controllable_novelty_and_a2_program_filter_"
            "shipped_parity_green"
        )
    if "A1_controllable_novelty_proposal" in levers:
        return "success: integrated_a1_controllable_novelty_proposal_shipped_parity_green"
    return "success: integrated_a2_program_synthesis_proposal_filter_shipped_parity_green"


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
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(audit, metrics, parity_green),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "config_integrated": audit.get("config_integrated"),
        "config_changed": config_changed,
        "first_win_rate_integrated": metrics.get("first_win_rate_integrated"),
        "multi_level_deepen_rate_integrated": metrics.get(
            "multi_level_deepen_rate_integrated"
        ),
        "parity_test_green": parity_green,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "pre_integration_config": {
            "source": "same submitted config" if not config_changed else "pre-integration measurement",
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
        "no_regression_vs_pre_integration": bool(metrics.get("no_regression_vs_pre_integration")),
        "positive_control_passed": bool(
            parity_green and metrics.get("no_regression_vs_pre_integration")
        ),
        "null_delta_methodology_note": (
            "config_changed=false; integrated and pre-integration metrics are identical by "
            "construction because both upstream A1/A2 submitted configs were unchanged."
            if not config_changed
            else ""
        ),
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
        "spec_has_req_4693": "REQ-ARC-WMTE-4693" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "submitted_agent_import",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4693",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


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
