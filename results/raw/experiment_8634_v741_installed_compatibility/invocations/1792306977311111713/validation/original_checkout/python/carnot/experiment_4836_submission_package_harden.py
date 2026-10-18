"""Experiment 4836: ARC-AGI-3 Kaggle submission-package hardening.

Spec refs: REQ-CAPSTONE-4836, SCENARIO-CAPSTONE-4836,
SCENARIO-CAPSTONE-4836-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4836-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4766_submission_package_harden as package_base


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
VramEstimator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
A1PriorResolver = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]

REPO_ROOT = package_base.REPO_ROOT
EXPERIMENT = "experiment_4836_submission_package_harden"
SCHEMA = "carnot.exp4836.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4836_submission_package_harden.json"
SPEC_RELATIVE_PATH = package_base.SPEC_RELATIVE_PATH
A1_PRIOR_RELATIVE_PATH = "results/experiment_4831_amortized_incontext_exploration_prior_live.json"
RANDOM_SEED = 4836
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VRAM_LIMIT_GB = package_base.VRAM_LIMIT_GB
A1_SUCCESS_VERDICT = "success_amortized_prior_raises_first_win_above_baseline"
SPEC_REFS = [
    "REQ-CAPSTONE-4836",
    "SCENARIO-CAPSTONE-4836",
    "SCENARIO-CAPSTONE-4836-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4836-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "complete_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; package builds is success_/complete_."},
    "submission_package_ready": {
        "principle": "True iff ready for the OPERATOR to submit; the task NEVER submits."
    },
    "vram_estimate_gb": {"principle": "must fit ~16GB Kaggle with KV + headroom."},
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "result_path",
    "preconditions_checked",
    "package_builds",
    "agent_config_resolution",
    "model_path_resolution",
    "vram_breakdown",
    "a1_prior_inclusion",
    "operator_checklist",
    "submitted_to_leaderboard",
    "operator_only",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

dry_build_package = package_base.dry_build_package
resolve_model_paths = package_base.resolve_model_paths
estimate_vram = package_base.estimate_vram
payload_checksum = package_base.payload_checksum


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - malformed external JSON.
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    kernel_dir = root_path / package_base.KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4836": "REQ-CAPSTONE-4836" in spec_text,
        "submission_kernel_present": (kernel_dir / package_base.KERNEL_MAIN).exists()
        and (kernel_dir / package_base.KERNEL_METADATA).exists(),
        "arc_competition_agent_present": (
            root_path / package_base.AGENT_RELATIVE_PATH
        ).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4836",
        "submission_kernel_present",
        "arc_competition_agent_present",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if checks["ok"] is not True:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


def resolve_agent_config(submitted_config: Mapping[str, Any]) -> JsonDict:
    config = dict(package_base.resolve_agent_config(submitted_config))
    config["amortized_first_contact_prior_enabled"] = bool(
        submitted_config.get("amortized_first_contact_prior_enabled")
    )
    config["amortized_first_contact_prior_mode"] = str(
        submitted_config.get("amortized_first_contact_prior_mode")
        or "in_context_exploration_prior"
    )
    config["go_explore_archive_enabled"] = bool(submitted_config.get("go_explore_archive_enabled"))
    config["go_explore_archive_mode"] = str(
        submitted_config.get("go_explore_archive_mode")
        or "return_then_explore_replayable_prefix_archive"
    )
    config["a1_prior_config_ready"] = bool(
        config["amortized_first_contact_prior_enabled"]
        and config["go_explore_archive_enabled"]
    )
    return config


def _runtime_agent_config() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return resolve_agent_config(SUBMITTED_AGENT_CONFIG)


def _a1_prior_passed(payload: Mapping[str, Any]) -> bool:
    archive = _mapping(payload.get("go_explore_archive_alive"))
    imitation = _mapping(payload.get("imitation_control_heldout_games"))
    ci = _mapping(payload.get("first_win_delta_ci95"))
    with_prior = _number(payload.get("first_win_rate_with_prior"))
    no_prior = _number(payload.get("first_win_rate_no_prior_ablation"))
    baseline = _number(payload.get("baseline_first_win_rate"), 0.04)
    archive_alive = bool(
        archive.get("alive", True) is not False
        and _number(archive.get("observations")) > 0.0
        and _number(archive.get("stored_cells")) > 0.0
        and _number(archive.get("prefixes_injected")) > 0.0
    )
    return bool(
        payload.get("honest_verdict") == A1_SUCCESS_VERDICT
        and payload.get("prior_changed_proposals") is True
        and archive_alive
        and with_prior > baseline
        and with_prior > no_prior
        and _number(ci.get("low")) > 0.0
        and imitation.get("heldout_not_in_distillation_set") is True
        and imitation.get("lift_holds") is True
    )


def resolve_a1_prior_inclusion(
    root: Path | str = REPO_ROOT,
    agent_config_resolution: Mapping[str, Any] | None = None,
) -> JsonDict:
    root_path = Path(root)
    agent_config = dict(agent_config_resolution or {})
    path = root_path / A1_PRIOR_RELATIVE_PATH
    if not path.exists():
        return {
            "source_artifact_path": A1_PRIOR_RELATIVE_PATH,
            "artifact_present": False,
            "passed": False,
            "included": False,
            "reason": "not_included_a1_prior_artifact_missing",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    passed = _a1_prior_passed(payload)
    config_ready = bool(agent_config.get("a1_prior_config_ready"))
    included = bool(passed and config_ready)
    if included:
        reason = "passed_and_included_in_frozen_config"
    elif passed:
        reason = "passed_but_not_in_frozen_config"
    else:
        reason = "not_included_a1_prior_did_not_pass"
    archive = _mapping(payload.get("go_explore_archive_alive"))
    imitation = _mapping(payload.get("imitation_control_heldout_games"))
    return {
        "source_artifact_path": A1_PRIOR_RELATIVE_PATH,
        "artifact_present": True,
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "passed": passed,
        "included": included,
        "reason": reason,
        "first_win_rate_with_prior": _number(payload.get("first_win_rate_with_prior")),
        "first_win_rate_no_prior_ablation": _number(
            payload.get("first_win_rate_no_prior_ablation")
        ),
        "first_win_delta_ci95": dict(_mapping(payload.get("first_win_delta_ci95"))),
        "prior_changed_proposals": payload.get("prior_changed_proposals") is True,
        "archive_alive": bool(
            archive.get("alive", True) is not False
            and _number(archive.get("observations")) > 0.0
            and _number(archive.get("stored_cells")) > 0.0
            and _number(archive.get("prefixes_injected")) > 0.0
        ),
        "imitation_lift_holds": imitation.get("lift_holds") is True,
        "frozen_config_includes_prior": config_ready,
    }


def _runtime_vram_estimate(
    paths: Mapping[str, Any],
    config: Mapping[str, Any],
) -> JsonDict:
    return package_base._runtime_vram_estimate(paths, config)


def _a1_inclusion_ready(a1_prior_inclusion: Mapping[str, Any]) -> bool:
    return bool(
        a1_prior_inclusion.get("passed") is not True
        or a1_prior_inclusion.get("included") is True
    )


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    a1_prior_inclusion: Mapping[str, Any],
) -> bool:
    return bool(
        package_base._package_ready(
            preconditions_checked,
            package_builds,
            agent_config_resolution,
            model_path_resolution,
            vram_breakdown,
        )
        and _a1_inclusion_ready(a1_prior_inclusion)
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    a1_prior_inclusion: Mapping[str, Any],
) -> str:
    if not package_base._package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
    ):
        return package_base._blocked_resource(
            preconditions_checked,
            package_builds,
            agent_config_resolution,
            model_path_resolution,
            vram_breakdown,
        )
    if not _a1_inclusion_ready(a1_prior_inclusion):
        return "a1_prior_inclusion"
    return "unknown"


def _operator_checklist(
    package_ready: bool,
    vram_estimate_gb: float,
    a1_prior_inclusion: Mapping[str, Any],
) -> list[str]:
    readiness = "ready" if package_ready else "blocked until this JSON reports success_"
    a1_reason = str(a1_prior_inclusion.get("reason") or "not_checked")
    return [
        (
            "OPERATOR-CHECK: Confirm "
            f"{RESULT_RELATIVE_PATH} is {readiness}, submitted_to_leaderboard=false, "
            "and operator_only=true."
        ),
        (
            "OPERATOR-CHECK: In Kaggle, use kernel iancblenke/carnot-arc-agi3-submission "
            "with GPU enabled, internet disabled, and the ARC-AGI-3 competition attached."
        ),
        (
            "OPERATOR-CHECK: Attach carnot-agent-code, carnot-llamacpp-mtp-binary, and "
            "carnot-qwen35-9b-mtp-gguf before Save & Run."
        ),
        (
            "OPERATOR-CHECK: Verify the rerun log resolves Qwen3.5-9B-MTP, draft-mtp, "
            "q8_0 KV, and the CUDA-12.8 llama-server."
        ),
        (
            "OPERATOR-CHECK: Verify VRAM estimate "
            f"{vram_estimate_gb:.3f}GB is below the ~16GB Kaggle budget with headroom."
        ),
        (
            "OPERATOR-CHECK: Verify A1 prior inclusion decision is "
            f"{a1_reason}; include A1 only when its own artifact passed."
        ),
        (
            "OPERATOR-CHECK: Submit only through the operator-controlled Kaggle UI or API; "
            "this task never submits."
        ),
    ]


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    a1_prior_inclusion: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    ready = _package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        a1_prior_inclusion,
    )
    blocked = _blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        a1_prior_inclusion,
    )
    vram_estimate_gb = float(vram_breakdown.get("vram_estimate_gb") or 0.0)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": (
            "success_package_builds_vram_gate_green"
            if ready
            else f"complete_package_not_ready_{blocked}"
        ),
        "submission_package_ready": ready,
        "vram_estimate_gb": vram_estimate_gb,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "package_builds": dict(package_builds),
        "agent_config_resolution": dict(agent_config_resolution),
        "model_path_resolution": dict(model_path_resolution),
        "vram_breakdown": dict(vram_breakdown),
        "a1_prior_inclusion": dict(a1_prior_inclusion),
        "operator_checklist": _operator_checklist(ready, vram_estimate_gb, a1_prior_inclusion),
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": int(random_seed),
        "duration_s": max(0.0001, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard")
    if artifact.get("operator_only") is not True:
        errors.append("operator_only")
    if artifact.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path")
    checklist = artifact.get("operator_checklist")
    if not (
        isinstance(checklist, list)
        and checklist
        and all(isinstance(step, str) and step.startswith("OPERATOR-CHECK:") for step in checklist)
    ):
        errors.append("operator_checklist")
    a1_prior_inclusion = artifact.get("a1_prior_inclusion")
    if not isinstance(a1_prior_inclusion, Mapping):
        errors.append("a1_prior_inclusion")
        a1_prior_inclusion = {}
    expected_ready = _package_ready(
        artifact.get("preconditions_checked", {}),
        artifact.get("package_builds", {}),
        artifact.get("agent_config_resolution", {}),
        artifact.get("model_path_resolution", {}),
        artifact.get("vram_breakdown", {}),
        a1_prior_inclusion,
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    if artifact.get("package_builds", {}).get("submitted_to_leaderboard") is not False:
        errors.append("package_builds_submitted_to_leaderboard")
    if artifact.get("vram_estimate_gb") != artifact.get("vram_breakdown", {}).get(
        "vram_estimate_gb"
    ):
        errors.append("vram_estimate_gb")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def run(
    root: Path | str = REPO_ROOT,
    *,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    package_builder: PackageBuilder = dry_build_package,
    agent_config_resolver: AgentConfigResolver = _runtime_agent_config,
    model_path_resolver: ModelPathResolver = resolve_model_paths,
    vram_estimator: VramEstimator = _runtime_vram_estimate,
    a1_prior_resolver: A1PriorResolver = resolve_a1_prior_inclusion,
    write: bool = True,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions = dict(preconditions_checker(root_path))
    package_builds = dict(package_builder(root_path))
    agent_config = dict(agent_config_resolver())
    model_paths = dict(model_path_resolver())
    vram = dict(vram_estimator(model_paths, agent_config))
    a1_prior = dict(a1_prior_resolver(root_path, agent_config))
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        a1_prior_inclusion=a1_prior,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; schema is tested directly.
        raise ValueError(f"invalid {EXPERIMENT} artifact: {errors}")
    if write:
        package_base._write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
