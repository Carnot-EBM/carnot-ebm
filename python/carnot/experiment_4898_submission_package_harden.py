"""Experiment 4898: ARC-AGI-3 final reserved-infra package hardening.

Spec refs: REQ-CAPSTONE-4898, SCENARIO-CAPSTONE-4898,
SCENARIO-CAPSTONE-4898-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4898-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(_PYTHON_ROOT))  # pragma: no cover - direct script guard.

from carnot import experiment_4888_submission_package_harden as previous_ready


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
VramEstimator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
RequirementsCrosschecker = Callable[..., Mapping[str, Any]]

REPO_ROOT = previous_ready.REPO_ROOT
PACKAGE_CORE = previous_ready.PACKAGE_CORE
EXPERIMENT = "experiment_4898_submission_package_harden"
SCHEMA = "carnot.exp4898.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4898_submission_package_harden.json"
PRIOR_READY_RELATIVE_PATH = previous_ready.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = previous_ready.SPEC_RELATIVE_PATH
REQUIREMENTS_RELATIVE_PATH = previous_ready.REQUIREMENTS_RELATIVE_PATH
RANDOM_SEED = 4898
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VRAM_LIMIT_GB = previous_ready.VRAM_LIMIT_GB

SPEC_REFS = [
    "REQ-CAPSTONE-4898",
    "SCENARIO-CAPSTONE-4898",
    "SCENARIO-CAPSTONE-4898-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4898-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "blocked_", "not_ready_")
READY_VERDICT = "success_submission_package_ready_final_pre_deadline"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; ready is success_submission_package_ready_final_pre_deadline."
    },
    "submission_package_ready": {
        "principle": (
            "true iff the package builds + config/model resolve + vram < 16GB + no "
            "regression vs .450 (the operator-submittable state)."
        )
    },
    "submitted_to_leaderboard": {
        "principle": (
            "MUST be false -- submission is OPERATOR-ONLY (CLAUDE.md Operator-Only "
            "External Publication); this task never submits."
        )
    },
    "operator_only": {
        "principle": (
            "true -- the task ends at a ready package + checklist; the operator performs "
            "the submission step."
        )
    },
    "vram_estimate_gb": {
        "principle": "the frozen iGPU live-stack VRAM under the ~16GB Kaggle budget."
    },
    "package_builds": {
        "principle": "the Kaggle ARC-AGI-3 package dry-build succeeded."
    },
    "operator_submission_checklist": {
        "principle": (
            "the final pre-deadline (6/30) operator action list -- the deliverable for "
            "the human submit step."
        )
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "result_path",
    "preconditions_checked",
    "agent_config_resolution",
    "model_path_resolution",
    "vram_breakdown",
    "packaging_requirements_crosscheck",
    "ready_package_regression_check",
    "blocked_resource",
    "operator_only",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

REQUIREMENTS_CHECK_KEYS = previous_ready.REQUIREMENTS_CHECK_KEYS
dry_build_package = previous_ready.dry_build_package
resolve_model_paths = previous_ready.resolve_model_paths
resolve_agent_config = previous_ready.resolve_agent_config
runtime_vram_estimate = previous_ready.runtime_vram_estimate
cross_check_packaging_requirements = previous_ready.cross_check_packaging_requirements
payload_checksum = previous_ready.payload_checksum


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_text = _text(root_path / SPEC_RELATIVE_PATH)
    kernel_dir = root_path / PACKAGE_CORE.KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4898": "REQ-CAPSTONE-4898" in spec_text,
        "packaging_requirements_doc_present": (root_path / REQUIREMENTS_RELATIVE_PATH).exists(),
        "prior_4888_ready_package_present": (root_path / PRIOR_READY_RELATIVE_PATH).exists(),
        "submission_packaging_scripts_present": (
            (kernel_dir / PACKAGE_CORE.KERNEL_MAIN).exists()
            and (kernel_dir / PACKAGE_CORE.KERNEL_METADATA).exists()
        ),
        "arc_competition_agent_present": (root_path / PACKAGE_CORE.AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4898",
        "packaging_requirements_doc_present",
        "prior_4888_ready_package_present",
        "submission_packaging_scripts_present",
        "arc_competition_agent_present",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if checks["ok"] is not True:
        checks["blocked_resource"] = next(key for key in required if not checks.get(key))
    return checks


def _runtime_agent_config() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return resolve_agent_config(SUBMITTED_AGENT_CONFIG)


def read_prior_ready_package(root: Path | str = REPO_ROOT) -> JsonDict:
    return _read_json_object(Path(root) / PRIOR_READY_RELATIVE_PATH)


def blocked_package_builds_payload(blocked_resource: str = "packaging_scripts_missing") -> JsonDict:
    payload = dict(previous_ready.blocked_package_builds_payload())
    payload["blocked_resource"] = blocked_resource
    return payload


def blocked_requirements_payload(blocked_resource: str = "packaging_scripts_missing") -> JsonDict:
    return {
        "requirements_doc_path": REQUIREMENTS_RELATIVE_PATH,
        "doc_present": False,
        "ok": False,
        "blocked_resource": blocked_resource,
        "checks": {key: False for key in REQUIREMENTS_CHECK_KEYS},
        "notes": ["requirements package cross-check blocked"],
    }


def blocked_vram_payload(blocked_resource: str) -> JsonDict:
    return {
        "vram_estimate_gb": 0.0,
        "fits_16gb": False,
        "limit_gb": VRAM_LIMIT_GB,
        "blocked_resource": blocked_resource,
    }


def _diff_value(prior_value: Any, current_value: Any) -> Any:
    if prior_value == current_value:
        return "unchanged"
    return {"prior": prior_value, "current": current_value}


def diff_against_ready_package(
    prior_ready_artifact: Mapping[str, Any],
    *,
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
) -> JsonDict:
    prior = _mapping(prior_ready_artifact)
    prior_package = _mapping(prior.get("package_builds"))
    prior_config = _mapping(prior.get("agent_config_resolution"))
    prior_paths = _mapping(prior.get("model_path_resolution"))
    prior_gguf = _mapping(prior_paths.get("gguf"))
    current_gguf = _mapping(model_path_resolution.get("gguf"))
    prior_server = _mapping(prior_paths.get("llama_server"))
    current_server = _mapping(model_path_resolution.get("llama_server"))
    current_vram = float(vram_breakdown.get("vram_estimate_gb") or 0.0)
    prior_vram = float(prior.get("vram_estimate_gb") or 0.0)

    checks = {
        "prior_artifact_present": bool(prior),
        "prior_submission_package_ready": prior.get("submission_package_ready") is True,
        "prior_submitted_to_leaderboard_false": prior.get("submitted_to_leaderboard") is False,
        "package_still_builds": bool(
            package_builds.get("package_builds") is True
            and package_builds.get("entrypoint_compiles") is True
            and package_builds.get("manifest_present") is True
            and package_builds.get("kernel_main_present") is True
        ),
        "frozen_config_still_resolves": agent_config_resolution.get("resolved") is True,
        "model_paths_still_resolve": model_path_resolution.get("resolved") is True,
        "requirements_still_pass": packaging_requirements_crosscheck.get("ok") is True,
        "vram_still_fits_16gb": bool(
            vram_breakdown.get("fits_16gb") is True and current_vram < VRAM_LIMIT_GB
        ),
        "submitted_still_false": package_builds.get("submitted_to_leaderboard") is False,
    }
    regressions = [key for key, passed in checks.items() if not passed]
    return {
        "prior_ready_artifact_path": PRIOR_READY_RELATIVE_PATH,
        "prior_experiment": prior.get("experiment"),
        "prior_artifact_present": checks["prior_artifact_present"],
        "prior_submission_package_ready": checks["prior_submission_package_ready"],
        "prior_submitted_to_leaderboard": prior.get("submitted_to_leaderboard"),
        "prior_vram_estimate_gb": prior_vram,
        "current_vram_estimate_gb": current_vram,
        "vram_delta_gb": round(current_vram - prior_vram, 3),
        "checks": checks,
        "regressions": regressions,
        "ok": not regressions,
        "diff": {
            "package_sha256": _diff_value(
                prior_package.get("package_sha256"),
                package_builds.get("package_sha256"),
            ),
            "model_id": _diff_value(
                prior_config.get("model_id"),
                agent_config_resolution.get("model_id"),
            ),
            "gguf_path": _diff_value(prior_gguf.get("path"), current_gguf.get("path")),
            "llama_server_path": _diff_value(
                prior_server.get("path"),
                current_server.get("path"),
            ),
            "vram_estimate_gb": _diff_value(prior_vram, current_vram),
        },
    }


def _packaging_scripts_missing(preconditions_checked: Mapping[str, Any]) -> bool:
    return bool(
        preconditions_checked.get("submission_packaging_scripts_present") is not True
        or preconditions_checked.get("arc_competition_agent_present") is not True
    )


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
) -> bool:
    return bool(
        preconditions_checked.get("ok") is True
        and package_builds.get("package_builds") is True
        and package_builds.get("entrypoint_compiles") is True
        and package_builds.get("manifest_present") is True
        and package_builds.get("kernel_main_present") is True
        and package_builds.get("submitted_to_leaderboard") is False
        and agent_config_resolution.get("resolved") is True
        and model_path_resolution.get("resolved") is True
        and vram_breakdown.get("fits_16gb") is True
        and float(vram_breakdown.get("vram_estimate_gb") or 0.0) < VRAM_LIMIT_GB
        and packaging_requirements_crosscheck.get("ok") is True
        and ready_package_regression_check.get("ok") is True
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("ok") is not True:
        if _packaging_scripts_missing(preconditions_checked):
            return "packaging_scripts_missing"
        return str(preconditions_checked.get("blocked_resource") or "precondition")
    if model_path_resolution.get("resolved") is not True:
        return "model_paths"
    if package_builds.get("package_builds") is not True:
        return str(package_builds.get("blocked_resource") or "dry_build")
    if agent_config_resolution.get("resolved") is not True:
        return "agent_config"
    if (
        vram_breakdown.get("fits_16gb") is not True
        or float(vram_breakdown.get("vram_estimate_gb") or 0.0) >= VRAM_LIMIT_GB
    ):
        return "vram"
    if packaging_requirements_crosscheck.get("ok") is not True:
        return "packaging_requirements"
    if ready_package_regression_check.get("ok") is not True:
        regressions = ready_package_regression_check.get("regressions") or ["regression"]
        return f"ready_package_regression_{regressions[0]}"
    return "unknown"


def _blocked_verdict_for_resource(resource: str) -> bool:
    return resource in {
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4898",
        "packaging_requirements_doc_present",
        "prior_4888_ready_package_present",
        "packaging_scripts_missing",
        "model_paths",
    }


def _operator_submission_checklist(
    package_ready: bool,
    vram_estimate_gb: float,
    ready_package_regression_check: Mapping[str, Any],
) -> list[str]:
    readiness = "ready" if package_ready else f"blocked until this JSON reports {READY_VERDICT}"
    regression_state = "passed" if ready_package_regression_check.get("ok") is True else "blocked"
    return [
        (
            "OPERATOR-CHECK: 6/30 final pre-deadline gate: confirm "
            f"{RESULT_RELATIVE_PATH} is {readiness}, submitted_to_leaderboard=false, "
            "and operator_only=true."
        ),
        (
            "OPERATOR-CHECK: Confirm the .450 ready-package regression check is "
            f"{regression_state} against {PRIOR_READY_RELATIVE_PATH}."
        ),
        (
            "OPERATOR-CHECK: Confirm the package dry-build compiled main.py and produced "
            "stable hashes for kernel-metadata.json and main.py."
        ),
        (
            "OPERATOR-CHECK: Confirm the requirements-doc package cross-check passed "
            f"against {REQUIREMENTS_RELATIVE_PATH}."
        ),
        (
            "OPERATOR-CHECK: Confirm Qwen3.5-9B-MTP GGUF resolves through "
            "CARNOT_ARC_GGUF_PATH and CUDA llama-server resolves through CARNOT_LLAMA_SERVER."
        ),
        (
            "OPERATOR-CHECK: Verify frozen iGPU live-stack VRAM estimate "
            f"{vram_estimate_gb:.3f}GB is below the ~16GB Kaggle budget with q8_0 KV headroom."
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
            "OPERATOR-CHECK: Inspect rerun logs for Qwen3.5-9B-MTP, iGPU/L4-pinned "
            "Kaggle-parity generator posture, q8_0 KV, CARNOT_ARC_MTP posture, and no hosted API."
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
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    ready = _package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        packaging_requirements_crosscheck,
        ready_package_regression_check,
    )
    blocked_resource = "" if ready else _blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        packaging_requirements_crosscheck,
        ready_package_regression_check,
    )
    if ready:
        verdict = READY_VERDICT
    elif _blocked_verdict_for_resource(blocked_resource):
        verdict = f"blocked_{blocked_resource}"
    else:
        verdict = f"not_ready_{blocked_resource}"
    vram_estimate_gb = float(vram_breakdown.get("vram_estimate_gb") or 0.0)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "submission_package_ready": ready,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": vram_estimate_gb,
        "package_builds": dict(package_builds),
        "operator_submission_checklist": _operator_submission_checklist(
            ready,
            vram_estimate_gb,
            ready_package_regression_check,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "agent_config_resolution": dict(agent_config_resolution),
        "model_path_resolution": dict(model_path_resolution),
        "vram_breakdown": dict(vram_breakdown),
        "packaging_requirements_crosscheck": dict(packaging_requirements_crosscheck),
        "ready_package_regression_check": dict(ready_package_regression_check),
        "blocked_resource": blocked_resource,
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
    if artifact.get("submission_package_ready") is True and verdict != READY_VERDICT:
        errors.append("honest_verdict_ready")
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
    checklist = artifact.get("operator_submission_checklist")
    if not (
        isinstance(checklist, list)
        and checklist
        and all(isinstance(step, str) and step.startswith("OPERATOR-CHECK:") for step in checklist)
    ):
        errors.append("operator_submission_checklist")
    expected_ready = _package_ready(
        _mapping(artifact.get("preconditions_checked")),
        _mapping(artifact.get("package_builds")),
        _mapping(artifact.get("agent_config_resolution")),
        _mapping(artifact.get("model_path_resolution")),
        _mapping(artifact.get("vram_breakdown")),
        _mapping(artifact.get("packaging_requirements_crosscheck")),
        _mapping(artifact.get("ready_package_regression_check")),
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    if _mapping(artifact.get("package_builds")).get("submitted_to_leaderboard") is not False:
        errors.append("package_builds_submitted_to_leaderboard")
    expected_vram = float(_mapping(artifact.get("vram_breakdown")).get("vram_estimate_gb") or 0.0)
    if artifact.get("vram_estimate_gb") != expected_vram:
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
    vram_estimator: VramEstimator = runtime_vram_estimate,
    requirements_crosschecker: RequirementsCrosschecker = cross_check_packaging_requirements,
    write: bool = True,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions = dict(preconditions_checker(root_path))
    if preconditions.get("ok") is not True:
        blocked = (
            "packaging_scripts_missing"
            if _packaging_scripts_missing(preconditions)
            else str(preconditions.get("blocked_resource") or "precondition")
        )
        package_builds = blocked_package_builds_payload(blocked)
        agent_config: JsonDict = {}
        model_paths: JsonDict = {}
        vram = blocked_vram_payload(blocked)
        requirements = blocked_requirements_payload(blocked)
    else:
        package_builds = dict(package_builder(root_path))
        agent_config = dict(agent_config_resolver())
        model_paths = dict(model_path_resolver())
        if model_paths.get("resolved") is True:
            vram = dict(vram_estimator(model_paths, agent_config))
        else:
            vram = blocked_vram_payload("model_paths")
        requirements = dict(
            requirements_crosschecker(
                root_path,
                package_builds=package_builds,
                agent_config_resolution=agent_config,
                model_path_resolution=model_paths,
            )
        )
    prior_artifact = read_prior_ready_package(root_path)
    regression = diff_against_ready_package(
        prior_artifact,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        packaging_requirements_crosscheck=requirements,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        packaging_requirements_crosscheck=requirements,
        ready_package_regression_check=regression,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; schema is tested directly.
        raise ValueError(f"invalid {EXPERIMENT} artifact: {errors}")
    if write:
        PACKAGE_CORE._write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
