"""Experiment 4997: FINAL pre-deadline ARC-AGI-3 operator package hardening.

Spec refs: REQ-CAPSTONE-4997, SCENARIO-CAPSTONE-4997,
SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4997-FIELD-PRINCIPLES.
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

from carnot import experiment_4986_submission_package_harden as previous_ready


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
RequirementsCrosschecker = Callable[..., Mapping[str, Any]]

REPO_ROOT = previous_ready.REPO_ROOT
PACKAGE_CORE = previous_ready.PACKAGE_CORE
EXPERIMENT = "experiment_4997_submission_package_harden"
SCHEMA = "carnot.exp4997.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4997_submission_package_harden.json"
PRIOR_READY_RELATIVE_PATH = previous_ready.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = previous_ready.SPEC_RELATIVE_PATH
REQUIREMENTS_RELATIVE_PATH = previous_ready.REQUIREMENTS_RELATIVE_PATH
RANDOM_SEED = 4997
INFERENCE_SUBSTRATE = previous_ready.INFERENCE_SUBSTRATE
AGGREGATION_DURATION_FLOOR_S = previous_ready.AGGREGATION_DURATION_FLOOR_S
VRAM_LIMIT_GB = previous_ready.VRAM_LIMIT_GB

SPEC_REFS = [
    "REQ-CAPSTONE-4997",
    "SCENARIO-CAPSTONE-4997",
    "SCENARIO-CAPSTONE-4997-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4997-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = previous_ready.TERMINAL_PREFIXES
READY_VERDICT = previous_ready.READY_VERDICT

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; ready is success_submission_package_ready_final_pre_deadline."
    },
    "submission_package_ready": {
        "principle": (
            "True iff ready for the OPERATOR to submit; the task itself NEVER submits "
            "(operator-only)."
        )
    },
    "submits": {
        "principle": (
            "false -- external publication is operator-only; the task prepares, never submits."
        )
    },
    "operator_only": {
        "principle": (
            "true -- the submission step is reserved for the operator (defense in depth: "
            "no leaderboard token used)."
        )
    },
    "peak_vram_gb": {
        "principle": ("must be < 16 (Kaggle validation env limit); 15.146 in .456/.457/.458/.459.")
    },
    "frozen_stack_loads": {
        "principle": "the frozen Qwen3.5-9B-MTP iGPU stack loads (Kaggle parity)."
    },
    "package_builds": {"principle": "the package dry-build succeeds."},
    "ready_package_regression_ok": {
        "principle": ("the ready package still passes its regression check before the deadline.")
    },
    "operator_submission_checklist": {
        "principle": "the FINAL operator-facing steps to submit (the task ends before submission)."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads the package + upstream; 0.0001s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records frozen-stack + build-input checks; a missing resource emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "result_path",
    "submitted_to_leaderboard",
    "package_build_check",
    "agent_config_resolution",
    "model_path_resolution",
    "vram_breakdown",
    "frozen_stack_load_check",
    "packaging_requirements_crosscheck",
    "ready_package_regression_check",
    "blocked_resource",
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
cross_check_packaging_requirements = previous_ready.cross_check_packaging_requirements
payload_checksum = previous_ready.payload_checksum
blocked_package_build_check = previous_ready.blocked_package_build_check
blocked_vram_breakdown = previous_ready.blocked_vram_breakdown
blocked_frozen_stack_load_check = previous_ready.blocked_frozen_stack_load_check
blocked_requirements_check = previous_ready.blocked_requirements_check
_diff_value = previous_ready._diff_value
_peak_vram_gb = previous_ready._peak_vram_gb
_package_build_check_ok = previous_ready._package_build_check_ok
_package_ready = previous_ready._package_ready
_blocked_resource = previous_ready._blocked_resource


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_prior_ready_package(root: Path | str = REPO_ROOT) -> JsonDict:
    return _read_json_object(Path(root) / PRIOR_READY_RELATIVE_PATH)


def _prior_has_frozen_stack_evidence(prior: Mapping[str, Any]) -> bool:
    stack = _mapping(prior.get("frozen_stack_load_check"))
    return bool(
        prior.get("submission_package_ready") is True
        and prior.get("submits") is False
        and prior.get("operator_only") is True
        and prior.get("frozen_stack_loads") is True
        and stack.get("frozen_stack_loads") is True
        and float(prior.get("peak_vram_gb") or stack.get("peak_vram_gb") or 0.0) < VRAM_LIMIT_GB
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_text = _text(root_path / SPEC_RELATIVE_PATH)
    kernel_dir = root_path / PACKAGE_CORE.KERNEL_RELATIVE_DIR
    prior_artifact = read_prior_ready_package(root_path)
    package_build_path_present = (kernel_dir / PACKAGE_CORE.KERNEL_MAIN).exists() and (
        kernel_dir / PACKAGE_CORE.KERNEL_METADATA
    ).exists()
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4997": "REQ-CAPSTONE-4997" in spec_text,
        "prior_4986_ready_package_present": bool(prior_artifact),
        "prior_4986_frozen_stack_evidence_present": _prior_has_frozen_stack_evidence(
            prior_artifact
        ),
        "packaging_requirements_doc_present": (root_path / REQUIREMENTS_RELATIVE_PATH).exists(),
        "package_build_path_present": package_build_path_present,
        "submission_packaging_scripts_present": package_build_path_present,
        "arc_competition_agent_present": (root_path / PACKAGE_CORE.AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4997",
        "prior_4986_ready_package_present",
        "prior_4986_frozen_stack_evidence_present",
        "packaging_requirements_doc_present",
        "package_build_path_present",
        "submission_packaging_scripts_present",
        "arc_competition_agent_present",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if checks["ok"] is not True:
        checks["blocked_resource"] = next(key for key in required if not checks.get(key))
        if checks["blocked_resource"] == "prior_4986_ready_package_present":
            checks["blocked_resource"] = "prior_4986_ready_package"
        elif checks["blocked_resource"] == "prior_4986_frozen_stack_evidence_present":
            checks["blocked_resource"] = "frozen_stack_evidence"
    return checks


def _runtime_agent_config() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return resolve_agent_config(SUBMITTED_AGENT_CONFIG)


def _prior_vram_breakdown(prior_ready_artifact: Mapping[str, Any]) -> JsonDict:
    vram = _mapping(prior_ready_artifact.get("vram_breakdown"))
    if vram:
        return dict(vram)
    return blocked_vram_breakdown("prior_4986_ready_package")


def _prior_frozen_stack_load_check(prior_ready_artifact: Mapping[str, Any]) -> JsonDict:
    stack = _mapping(prior_ready_artifact.get("frozen_stack_load_check"))
    if stack:
        return dict(stack)
    return blocked_frozen_stack_load_check("frozen_stack_evidence")


def _augment_preconditions(
    preconditions_checked: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
) -> JsonDict:
    out = dict(preconditions_checked)
    gguf = _mapping(model_path_resolution.get("gguf"))
    server = _mapping(model_path_resolution.get("llama_server"))
    if model_path_resolution:
        out["frozen_gguf_present"] = gguf.get("present") is True
        out["kaggle_cuda_server_present"] = server.get("present") is True
        out["model_paths_resolved"] = model_path_resolution.get("resolved") is True
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4997",
        "prior_4986_ready_package_present",
        "prior_4986_frozen_stack_evidence_present",
        "packaging_requirements_doc_present",
        "package_build_path_present",
        "submission_packaging_scripts_present",
        "arc_competition_agent_present",
        "frozen_gguf_present",
        "kaggle_cuda_server_present",
        "model_paths_resolved",
    )
    missing = [key for key in required if out.get(key) is not True]
    out["ok"] = not missing
    if missing:
        resource = out.get("blocked_resource") or missing[0]
        if resource in {
            "frozen_gguf_present",
            "kaggle_cuda_server_present",
            "model_paths_resolved",
        }:
            out["blocked_resource"] = "model_paths"
        else:
            out["blocked_resource"] = resource
    else:
        out.pop("blocked_resource", None)
    return out


def confirm_ready_package_regression(
    prior_ready_artifact: Mapping[str, Any],
    *,
    package_build_check: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
) -> JsonDict:
    prior = _mapping(prior_ready_artifact)
    prior_package = _mapping(prior.get("package_build_check"))
    prior_config = _mapping(prior.get("agent_config_resolution"))
    prior_paths = _mapping(prior.get("model_path_resolution"))
    prior_gguf = _mapping(prior_paths.get("gguf"))
    current_gguf = _mapping(model_path_resolution.get("gguf"))
    prior_server = _mapping(prior_paths.get("llama_server"))
    current_server = _mapping(model_path_resolution.get("llama_server"))
    current_vram = float(vram_breakdown.get("vram_estimate_gb") or 0.0)
    prior_vram = float(_mapping(prior.get("vram_breakdown")).get("vram_estimate_gb") or 0.0)
    current_peak = _peak_vram_gb(frozen_stack_load_check, vram_breakdown)
    prior_peak = float(prior.get("peak_vram_gb") or prior_vram)

    checks = {
        "prior_artifact_present": bool(prior),
        "prior_submission_package_ready": prior.get("submission_package_ready") is True,
        "prior_submits_false": bool(
            prior.get("submits") is False
            and prior.get("submitted_to_leaderboard") is False
            and prior.get("operator_only") is True
        ),
        "package_still_builds": _package_build_check_ok(package_build_check),
        "frozen_config_still_resolves": agent_config_resolution.get("resolved") is True,
        "model_paths_still_resolve": model_path_resolution.get("resolved") is True,
        "requirements_still_pass": packaging_requirements_crosscheck.get("ok") is True,
        "vram_still_fits_16gb": bool(
            vram_breakdown.get("fits_16gb") is True and current_vram < VRAM_LIMIT_GB
        ),
        "frozen_stack_still_loads": bool(
            frozen_stack_load_check.get("ok") is True
            and frozen_stack_load_check.get("frozen_stack_loads") is True
            and frozen_stack_load_check.get("uses_3090") is not True
        ),
        "peak_vram_still_fits_16gb": bool(
            frozen_stack_load_check.get("fits_16gb") is True and current_peak < VRAM_LIMIT_GB
        ),
        "submits_still_false": bool(
            package_build_check.get("submitted_to_leaderboard") is False
            and prior.get("submits") is False
            and prior.get("submitted_to_leaderboard") is False
        ),
    }
    regressions = [key for key, passed in checks.items() if not passed]
    return {
        "prior_ready_artifact_path": PRIOR_READY_RELATIVE_PATH,
        "prior_experiment": prior.get("experiment"),
        "prior_artifact_present": checks["prior_artifact_present"],
        "prior_submission_package_ready": checks["prior_submission_package_ready"],
        "prior_submits": prior.get("submits"),
        "prior_operator_only": prior.get("operator_only"),
        "prior_peak_vram_gb": prior_peak,
        "current_peak_vram_gb": current_peak,
        "peak_vram_delta_gb": round(current_peak - prior_peak, 3),
        "checks": checks,
        "regressions": regressions,
        "ok": not regressions,
        "diff": {
            "package_sha256": _diff_value(
                prior_package.get("package_sha256"),
                package_build_check.get("package_sha256"),
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
            "peak_vram_gb": _diff_value(prior_peak, current_peak),
        },
    }


def _packaging_scripts_missing(preconditions_checked: Mapping[str, Any]) -> bool:
    return bool(
        preconditions_checked.get("submission_packaging_scripts_present") is not True
        or preconditions_checked.get("package_build_path_present") is not True
        or preconditions_checked.get("arc_competition_agent_present") is not True
    )


def _blocked_verdict_for_resource(resource: str) -> bool:
    return resource in {
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4997",
        "prior_4986_ready_package",
        "frozen_stack_evidence",
        "packaging_requirements_doc_present",
        "packaging_scripts_missing",
        "model_paths",
    }


def _operator_submission_checklist(
    package_ready: bool,
    peak_vram_gb: float,
    ready_package_regression_check: Mapping[str, Any],
) -> list[str]:
    readiness = "ready" if package_ready else f"blocked until this JSON reports {READY_VERDICT}"
    regression_state = "passed" if ready_package_regression_check.get("ok") is True else "blocked"
    return [
        (
            "OPERATOR-CHECK: FINAL pre-deadline gate: confirm "
            f"{RESULT_RELATIVE_PATH} is {readiness}, submits=false, and operator_only=true."
        ),
        (
            "OPERATOR-CHECK: Confirm the .459 ready-package regression check is "
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
            "OPERATOR-CHECK: Verify the frozen .459 load evidence reports draft-MTP, "
            "q8_0 KV, /no_think, no 3090 use, and peak VRAM "
            f"{peak_vram_gb:.3f}GB below the 16GB Kaggle budget."
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
            "OPERATOR-CHECK: Submit only through the operator-controlled Kaggle UI or API; "
            "this task never submits."
        ),
    ]


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    package_build_check: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    ready = _package_ready(
        preconditions_checked,
        package_build_check,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        frozen_stack_load_check,
        packaging_requirements_crosscheck,
        ready_package_regression_check,
    )
    blocked_resource = (
        ""
        if ready
        else _blocked_resource(
            preconditions_checked,
            package_build_check,
            agent_config_resolution,
            model_path_resolution,
            vram_breakdown,
            frozen_stack_load_check,
            packaging_requirements_crosscheck,
            ready_package_regression_check,
        )
    )
    if ready:
        verdict = READY_VERDICT
    elif _blocked_verdict_for_resource(blocked_resource):
        verdict = f"blocked_{blocked_resource}"
    else:
        verdict = f"not_ready_{blocked_resource}"
    peak = _peak_vram_gb(frozen_stack_load_check, vram_breakdown)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "submission_package_ready": ready,
        "submits": False,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "peak_vram_gb": peak,
        "frozen_stack_loads": frozen_stack_load_check.get("frozen_stack_loads") is True,
        "package_builds": _package_build_check_ok(package_build_check),
        "ready_package_regression_ok": ready_package_regression_check.get("ok") is True,
        "operator_submission_checklist": _operator_submission_checklist(
            ready,
            peak,
            ready_package_regression_check,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "package_build_check": dict(package_build_check),
        "agent_config_resolution": dict(agent_config_resolution),
        "model_path_resolution": dict(model_path_resolution),
        "vram_breakdown": dict(vram_breakdown),
        "frozen_stack_load_check": dict(frozen_stack_load_check),
        "packaging_requirements_crosscheck": dict(packaging_requirements_crosscheck),
        "ready_package_regression_check": dict(ready_package_regression_check),
        "blocked_resource": blocked_resource,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": int(random_seed),
        "duration_s": max(AGGREGATION_DURATION_FLOOR_S, round(float(duration_s), 6)),
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
    if float(artifact.get("duration_s") or 0.0) < AGGREGATION_DURATION_FLOOR_S:
        errors.append("duration_s_aggregation_floor")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submits") is not False:
        errors.append("submits")
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
        _mapping(artifact.get("package_build_check")),
        _mapping(artifact.get("agent_config_resolution")),
        _mapping(artifact.get("model_path_resolution")),
        _mapping(artifact.get("vram_breakdown")),
        _mapping(artifact.get("frozen_stack_load_check")),
        _mapping(artifact.get("packaging_requirements_crosscheck")),
        _mapping(artifact.get("ready_package_regression_check")),
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    expected_package_builds = _package_build_check_ok(_mapping(artifact.get("package_build_check")))
    if artifact.get("package_builds") is not expected_package_builds:
        errors.append("package_builds")
    if _mapping(artifact.get("package_build_check")).get("submitted_to_leaderboard") is not False:
        errors.append("package_build_check_submitted_to_leaderboard")
    expected_peak = _peak_vram_gb(
        _mapping(artifact.get("frozen_stack_load_check")),
        _mapping(artifact.get("vram_breakdown")),
    )
    if artifact.get("peak_vram_gb") != expected_peak:
        errors.append("peak_vram_gb")
    expected_loads = (
        _mapping(artifact.get("frozen_stack_load_check")).get("frozen_stack_loads") is True
    )
    if artifact.get("frozen_stack_loads") is not expected_loads:
        errors.append("frozen_stack_loads")
    expected_regression = _mapping(artifact.get("ready_package_regression_check")).get("ok") is True
    if artifact.get("ready_package_regression_ok") is not expected_regression:
        errors.append("ready_package_regression_ok")
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
    requirements_crosschecker: RequirementsCrosschecker = cross_check_packaging_requirements,
    write: bool = True,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions = dict(preconditions_checker(root_path))
    prior_artifact = read_prior_ready_package(root_path)
    if preconditions.get("ok") is not True:
        blocked = (
            "packaging_scripts_missing"
            if _packaging_scripts_missing(preconditions)
            else str(preconditions.get("blocked_resource") or "precondition")
        )
        package_build = blocked_package_build_check(blocked)
        agent_config: JsonDict = {}
        model_paths: JsonDict = {}
        vram = blocked_vram_breakdown(blocked)
        stack_load = blocked_frozen_stack_load_check(blocked)
        requirements = blocked_requirements_check(blocked)
    else:
        package_build = dict(package_builder(root_path))
        agent_config = dict(agent_config_resolver())
        model_paths = dict(model_path_resolver())
        vram = _prior_vram_breakdown(prior_artifact)
        stack_load = _prior_frozen_stack_load_check(prior_artifact)
        requirements = dict(
            requirements_crosschecker(
                root_path,
                package_builds=package_build,
                agent_config_resolution=agent_config,
                model_path_resolution=model_paths,
            )
        )
    preconditions = _augment_preconditions(preconditions, model_paths)
    regression = confirm_ready_package_regression(
        prior_artifact,
        package_build_check=package_build,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        frozen_stack_load_check=stack_load,
        packaging_requirements_crosscheck=requirements,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_build_check=package_build,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        frozen_stack_load_check=stack_load,
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
