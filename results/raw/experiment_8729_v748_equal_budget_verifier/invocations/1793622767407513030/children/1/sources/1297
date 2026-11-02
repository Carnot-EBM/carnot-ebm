"""Experiment 4909: final ARC-AGI-3 operator submission-package hardening.

Spec refs: REQ-CAPSTONE-4909, SCENARIO-CAPSTONE-4909,
SCENARIO-CAPSTONE-4909-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4909-FIELD-PRINCIPLES.
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

from carnot import experiment_4898_submission_package_harden as previous_ready


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
VramEstimator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
StackLoadChecker = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
RequirementsCrosschecker = Callable[..., Mapping[str, Any]]

REPO_ROOT = previous_ready.REPO_ROOT
PACKAGE_CORE = previous_ready.PACKAGE_CORE
EXPERIMENT = "experiment_4909_submission_package_harden"
SCHEMA = "carnot.exp4909.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4909_submission_package_harden.json"
PRIOR_READY_RELATIVE_PATH = previous_ready.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = previous_ready.SPEC_RELATIVE_PATH
REQUIREMENTS_RELATIVE_PATH = previous_ready.REQUIREMENTS_RELATIVE_PATH
RANDOM_SEED = 4909
INFERENCE_SUBSTRATE = "live_llm_inference"
LIVE_LLM_DURATION_FLOOR_S = 60.0
VRAM_LIMIT_GB = previous_ready.VRAM_LIMIT_GB

SPEC_REFS = [
    "REQ-CAPSTONE-4909",
    "SCENARIO-CAPSTONE-4909",
    "SCENARIO-CAPSTONE-4909-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4909-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "blocked_", "not_ready_")
READY_VERDICT = "success_submission_package_ready_final_pre_deadline"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success_submission_package_ready_final_pre_deadline."
    },
    "submission_package_ready": {
        "principle": (
            "true iff the package is ready for the OPERATOR to submit; this task never "
            "submits (Operator-Only External Publication)."
        )
    },
    "peak_vram_gb": {
        "principle": (
            "the frozen Qwen3.5-9B-MTP stack peak VRAM; must be < ~16GB "
            "(Kaggle constraint)."
        )
    },
    "frozen_stack_loads": {
        "principle": (
            "true iff the iGPU MTP stack loads -- the live submission generator "
            "(NOT the 3090s)."
        )
    },
    "operator_checklist": {
        "principle": (
            "the steps the OPERATOR performs to submit (this task stops at package-ready)."
        )
    },
    "submits": {
        "principle": "MUST be false -- external submission is operator-only."
    },
    "inference_substrate": {
        "principle": "live_llm_inference (the frozen GGUF load check on the iGPU; 60s floor)."
    },
    "preconditions_checked": {
        "principle": "records package/GGUF/iGPU checks; a missing resource emits blocked_."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "result_path",
    "submitted_to_leaderboard",
    "operator_only",
    "package_builds",
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
    package_build_path_present = (
        (kernel_dir / PACKAGE_CORE.KERNEL_MAIN).exists()
        and (kernel_dir / PACKAGE_CORE.KERNEL_METADATA).exists()
    )
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4909": "REQ-CAPSTONE-4909" in spec_text,
        "packaging_requirements_doc_present": (root_path / REQUIREMENTS_RELATIVE_PATH).exists(),
        "prior_4898_ready_package_present": (root_path / PRIOR_READY_RELATIVE_PATH).exists(),
        "package_build_path_present": package_build_path_present,
        "submission_packaging_scripts_present": package_build_path_present,
        "arc_competition_agent_present": (root_path / PACKAGE_CORE.AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4909",
        "packaging_requirements_doc_present",
        "prior_4898_ready_package_present",
        "package_build_path_present",
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
    payload = dict(previous_ready.blocked_package_builds_payload(blocked_resource))
    payload["blocked_resource"] = blocked_resource
    return payload


def blocked_requirements_payload(blocked_resource: str = "packaging_scripts_missing") -> JsonDict:
    return previous_ready.blocked_requirements_payload(blocked_resource)


def blocked_vram_payload(blocked_resource: str) -> JsonDict:
    return previous_ready.blocked_vram_payload(blocked_resource)


def blocked_stack_load_payload(blocked_resource: str) -> JsonDict:
    return {
        "ok": False,
        "frozen_stack_loads": False,
        "blocked_resource": blocked_resource,
        "peak_vram_gb": 0.0,
        "fits_16gb": False,
        "limit_gb": VRAM_LIMIT_GB,
        "server_reachable": False,
        "igpu_hip_server_present": False,
        "igpu_hip_server_path": "",
        "generator_backend": None,
        "uses_3090": False,
        "mtp": True,
        "kv_quant": "q8_0",
        "n_predict_min": 2048,
        "no_think_prefix": True,
        "measurement_source": "blocked",
    }


def _candidate_igpu_hip_server() -> Path:
    return Path.home() / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server"


def _load_frozen_stack_live(
    model_path_resolution: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live model boundary.
    import os

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    gguf_path = Path(str(_mapping(model_path_resolution.get("gguf")).get("path") or ""))
    server_path = _candidate_igpu_hip_server()
    base: JsonDict = {
        "igpu_hip_server_path": str(server_path),
        "generator_backend": "igpu_hip",
        "uses_3090": False,
        "mtp": bool(agent_config_resolution.get("mtp", True)),
        "kv_quant": str(agent_config_resolution.get("kv_quant") or "q8_0"),
        "n_predict_min": int(agent_config_resolution.get("n_predict_min") or 2048),
        "no_think_prefix": True,
        "limit_gb": VRAM_LIMIT_GB,
        "peak_vram_gb": float(vram_breakdown.get("vram_estimate_gb") or 0.0),
        "measurement_source": "runtime_vram_estimate_after_live_load",
    }
    if not gguf_path.exists():
        return {
            **base,
            "ok": False,
            "frozen_stack_loads": False,
            "blocked_resource": "model_paths",
            "fits_16gb": False,
            "server_reachable": False,
            "igpu_hip_server_present": server_path.exists(),
        }
    if not server_path.exists():
        return {
            **base,
            "ok": False,
            "frozen_stack_loads": False,
            "blocked_resource": "igpu_hip_server",
            "fits_16gb": False,
            "server_reachable": False,
            "igpu_hip_server_present": False,
        }

    old_server = os.environ.get("CARNOT_LLAMA_SERVER")
    os.environ["CARNOT_LLAMA_SERVER"] = str(server_path)
    proposer: Any | None = None
    try:
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP",
            model_path=str(gguf_path),
            mtp=True,
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=2560,
            n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
            timeout=int(os.environ.get("CARNOT_ARC_4909_LLM_TIMEOUT", "300")),
            tries=1,
            port=int(os.environ.get("CARNOT_ARC_4909_LLM_PORT", "8949")),
        )
        ok = bool(proposer._ensure_server())
    finally:
        proc = getattr(proposer, "_proc", None) if proposer is not None else None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        if old_server is None:
            os.environ.pop("CARNOT_LLAMA_SERVER", None)
        else:
            os.environ["CARNOT_LLAMA_SERVER"] = old_server

    peak = float(base["peak_vram_gb"])
    return {
        **base,
        "ok": ok and peak < VRAM_LIMIT_GB,
        "frozen_stack_loads": ok,
        "blocked_resource": "" if ok else "igpu_hip_server",
        "fits_16gb": ok and peak < VRAM_LIMIT_GB,
        "server_reachable": ok,
        "igpu_hip_server_present": True,
    }


def load_frozen_stack(
    model_path_resolution: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live model boundary.
    return _load_frozen_stack_live(model_path_resolution, agent_config_resolution, vram_breakdown)


def _augment_preconditions(
    preconditions_checked: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
) -> JsonDict:
    out = dict(preconditions_checked)
    gguf = _mapping(model_path_resolution.get("gguf"))
    server = _mapping(model_path_resolution.get("llama_server"))
    out["frozen_gguf_present"] = gguf.get("present") is True
    out["kaggle_cuda_server_present"] = server.get("present") is True
    out["igpu_hip_server_present"] = frozen_stack_load_check.get("igpu_hip_server_present") is True
    out["igpu_hip_server_reachable"] = frozen_stack_load_check.get("server_reachable") is True
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4909",
        "packaging_requirements_doc_present",
        "prior_4898_ready_package_present",
        "package_build_path_present",
        "submission_packaging_scripts_present",
        "arc_competition_agent_present",
        "frozen_gguf_present",
        "kaggle_cuda_server_present",
        "igpu_hip_server_present",
        "igpu_hip_server_reachable",
    )
    missing = [key for key in required if out.get(key) is not True]
    out["ok"] = not missing
    if missing:
        out["blocked_resource"] = out.get("blocked_resource") or missing[0]
    else:
        out.pop("blocked_resource", None)
    return out


def _diff_value(prior_value: Any, current_value: Any) -> Any:
    if prior_value == current_value:
        return "unchanged"
    return {"prior": prior_value, "current": current_value}


def _peak_vram_gb(
    frozen_stack_load_check: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> float:
    return float(
        frozen_stack_load_check.get("peak_vram_gb")
        or vram_breakdown.get("vram_estimate_gb")
        or 0.0
    )


def diff_against_ready_package(
    prior_ready_artifact: Mapping[str, Any],
    *,
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
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
    current_peak = _peak_vram_gb(frozen_stack_load_check, vram_breakdown)
    prior_peak = float(prior.get("peak_vram_gb") or prior_vram)

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
        "frozen_stack_still_loads": bool(
            frozen_stack_load_check.get("ok") is True
            and frozen_stack_load_check.get("frozen_stack_loads") is True
            and frozen_stack_load_check.get("uses_3090") is not True
        ),
        "peak_vram_still_fits_16gb": bool(
            frozen_stack_load_check.get("fits_16gb") is True and current_peak < VRAM_LIMIT_GB
        ),
        "submits_still_false": bool(
            package_builds.get("submitted_to_leaderboard") is False
            and prior.get("submits", False) is False
        ),
    }
    regressions = [key for key, passed in checks.items() if not passed]
    return {
        "prior_ready_artifact_path": PRIOR_READY_RELATIVE_PATH,
        "prior_experiment": prior.get("experiment"),
        "prior_artifact_present": checks["prior_artifact_present"],
        "prior_submission_package_ready": checks["prior_submission_package_ready"],
        "prior_submitted_to_leaderboard": prior.get("submitted_to_leaderboard"),
        "prior_submits": prior.get("submits", False),
        "prior_vram_estimate_gb": prior_vram,
        "prior_peak_vram_gb": prior_peak,
        "current_vram_estimate_gb": current_vram,
        "current_peak_vram_gb": current_peak,
        "vram_delta_gb": round(current_vram - prior_vram, 3),
        "peak_vram_delta_gb": round(current_peak - prior_peak, 3),
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
            "peak_vram_gb": _diff_value(prior_peak, current_peak),
        },
    }


def _packaging_scripts_missing(preconditions_checked: Mapping[str, Any]) -> bool:
    return bool(
        preconditions_checked.get("submission_packaging_scripts_present") is not True
        or preconditions_checked.get("package_build_path_present") is not True
        or preconditions_checked.get("arc_competition_agent_present") is not True
    )


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
) -> bool:
    peak = _peak_vram_gb(frozen_stack_load_check, vram_breakdown)
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
        and frozen_stack_load_check.get("ok") is True
        and frozen_stack_load_check.get("frozen_stack_loads") is True
        and frozen_stack_load_check.get("uses_3090") is not True
        and frozen_stack_load_check.get("fits_16gb") is True
        and peak < VRAM_LIMIT_GB
        and packaging_requirements_crosscheck.get("ok") is True
        and ready_package_regression_check.get("ok") is True
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    frozen_stack_load_check: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
    ready_package_regression_check: Mapping[str, Any],
) -> str:
    core_resource = str(preconditions_checked.get("blocked_resource") or "")
    if _packaging_scripts_missing(preconditions_checked):
        return "packaging_scripts_missing"
    if core_resource in {
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4909",
        "packaging_requirements_doc_present",
        "prior_4898_ready_package_present",
    }:
        return core_resource
    if model_path_resolution.get("resolved") is not True:
        return "model_paths"
    stack_resource = str(frozen_stack_load_check.get("blocked_resource") or "")
    if stack_resource in {"igpu_hip_server", "model_paths"}:
        return stack_resource
    if preconditions_checked.get("ok") is not True and core_resource:
        return core_resource
    if package_builds.get("package_builds") is not True:
        return str(package_builds.get("blocked_resource") or "dry_build")
    if agent_config_resolution.get("resolved") is not True:
        return "agent_config"
    if (
        vram_breakdown.get("fits_16gb") is not True
        or float(vram_breakdown.get("vram_estimate_gb") or 0.0) >= VRAM_LIMIT_GB
    ):
        return "vram"
    if frozen_stack_load_check.get("frozen_stack_loads") is not True:
        return str(frozen_stack_load_check.get("blocked_resource") or "frozen_stack_load")
    if _peak_vram_gb(frozen_stack_load_check, vram_breakdown) >= VRAM_LIMIT_GB:
        return "peak_vram"
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
        "spec_has_req_4909",
        "packaging_requirements_doc_present",
        "prior_4898_ready_package_present",
        "packaging_scripts_missing",
        "model_paths",
        "igpu_hip_server",
        "frozen_gguf_present",
        "kaggle_cuda_server_present",
    }


def _operator_checklist(
    package_ready: bool,
    peak_vram_gb: float,
    ready_package_regression_check: Mapping[str, Any],
) -> list[str]:
    readiness = (
        "ready"
        if package_ready
        else f"blocked until this JSON reports {READY_VERDICT}"
    )
    regression_state = "passed" if ready_package_regression_check.get("ok") is True else "blocked"
    return [
        (
            "OPERATOR-CHECK: 6/30 final pre-deadline gate: confirm "
            f"{RESULT_RELATIVE_PATH} is {readiness}, submits=false, "
            "and operator_only=true."
        ),
        (
            "OPERATOR-CHECK: Confirm the .451 ready-package regression check is "
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
            "OPERATOR-CHECK: Verify the frozen iGPU HIP live-stack load used draft-MTP, "
            "q8_0 KV, /no_think, n_predict>=2048, and never targeted the 3090s."
        ),
        (
            "OPERATOR-CHECK: Verify frozen stack peak VRAM "
            f"{peak_vram_gb:.3f}GB is below the ~16GB Kaggle budget."
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
    package_builds: Mapping[str, Any],
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
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        frozen_stack_load_check,
        packaging_requirements_crosscheck,
        ready_package_regression_check,
    )
    blocked_resource = "" if ready else _blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        frozen_stack_load_check,
        packaging_requirements_crosscheck,
        ready_package_regression_check,
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
        "peak_vram_gb": peak,
        "frozen_stack_loads": frozen_stack_load_check.get("frozen_stack_loads") is True,
        "operator_checklist": _operator_checklist(ready, peak, ready_package_regression_check),
        "submits": False,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "package_builds": dict(package_builds),
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
        "duration_s": max(LIVE_LLM_DURATION_FLOOR_S, round(float(duration_s), 6)),
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
    if float(artifact.get("duration_s") or 0.0) < LIVE_LLM_DURATION_FLOOR_S:
        errors.append("duration_s_live_floor")
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
    checklist = artifact.get("operator_checklist")
    if not (
        isinstance(checklist, list)
        and checklist
        and all(isinstance(step, str) and step.startswith("OPERATOR-CHECK:") for step in checklist)
    ):
        errors.append("operator_checklist")
    expected_ready = _package_ready(
        _mapping(artifact.get("preconditions_checked")),
        _mapping(artifact.get("package_builds")),
        _mapping(artifact.get("agent_config_resolution")),
        _mapping(artifact.get("model_path_resolution")),
        _mapping(artifact.get("vram_breakdown")),
        _mapping(artifact.get("frozen_stack_load_check")),
        _mapping(artifact.get("packaging_requirements_crosscheck")),
        _mapping(artifact.get("ready_package_regression_check")),
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    if _mapping(artifact.get("package_builds")).get("submitted_to_leaderboard") is not False:
        errors.append("package_builds_submitted_to_leaderboard")
    expected_peak = _peak_vram_gb(
        _mapping(artifact.get("frozen_stack_load_check")),
        _mapping(artifact.get("vram_breakdown")),
    )
    if artifact.get("peak_vram_gb") != expected_peak:
        errors.append("peak_vram_gb")
    expected_loads = _mapping(artifact.get("frozen_stack_load_check")).get(
        "frozen_stack_loads"
    ) is True
    if artifact.get("frozen_stack_loads") is not expected_loads:
        errors.append("frozen_stack_loads")
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
    stack_load_checker: StackLoadChecker = load_frozen_stack,
    requirements_crosschecker: RequirementsCrosschecker = cross_check_packaging_requirements,
    write: bool = True,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions = dict(preconditions_checker(root_path))
    if preconditions.get("ok") is not True and _packaging_scripts_missing(preconditions):
        blocked = "packaging_scripts_missing"
        package_builds = blocked_package_builds_payload(blocked)
        agent_config: JsonDict = {}
        model_paths: JsonDict = {}
        vram = blocked_vram_payload(blocked)
        stack_load = blocked_stack_load_payload(blocked)
        requirements = blocked_requirements_payload(blocked)
    else:
        package_builds = dict(package_builder(root_path))
        agent_config = dict(agent_config_resolver())
        model_paths = dict(model_path_resolver())
        if model_paths.get("resolved") is True:
            vram = dict(vram_estimator(model_paths, agent_config))
            stack_load = dict(stack_load_checker(model_paths, agent_config, vram))
        else:
            vram = blocked_vram_payload("model_paths")
            stack_load = blocked_stack_load_payload("model_paths")
        requirements = dict(
            requirements_crosschecker(
                root_path,
                package_builds=package_builds,
                agent_config_resolution=agent_config,
                model_path_resolution=model_paths,
            )
        )
    preconditions = _augment_preconditions(preconditions, model_paths, stack_load)
    prior_artifact = read_prior_ready_package(root_path)
    regression = diff_against_ready_package(
        prior_artifact,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        frozen_stack_load_check=stack_load,
        packaging_requirements_crosscheck=requirements,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_builds=package_builds,
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
