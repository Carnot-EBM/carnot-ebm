"""Experiment 4776: ARC-AGI-3 Kaggle submission-package hardening.

Spec refs: REQ-CAPSTONE-4776, SCENARIO-CAPSTONE-4776,
SCENARIO-CAPSTONE-4776-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4776-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4766_submission_package_harden as base

# The canonical generator pin, imported rather than assumed. The 2026-07-28 gemma migration
# introduced these names into this module's OPERATOR-CHECK strings without importing them, so
# `build_artifact()` raised NameError on a real code path -- the readiness capstone could not run
# at all. `ast.parse`/ruff-format are both blind to this; only executing it (or F821) finds it.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
VramEstimator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

REPO_ROOT = base.REPO_ROOT
EXPERIMENT = "experiment_4776_submission_package_harden"
SCHEMA = "carnot.exp4776.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4776_submission_package_harden.json"
SPEC_RELATIVE_PATH = base.SPEC_RELATIVE_PATH
RANDOM_SEED = 4776
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts; 0.0001s floor."
VRAM_LIMIT_GB = base.VRAM_LIMIT_GB
SPEC_REFS = [
    "REQ-CAPSTONE-4776",
    "SCENARIO-CAPSTONE-4776",
    "SCENARIO-CAPSTONE-4776-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4776-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "complete_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; package builds is success_/complete_."},
    "submission_package_ready": {
        "principle": "True iff ready for the OPERATOR to submit; the task itself NEVER submits."
    },
    "vram_estimate_gb": {
        "principle": "must fit ~16GB Kaggle with KV + headroom -- the deployment gate."
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts; 0.0001s floor."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "preconditions_checked",
    "package_builds",
    "agent_config_resolution",
    "model_path_resolution",
    "vram_breakdown",
    "operator_checklist",
    "submitted_to_leaderboard",
    "operator_only",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

dry_build_package = base.dry_build_package
resolve_agent_config = base.resolve_agent_config
resolve_model_paths = base.resolve_model_paths
estimate_vram = base.estimate_vram


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    kernel_dir = root_path / base.KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4776": "REQ-CAPSTONE-4776" in spec_text,
        "submission_kernel_present": (kernel_dir / base.KERNEL_MAIN).exists()
        and (kernel_dir / base.KERNEL_METADATA).exists(),
        "arc_competition_agent_present": (root_path / base.AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4776",
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


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> bool:
    return base._package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> str:
    return base._blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
    )


def _operator_checklist(package_ready: bool, vram_estimate_gb: float) -> list[str]:
    readiness = "ready" if package_ready else "blocked until this JSON reports success_"
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
            "carnot-gemma4-31b-it-gguf before Save & Run."
        ),
        (
            f"OPERATOR-CHECK: Verify the rerun log resolves {ARC_LIVE_GENERATOR_REPO_SUBSTR}, no draft-mtp, "
            "q8_0 KV, and the CUDA-12.8 llama-server."
        ),
        (
            "OPERATOR-CHECK: Verify VRAM estimate "
            f"{vram_estimate_gb:.3f}GB is below the ~16GB Kaggle budget with headroom."
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
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    ready = _package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
    )
    blocked = _blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
    )
    vram_estimate_gb = float(vram_breakdown.get("vram_estimate_gb") or 0.0)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
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
        "operator_checklist": _operator_checklist(ready, vram_estimate_gb),
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
    checklist = artifact.get("operator_checklist")
    if not (
        isinstance(checklist, list)
        and checklist
        and all(isinstance(step, str) and step.startswith("OPERATOR-CHECK:") for step in checklist)
    ):
        errors.append("operator_checklist")
    expected_ready = _package_ready(
        artifact.get("preconditions_checked", {}),
        artifact.get("package_builds", {}),
        artifact.get("agent_config_resolution", {}),
        artifact.get("model_path_resolution", {}),
        artifact.get("vram_breakdown", {}),
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
    agent_config_resolver: AgentConfigResolver = base._runtime_agent_config,
    model_path_resolver: ModelPathResolver = resolve_model_paths,
    vram_estimator: VramEstimator = base._runtime_vram_estimate,
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
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; schema is tested directly.
        raise ValueError(f"invalid {EXPERIMENT} artifact: {errors}")
    if write:
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
