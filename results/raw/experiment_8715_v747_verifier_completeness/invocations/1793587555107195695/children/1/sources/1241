"""Experiment 4846: ARC-AGI-3 Kaggle submission-package hardening.

Spec refs: REQ-CAPSTONE-4846, SCENARIO-CAPSTONE-4846,
SCENARIO-CAPSTONE-4846-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4846-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4766_submission_package_harden as package_core
from carnot import experiment_4836_submission_package_harden as package_base

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
A1PriorResolver = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
RequirementsCrosschecker = Callable[..., Mapping[str, Any]]

REPO_ROOT = package_base.REPO_ROOT
EXPERIMENT = "experiment_4846_submission_package_harden"
SCHEMA = "carnot.exp4846.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4846_submission_package_harden.json"
SPEC_RELATIVE_PATH = package_base.SPEC_RELATIVE_PATH
REQUIREMENTS_RELATIVE_PATH = (
    "docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md"
)
A1_PRIOR_RELATIVE_PATH = package_base.A1_PRIOR_RELATIVE_PATH
RANDOM_SEED = 4846
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VRAM_LIMIT_GB = package_base.VRAM_LIMIT_GB
DEFAULT_QWEN35_Q4_BYTES = 5_868_826_976

SPEC_REFS = [
    "REQ-CAPSTONE-4846",
    "SCENARIO-CAPSTONE-4846",
    "SCENARIO-CAPSTONE-4846-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4846-FIELD-PRINCIPLES",
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

REQUIREMENTS_CHECK_KEYS = (
    "agent_submission_shape",
    "internet_disabled",
    "competition_attached",
    "placeholder_non_rerun_submission",
    "agent_code_dataset_attached",
    "gguf_dataset_attached",
    "llama_server_binary_dataset_attached",
    "env_paths_declared",
    "mtp_q8_declared",
    "kaggle_probe_script_present",
    "operator_only_external_publication",
)

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
    "packaging_requirements_crosscheck",
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
resolve_agent_config = package_base.resolve_agent_config
resolve_model_paths = package_base.resolve_model_paths
resolve_a1_prior_inclusion = package_base.resolve_a1_prior_inclusion
estimate_vram = package_base.estimate_vram
payload_checksum = package_base.payload_checksum


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - malformed metadata is a defensive path.
        return {}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = _text(spec_path)
    kernel_dir = root_path / package_core.KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4846": "REQ-CAPSTONE-4846" in spec_text,
        "packaging_requirements_doc_present": (root_path / REQUIREMENTS_RELATIVE_PATH).exists(),
        "submission_kernel_present": (kernel_dir / package_core.KERNEL_MAIN).exists()
        and (kernel_dir / package_core.KERNEL_METADATA).exists(),
        "arc_competition_agent_present": (root_path / package_core.AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4846",
        "packaging_requirements_doc_present",
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


def cross_check_packaging_requirements(
    root: Path | str = REPO_ROOT,
    *,
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
) -> JsonDict:
    root_path = Path(root)
    doc_path = root_path / REQUIREMENTS_RELATIVE_PATH
    kernel_dir = root_path / package_core.KERNEL_RELATIVE_DIR
    metadata = _read_json(kernel_dir / package_core.KERNEL_METADATA)
    kernel_main = _text(kernel_dir / package_core.KERNEL_MAIN)
    doc_text = _text(doc_path)
    doc_lower = doc_text.lower()
    kernel_lower = kernel_main.lower()
    datasets = tuple(str(item) for item in metadata.get("dataset_sources") or ())
    competitions = tuple(str(item) for item in metadata.get("competition_sources") or ())
    checks_map = _mapping(agent_config_resolution.get("checks"))
    gguf_info = _mapping(model_path_resolution.get("gguf"))
    model_filename = str(
        gguf_info.get("filename") or agent_config_resolution.get("model_filename") or ""
    )

    checks = {
        "agent_submission_shape": bool(
            doc_path.exists()
            and "submit an agent" in doc_lower
            and "kaggle_is_competition_rerun" in kernel_lower
        ),
        "internet_disabled": bool(
            "no internet" in doc_lower and metadata.get("enable_internet") is False
        ),
        "competition_attached": "arc-prize-2026-arc-agi-3" in competitions,
        "placeholder_non_rerun_submission": "submission.parquet" in kernel_lower,
        "agent_code_dataset_attached": any("carnot-agent-code" in item for item in datasets),
        "gguf_dataset_attached": bool(
            # Canonical pin, 2026-07-28 (was the retired Qwen3.5-9B-MTP triple).
            ARC_LIVE_GENERATOR_MODEL_FILENAME in doc_text
            and model_filename == ARC_LIVE_GENERATOR_MODEL_FILENAME
            and any("carnot-gemma4-31b-it-gguf" in item for item in datasets)
        ),
        "llama_server_binary_dataset_attached": bool(
            "llama-server" in doc_lower
            and checks_map.get("binary_not_wheel") is True
            and any("carnot-llamacpp-mtp-binary" in item for item in datasets)
        ),
        "env_paths_declared": bool(
            "CARNOT_ARC_GGUF_PATH" in doc_text
            and "CARNOT_LLAMA_SERVER" in doc_text
            and "CARNOT_ARC_GGUF_PATH" in kernel_main
            and "CARNOT_LLAMA_SERVER" in kernel_main
            and agent_config_resolution.get("model_path_env") == "CARNOT_ARC_GGUF_PATH"
            and agent_config_resolution.get("server_path_env") == "CARNOT_LLAMA_SERVER"
        ),
        "mtp_q8_declared": bool(
            "draft-mtp" in doc_lower
            and "q8_0" in doc_text
            and "q8_0" in kernel_main
            and "CARNOT_ARC_MTP" in kernel_main
            and checks_map.get("mtp_enabled") is True
            and checks_map.get("q8_kv") is True
        ),
        "kaggle_probe_script_present": bool(
            _contains_any(doc_lower, ("build_verify_llamacpp_mtp.py", "one-shot kaggle notebook"))
            and (root_path / "scripts" / "kaggle" / "build_verify_llamacpp_mtp.py").exists()
        ),
        "operator_only_external_publication": bool(
            "operator" in doc_lower and package_builds.get("submitted_to_leaderboard") is False
        ),
    }
    ok = bool(doc_path.exists() and all(checks.values()))
    notes = [
        "operator-only requirements package cross-check passed"
        if ok
        else "requirements package cross-check blocked"
    ]
    if "CARNOT_ARC_MTP" in kernel_main:
        notes.append("kernel records CARNOT_ARC_MTP tight-VRAM override path")
    return {
        "requirements_doc_path": REQUIREMENTS_RELATIVE_PATH,
        "doc_present": doc_path.exists(),
        "ok": ok,
        "blocked_resource": "" if ok else "packaging_requirements",
        "checks": checks,
        "notes": notes,
    }


def _runtime_agent_config() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return resolve_agent_config(SUBMITTED_AGENT_CONFIG)


def runtime_vram_estimate(
    paths: Mapping[str, Any],
    config: Mapping[str, Any],
) -> JsonDict:
    gguf = _mapping(paths.get("gguf"))
    size_bytes = int(gguf.get("size_bytes") or 0)
    source = "resolved_model_path" if size_bytes > 0 else "packaging_spec_default"
    selected_size = size_bytes if size_bytes > 0 else DEFAULT_QWEN35_Q4_BYTES
    estimate = estimate_vram(
        model_size_bytes=selected_size,
        mtp_enabled=bool(config.get("mtp")),
        kv_quant=str(config.get("kv_quant") or "q8_0"),
        context_tokens=package_core.CONTEXT_TOKENS,
    )
    estimate["model_size_source"] = source
    estimate["selected_model_size_bytes"] = selected_size
    return estimate


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    a1_prior_inclusion: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
) -> bool:
    return bool(
        package_base._package_ready(
            preconditions_checked,
            package_builds,
            agent_config_resolution,
            model_path_resolution,
            vram_breakdown,
            a1_prior_inclusion,
        )
        and packaging_requirements_crosscheck.get("ok") is True
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
    a1_prior_inclusion: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
) -> str:
    if not package_base._package_ready(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        a1_prior_inclusion,
    ):
        return package_base._blocked_resource(
            preconditions_checked,
            package_builds,
            agent_config_resolution,
            model_path_resolution,
            vram_breakdown,
            a1_prior_inclusion,
        )
    if packaging_requirements_crosscheck.get("ok") is not True:
        return "packaging_requirements"
    return "unknown"


def _operator_checklist(
    package_ready: bool,
    vram_estimate_gb: float,
    a1_prior_inclusion: Mapping[str, Any],
    packaging_requirements_crosscheck: Mapping[str, Any],
) -> list[str]:
    readiness = "ready" if package_ready else "blocked until this JSON reports success_"
    a1_reason = str(a1_prior_inclusion.get("reason") or "not_checked")
    requirements_state = (
        "passed" if packaging_requirements_crosscheck.get("ok") is True else "blocked"
    )
    return [
        (
            "OPERATOR-CHECK: Confirm "
            f"{RESULT_RELATIVE_PATH} is {readiness}, submitted_to_leaderboard=false, "
            "and operator_only=true."
        ),
        (
            "OPERATOR-CHECK: Confirm the requirements-doc package cross-check is "
            f"{requirements_state} against {REQUIREMENTS_RELATIVE_PATH}."
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
            f"OPERATOR-CHECK: Verify the rerun log resolves {ARC_LIVE_GENERATOR_REPO_SUBSTR}, no draft-mtp or "
            "the tight-VRAM CARNOT_ARC_MTP=0 override, q8_0 KV, and CUDA llama-server."
        ),
        (
            "OPERATOR-CHECK: Verify VRAM estimate "
            f"{vram_estimate_gb:.3f}GB is below the ~16GB Kaggle budget with KV + headroom."
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
    packaging_requirements_crosscheck: Mapping[str, Any],
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
        packaging_requirements_crosscheck,
    )
    blocked = _blocked_resource(
        preconditions_checked,
        package_builds,
        agent_config_resolution,
        model_path_resolution,
        vram_breakdown,
        a1_prior_inclusion,
        packaging_requirements_crosscheck,
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
        "packaging_requirements_crosscheck": dict(packaging_requirements_crosscheck),
        "operator_checklist": _operator_checklist(
            ready,
            vram_estimate_gb,
            a1_prior_inclusion,
            packaging_requirements_crosscheck,
        ),
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
    requirements = artifact.get("packaging_requirements_crosscheck")
    if not isinstance(requirements, Mapping):
        errors.append("packaging_requirements_crosscheck")
        requirements = {}

    expected_ready = _package_ready(
        _mapping(artifact.get("preconditions_checked")),
        _mapping(artifact.get("package_builds")),
        _mapping(artifact.get("agent_config_resolution")),
        _mapping(artifact.get("model_path_resolution")),
        _mapping(artifact.get("vram_breakdown")),
        a1_prior_inclusion,
        requirements,
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    if _mapping(artifact.get("package_builds")).get("submitted_to_leaderboard") is not False:
        errors.append("package_builds_submitted_to_leaderboard")
    if artifact.get("vram_estimate_gb") != _mapping(artifact.get("vram_breakdown")).get(
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
    vram_estimator: VramEstimator = runtime_vram_estimate,
    a1_prior_resolver: A1PriorResolver = resolve_a1_prior_inclusion,
    requirements_crosschecker: RequirementsCrosschecker = cross_check_packaging_requirements,
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
    requirements = dict(
        requirements_crosschecker(
            root_path,
            package_builds=package_builds,
            agent_config_resolution=agent_config,
            model_path_resolution=model_paths,
        )
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_builds=package_builds,
        agent_config_resolution=agent_config,
        model_path_resolution=model_paths,
        vram_breakdown=vram,
        a1_prior_inclusion=a1_prior,
        packaging_requirements_crosscheck=requirements,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; schema is tested directly.
        raise ValueError(f"invalid {EXPERIMENT} artifact: {errors}")
    if write:
        package_core._write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
