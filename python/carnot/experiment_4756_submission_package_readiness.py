"""Experiment 4756: Kaggle submission-package readiness validation.

Spec refs: REQ-CAPSTONE-4756, SCENARIO-CAPSTONE-4756,
SCENARIO-CAPSTONE-4756-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4756-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import glob
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

# THE CANONICAL GENERATOR PIN, imported rather than re-typed. This module's generator assertions
# were hardcoded to the retired Qwen3.5-9B-MTP and went stale the moment the 2026-07-28 operator
# directive re-pinned the live generator to gemma-4-31B-it. Reading the constants means a future
# switch updates this readiness gate for free instead of leaving it asserting a model nothing runs.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_MODEL_ID,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
ManifestInspector = Callable[[Path], Mapping[str, Any]]
RequirementsInspector = Callable[[Path], Mapping[str, Any]]
PackageValidator = Callable[
    [Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]

EXPERIMENT = "experiment_4756_submission_package_readiness"
SCHEMA = "carnot.exp4756.submission_package_readiness.v1"
RESULT_RELATIVE_PATH = "results/experiment_4756_submission_package_readiness.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
KERNEL_RELATIVE_DIR = "scripts/kaggle/submission_kernel"
KERNEL_MAIN = "main.py"
KERNEL_METADATA = "kernel-metadata.json"
RANDOM_SEED = 4756
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts; 100us floor."
SPEC_REFS = [
    "REQ-CAPSTONE-4756",
    "SCENARIO-CAPSTONE-4756",
    "SCENARIO-CAPSTONE-4756-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4756-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "passed_", "complete_", "blocked_")
# THE KAGGLE DATASET SET THE KERNEL MUST ATTACH. Re-pinned 2026-07-28 with the generator switch.
# This was a genuine PRODUCTION FAILURE, not a stale comment: `datasets_attached` is
# `REQUIRED_DATASETS.issubset(dataset_sources)`, and kernel-metadata.json now requests
# `carnot-gemma4-31b-it-gguf`, so the readiness gate reported the submission package as blocked on
# a dataset the kernel deliberately no longer attaches. Verified failing before this fix
# (missing: ['iancblenke/carnot-qwen35-9b-mtp-gguf']) and passing after.
#
# NOTE FOR THE OPERATOR: the gemma dataset is an 18.3 GB upload that DOES NOT EXIST YET, and only
# the operator can create it. `SUBMITTED_AGENT_CONFIG["frozen_generator"]["kaggle_dataset_uploaded"]`
# is False for exactly this reason. This constant naming it is what keeps that from being silently
# forgotten between here and a submission.
#
# ^^^ RESOLVED 2026-07-28 (same day; preserved per never-prune because it records WHY this constant
# exists). Both datasets have since been created, privately, by the operator:
#     iancblenke/carnot-gemma4-31b-it-gguf    (17 GB, main weights)
#     iancblenke/carnot-gemma4-31b-mtp-head   (491 MB, the draft head)
# `kaggle_dataset_uploaded` is now True. The paragraph above is therefore HISTORY, not a live
# blocker -- read it as the reason this constant is written out by name rather than as a
# description of today's state.
#
# 2026-07-28 (second pass): the MTP DRAFT HEAD is now a required attachment too. It is a separate
# 491 MB dataset because the head is a separate GGUF -- gemma-4-31B's MTP is not embedded in the
# main weights. It is REQUIRED rather than optional because the scored kernel enables MTP, and the
# failure mode of a missing head is a silent ~1.4x slowdown, not an error: the kernel correctly
# drops the flags, the run completes, and nothing downstream distinguishes it from a fast run.
# Making it a gate precondition is the only place that difference becomes visible before submission.
REQUIRED_DATASETS = {
    "iancblenke/carnot-agent-code",
    "iancblenke/carnot-llamacpp-mtp-binary",
    "iancblenke/carnot-gemma4-31b-it-gguf",
    "iancblenke/carnot-gemma4-31b-mtp-head",
}
REQUIRED_COMPETITION = "arc-prize-2026-arc-agi-3"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; a package-ready run is success_/passed_."},
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts; 100us floor."},
    "preconditions_checked": {"principle": "records the GGUF/binary checks."},
    "package_builds": {
        "principle": (
            "the package assembles + the entrypoint smoke-runs offline -- the readiness gate."
        )
    },
    "submission_package_ready": {
        "principle": (
            "True only if OPERATOR-ready; this task NEVER submits "
            "(operator-only external publication)."
        )
    },
    "operator_checklist": {
        "principle": (
            "the steps the OPERATOR performs to submit -- the task ends at the checklist, "
            "not the submission."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "package_manifest",
    "requirements_check",
    "submitted_to_leaderboard",
    "operator_only",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _default_gguf_finder() -> list[str]:  # pragma: no cover - local filesystem boundary.
    candidates: list[str] = []
    env_path = os.environ.get("CARNOT_ARC_GGUF_PATH")
    if env_path and Path(env_path).is_file():
        candidates.append(env_path)
    pattern = (
        f"~/.cache/huggingface/hub/models--unsloth--{ARC_LIVE_GENERATOR_MODEL_ID.split('/')[-1]}/"
        f"snapshots/*/{ARC_LIVE_GENERATOR_MODEL_FILENAME}"
    )
    candidates.extend(glob.glob(os.path.expanduser(pattern)))
    return sorted(dict.fromkeys(candidates))


def _default_llama_server_finder() -> list[str]:  # pragma: no cover - local filesystem boundary.
    candidates: list[str] = []
    env_path = os.environ.get("CARNOT_LLAMA_SERVER")
    if env_path and Path(env_path).is_file():
        candidates.append(env_path)
    home = Path.home()
    for path in (
        REPO_ROOT / "build-hip" / "bin" / "llama-server",
        REPO_ROOT / "build" / "bin" / "llama-server",
        home / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server",
        home / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server",
    ):
        if path.is_file():
            candidates.append(str(path))
    return sorted(dict.fromkeys(candidates))


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    gguf_finder: Callable[[], list[str]] | None = None,
    llama_server_finder: Callable[[], list[str]] | None = None,
) -> JsonDict:
    root_path = Path(root)
    ggufs = list((gguf_finder or _default_gguf_finder)())
    servers = list((llama_server_finder or _default_llama_server_finder)())
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    kernel_dir = root_path / KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4756": "REQ-CAPSTONE-4756" in spec_text,
        "submission_kernel_present": (kernel_dir / KERNEL_MAIN).exists()
        and (kernel_dir / KERNEL_METADATA).exists(),
        "qwen35_mtp_gguf_cached": bool(ggufs),
        "qwen35_mtp_gguf_paths": ggufs,
        "llama_server_binary_present": bool(servers),
        "llama_server_paths": servers,
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4756",
        "submission_kernel_present",
        "qwen35_mtp_gguf_cached",
        "llama_server_binary_present",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if checks["ok"] is not True:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


def _blocked_fields(checks: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in checks.items() if passed is not True]


def inspect_package_manifest(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    kernel_dir = root_path / KERNEL_RELATIVE_DIR
    metadata_path = kernel_dir / KERNEL_METADATA
    main_path = kernel_dir / KERNEL_MAIN
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata = {"error": repr(exc)}
    dataset_sources = set(metadata.get("dataset_sources") or [])
    competition_sources = set(metadata.get("competition_sources") or [])
    checks = {
        "metadata_present": metadata_path.exists(),
        "main_present": main_path.exists(),
        "code_file_main": metadata.get("code_file") == KERNEL_MAIN,
        "script_kernel": metadata.get("kernel_type") == "script",
        "gpu_enabled": metadata.get("enable_gpu") is True,
        "internet_off": metadata.get("enable_internet") is False,
        "datasets_attached": REQUIRED_DATASETS.issubset(dataset_sources),
        "competition_attached": REQUIRED_COMPETITION in competition_sources,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "blocked_resources": _blocked_fields(checks),
        "metadata_path": f"{KERNEL_RELATIVE_DIR}/{KERNEL_METADATA}",
        "code_file": str(metadata.get("code_file") or ""),
        "dataset_sources": sorted(dataset_sources),
        "competition_sources": sorted(competition_sources),
        "machine_shape": str(metadata.get("machine_shape") or ""),
    }


def inspect_requirements(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    main_path = root_path / KERNEL_RELATIVE_DIR / KERNEL_MAIN
    source = main_path.read_text(encoding="utf-8") if main_path.exists() else ""
    checks = {
        "offline_pip_no_index": "--no-index" in source and "arc_agi_3_wheels" in source,
        "arc_agi_wheel": "arc-agi" in source,
        "dotenv_wheel": "python-dotenv" in source,
        "rerun_gateway_mode": "KAGGLE_IS_COMPETITION_RERUN" in source and "gateway:8001" in source,
        "agent_framework_copy": "ARC-AGI-3-Agents" in source and "my_agent.py" in source,
        "llama_server_env": "CARNOT_LLAMA_SERVER" in source and "llama-server" in source,
        "gguf_env": "CARNOT_ARC_GGUF_PATH" in source and ".gguf" in source,
        "generator_resolution": ARC_LIVE_GENERATOR_REPO_SUBSTR in source or "Q4_K_M" in source,
        "placeholder_parquet": "submission.parquet" in source
        and "pandas" in source
        and "to_parquet" in source,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "blocked_resources": _blocked_fields(checks),
        "source_path": f"{KERNEL_RELATIVE_DIR}/{KERNEL_MAIN}",
        "offline_dependency_sources": [
            "competition arc_agi_3_wheels",
            "kaggle base image pandas/parquet stack",
            "attached carnot-agent-code dataset",
            "attached llama-server binary dataset",
            f"attached {ARC_LIVE_GENERATOR_REPO_SUBSTR} GGUF dataset",
        ],
    }


def assemble_submission_package(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - filesystem boundary.
    root_path = Path(root)
    kernel_dir = root_path / KERNEL_RELATIVE_DIR
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp4756_") as tmp:
            tmp_path = Path(tmp)
            files: list[str] = []
            hashes: dict[str, str] = {}
            for name in (KERNEL_METADATA, KERNEL_MAIN):
                src = kernel_dir / name
                dst = tmp_path / name
                shutil.copy2(src, dst)
                files.append(name)
                hashes[name] = _file_sha256(dst)
            py_compile.compile(str(tmp_path / KERNEL_MAIN), doraise=True)
            return {
                "assembled": True,
                "entrypoint_compiles": True,
                "files": files,
                "file_hashes": hashes,
                "sha256": "sha256:"
                + hashlib.sha256(_stable_json(hashes).encode("utf-8")).hexdigest(),
                "blocked_resource": "",
            }
    except Exception as exc:
        return {
            "assembled": False,
            "entrypoint_compiles": False,
            "files": [],
            "file_hashes": {},
            "sha256": "",
            "blocked_resource": "package_assembly",
            "error": repr(exc)[:500],
        }


def run_clean_env_smoke(
    root: Path | str,
    preconditions: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - subprocess/import boundary.
    if preconditions.get("ok") is not True:
        return {
            "passed": False,
            "clean_env": True,
            "blocked_resource": str(preconditions.get("blocked_resource") or "precondition"),
        }
    root_path = Path(root)
    code = """
import json
from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.experiment_4605_live_integration_scored_agent import _NoOpProposer

class _Base:
    def __init__(self, *args, **kwargs):
        self.game_id = str(kwargs.get("game_id") or "paritytest")

agent_cls = make_carnot_agent(_Base, proposer=_NoOpProposer())
agent = agent_cls(game_id="paritytest")
policy = getattr(agent, "_policy", None)
print(json.dumps({
    "constructed": isinstance(policy, E3AgentPolicy),
    "policy_class": policy.__class__.__name__ if policy is not None else "",
}))
"""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(root_path / "python"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CARNOT_ARC_GGUF_PATH": str((preconditions.get("qwen35_mtp_gguf_paths") or [""])[0]),
        "CARNOT_LLAMA_SERVER": str((preconditions.get("llama_server_paths") or [""])[0]),
        "CARNOT_ARC_MTP": "0",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception:
        payload = {}
    passed = completed.returncode == 0 and payload.get("constructed") is True
    return {
        "passed": passed,
        "clean_env": True,
        "returncode": int(completed.returncode),
        "policy_class": str(payload.get("policy_class") or ""),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
        "blocked_resource": "" if passed else "clean_env_smoke",
    }


def validate_package(
    root: Path,
    preconditions: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    requirements_check: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - filesystem/subprocess boundary.
    assembly = assemble_submission_package(root)
    smoke = run_clean_env_smoke(root, preconditions)
    return {
        "assembled": assembly.get("assembled") is True,
        "entrypoint_compiles": assembly.get("entrypoint_compiles") is True,
        "manifest_complete": package_manifest.get("complete") is True,
        "requirements_complete": requirements_check.get("complete") is True,
        "clean_env_smoke_ran": smoke.get("passed") is True,
        "clean_env": smoke.get("clean_env") is True,
        "submitted_to_leaderboard": False,
        "blocked_resource": str(
            assembly.get("blocked_resource") or smoke.get("blocked_resource") or ""
        ),
        "assembly": assembly,
        "entrypoint_smoke": smoke,
    }


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("ok") is not True:
        return str(preconditions_checked.get("blocked_resource") or "precondition")
    for field, resource in (
        ("assembled", "package_assembly"),
        ("entrypoint_compiles", "entrypoint_compile"),
        ("manifest_complete", "manifest"),
        ("requirements_complete", "requirements"),
        ("clean_env_smoke_ran", "clean_env_smoke"),
    ):
        if package_builds.get(field) is not True:
            return resource
    return "unknown"


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
) -> bool:
    return bool(
        preconditions_checked.get("ok") is True
        and package_builds.get("assembled") is True
        and package_builds.get("entrypoint_compiles") is True
        and package_builds.get("manifest_complete") is True
        and package_builds.get("requirements_complete") is True
        and package_builds.get("clean_env_smoke_ran") is True
        and package_builds.get("submitted_to_leaderboard") is False
    )


def _operator_checklist(package_ready: bool) -> list[str]:
    prefix = "OPERATOR-ACTION:"
    readiness = "ready" if package_ready else "blocked until the readiness JSON is success_"
    return [
        f"{prefix} Confirm results/experiment_4756_submission_package_readiness.json is {readiness} and submitted_to_leaderboard is false.",
        f"{prefix} In Kaggle, open kernel iancblenke/carnot-arc-agi3-submission with GPU enabled and internet disabled.",
        f"{prefix} Attach the ARC-AGI-3 competition plus datasets carnot-agent-code, carnot-llamacpp-mtp-binary, and carnot-gemma4-31b-it-gguf.",
        f"{prefix} Save & Run the kernel and wait for the non-rerun submission.parquet output before any external publication step.",
        f"{prefix} Review logs for LLM TIER RESOLVED or LLM GENERATOR HEALTHY, and record any CPU-only degradation before submit.",
        f"{prefix} Submit only through the operator-controlled Kaggle UI or API after the above checks; this experiment never submits.",
    ]


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    requirements_check: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    ready = _package_ready(preconditions_checked, package_builds)
    blocked = _blocked_resource(preconditions_checked, package_builds)
    verdict = (
        "success_package_ready_offline_smoke_green"
        if ready
        else (
            f"blocked_{blocked}"
            if preconditions_checked.get("ok") is not True
            else f"complete_package_not_ready_{blocked}"
        )
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "package_builds": dict(package_builds),
        "submission_package_ready": ready,
        "operator_checklist": _operator_checklist(ready),
        "package_manifest": dict(package_manifest),
        "requirements_check": dict(requirements_check),
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
        and all(isinstance(step, str) and step.startswith("OPERATOR-ACTION:") for step in checklist)
    ):
        errors.append("operator_checklist")
    expected_ready = _package_ready(
        artifact.get("preconditions_checked", {}),
        artifact.get("package_builds", {}),
    )
    if artifact.get("submission_package_ready") is not expected_ready:
        errors.append("submission_package_ready_gate")
    if artifact.get("package_builds", {}).get("submitted_to_leaderboard") is not False:
        errors.append("package_builds_submitted_to_leaderboard")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def run(
    root: Path | str = REPO_ROOT,
    *,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    manifest_inspector: ManifestInspector = inspect_package_manifest,
    requirements_inspector: RequirementsInspector = inspect_requirements,
    package_validator: PackageValidator = validate_package,
    write: bool = True,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions = dict(preconditions_checker(root_path))
    manifest = dict(manifest_inspector(root_path))
    requirements = dict(requirements_inspector(root_path))
    package_builds = dict(package_validator(root_path, preconditions, manifest, requirements))
    artifact = build_artifact(
        preconditions_checked=preconditions,
        package_manifest=manifest,
        requirements_check=requirements,
        package_builds=package_builds,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; tests cover schema directly.
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
