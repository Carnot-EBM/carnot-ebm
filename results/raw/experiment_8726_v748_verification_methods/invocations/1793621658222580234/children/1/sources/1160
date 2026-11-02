"""Experiment 4766: ARC-AGI-3 Kaggle submission-package hardening.

Spec refs: REQ-CAPSTONE-4766, SCENARIO-CAPSTONE-4766,
SCENARIO-CAPSTONE-4766-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4766-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

# THE CANONICAL GENERATOR PIN, imported rather than re-typed -- the 2026-07-28 operator directive
# re-pinned the ARC generator from Qwen3.5-9B-MTP to gemma-4-31B-it, and every hardening gate that
# spelled the old name as a literal kept asserting a model nothing runs.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_MODEL_ID,
    ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
PackageBuilder = Callable[[Path], Mapping[str, Any]]
AgentConfigResolver = Callable[[], Mapping[str, Any]]
ModelPathResolver = Callable[[], Mapping[str, Any]]
VramEstimator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

EXPERIMENT = "experiment_4766_submission_package_harden"
SCHEMA = "carnot.exp4766.submission_package_harden.v1"
RESULT_RELATIVE_PATH = "results/experiment_4766_submission_package_harden.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
KERNEL_RELATIVE_DIR = "scripts/kaggle/submission_kernel"
KERNEL_MAIN = "main.py"
KERNEL_METADATA = "kernel-metadata.json"
AGENT_RELATIVE_PATH = "python/carnot/agentic/arc_competition_agent.py"
RANDOM_SEED = 4766
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts; 0.0001s floor."
VRAM_LIMIT_GB = 16.0
RUNTIME_OVERHEAD_GB = 0.7
REQUIRED_HEADROOM_GB = 1.5
KV_LAYERS = 36
KV_EMBEDDING_DIM = 1024
CONTEXT_TOKENS = 16_384
BYTES_PER_GB = 1_000_000_000

SPEC_REFS = [
    "REQ-CAPSTONE-4766",
    "SCENARIO-CAPSTONE-4766",
    "SCENARIO-CAPSTONE-4766-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4766-FIELD-PRINCIPLES",
]
TERMINAL_PREFIXES = ("success_", "complete_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; package builds is success_/complete_."},
    "submission_package_ready": {
        "principle": (
            "True iff ready for the OPERATOR to submit; the task itself NEVER submits "
            "(operator-only)."
        )
    },
    "vram_estimate_gb": {
        "principle": (
            "must fit the ~16GB Kaggle constraint with KV + headroom -- the deployment gate."
        )
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
        home / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server",
        home / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server",
        REPO_ROOT / "build" / "bin" / "llama-server",
        REPO_ROOT / "build-hip" / "bin" / "llama-server",
    ):
        if path.is_file():
            candidates.append(str(path))
    return sorted(dict.fromkeys(candidates))


def _default_cuda_inspector(path: Path) -> bool:  # pragma: no cover - binary/linker boundary.
    directory = path.parent
    if any(child.name.startswith("libggml-cuda") for child in directory.glob("libggml-cuda*")):
        return True
    try:
        completed = subprocess.run(
            ["ldd", str(path)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return "libggml-cuda" in completed.stdout or "libcudart" in completed.stdout


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    kernel_dir = root_path / KERNEL_RELATIVE_DIR
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4766": "REQ-CAPSTONE-4766" in spec_text,
        "submission_kernel_present": (kernel_dir / KERNEL_MAIN).exists()
        and (kernel_dir / KERNEL_METADATA).exists(),
        "arc_competition_agent_present": (root_path / AGENT_RELATIVE_PATH).exists(),
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4766",
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


def dry_build_package(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    kernel_dir = root_path / KERNEL_RELATIVE_DIR
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp4766_") as tmp:
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
                "dry_build_ran": True,
                "package_builds": True,
                "entrypoint_compiles": True,
                "manifest_present": True,
                "kernel_main_present": True,
                "submitted_to_leaderboard": False,
                "blocked_resource": "",
                "files": files,
                "file_hashes": hashes,
                "package_sha256": "sha256:"
                + hashlib.sha256(_stable_json(hashes).encode("utf-8")).hexdigest(),
            }
    except Exception as exc:  # pragma: no cover - defensive filesystem failure path.
        return {
            "dry_build_ran": True,
            "package_builds": False,
            "entrypoint_compiles": False,
            "manifest_present": (kernel_dir / KERNEL_METADATA).exists(),
            "kernel_main_present": (kernel_dir / KERNEL_MAIN).exists(),
            "submitted_to_leaderboard": False,
            "blocked_resource": "dry_build",
            "files": [],
            "file_hashes": {},
            "package_sha256": "",
            "error": repr(exc)[:500],
        }


def resolve_agent_config(submitted_config: Mapping[str, Any]) -> JsonDict:
    frozen = dict(submitted_config.get("frozen_generator") or {})
    checks = {
        "submitted_policy_e3": submitted_config.get("policy") == "E3AgentPolicy",
        "submitted_cascade": submitted_config.get("cascade") is True,
        # Canonical pin, 2026-07-28. Renamed off the dead `qwen35` key name: a check whose NAME
        # asserts a retired model is unreadable even when its logic is right.
        "model_is_pinned_generator": frozen.get("model_id") == ARC_LIVE_GENERATOR_MODEL_ID
        and frozen.get("repo_substr") == ARC_LIVE_GENERATOR_REPO_SUBSTR,
        "model_filename": frozen.get("model_filename") == ARC_LIVE_GENERATOR_MODEL_FILENAME,
        "mtp_enabled": frozen.get("mtp") is True and frozen.get("spec_type") == "draft-mtp",
        "q8_kv": frozen.get("kv_quant") == "q8_0",
        # CANONICAL PIN, not the retired Qwen literal (fixed 2026-07-28, third pass). This read
        # `== "/no_think\n"` -- a Qwen3 hybrid-thinking control token. Gemma-4 has no such token
        # and would consume it as literal prompt text, which is why the canonical pin is the EMPTY
        # string. The consequence was not cosmetic: the LIVE, correctly-configured
        # `SUBMITTED_AGENT_CONFIG` FAILED this check, so `resolved` was False and every harden
        # capstone in this family reported `submission_package_ready: False` for a submission that
        # is in fact correctly configured. A readiness gate that goes red exactly when the thing it
        # gates is right is worse than no gate -- it trains the reader to ignore it.
        "no_think": frozen.get("no_think_prefix") == ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
        "n_predict_floor": int(frozen.get("max_tokens") or 0)
        >= int(frozen.get("n_predict_min") or 0)
        >= 2048,
        "cuda_128_server": frozen.get("llama_server_kind") == "cuda-12.8-binary",
        "binary_not_wheel": frozen.get("binary_not_wheel") is True
        and frozen.get("wheel_fallback_allowed") is False,
    }
    resolved = all(checks.values())
    return {
        "resolved": resolved,
        "checks": checks,
        "blocked_resource": "" if resolved else "agent_config",
        "model_id": str(frozen.get("model_id") or ""),
        "repo_substr": str(frozen.get("repo_substr") or ""),
        "model_filename": str(frozen.get("model_filename") or ""),
        "mtp": bool(frozen.get("mtp")),
        "spec_type": str(frozen.get("spec_type") or ""),
        "kv_quant": str(frozen.get("kv_quant") or ""),
        "max_tokens": int(frozen.get("max_tokens") or 0),
        "n_predict_min": int(frozen.get("n_predict_min") or 0),
        "llama_server_kind": str(frozen.get("llama_server_kind") or ""),
        "model_path_env": str(frozen.get("model_path_env") or "CARNOT_ARC_GGUF_PATH"),
        "server_path_env": str(frozen.get("server_path_env") or "CARNOT_LLAMA_SERVER"),
    }


def _runtime_agent_config() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return resolve_agent_config(SUBMITTED_AGENT_CONFIG)


def resolve_model_paths(
    *,
    gguf_paths: Sequence[str] | None = None,
    llama_server_paths: Sequence[str] | None = None,
    cuda_inspector: Callable[[Path], bool] = _default_cuda_inspector,
    model_size_bytes: int | None = None,
) -> JsonDict:
    gguf_candidates = list(_default_gguf_finder() if gguf_paths is None else gguf_paths)
    server_candidates = list(
        _default_llama_server_finder() if llama_server_paths is None else llama_server_paths
    )
    gguf_path = next((Path(item) for item in gguf_candidates if Path(item).is_file()), None)
    server_path: Path | None = None
    cuda_capable = False
    server_checks: list[JsonDict] = []
    for candidate in server_candidates:
        path = Path(candidate)
        present = path.is_file()
        candidate_cuda = bool(present and cuda_inspector(path))
        server_checks.append(
            {
                "path": str(path),
                "present": present,
                "cuda_12_8_capable": candidate_cuda,
            }
        )
        if server_path is None and present:
            server_path = path
            cuda_capable = candidate_cuda
        if candidate_cuda:
            server_path = path
            cuda_capable = True
            break
    gguf_present = gguf_path is not None and gguf_path.is_file()
    server_present = server_path is not None and server_path.is_file()
    size_bytes = int(model_size_bytes if model_size_bytes is not None else 0)
    if model_size_bytes is None and gguf_present:
        size_bytes = int(gguf_path.stat().st_size)
    resolved = bool(gguf_present and server_present and cuda_capable and size_bytes > 0)
    return {
        "resolved": resolved,
        "blocked_resource": "" if resolved else "model_paths",
        "gguf": {
            "path": str(gguf_path or ""),
            "filename": gguf_path.name if gguf_path is not None else "",
            "present": gguf_present,
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / BYTES_PER_GB, 6),
        },
        "llama_server": {
            "path": str(server_path or ""),
            "filename": server_path.name if server_path is not None else "",
            "present": server_present,
            "cuda_12_8_capable": cuda_capable,
            "kind": "cuda-12.8-binary",
        },
        "all_gguf_candidates": [str(item) for item in gguf_candidates],
        "all_llama_server_candidates": [str(item) for item in server_candidates],
        "llama_server_candidate_checks": server_checks,
    }


def estimate_vram(
    *,
    model_size_bytes: int,
    mtp_enabled: bool,
    kv_quant: str,
    context_tokens: int = CONTEXT_TOKENS,
    limit_gb: float = VRAM_LIMIT_GB,
) -> JsonDict:
    model_gb = float(model_size_bytes) / BYTES_PER_GB
    model_copies = 2 if mtp_enabled else 1
    draft_model_gb = model_gb if mtp_enabled else 0.0
    kv_bytes_per_value = 1 if kv_quant == "q8_0" else 2
    kv_cache_gb = (
        KV_LAYERS * int(context_tokens) * KV_EMBEDDING_DIM * 2 * kv_bytes_per_value / BYTES_PER_GB
    )
    total_with_headroom = (
        model_gb * model_copies + kv_cache_gb + RUNTIME_OVERHEAD_GB + REQUIRED_HEADROOM_GB
    )
    remaining = float(limit_gb) - total_with_headroom
    estimate = {
        "vram_estimate_gb": round(total_with_headroom, 3),
        "fits_16gb": remaining > 0,
        "limit_gb": float(limit_gb),
        "remaining_headroom_gb": round(remaining, 3),
        "model_copies": model_copies,
        "model_weights_gb": round(model_gb, 3),
        "draft_model_weights_gb": round(draft_model_gb, 3),
        "kv_cache_gb": round(kv_cache_gb, 3),
        "kv_quant": kv_quant,
        "context_tokens": int(context_tokens),
        "runtime_overhead_gb": RUNTIME_OVERHEAD_GB,
        "required_headroom_gb": REQUIRED_HEADROOM_GB,
        "total_with_headroom_gb": round(total_with_headroom, 3),
    }
    return estimate


def _runtime_vram_estimate(paths: Mapping[str, Any], config: Mapping[str, Any]) -> JsonDict:
    return estimate_vram(
        model_size_bytes=int(paths.get("gguf", {}).get("size_bytes") or 0),
        mtp_enabled=bool(config.get("mtp")),
        kv_quant=str(config.get("kv_quant") or "q8_0"),
        context_tokens=CONTEXT_TOKENS,
    )


def _package_ready(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
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
    )


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    package_builds: Mapping[str, Any],
    agent_config_resolution: Mapping[str, Any],
    model_path_resolution: Mapping[str, Any],
    vram_breakdown: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("ok") is not True:
        return str(preconditions_checked.get("blocked_resource") or "precondition")
    if package_builds.get("package_builds") is not True:
        return str(package_builds.get("blocked_resource") or "dry_build")
    if agent_config_resolution.get("resolved") is not True:
        return "agent_config"
    if model_path_resolution.get("resolved") is not True:
        return "model_paths"
    if (
        vram_breakdown.get("fits_16gb") is not True
        or float(vram_breakdown.get("vram_estimate_gb") or 0.0) >= VRAM_LIMIT_GB
    ):
        return "vram"
    return "unknown"


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
    agent_config_resolver: AgentConfigResolver = _runtime_agent_config,
    model_path_resolver: ModelPathResolver = resolve_model_paths,
    vram_estimator: VramEstimator = _runtime_vram_estimate,
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
