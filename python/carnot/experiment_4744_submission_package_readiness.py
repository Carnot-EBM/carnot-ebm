"""Experiment 4744: Kaggle submission-package readiness validation.

Spec refs: REQ-CAPSTONE-4744, SCENARIO-CAPSTONE-4744,
SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4744-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# THE CANONICAL GENERATOR PIN. Read, never re-typed: this module's checks were hardcoded to the
# retired Qwen3.5-9B-MTP and silently went red in production the moment the operator directive of
# 2026-07-28 moved the pin. Importing the constants means a future switch cannot leave this gate
# behind again. Imported from the leaf module (`arc_executable_world_model`), not from
# `arc_competition_agent`, because the latter costs ~590 MB to import and this module is also run
# as a standalone readiness script.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_MODEL_ID,
    ARC_LIVE_GENERATOR_MTP_DEFAULT,  # noqa: F401  (local default; kept for provenance/audit)
    ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT,
    ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]
RESULT_RELATIVE_PATH = "results/experiment_4744_submission_package_readiness.json"
EXPERIMENT = "experiment_4744_submission_package_readiness"
SCHEMA = "carnot.exp4744.submission_package_readiness.v1"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- package validation + an offline smoke, "
    "no headline live-model load (100us/1s floor)."
)
RANDOM_SEED = 4744
SMOKE_ACTION_BUDGET = 5
RPM_CAP = 600
SPEC_REFS = [
    "REQ-CAPSTONE-4744",
    "SCENARIO-CAPSTONE-4744",
    "SCENARIO-CAPSTONE-4744-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4744-FIELD-PRINCIPLES",
]
REQUIRED_SHARED_LIBRARIES = (
    "libllama-common",
    "libllama",
    "libggml",
    "libggml-cuda",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: submission_package_ready OR complete: "
            "submission_package_blocked_<resource>."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- package validation + an offline smoke, "
            "no headline live-model load (100us/1s floor)."
        )
    },
    "submission_package_ready": {
        "principle": (
            "True if the package is ready for the OPERATOR to submit; the task itself "
            "never submits (Operator-Only External Publication)."
        )
    },
    "readiness_checklist": {
        "principle": (
            "per-item pass/blocked (entrypoint imports, frozen generator wired, manifest "
            "complete, smoke episode ran, parity green) -- the operator's pre-submission checklist."
        )
    },
    "frozen_generator_confirmed": {
        "principle": (
            "the generator wired into SUBMITTED_AGENT_CONFIG is the CANONICAL pinned "
            "generator (gemma-4-31B-it as of the 2026-07-28 operator directive; was "
            "Qwen3.5-9B-MTP before that) -- the frozen-stack guard. NEVER gemma-8919, NEVER "
            "the 3090s."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- the validated agent is byte-for-byte the SUBMITTED_AGENT_CONFIG."
        )
    },
    "verifier_is_oracle": {"principle": "false -- a packaging validation invokes no oracle."},
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash of the validated package manifest."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (agent importable, offline arcade); pre-empts "
            "missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "honest_verdict",
    "inference_substrate",
    "submission_package_ready",
    "readiness_checklist",
    "frozen_generator_confirmed",
    "parity_test_green",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "package_manifest",
    "entrypoint_validation",
    "frozen_generator_config",
    "smoke_episode",
    "parity_test",
    "submitted_agent_config",
    "submitted_to_leaderboard",
    "operator_only",
    "field_principles",
    "spec_refs",
    "duration_s",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def _jsonable(value: Any) -> JsonDict:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def frozen_generator_config_from_submitted(submitted_config: Mapping[str, Any]) -> JsonDict:
    raw = dict(submitted_config.get("frozen_generator") or {})
    checks = {
        "submitted_policy_e3": submitted_config.get("policy") == "E3AgentPolicy",
        "submitted_cascade": submitted_config.get("cascade") is True,
        # RE-PINNED 2026-07-28 to gemma-4-31B-it, and re-pinned to the CANONICAL CONSTANTS rather
        # than to a fresh set of string literals. These three checks were hardcoded to the retired
        # Qwen3.5-9B-MTP, so once `SUBMITTED_AGENT_CONFIG` moved, this gate returned
        # `confirmed=False` in production while its own unit tests stayed green -- the tests build
        # SYNTHETIC fixture dicts and never evaluate the live config. A readiness gate that fails
        # closed on the shipped configuration is not a safety net, it is a permanently-red light
        # the operator learns to ignore.
        #
        # The key name `model_is_qwen35_mtp` is DEAD and has been renamed. Historical artifacts
        # that recorded it keep their key (never-prune); nothing reads it forward.
        "model_is_pinned_generator": raw.get("model_id") == ARC_LIVE_GENERATOR_MODEL_ID
        and raw.get("repo_substr") == ARC_LIVE_GENERATOR_REPO_SUBSTR,
        # SUPERSEDED 2026-07-28 (same day, measured) -- preserved per never-prune:
        #   "MTP IS NOW CORRECTLY OFF ... gemma-4-31B-it declares no `nextn_predict_layers`, so
        #    `--spec-type draft-mtp` would make llama-server load the same 18.3 GB file twice and
        #    cudaMalloc-fail."
        # The premise was wrong: this model's MTP head is a SEPARATE 491 MiB GGUF (arch
        # `gemma4-assistant`), which is why none was found inside the main file. Enabling MTP loads
        # the head, not a second copy of the weights.
        #
        # THE CHECK IS NOW READ AGAINST THE **SCORED** CONSTANT, WHICH IS THE BUG THIS FIXES.
        # `SUBMITTED_AGENT_CONFIG` describes the KAGGLE launch, and the previous line compared it
        # to `ARC_LIVE_GENERATOR_MTP_DEFAULT` -- the LOCAL dev-box default. Those two are correctly
        # DIFFERENT (MTP is a net loss on a 24 GB card that must offload to fit it, and a pure
        # ~1.4x win on the 96 GB scored card), so comparing the scored config against the local
        # constant makes the gate red exactly when the submission is configured correctly.
        "mtp_matches_model": (raw.get("mtp") is True and raw.get("spec_type") == "draft-mtp")
        if ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"
        else (raw.get("mtp") is False and raw.get("spec_type") is None),
        "q8_kv": raw.get("kv_quant") == "q8_0",
        "n_predict_floor": int(raw.get("max_tokens") or 0)
        >= int(raw.get("n_predict_min") or 2048)
        >= 2048,
        # `/no_think` is a Qwen3 hybrid-thinking control token. On gemma-4 it is not a token at
        # all, it is literal prompt text -- so the correct frozen value is the empty string, and
        # the check is "matches the canonical pin", not "is /no_think".
        "no_think_matches_model": raw.get("no_think_prefix") == ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
        "binary_not_wheel": raw.get("binary_not_wheel") is True
        and raw.get("wheel_fallback_allowed") is False,
        "free_non_8919_port": raw.get("port_strategy") == "free_non_8919",
        "props_verify": raw.get("props_verify_endpoint") == "/props",
        "not_gemma_8919": "gemma-8919" in list(raw.get("forbidden_models") or []),
        "not_3090": "3090" in list(raw.get("forbidden_gpu_targets") or []),
    }
    out = {
        "confirmed": all(checks.values()),
        "checks": checks,
        "model_id": raw.get("model_id", ""),
        "repo_substr": raw.get("repo_substr", ""),
        "model_filename": raw.get("model_filename", ""),
        "model_path_env": raw.get("model_path_env", "CARNOT_ARC_GGUF_PATH"),
        "server_path_env": raw.get("server_path_env", "CARNOT_LLAMA_SERVER"),
        "llama_server_kind": raw.get("llama_server_kind", ""),
        "binary_not_wheel": bool(raw.get("binary_not_wheel")),
        "required_shared_libraries": list(
            raw.get("required_shared_libraries") or REQUIRED_SHARED_LIBRARIES
        ),
        "mtp": bool(raw.get("mtp")),
        "spec_type": raw.get("spec_type", ""),
        "kv_quant": raw.get("kv_quant", ""),
        "no_think_prefix": raw.get("no_think_prefix", ""),
        "max_tokens": int(raw.get("max_tokens") or 0),
        "n_predict_min": int(raw.get("n_predict_min") or 0),
        "port_strategy": raw.get("port_strategy", ""),
        "props_verify_endpoint": raw.get("props_verify_endpoint", ""),
        "wheel_fallback_allowed": bool(raw.get("wheel_fallback_allowed")),
        "forbidden_models": list(raw.get("forbidden_models") or []),
        "forbidden_gpu_targets": list(raw.get("forbidden_gpu_targets") or []),
        "gpu_target": raw.get("gpu_target", ""),
    }
    return out


def _shared_library_present(name: str, dirs: Sequence[Path]) -> bool:
    for directory in dirs:
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            child_name = child.name
            if child_name == name or child_name.startswith(f"{name}."):
                return True
    return False


def build_package_manifest(
    *,
    env: Mapping[str, str] | None = None,
    shared_library_search_dirs: Sequence[Path | str] = (),
) -> JsonDict:
    source = os.environ if env is None else env
    model_raw = str(source.get("CARNOT_ARC_GGUF_PATH", "") or "").strip()
    server_raw = str(source.get("CARNOT_LLAMA_SERVER", "") or "").strip()
    model_path = Path(model_raw) if model_raw else None
    server_path = Path(server_raw) if server_raw else None
    search_dirs = [Path(item) for item in shared_library_search_dirs]
    if server_path is not None:
        search_dirs.append(server_path.parent)
        search_dirs.append(server_path.parent / "lib")
    shared_libraries = [
        {
            "name": name,
            "required": True,
            "present": _shared_library_present(name, search_dirs),
        }
        for name in REQUIRED_SHARED_LIBRARIES
    ]
    blocked_resources: list[str] = []
    if not (model_path is not None and model_path.is_file()):
        blocked_resources.append("model_file")
    if not (server_path is not None and server_path.is_file()):
        blocked_resources.append("llama_server_binary")
    blocked_resources.extend(
        item["name"] for item in shared_libraries if item["present"] is not True
    )
    return {
        "complete": not blocked_resources,
        "blocked_resources": blocked_resources,
        "model_files": [
            {
                # Canonical pin, not a literal -- see the import block at the top of this
                # module for why this file no longer re-types the generator identity.
                "model_id": ARC_LIVE_GENERATOR_MODEL_ID,
                "filename": ARC_LIVE_GENERATOR_MODEL_FILENAME,
                "path_env": "CARNOT_ARC_GGUF_PATH",
                "path": str(model_path) if model_path is not None else "",
                "present": model_path is not None and model_path.is_file(),
                "required": True,
            }
        ],
        "llama_server": {
            "path_env": "CARNOT_LLAMA_SERVER",
            "path": str(server_path) if server_path is not None else "",
            "present": server_path is not None and server_path.is_file(),
            "kind": "cuda-12.8-binary",
            "binary_not_wheel": True,
            "required": True,
        },
        "shared_libraries": shared_libraries,
        "entrypoint": {
            "module": "carnot.agentic.arc_competition_agent",
            "factory": "make_carnot_agent",
            "policy": "E3AgentPolicy",
        },
        "kaggle_constraints": {
            "vram_gb": 16,
            "rpm_cap": RPM_CAP,
            "runtime_hours": 12,
        },
    }


def _checklist_item(item_id: str, label: str, passed: bool, detail: str) -> JsonDict:
    return {
        "id": item_id,
        "label": label,
        "status": "pass" if passed else "blocked",
        "detail": detail,
    }


def _readiness_checklist(
    *,
    entrypoint_validation: Mapping[str, Any],
    frozen_generator_config: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    smoke_episode: Mapping[str, Any],
    parity_test: Mapping[str, Any],
) -> list[JsonDict]:
    smoke_passed = (
        smoke_episode.get("ran") is True
        and smoke_episode.get("within_action_budget") is True
        and smoke_episode.get("within_rpm_budget") is True
        and smoke_episode.get("solve_claim_made") is False
    )
    return [
        _checklist_item(
            "entrypoint_imports",
            "entrypoint imports and constructs",
            entrypoint_validation.get("imported") is True
            and entrypoint_validation.get("constructed") is True
            and entrypoint_validation.get("policy_class") == "E3AgentPolicy",
            str(
                entrypoint_validation.get("blocked_resource")
                or "make_carnot_agent -> E3AgentPolicy"
            ),
        ),
        _checklist_item(
            "frozen_generator_wired",
            "frozen generator wired",
            frozen_generator_config.get("confirmed") is True,
            f"{ARC_LIVE_GENERATOR_REPO_SUBSTR} with mtp/q8/no_think/n_predict guard",
        ),
        _checklist_item(
            "manifest_complete",
            "package manifest complete",
            package_manifest.get("complete") is True,
            ",".join(package_manifest.get("blocked_resources") or [])
            or "model,binary,entrypoint present",
        ),
        _checklist_item(
            "smoke_episode_ran",
            "offline smoke episode ran",
            smoke_passed,
            str(smoke_episode.get("blocked_resource") or "offline smoke within budget"),
        ),
        _checklist_item(
            "parity_green",
            "submitted-agent parity green",
            parity_test.get("passed") is True,
            str(parity_test.get("command") or "tests/python/test_arc_submitted_agent_parity.py"),
        ),
    ]


def _blocked_resource(
    preconditions_checked: Mapping[str, Any],
    checklist: Sequence[Mapping[str, Any]],
) -> str:
    if preconditions_checked.get("ok") is not True:
        return str(preconditions_checked.get("blocked_resource") or "preconditions")
    for item in checklist:
        if item.get("status") == "blocked":
            if item.get("id") == "manifest_complete":
                return "manifest_resources"
            if item.get("id") == "smoke_episode_ran":
                return "offline_smoke"
            if item.get("id") == "parity_green":
                return "parity_test"
            if item.get("id") == "frozen_generator_wired":
                return "frozen_generator"
            return "entrypoint"
    return "unknown"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    entrypoint_validation: Mapping[str, Any],
    frozen_generator_config: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    smoke_episode: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    submitted_agent_config: Mapping[str, Any] | None = None,
) -> JsonDict:
    checklist = _readiness_checklist(
        entrypoint_validation=entrypoint_validation,
        frozen_generator_config=frozen_generator_config,
        package_manifest=package_manifest,
        smoke_episode=smoke_episode,
        parity_test=parity_test,
    )
    ready = preconditions_checked.get("ok") is True and all(
        item["status"] == "pass" for item in checklist
    )
    blocked_resource = _blocked_resource(preconditions_checked, checklist)
    honest_verdict = (
        "success: submission_package_ready"
        if ready
        else f"complete: submission_package_blocked_{blocked_resource}"
    )
    submitted_snapshot = _jsonable(submitted_agent_config or {})
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "submission_package_ready": bool(ready),
        "readiness_checklist": checklist,
        "frozen_generator_confirmed": frozen_generator_config.get("confirmed") is True,
        "parity_test_green": parity_test.get("passed") is True,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "package_manifest": dict(package_manifest),
        "entrypoint_validation": dict(entrypoint_validation),
        "frozen_generator_config": dict(frozen_generator_config),
        "smoke_episode": dict(smoke_episode),
        "parity_test": dict(parity_test),
        "submitted_agent_config": submitted_snapshot,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard")
    if artifact.get("operator_only") is not True:
        errors.append("operator_only")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    checklist = artifact.get("readiness_checklist")
    checklist_pass = isinstance(checklist, list) and all(
        isinstance(item, Mapping) and item.get("status") == "pass" for item in checklist
    )
    if artifact.get("submission_package_ready") is not (
        artifact.get("preconditions_checked", {}).get("ok") is True and checklist_pass
    ):
        errors.append("submission_package_ready_gate")
    if artifact.get("frozen_generator_confirmed") is not (
        artifact.get("frozen_generator_config", {}).get("confirmed") is True
    ):
        errors.append("frozen_generator_confirmed")
    if artifact.get("parity_test_green") is not (
        artifact.get("parity_test", {}).get("passed") is True
    ):
        errors.append("parity_test_green")
    if artifact.get("smoke_episode", {}).get("solve_claim_made") is not False:
        errors.append("smoke_episode_solve_claim")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("submission_package_ready") is True:
        if verdict != "success: submission_package_ready":
            errors.append("honest_verdict")
    elif not verdict.startswith("complete: submission_package_blocked_"):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def validate_entrypoint() -> JsonDict:  # pragma: no cover - ARC package boundary.
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
        from carnot.experiment_4605_live_integration_scored_agent import _NoOpProposer

        class _Base:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.game_id = str(kwargs.get("game_id") or "ar25")

        agent_cls = make_carnot_agent(_Base, proposer=_NoOpProposer())
        agent = agent_cls(game_id="ar25")
        policy = getattr(agent, "_policy", None)
        return {
            "imported": True,
            "constructed": isinstance(policy, E3AgentPolicy),
            "entrypoint": "carnot.agentic.arc_competition_agent.make_carnot_agent",
            "policy_class": policy.__class__.__name__ if policy is not None else "",
            "max_actions": int(getattr(agent_cls, "MAX_ACTIONS", 0)),
            "blocked_resource": "",
        }
    except Exception as exc:
        return {
            "imported": False,
            "constructed": False,
            "entrypoint": "carnot.agentic.arc_competition_agent.make_carnot_agent",
            "policy_class": "",
            "max_actions": 0,
            "blocked_resource": "entrypoint",
            "error": repr(exc)[:300],
        }


def _runtime_frozen_generator_config() -> tuple[
    JsonDict, JsonDict
]:  # pragma: no cover - import boundary.
    try:
        from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

        submitted = _jsonable(SUBMITTED_AGENT_CONFIG)
        return frozen_generator_config_from_submitted(SUBMITTED_AGENT_CONFIG), submitted
    except Exception as exc:
        return (
            {
                "confirmed": False,
                "checks": {"submitted_config_importable": False},
                "error": repr(exc)[:300],
            },
            {},
        )


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - filesystem boundary.
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "spec_has_req_4744": False,
        "arc_competition_agent_importable": False,
        "offline_arcade_importable": False,
        "ok": False,
    }
    spec_path = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4744"] = spec_path.exists() and "REQ-CAPSTONE-4744" in spec_path.read_text(
        encoding="utf-8"
    )
    try:
        import carnot.agentic.arc_competition_agent as _agent

        checks["arc_competition_agent_importable"] = _agent is not None
    except Exception as exc:
        checks["blocked_resource"] = "arc_competition_agent"
        checks["error"] = repr(exc)[:300]
        return checks
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_importable"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:300]
        return checks
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "spec_has_req_4744",
            "arc_competition_agent_importable",
            "offline_arcade_importable",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "preconditions"
    return checks


def run_smoke_episode(
    root: Path | str = REPO_ROOT,
    *,
    action_budget: int = SMOKE_ACTION_BUDGET,
) -> JsonDict:  # pragma: no cover - ARC runtime boundary.
    try:
        from arcengine import GameAction
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_competition_agent import make_carnot_agent
        from carnot.experiment_4605_live_integration_scored_agent import _NoOpProposer

        class _Base:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.game_id = str(kwargs.get("game_id") or "ar25")

        env_root = Path(root) / "environment_files"
        games = sorted(path.name for path in env_root.iterdir() if path.is_dir())
        game = os.environ.get("CARNOT_ARC_SMOKE_GAME") or (games[0] if games else "ar25")
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        agent = make_carnot_agent(_Base, proposer=_NoOpProposer())(game_id=game)
        frames: list[Any] = []
        latest = None
        actions_taken = 0
        resets = 0
        start = time.monotonic()
        for _ in range(int(action_budget)):
            if agent.is_done(frames, latest):
                break
            action = agent.choose_action(frames, latest)
            if action is GameAction.RESET:
                latest = env.reset()
                resets += 1
            else:
                raw_data = action.action_data.model_dump()
                data = {key: raw_data[key] for key in ("x", "y") if key in raw_data}
                latest = env.step(action, data=data or None)
                actions_taken += 1
            frames.append(latest)
            if actions_taken >= 1:
                break
        elapsed = max(time.monotonic() - start, 1e-6)
        return {
            "ran": True,
            "game": game,
            "actions_taken": actions_taken,
            "resets": resets,
            "action_budget": int(action_budget),
            "rpm_cap": RPM_CAP,
            "estimated_rpm": round((actions_taken / elapsed) * 60.0, 6),
            "within_action_budget": actions_taken <= int(action_budget),
            "within_rpm_budget": actions_taken <= RPM_CAP,
            "solve_claim_made": False,
            "blocked_resource": "",
        }
    except Exception as exc:
        return {
            "ran": False,
            "game": "",
            "actions_taken": 0,
            "action_budget": int(action_budget),
            "rpm_cap": RPM_CAP,
            "within_action_budget": False,
            "within_rpm_budget": False,
            "solve_claim_made": False,
            "blocked_resource": "offline_smoke",
            "error": repr(exc)[:300],
        }


def run_parity_test(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - subprocess boundary.
    root_path = Path(root)
    pytest_bin = root_path / ".venv" / "bin" / "pytest"
    if pytest_bin.exists():
        cmd = [
            str(pytest_bin),
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
        ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        return {
            "passed": completed.returncode == 0,
            "command": " ".join(cmd),
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "duration_s": round(time.monotonic() - start, 6),
        }
    except Exception as exc:
        return {
            "passed": False,
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": repr(exc)[:1000],
            "duration_s": round(time.monotonic() - start, 6),
        }


PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
EntrypointValidator = Callable[[], Mapping[str, Any]]
FrozenGeneratorLoader = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
ManifestBuilder = Callable[[], Mapping[str, Any]]
SmokeRunner = Callable[[], Mapping[str, Any]]
ParityRunner = Callable[[Path], Mapping[str, Any]]


def run(
    root: Path | str = REPO_ROOT,
    *,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    entrypoint_validator: EntrypointValidator = validate_entrypoint,
    frozen_generator_loader: FrozenGeneratorLoader = _runtime_frozen_generator_config,
    manifest_builder: ManifestBuilder = build_package_manifest,
    smoke_runner: SmokeRunner = run_smoke_episode,
    parity_runner: ParityRunner = run_parity_test,
    write: bool = True,
) -> JsonDict:
    root_path = Path(root)
    start = time.monotonic()
    preconditions = dict(preconditions_checker(root_path))
    frozen_generator_config, submitted_config = frozen_generator_loader()
    if preconditions.get("ok") is True:
        entrypoint = dict(entrypoint_validator())
        manifest = dict(manifest_builder())
        smoke = dict(smoke_runner())
        parity = dict(parity_runner(root_path))
    else:
        blocked = str(preconditions.get("blocked_resource") or "preconditions")
        entrypoint = {
            "imported": False,
            "constructed": False,
            "entrypoint": "carnot.agentic.arc_competition_agent.make_carnot_agent",
            "policy_class": "",
            "max_actions": 0,
            "blocked_resource": blocked,
        }
        manifest = dict(manifest_builder())
        smoke = {
            "ran": False,
            "game": "",
            "actions_taken": 0,
            "action_budget": SMOKE_ACTION_BUDGET,
            "rpm_cap": RPM_CAP,
            "within_action_budget": False,
            "within_rpm_budget": False,
            "solve_claim_made": False,
            "blocked_resource": blocked,
        }
        parity = {
            "passed": False,
            "command": ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
            "returncode": -1,
            "blocked_resource": blocked,
        }
    artifact = build_artifact(
        preconditions_checked=preconditions,
        entrypoint_validation=entrypoint,
        frozen_generator_config=frozen_generator_config,
        package_manifest=manifest,
        smoke_episode=smoke,
        parity_test=parity,
        duration_s=time.monotonic() - start,
        submitted_agent_config=submitted_config,
    )
    if write:
        output = root_path / RESULT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
