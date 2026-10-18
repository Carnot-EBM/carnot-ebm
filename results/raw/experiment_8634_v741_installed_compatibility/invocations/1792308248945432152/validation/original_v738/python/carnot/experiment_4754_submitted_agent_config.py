"""Experiment 4754: confirm submitted `.437` lever config integration.

Spec refs: REQ-ARC-WMTE-4754, SCENARIO-ARC-WMTE-4754.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# THE CANONICAL GENERATOR PIN, imported rather than re-typed. This module's generator assertions
# were hardcoded to the retired Qwen3.5-9B-MTP and went stale the moment the 2026-07-28 operator
# directive re-pinned the live generator to gemma-4-31B-it. Reading the constants means a future
# switch updates this readiness gate for free instead of leaving it asserting a model nothing runs.
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MODEL_FILENAME,
    ARC_LIVE_GENERATOR_MODEL_ID,
    ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT,
    ARC_LIVE_GENERATOR_REPO_SUBSTR,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
GGUFFinder = Callable[[], list[str]]
StatusCheck = Callable[[], Mapping[str, Any]]
SmokeRunner = Callable[[Mapping[str, str]], Mapping[str, Any]]
LintRunner = Callable[[Path], Mapping[str, Any]]
ConfigLoader = Callable[[], Mapping[str, Any]]

EXPERIMENT = "experiment_4754_submitted_agent_config"
SCHEMA = "carnot.exp4754.submitted_agent_config.v1"
RESULT_RELATIVE_PATH = "results/experiment_4754_submitted_agent_config.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARC_AGENT_RELATIVE_PATH = "python/carnot/agentic/arc_competition_agent.py"
SUBMISSION_ENTRYPOINT_RELATIVE_PATH = "scripts/kaggle/submission_kernel/main.py"
A1_RELATIVE_PATH = "results/experiment_4749_structured_engine_vs_freeform.json"
A2_RELATIVE_PATH = "results/experiment_4750_structural_alignment_detector_fix.json"

RANDOM_SEED = 4754
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ["REQ-ARC-WMTE-4754", "SCENARIO-ARC-WMTE-4754"]
TERMINAL_PREFIXES = (
    "success_",
    "complete_",
    "blocked_",
    "success:",
    "complete:",
    "passed:",
    "shipped:",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; an integration-confirmed run is success_/complete_."
    },
    "inference_substrate": {"principle": "live_llm_inference."},
    "preconditions_checked": {"principle": "records GGUF/arcade checks."},
    "agent_constructs_and_smoke_runs": {
        "principle": (
            "the agent entrypoint builds + a smoke step runs offline -- the integration gate."
        )
    },
    "submission_package_ready": {
        "principle": (
            "True only if OPERATOR-ready; this task NEVER submits "
            "(operator-only external publication)."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "env_gate_state",
    "levers_integrated",
    "upstream_validation",
    "live_path_hooks",
    "frozen_generator_intact",
    "arc_orphan_solver_lint",
    "submission_package",
    "submitted_agent_config",
    "submitted_to_leaderboard",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "field_principles",
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


def _load_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _is_success_verdict(value: Any) -> bool:
    return str(value or "").startswith(("success:", "success_"))


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _structured_engine_validated(artifact: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(artifact, Mapping):
        return False, "a1_artifact_missing_or_not_mapping"
    if not _is_success_verdict(artifact.get("honest_verdict")):
        return False, "a1_upstream_not_success"
    if artifact.get("verifier_is_oracle") is not False:
        return False, "a1_verifier_oracle_not_false"
    if artifact.get("structured_engine_non_degenerate") is not True:
        return False, "a1_structured_engine_degenerate_or_unproven"
    structured = _as_float(artifact.get("structured_heldout_accuracy"))
    freeform = _as_float(artifact.get("freeform_heldout_accuracy"))
    if structured < 0.5 and structured <= freeform:
        return False, "a1_no_accuracy_or_l2_validation"
    return True, "validated"


def _fixed_detector_validated(artifact: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(artifact, Mapping):
        return False, "a2_artifact_missing_or_not_mapping"
    if not _is_success_verdict(artifact.get("honest_verdict")):
        return False, "a2_upstream_not_success"
    if artifact.get("verifier_is_oracle") is not False:
        return False, "a2_verifier_oracle_not_false"
    fixed = (
        artifact.get("structural_alignment_detector_fixed") is True
        or artifact.get("fixed_detector_validated") is True
        or artifact.get("structural_goal_provider_fixed") is True
    )
    if not fixed:
        return False, "a2_fixed_detector_unproven"
    return True, "validated"


def audit_env_gate_state(
    *,
    a1_artifact: Mapping[str, Any] | None,
    a2_artifact: Mapping[str, Any] | None,
    live_source_text: str,
) -> JsonDict:
    structured_hook = (
        "CARNOT_ARC_STRUCTURED_ENGINE" in live_source_text
        and "arc_structured_world_model" in live_source_text
    )
    detector_hook = (
        "structural_alignment_goal_candidate" in live_source_text
        and "CARNOT_ARC_TRUST_METRIC" in live_source_text
    )
    a1_valid, a1_reason = _structured_engine_validated(a1_artifact)
    a2_valid, a2_reason = _fixed_detector_validated(a2_artifact)
    levers: list[str] = []

    if a1_valid and structured_hook:
        structured_gate = "1"
        levers.append("A1_structured_engine")
        a1_reason = "validated_and_live_hook_present"
    else:
        structured_gate = "0"
        if a1_valid:
            a1_reason = "structured_engine_live_hook_missing"

    if a2_valid and detector_hook:
        trust_metric = "cell_recall"
        levers.append("A2_fixed_structural_alignment_detector")
        a2_reason = "validated_and_live_hook_present"
    else:
        trust_metric = "exact"
        if a2_valid:
            a2_reason = "fixed_detector_live_hook_missing"

    return {
        "env_gate_state": {
            "CARNOT_ARC_STRUCTURED_ENGINE": structured_gate,
            "CARNOT_ARC_TRUST_METRIC": trust_metric,
        },
        "levers_integrated": levers,
        "a1": {"validated": a1_valid, "reason": a1_reason},
        "a2": {"validated": a2_valid, "reason": a2_reason},
        "live_path_hooks": {
            "structured_engine_hook_present": structured_hook,
            "fixed_detector_hook_present": detector_hook,
        },
    }


def _default_gguf_finder() -> list[str]:  # pragma: no cover - filesystem boundary.
    pattern = (
        f"~/.cache/huggingface/hub/models--unsloth--{ARC_LIVE_GENERATOR_MODEL_ID.split('/')[-1]}/"
        "snapshots/*/*.gguf"
    )
    return sorted(glob.glob(os.path.expanduser(pattern)))


def _default_offline_arcade_checker() -> JsonDict:  # pragma: no cover - import boundary.
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        return {"offline_arcade_ok": True}
    except Exception as exc:
        return {"offline_arcade_ok": False, "offline_arcade_error": repr(exc)}


def _default_agent_import_checker() -> JsonDict:  # pragma: no cover - import boundary.
    try:
        from carnot.agentic.arc_competition_agent import make_carnot_agent

        return {"make_carnot_agent_import_ok": callable(make_carnot_agent)}
    except Exception as exc:
        return {"make_carnot_agent_import_ok": False, "make_carnot_agent_import_error": repr(exc)}


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    gguf_finder: GGUFFinder | None = None,
    offline_arcade_checker: StatusCheck | None = None,
    agent_import_checker: StatusCheck | None = None,
) -> JsonDict:
    root_path = Path(root)
    ggufs = list((gguf_finder or _default_gguf_finder)())
    offline = dict((offline_arcade_checker or _default_offline_arcade_checker)())
    agent_import = dict((agent_import_checker or _default_agent_import_checker)())
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "pinned_generator_gguf_cached": bool(ggufs),
        "pinned_generator_gguf_paths": ggufs,
        "spec_has_req_4754": "REQ-ARC-WMTE-4754" in spec_text,
        "submission_entrypoint_present": (root_path / SUBMISSION_ENTRYPOINT_RELATIVE_PATH).exists(),
    }
    checks.update(offline)
    checks.update(agent_import)
    required = (
        "pinned_generator_gguf_cached",
        "offline_arcade_ok",
        "make_carnot_agent_import_ok",
        "agents_md_read",
        "codex_md_read",
        "spec_has_req_4754",
        "submission_entrypoint_present",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


def _default_submitted_agent_config_loader() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))


def frozen_generator_status(
    submitted_agent_config: Mapping[str, Any],
    gguf_paths: list[str],
    submission_entrypoint_text: str,
) -> JsonDict:
    frozen = submitted_agent_config.get("frozen_generator")
    frozen_map = frozen if isinstance(frozen, Mapping) else {}
    model_id = str(frozen_map.get("model_id") or "")
    filename = str(frozen_map.get("model_filename") or "")
    cached_match = any(filename and filename in path for path in gguf_paths)
    # THE MTP EXPECTATION IS READ FROM THE **SCORED** CONSTANT, and that choice is the whole point.
    #
    # `SUBMITTED_AGENT_CONFIG` describes the KAGGLE launch, so the only constant it can legitimately
    # be compared against is `ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT`. The local dev default
    # (`ARC_LIVE_GENERATOR_MTP_DEFAULT`) is correctly DIFFERENT -- MTP is a net throughput loss on a
    # 24 GB card that has to offload FFN blocks to fit the head, and a ~1.4x win on the 96 GB scored
    # card -- so comparing the scored config against the local constant makes this gate red exactly
    # when the submission is configured correctly. `experiment_4744` already reads the scored
    # constant here (see its `mtp_matches_model`); this module read a hardcoded OFF, so the two
    # readiness gates over the SAME config demanded contradictory values and one of them was
    # guaranteed red no matter how the submission was configured.
    #
    # HISTORICAL NOTE, PRESERVED BECAUSE THE PREMISE WAS FALSIFIED THE SAME DAY. An earlier
    # 2026-07-28 pass hardcoded `mtp is False` / `spec_type is None` here on the stated ground that
    # "gemma-4-31B-it has no MTP heads, so `--spec-type draft-mtp` would double-load 18.3 GB of
    # weights and cudaMalloc-fail". That premise is WRONG: this model's MTP head is a SEPARATE
    # 491 MiB GGUF (`mtp-gemma-4-31B-it-Q8_0.gguf`, whose header declares
    # `general.architecture = gemma4-assistant`), which is exactly why no `nextn_predict_layers`
    # key was found inside the MAIN file. Enabling MTP loads the head, not a second copy of the
    # weights. What survives from that reasoning is the narrower and still-correct warning that
    # `--model-draft <the main gguf>` IS the double-load, which `_resolve_mtp_head()` now prevents
    # structurally rather than by leaving MTP permanently off.
    scored_mtp_on = ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"
    declared = (
        # Re-pinned 2026-07-28: canonical constants, not literals (see the import block above).
        model_id == ARC_LIVE_GENERATOR_MODEL_ID
        and frozen_map.get("repo_substr") == ARC_LIVE_GENERATOR_REPO_SUBSTR
        and frozen_map.get("mtp") is scored_mtp_on
        and frozen_map.get("spec_type") == ("draft-mtp" if scored_mtp_on else None)
        and frozen_map.get("kv_quant") == "q8_0"
        and frozen_map.get("wheel_fallback_allowed") is False
    )
    entrypoint_keeps_generator = (
        ARC_LIVE_GENERATOR_REPO_SUBSTR in submission_entrypoint_text
        and "CARNOT_ARC_GGUF_PATH" in submission_entrypoint_text
    )
    return {
        "intact": bool(declared and cached_match and entrypoint_keeps_generator),
        "submitted_config_declares_pinned_generator": bool(declared),
        "cached_gguf_matches": bool(cached_match),
        "submission_entrypoint_resolves_generator": bool(entrypoint_keeps_generator),
        "model_id": model_id,
        "model_filename": filename,
    }


def submission_package_status(
    root: Path | str,
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    rel = str(submitted_agent_config.get("live_submit_package_path") or "")
    package_path = Path(root) / rel if rel else Path(root) / "__missing_package__"
    return {
        "operator_package_present": bool(rel and package_path.exists()),
        "path": rel,
    }


def _default_agent_smoke_runner(
    env_gate_state: Mapping[str, str],
) -> JsonDict:  # pragma: no cover - import boundary.
    old_env = {key: os.environ.get(key) for key in env_gate_state}
    try:
        for key, value in env_gate_state.items():
            os.environ[key] = value
        from carnot.agentic.arc_competition_agent import make_carnot_agent

        class _Base:
            def __init__(self) -> None:
                self.game_id = "paritytest"

        agent_cls = make_carnot_agent(_Base)
        agent = agent_cls()
        policy = getattr(agent, "_policy", None)
        move = policy.next_move([], None) if policy is not None else (None, None)
        return {
            "constructed": True,
            "smoke_step_ran": move[0] == "RESET" or isinstance(move[0], int),
            "policy": type(policy).__name__,
            "move": [move[0], move[1]],
            "env_gates_applied": dict(env_gate_state),
        }
    except Exception as exc:
        return {
            "constructed": False,
            "smoke_step_ran": False,
            "error": repr(exc),
            "env_gates_applied": dict(env_gate_state),
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _default_orphan_lint_runner(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    proc = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _verdict(
    gate_audit: Mapping[str, Any],
    smoke: Mapping[str, Any],
    frozen: Mapping[str, Any],
    lint: Mapping[str, Any],
) -> str:
    smoke_green = smoke.get("constructed") is True and smoke.get("smoke_step_ran") is True
    gate_green = frozen.get("intact") is True and lint.get("passed") is True and smoke_green
    if not gate_green:
        return "complete_437_submitted_config_confirmation_failed_gate"
    if gate_audit.get("levers_integrated"):
        return "success_437_validated_levers_integrated_entrypoint_green"
    return "complete_437_levers_unvalidated_config_unchanged_entrypoint_green"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    gate_audit: Mapping[str, Any],
    agent_smoke: Mapping[str, Any],
    frozen_generator_intact: Mapping[str, Any],
    orphan_lint: Mapping[str, Any],
    submission_package: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    smoke_green = (
        agent_smoke.get("constructed") is True and agent_smoke.get("smoke_step_ran") is True
    )
    package_ready = bool(
        preconditions_checked.get("ok") is True
        and smoke_green
        and frozen_generator_intact.get("intact") is True
        and orphan_lint.get("passed") is True
        and submission_package.get("operator_package_present") is True
    )
    verdict = _verdict(gate_audit, agent_smoke, frozen_generator_intact, orphan_lint)
    if preconditions_checked.get("ok") is not True:
        verdict = f"blocked_{preconditions_checked.get('blocked_resource', 'precondition')}"
        package_ready = False
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "agent_constructs_and_smoke_runs": dict(agent_smoke),
        "submission_package_ready": package_ready,
        "env_gate_state": dict(gate_audit.get("env_gate_state") or {}),
        "levers_integrated": list(gate_audit.get("levers_integrated") or []),
        "upstream_validation": {
            "a1": dict(gate_audit.get("a1") or {}),
            "a2": dict(gate_audit.get("a2") or {}),
        },
        "live_path_hooks": dict(gate_audit.get("live_path_hooks") or {}),
        "frozen_generator_intact": dict(frozen_generator_intact),
        "arc_orphan_solver_lint": dict(orphan_lint),
        "submission_package": dict(submission_package),
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "submitted_to_leaderboard": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": max(1.0, round(float(duration_s), 6)),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
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
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    smoke = artifact.get("agent_constructs_and_smoke_runs")
    if not (
        isinstance(smoke, Mapping)
        and type(smoke.get("constructed")) is bool
        and type(smoke.get("smoke_step_ran")) is bool
    ):
        errors.append("agent_constructs_and_smoke_runs")
    if type(artifact.get("submission_package_ready")) is not bool:
        errors.append("submission_package_ready_bool")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def run(
    root: Path | str = REPO_ROOT,
    *,
    gguf_finder: GGUFFinder | None = None,
    offline_arcade_checker: StatusCheck | None = None,
    agent_import_checker: StatusCheck | None = None,
    agent_smoke_runner: SmokeRunner | None = None,
    orphan_lint_runner: LintRunner | None = None,
    submitted_agent_config_loader: ConfigLoader | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    checks = check_preconditions(
        root_path,
        gguf_finder=gguf_finder,
        offline_arcade_checker=offline_arcade_checker,
        agent_import_checker=agent_import_checker,
    )
    if checks.get("ok") is True:
        submitted_config = dict(
            (submitted_agent_config_loader or _default_submitted_agent_config_loader)()
        )
    else:
        try:
            submitted_config = dict(
                (submitted_agent_config_loader or _default_submitted_agent_config_loader)()
            )
        except Exception:
            submitted_config = {}

    agent_source = (root_path / ARC_AGENT_RELATIVE_PATH).read_text(encoding="utf-8")
    submission_source = (root_path / SUBMISSION_ENTRYPOINT_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    a1 = _load_optional_json(root_path / A1_RELATIVE_PATH)
    a2 = _load_optional_json(root_path / A2_RELATIVE_PATH)
    gate_audit = audit_env_gate_state(
        a1_artifact=a1,
        a2_artifact=a2,
        live_source_text=agent_source,
    )
    env_gate_state = dict(gate_audit["env_gate_state"])
    gguf_paths = list(checks.get("pinned_generator_gguf_paths") or [])
    frozen = frozen_generator_status(submitted_config, gguf_paths, submission_source)
    package = submission_package_status(root_path, submitted_config)

    if checks.get("ok") is True:
        smoke = dict((agent_smoke_runner or _default_agent_smoke_runner)(env_gate_state))
        lint = dict((orphan_lint_runner or _default_orphan_lint_runner)(root_path))
    else:
        smoke = {
            "constructed": False,
            "smoke_step_ran": False,
            "blocked": str(checks.get("blocked_resource") or "precondition"),
            "env_gates_applied": env_gate_state,
        }
        lint = {"passed": False, "blocked": str(checks.get("blocked_resource") or "precondition")}

    artifact = build_artifact(
        preconditions_checked=checks,
        gate_audit=gate_audit,
        agent_smoke=smoke,
        frozen_generator_intact=frozen,
        orphan_lint=lint,
        submission_package=package,
        submitted_agent_config=submitted_config,
        duration_s=now() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive guard; tests assert schema before writes.
        raise ValueError(f"invalid {EXPERIMENT} artifact: {errors}")
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI boundary.
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
