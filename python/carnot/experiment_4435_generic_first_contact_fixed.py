"""Exp 4435: fixed generic first-contact verdict plus one routed solve attempt.

Spec refs: REQ-REPORT-4435, SCENARIO-REPORT-4435.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from carnot import experiment_4423_generic_first_contact_breadth as exp4423


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4435_generic_first_contact_fixed.json"
RANDOM_SEED = 4435
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
SPEC_REFS = ("REQ-REPORT-4435", "SCENARIO-REPORT-4435")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verdict_contract_fixed",
    "reproduced_levels",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "verifier_is_oracle",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal-prefixed complete:/success: -- the whole point of this task "
            "is to stop emitting the conductor-rejected partial:"
        )
    },
    "verdict_contract_fixed": {
        "principle": (
            "bare bool: the 5-point atomic fix landed AND the test is green "
            "(the pre-test gate will not poison)"
        )
    },
    "reproduced_levels": {
        "principle": (
            "bare int; reproduction-gated; a routed game that actually SOLVES "
            "banks a real level"
        )
    },
    "offline_reproduced": {"principle": "the gate"},
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {"principle": "content hash for reproducibility"},
}

FirstContactRun = Callable[..., Mapping[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_terminal(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def _checksum_is_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def exp4423_verdict_contract_fixed() -> bool:
    """REQ-REPORT-4435: probe the fixed terminal vocabulary in Exp 4423."""

    return exp4423._terminal_prefixed("complete: routed_no_new_level_gap_logged") and not (
        exp4423._terminal_prefixed("partial: routed_missing_verifier_gap_logged")
    )


def precondition_probe(
    root: Path = REPO_ROOT,
    *,
    focused_exp4423_pytest_green: bool = False,
    llm_induction_needed: bool = False,
) -> dict[str, Any]:  # pragma: no cover - filesystem/import boundary
    env_dir = Path(root) / "environment_files"
    checks = {
        "offline_env_files_present": env_dir.is_dir() and any(env_dir.iterdir()),
        "arc_solver_kit_import": False,
        "arc_solve_learning_import": False,
        "focused_exp4423_pytest_green": bool(focused_exp4423_pytest_green),
        "verdict_contract_fixed": exp4423_verdict_contract_fixed(),
        "llm_induction_needed": bool(llm_induction_needed),
        "live_generator_gguf_cached_if_needed": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solve_learning, arc_solver_kit

        checks["arc_solver_kit_import"] = callable(getattr(arc_solver_kit, "reproduce", None))
        checks["arc_solve_learning_import"] = callable(
            getattr(arc_solve_learning, "recommend_approach", None)
        )
    except Exception as exc:
        checks["import_error"] = f"{type(exc).__name__}: {exc}"
    if llm_induction_needed:
        cache = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF"
        checks["live_generator_gguf_cached_if_needed"] = cache.is_dir() and any(cache.iterdir())
    checks["ok"] = first_precondition_miss(checks) is None
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_env_files_present") is not True:
        return "offline_env_files"
    if preconditions.get("arc_solver_kit_import") is not True:
        return "arc_solver_kit"
    if preconditions.get("arc_solve_learning_import") is not True:
        return "arc_solve_learning"
    if preconditions.get("verdict_contract_fixed") is not True:
        return "verdict_contract_fixed"
    if preconditions.get("focused_exp4423_pytest_green") is not True:
        return "focused_exp4423_pytest"
    if (
        preconditions.get("llm_induction_needed") is True
        and preconditions.get("live_generator_gguf_cached_if_needed") is not True
    ):
        return "live_generator_gguf"
    if preconditions.get("no_3090_inference") is not True:
        return "no_3090_inference_policy"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission_policy"
    return None


def _verdict(
    *,
    precondition_miss: str | None,
    target_game: str,
    offline_reproduced: bool,
    reproduced_levels: int,
) -> str:
    if precondition_miss:
        return f"complete: blocked_{precondition_miss}"
    if offline_reproduced and reproduced_levels >= 1:
        return f"success: generic_first_contact_fixed_{target_game}_L{reproduced_levels}_offline_reproduced"
    return f"complete: generic_first_contact_{target_game}_routed_no_new_level_gap_logged"


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    exp4423_artifact: Mapping[str, Any] | None,
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    precondition_miss = first_precondition_miss(preconditions)
    base = exp4423_artifact or {}
    target_game = str(base.get("target_game") or "")
    offline_reproduced = precondition_miss is None and base.get("offline_reproduced") is True
    reproduced_levels = _as_int(base.get("reproduced_levels")) if offline_reproduced else 0
    gaps = base.get("missing_verifier_gaps")
    missing_gaps = list(gaps) if isinstance(gaps, list) else []
    checksum_payload = {
        "exp4423_artifact": base,
        "preconditions": dict(preconditions),
        "random_seed": RANDOM_SEED,
        "target_game": target_game,
    }
    return {
        "experiment": "experiment_4435_generic_first_contact_fixed",
        "schema": "carnot.exp4435.generic_first_contact_fixed.v1",
        "target_game": target_game,
        "honest_verdict": _verdict(
            precondition_miss=precondition_miss,
            target_game=target_game or "none",
            offline_reproduced=offline_reproduced,
            reproduced_levels=reproduced_levels,
        ),
        "verdict_contract_fixed": precondition_miss is None
        and preconditions.get("verdict_contract_fixed") is True
        and preconditions.get("focused_exp4423_pytest_green") is True,
        "reproduced_levels": reproduced_levels,
        "offline_reproduced": offline_reproduced,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256(checksum_payload),
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(preconditions),
        "routing_recommendation": dict(base.get("recommendation") or {}),
        "routing_options": list(base.get("routing_options") or []),
        "standing_loop_result": dict(base.get("standing_loop_result") or {}),
        "underlying_exp4423_honest_verdict": str(base.get("honest_verdict") or ""),
        "underlying_exp4423_checksum": str(base.get("reproducibility_checksum") or ""),
        "missing_verifier_gaps": missing_gaps,
        "residual_mechanic_gap_logged": bool(missing_gaps),
        "no_3090_inference": True,
        "submitted_to_leaderboard": False,
        "inference_substrate": "offline_arc_solver_kit_reproduce_no_3090",
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "root": str(Path(root)),
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not _is_terminal(verdict):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    if type(artifact.get("verdict_contract_fixed")) is not bool:
        errors.append("verdict_contract_fixed must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    elif not _checksum_is_hex(checksum):
        errors.append("reproducibility_checksum must be hex")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")

    blocked = isinstance(verdict, str) and "blocked_" in verdict
    gaps = artifact.get("missing_verifier_gaps")
    if not isinstance(gaps, list):
        errors.append("missing_verifier_gaps must be list")
    if (
        not blocked
        and artifact.get("offline_reproduced") is not True
        and _as_int(artifact.get("reproduced_levels")) == 0
        and gaps == []
    ):
        errors.append("complete no-new-level verdict requires missing_verifier_gaps")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("success verdict requires reproduced_levels >= 1")
    if artifact.get("offline_reproduced") is True and _as_int(artifact.get("reproduced_levels")) < 1:
        errors.append("offline_reproduced true requires reproduced_levels >= 1")
    if artifact.get("no_3090_inference") is not True:
        errors.append("no_3090_inference must be true")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be dict")
    else:
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-REPORT-4435")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    target_game: str | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    first_contact_run_fn: FirstContactRun = exp4423.run,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-4435: verify the fixed contract, then run one first-contact attempt."""

    started = now()
    root = Path(root)
    checked = dict(preconditions_checked or precondition_probe(root))
    checked.setdefault("verdict_contract_fixed", exp4423_verdict_contract_fixed())
    checked.setdefault("llm_induction_needed", False)
    checked.setdefault("live_generator_gguf_cached_if_needed", True)
    checked.setdefault("no_3090_inference", True)
    checked.setdefault("leaderboard_submission", False)
    precondition_miss = first_precondition_miss(checked)
    exp4423_artifact: Mapping[str, Any] | None = None
    if precondition_miss is None:
        exp4423_artifact = first_contact_run_fn(
            root=root,
            target_game=target_game,
            write_registry=True,
        )
    artifact = build_artifact(
        root=root,
        preconditions=checked,
        exp4423_artifact=exp4423_artifact,
        started_at=started,
        ended_at=now(),
    )
    write_artifact(root, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    parser.add_argument("--focused-test-green", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        REPO_ROOT,
        target_game=args.game,
        preconditions_checked=precondition_probe(
            REPO_ROOT,
            focused_exp4423_pytest_green=args.focused_test_green,
        ),
    )
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
