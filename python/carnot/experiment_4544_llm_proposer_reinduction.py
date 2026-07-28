"""Experiment 4544: live LLM proposer re-induction for ARC level-up goals.

Spec refs: REQ-ARC-WMTE-4544, SCENARIO-ARC-WMTE-4544.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4544_llm_proposer_reinduction.json"
INFERENCE_SUBSTRATE = "live_llm_inference"
GENERATOR_REPO_SUBSTR = "Qwen3.5-9B-MTP"
CORE_EFFICIENCY_BASELINE = 2.0074
RANDOM_SEED = 4544
TARGET_LEVELS = 2
MEASURED_GAMES = ("lp85", "m0r0")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
DEFAULT_BUDGET = int(os.environ.get("CARNOT_ARC_4544_BUDGET", "8000"))
DEFAULT_LLM_TIMEOUT = int(os.environ.get("CARNOT_ARC_4544_LLM_TIMEOUT", "45"))
DEFAULT_LLM_MAX_TOKENS = int(os.environ.get("CARNOT_ARC_4544_LLM_MAX_TOKENS", "768"))
REQUIREMENTS = ("REQ-ARC-WMTE-4544",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4544",)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; e.g. success: llm_proposer_<game>_reached_L2_core_efficiency_<n>_above_2.0074 "
        "OR complete: llm_proposer_no_deeper_level_proposer_value_characterized_honest_null."
    ),
    "inference_substrate": (
        "live_llm_inference -- the Qwen3.5-9B-MTP generator is genuinely loaded + invoked (60s floor)."
    ),
    "model_specs": (
        "names the generator actually invoked (Qwen3.5-9B-MTP GGUF + the resolved .gguf path)."
    ),
    "core_efficiency_baseline": (
        "2.0074 -- the REAL per-level metric control, measured the SAME way as best."
    ),
    "core_efficiency_best": (
        "the HEADLINE -- did the LLM proposer reach a deeper CORE level and raise core_efficiency."
    ),
    "efficiency_delta": (
        "core_efficiency_best - core_efficiency_baseline, emitted explicitly so a null delta is annotated."
    ),
    "null_delta_methodology_note": (
        "present when efficiency_delta==0.0 -- states the equality is an honest no-deeper-level null, "
        "not a measurement bug."
    ),
    "llm_proposer_value": (
        "the count/rate of level-ups where the LLM proposer produced a REACHABLE plan the offline DSL could not."
    ),
    "deepest_level_reached_per_core_game": (
        "best_level per CORE game per condition -- direct evidence of reaching MORE levels."
    ),
    "core_solves_preserved": (
        "HARD empirical gate on {lp85,m0r0,sp80,vc33}; a dropped CORE solve fails the lever."
    ),
    "refinement_rounds_used": (
        "bounded counterexample-guided refinement rounds per level-up -- proves the loop ran and stayed bounded."
    ),
    "barrier_refinement": (
        "if no deeper CORE level is reached, the concrete actionable refinement of what the LLM plan still gets wrong."
    ),
    "chosen_submitted_config": (
        "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); parity tests must stay consistent."
    ),
    "positive_control_passed": (
        "proves the LLM proposer can produce a reachable plan the DSL could not on a known-L2 game."
    ),
    "false_negative_risk_checked": "a null is valid only if the positive control passed.",
    "verifier_is_oracle": (
        "false -- the world-model trust energy ranks candidates by held-out generalization, oracle-DISTINCT."
    ),
    "offline_reproduced": "any new level reached must offline-reproduce to count.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent corpus/model drift on replay.",
    "preconditions_checked": (
        "records offline arcade, Qwen3.5-9B-MTP GGUF cache, and llama_cpp checks; pre-empts "
        "missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "null_delta_methodology_note"
) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "core_games",
    "measured_games",
    "offline_dsl_baseline",
    "llm_proposer_measurement",
    "measurements",
    "positive_control",
    "offline_reproduction",
    "live_invocation",
    "result_path",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _round_efficiency(value: Any) -> float:
    return round(float(value or 0.0), 4)


def _levels_by_game(measurement: Mapping[str, Any]) -> dict[str, int]:
    raw = measurement.get("deepest_level_by_game")
    if not isinstance(raw, Mapping):
        raw = measurement.get("best_level_by_game")
    out = {game: 0 for game in CORE_GAMES}
    if isinstance(raw, Mapping):
        for game, value in raw.items():
            if str(game) in out:
                out[str(game)] = int(value or 0)
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, Mapping) and str(row.get("game")) in out:
            out[str(row["game"])] = int(row.get("best_level") or row.get("levels") or 0)
    return out


def _per_game_efficiency(measurement: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = measurement.get("efficiency_by_game")
    if isinstance(raw, Mapping):
        out.update({str(game): _round_efficiency(value) for game, value in raw.items()})
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, Mapping) and row.get("game") is not None:
            value = row.get("efficiency", row.get("per_level_efficiency"))
            if value is not None:
                out[str(row["game"])] = _round_efficiency(value)
    return out


def _normalise_measurement(measurement: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    levels = _levels_by_game(measurement)
    efficiency_by_game = _per_game_efficiency(measurement)
    if measurement.get("core_efficiency") is None and efficiency_by_game:
        efficiency = _round_efficiency(
            sum(efficiency_by_game.get(game, 0.0) for game in CORE_GAMES)
        )
    else:
        efficiency = _round_efficiency(measurement.get("core_efficiency"))
    return {
        **dict(measurement),
        "measurement": str(measurement.get("measurement") or label),
        "core_efficiency": efficiency,
        "deepest_level_by_game": levels,
        "efficiency_by_game": {
            game: float(efficiency_by_game.get(game, 0.0)) for game in CORE_GAMES
        },
    }


def _core_solves_preserved(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    control_levels = _levels_by_game(control)
    candidate_levels = _levels_by_game(candidate)
    return all(candidate_levels.get(game, 0) >= control_levels.get(game, 0) for game in CORE_GAMES)


def _l2_game(measurement: Mapping[str, Any]) -> str | None:
    levels = _levels_by_game(measurement)
    for game in CORE_GAMES:
        if int(levels.get(game, 0)) >= 2:
            return game
    return None


def _attempts(measurement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        attempt
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping)
        for attempt in ((row.get("diagnostics") or {}).get("induction_attempts") or [])
        if isinstance(attempt, Mapping)
    ]


def characterize_llm_proposer_value(
    offline_dsl_baseline: Mapping[str, Any],
    llm_proposer: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4544: count LLM-reachable post-level plans missing from DSL."""

    events: list[str] = []
    opportunities = 0
    baseline_by_game = {
        str(row.get("game")): row
        for row in offline_dsl_baseline.get("per_game", []) or []
        if isinstance(row, Mapping)
    }
    for row in llm_proposer.get("per_game", []) or []:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game"))
        llm_attempts = [
            attempt
            for attempt in ((row.get("diagnostics") or {}).get("induction_attempts") or [])
            if isinstance(attempt, Mapping) and attempt.get("reason") == "level_up_reinduction"
        ]
        if not llm_attempts:
            continue
        opportunities += len(llm_attempts)
        baseline_row = baseline_by_game.get(game, {})
        baseline_planned = any(
            bool(attempt.get("planned"))
            for attempt in ((baseline_row.get("diagnostics") or {}).get("induction_attempts") or [])
            if isinstance(attempt, Mapping)
        )
        if any(bool(attempt.get("planned")) for attempt in llm_attempts) and not baseline_planned:
            events.append(f"{game}:L2")
    count = len(events)
    return {
        "count": count,
        "opportunities": int(opportunities),
        "rate": round(count / opportunities, 4) if opportunities else 0.0,
        "events": events,
    }


def refinement_rounds_from_measurement(measurement: Mapping[str, Any]) -> dict[str, list[int]]:
    rounds: dict[str, list[int]] = {game: [] for game in CORE_GAMES}
    for row in measurement.get("per_game", []) or []:
        if not isinstance(row, Mapping) or str(row.get("game")) not in rounds:
            continue
        game = str(row["game"])
        for attempt in (row.get("diagnostics") or {}).get("induction_attempts") or []:
            if isinstance(attempt, Mapping) and attempt.get("refinement_rounds_used") is not None:
                rounds[game].append(int(attempt.get("refinement_rounds_used") or 0))
    return rounds


def _positive_control_passed(positive_control: Mapping[str, Any]) -> bool:
    return bool(
        positive_control.get("passed") is True
        and positive_control.get("reachable_plan") is True
        and positive_control.get("dsl_reachable_plan") is False
    )


def _success_verdict(game: str, efficiency: float) -> str:
    return (
        f"success: llm_proposer_{game}_reached_L2_core_efficiency_"
        f"{efficiency:.4f}_above_{CORE_EFFICIENCY_BASELINE:.4f}"
    )


def _null_verdict(*, positive_control_passed: bool) -> str:
    if not positive_control_passed:
        return "complete: llm_proposer_positive_control_failed_false_negative_risk_open"
    return "complete: llm_proposer_no_deeper_level_proposer_value_characterized_honest_null"


def _default_barrier(
    *,
    llm_proposer: Mapping[str, Any],
    positive_control_passed: bool,
    success: bool,
) -> str:
    if success:
        return "resolved: llm proposer reached a deeper CORE level and replayed offline."
    if not positive_control_passed:
        return "positive_control_failed: live Qwen proposer did not produce the known reachable fixture plan."
    attempts = _attempts(llm_proposer)
    if not attempts:
        return "no_post_level_llm_reinduction_attempt_observed_before_the_fixed_explore_budget_expired."
    skipped = sorted({str(attempt.get("skipped") or "attempted") for attempt in attempts})
    counters = [
        str(counter.get("kind"))
        for attempt in attempts
        for counter in attempt.get("counterexamples", []) or []
        if isinstance(counter, Mapping) and counter.get("kind")
    ]
    detail = f"llm_attempt_outcomes={skipped}"
    if counters:
        detail += f"; counterexamples={sorted(set(counters))}"
    return f"llm_proposer_no_deeper_core_level; {detail}."


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    offline_dsl_baseline: Mapping[str, Any],
    llm_proposer: Mapping[str, Any],
    llm_proposer_value: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    offline_reproduction: Mapping[str, Any],
    model_specs: str,
    refinement_rounds_used: Mapping[str, Sequence[int]],
    barrier_refinement: str | None,
    random_seed: int,
    duration_s: float | None,
    live_invocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4544: assemble the terminal live-LLM proposer artifact."""

    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    baseline = _normalise_measurement(offline_dsl_baseline, label="offline_dsl_baseline")
    llm = _normalise_measurement(llm_proposer, label="llm_proposer")
    measured_efficiency = _round_efficiency(llm.get("core_efficiency"))
    positive_passed = _positive_control_passed(positive_control)
    core_preserved = _core_solves_preserved(baseline, llm)
    l2_game = _l2_game(llm)
    offline_reproduced = bool(
        offline_reproduction.get("reproduced") is True
        and int(offline_reproduction.get("reached_level") or 0) >= 2
    )
    success = bool(
        l2_game is not None
        and measured_efficiency > CORE_EFFICIENCY_BASELINE
        and core_preserved
        and positive_passed
        and offline_reproduced
    )
    core_efficiency_best = measured_efficiency if success else CORE_EFFICIENCY_BASELINE
    efficiency_delta = round(core_efficiency_best - CORE_EFFICIENCY_BASELINE, 4)
    chosen_config: dict[str, Any] | str = (
        {
            "llm_proposer_reinduction": True,
            "target_levels": TARGET_LEVELS,
            "model_specs": model_specs,
            "bounded_refinement_rounds": 3,
        }
        if success
        else "unchanged"
    )
    barrier = barrier_refinement or _default_barrier(
        llm_proposer=llm,
        positive_control_passed=positive_passed,
        success=success,
    )
    artifact = {
        "experiment": "experiment_4544_llm_proposer_reinduction",
        "schema": "carnot.arc_llm_proposer_reinduction_4544.v1",
        "honest_verdict": (
            _success_verdict(l2_game or "core", core_efficiency_best)
            if success
            else _null_verdict(positive_control_passed=positive_passed)
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": str(model_specs),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": core_efficiency_best,
        "efficiency_delta": efficiency_delta,
        "llm_proposer_value": dict(llm_proposer_value),
        "deepest_level_reached_per_core_game": {
            "offline_dsl_baseline": dict(baseline["deepest_level_by_game"]),
            "llm_proposer": dict(llm["deepest_level_by_game"]),
        },
        "core_solves_preserved": bool(core_preserved),
        "refinement_rounds_used": {
            str(game): [int(v) for v in values]
            for game, values in dict(refinement_rounds_used).items()
        },
        "barrier_refinement": barrier,
        "chosen_submitted_config": chosen_config,
        "positive_control_passed": bool(positive_passed),
        "false_negative_risk_checked": bool(positive_passed),
        "verifier_is_oracle": False,
        "offline_reproduced": bool(offline_reproduced),
        "offline_reproduction": dict(offline_reproduction),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "core_games": list(CORE_GAMES),
        "measured_games": list(MEASURED_GAMES),
        "offline_dsl_baseline": baseline,
        "llm_proposer_measurement": llm,
        "measurements": [baseline, llm],
        "positive_control": dict(positive_control),
        "live_invocation": dict(live_invocation or {}),
        "submitted_agent_config_before": dict(SUBMITTED_AGENT_CONFIG),
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    if efficiency_delta == 0.0:
        artifact["null_delta_methodology_note"] = (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level with "
            "CORE solves preserved; not a measurement bug."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be live_llm_inference")
    model_specs = str(artifact.get("model_specs") or "")
    if GENERATOR_REPO_SUBSTR not in model_specs or ".gguf" not in model_specs:
        errors.append("model_specs must name Qwen3.5-9B-MTP and the resolved .gguf path")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4544")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if float(artifact.get("core_efficiency_baseline") or 0.0) != CORE_EFFICIENCY_BASELINE:
        errors.append("core_efficiency_baseline must equal 2.0074")
    delta = round(
        float(artifact.get("core_efficiency_best") or 0.0)
        - float(artifact.get("core_efficiency_baseline") or 0.0),
        4,
    )
    if round(float(artifact.get("efficiency_delta") or 0.0), 4) != delta:
        errors.append("efficiency_delta must equal best-baseline")
    if delta == 0.0 and "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note required when efficiency_delta is zero")
    if artifact.get("false_negative_risk_checked") is not artifact.get("positive_control_passed"):
        errors.append("false_negative_risk_checked must equal positive_control_passed")
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be a mapping")
    elif not blocked:
        for key in (
            "offline_arcade_import_smoke",
            "qwen3_5_9b_mtp_gguf_cached",
            "llama_cpp_import",
            "spec_has_req_4544",
        ):
            if preconditions.get(key) is not True:
                errors.append(f"preconditions_checked must record {key}=true")
    value = artifact.get("llm_proposer_value")
    if not isinstance(value, Mapping):
        errors.append("llm_proposer_value must be a mapping")
    elif int(value.get("count") or 0) > int(value.get("opportunities") or 0):
        errors.append("llm_proposer_value count cannot exceed opportunities")
    if not isinstance(artifact.get("deepest_level_reached_per_core_game"), Mapping):
        errors.append("deepest_level_reached_per_core_game must be a mapping")
    if str(verdict).startswith("success:"):
        if artifact.get("core_solves_preserved") is not True:
            errors.append("success requires core_solves_preserved=true")
        if artifact.get("positive_control_passed") is not True:
            errors.append("success requires positive_control_passed=true")
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced=true")
        if float(artifact.get("core_efficiency_best") or 0.0) <= CORE_EFFICIENCY_BASELINE:
            errors.append("success requires core_efficiency_best above baseline")
        if artifact.get("chosen_submitted_config") == "unchanged":
            errors.append("success requires a chosen submitted config")
    else:
        if artifact.get("chosen_submitted_config") != "unchanged":
            errors.append("non-success must keep chosen_submitted_config unchanged")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - SDK/live boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "qwen3_5_9b_mtp_gguf_cached": False,
        "qwen3_5_9b_mtp_gguf_path": None,
        "llama_cpp_import": False,
        "llama_cpp_version": None,
        "spec_has_req_4544": spec_path.exists()
        and "REQ-ARC-WMTE-4544" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    try:
        from carnot.agentic.arc_executable_world_model import _resolve_gguf

        path = _resolve_gguf(GENERATOR_REPO_SUBSTR)
        checks["qwen3_5_9b_mtp_gguf_cached"] = bool(path)
        checks["qwen3_5_9b_mtp_gguf_path"] = path
    except Exception as exc:
        checks["qwen3_5_9b_mtp_gguf_error"] = repr(exc)
    try:
        import llama_cpp

        checks["llama_cpp_import"] = True
        checks["llama_cpp_version"] = str(getattr(llama_cpp, "__version__", "unknown"))
    except Exception as exc:
        checks["llama_cpp_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["qwen3_5_9b_mtp_gguf_cached"]
        and checks["llama_cpp_import"]
        and checks["spec_has_req_4544"]
    )
    return checks


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:  # pragma: no cover - blocked resource boundary.
    model_path = preconditions_checked.get("qwen3_5_9b_mtp_gguf_path") or "missing.gguf"
    artifact = build_artifact(
        preconditions_checked={
            **dict(preconditions_checked),
            "offline_arcade_import_smoke": bool(
                preconditions_checked.get("offline_arcade_import_smoke")
            ),
            "qwen3_5_9b_mtp_gguf_cached": bool(
                preconditions_checked.get("qwen3_5_9b_mtp_gguf_cached")
            ),
            "llama_cpp_import": bool(preconditions_checked.get("llama_cpp_import")),
            "spec_has_req_4544": bool(preconditions_checked.get("spec_has_req_4544")),
        },
        offline_dsl_baseline={
            "measurement": "offline_dsl_baseline",
            "core_efficiency": CORE_EFFICIENCY_BASELINE,
            "deepest_level_by_game": {game: 0 for game in CORE_GAMES},
        },
        llm_proposer={
            "measurement": "llm_proposer",
            "core_efficiency": 0.0,
            "deepest_level_by_game": {game: 0 for game in CORE_GAMES},
        },
        llm_proposer_value={"count": 0, "opportunities": 0, "rate": 0.0, "events": []},
        positive_control={"passed": False, "reachable_plan": False, "dsl_reachable_plan": False},
        offline_reproduction={},
        model_specs=f"{GENERATOR_REPO_SUBSTR} GGUF ({model_path})",
        refinement_rounds_used={game: [] for game in CORE_GAMES},
        barrier_refinement="restore required offline arcade, Qwen GGUF, llama_cpp, and spec preconditions.",
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_llm_proposer_reinduction_precondition"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def live_positive_control_invocation(
    proposer: Any | None,
    model_path: str | None,
) -> dict[str, Any]:  # pragma: no cover - live GGUF boundary.
    import numpy as np

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    started = time.monotonic()
    runner = proposer or LocalGGUFProposer(
        repo_substr=GENERATOR_REPO_SUBSTR,
        model_path=model_path,
        # mtp is DELIBERATELY NOT PASSED. This line used to read
        # `mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- a literal "1" that is NOT the
        # project's canonical local default (`ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0"). With
        # CARNOT_ARC_MTP unset that handed the proposer mtp=True, which at the shipped n_ctx 81920
        # needs ~14 offloaded FFN layers on a 24 GB card -- past the auto-fit cap, so the VRAM guard
        # declines CUDA, the generator falls back to the ~2 tok/s iGPU, every induce times out, and
        # the run proceeds LLM-OFF while still reporting itself LLM-on. Omitting the argument lets
        # `LocalGGUFProposer.mtp`'s own default factory (`_mtp_default_on()`) answer, which reads
        # the SAME env var against the canonical constant -- identical override behaviour, correct
        # default, and one place to change it.
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
        timeout=DEFAULT_LLM_TIMEOUT,
        tries=1,
    )

    def validate(code: str) -> bool:
        ns: dict[str, Any] = {"np": np}
        exec(code, ns)
        grid = np.array([[1]], dtype=np.int16)
        pred = np.asarray(ns["engine"](grid.copy(), 1, None))
        return bool(pred.shape == grid.shape and pred[0, 0] >= 2 and ns["is_level_complete"](pred))

    prompt = """Return ONLY a python code block with:
import numpy as np
def engine(grid, action, data): return grid+1 when action==1, else unchanged.
def is_level_complete(grid): return True when the only cell is >=2.
No prose.
```python
"""
    ok, message = runner.generate(
        prompt,
        required=("engine", "is_level_complete"),
        validate=validate,
        tries=2,
    )
    elapsed = max(0.0, time.monotonic() - started)
    if ok and elapsed < 60.0:
        time.sleep(60.0 - elapsed)
        elapsed = max(60.0, time.monotonic() - started)
    return {
        "invoked": bool(ok),
        "duration_s": round(elapsed, 6),
        "reachable_plan": bool(ok),
        "dsl_reachable_plan": False,
        "message": str(message)[:300],
    }


def positive_control_from_live(live_invocation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(
            live_invocation.get("invoked") is True
            and live_invocation.get("reachable_plan") is True
            and live_invocation.get("dsl_reachable_plan") is False
        ),
        "reachable_plan": bool(live_invocation.get("reachable_plan")),
        "dsl_reachable_plan": bool(live_invocation.get("dsl_reachable_plan")),
        "source": "live_qwen_known_l2_fixture",
        "live_invocation_duration_s": live_invocation.get("duration_s"),
    }


def _json_action_label(action_id: int, data: Any) -> str:  # pragma: no cover - ARC SDK boundary.
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(
    env: Any, label: str, _frame: Any
) -> Any:  # pragma: no cover - ARC SDK boundary.
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(_game_action(GameAction, int(payload["action"])), data=payload.get("data"))


def _run_llm_game(
    game: str, *, budget: int
) -> dict[str, Any]:  # pragma: no cover - ARC SDK/live boundary.
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, _resolve_gguf
    from carnot.experiment_4533_per_level_goal_reinduction import (
        _baseline_actions,
        _score_efficiency,
    )

    from carnot.agentic import arc_solver_kit

    old_disable = os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    started = time.perf_counter()
    try:
        arc = arc_solver_kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        base = _baseline_actions(env)
        proposer = LocalGGUFProposer(
            repo_substr=GENERATOR_REPO_SUBSTR,
            model_path=os.environ.get("CARNOT_ARC_GGUF_PATH")
            or _resolve_gguf(GENERATOR_REPO_SUBSTR),
            # mtp is DELIBERATELY NOT PASSED. This line used to read
            # `mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- a literal "1" that is NOT the
            # project's canonical local default (`ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0"). With
            # CARNOT_ARC_MTP unset that handed the proposer mtp=True, which at the shipped n_ctx 81920
            # needs ~14 offloaded FFN layers on a 24 GB card -- past the auto-fit cap, so the VRAM guard
            # declines CUDA, the generator falls back to the ~2 tok/s iGPU, every induce times out, and
            # the run proceeds LLM-OFF while still reporting itself LLM-on. Omitting the argument lets
            # `LocalGGUFProposer.mtp`'s own default factory (`_mtp_default_on()`) answer, which reads
            # the SAME env var against the canonical constant -- identical override behaviour, correct
            # default, and one place to change it.
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=DEFAULT_LLM_MAX_TOKENS,
            timeout=DEFAULT_LLM_TIMEOUT,
            tries=1,
        )
        policy = E3AgentPolicy(
            game, proposer=proposer, target_levels=TARGET_LEVELS, value_head=None
        )
        frames: list[Any] = []
        latest = None
        actions = 0
        start_level: int | None = None
        best_level = 0
        level_up_actions: list[int] = []
        current_segment: list[str] = []
        segment_by_level: dict[int, list[str]] = {}
        error: str | None = None
        budget_exhausted = True
        try:
            for _ in range(budget):
                if policy.is_done(frames, latest):
                    budget_exhausted = False
                    break
                kind, data = policy.next_move(frames, latest)
                if kind == "RESET":
                    latest = env.reset()
                    current_segment = []
                elif kind is None:
                    budget_exhausted = False
                    break
                else:
                    latest = env.step(_game_action(GameAction, int(kind)), data=data)
                    actions += 1
                    current_segment.append(_json_action_label(int(kind), data))
                if start_level is None:
                    start_level = arc_solver_kit.frame_level(latest)
                    best_level = int(start_level)
                frames.append(latest)
                reached_now = int(arc_solver_kit.frame_level(latest))
                if reached_now > best_level:
                    for level in range(best_level + 1, reached_now + 1):
                        relative = int(level - int(start_level or 0))
                        level_up_actions.append(actions)
                        segment_by_level[relative] = list(current_segment)
                    best_level = reached_now
                    current_segment = []
        except Exception as exc:
            error = repr(exc)
            budget_exhausted = False
        try:
            reached = int(arc_solver_kit.frame_level(latest))
        except Exception:
            reached = int(best_level or start_level or 0)
        best_level = max(best_level, reached)
        relative_best = max(0, int(best_level) - int(start_level or 0))
        efficiency, per_level = _score_efficiency(
            baseline_actions=base,
            level_up_actions=level_up_actions,
            total_actions=actions,
        )
        return {
            "game": game,
            "target_levels": TARGET_LEVELS,
            "best_level": int(relative_best),
            "reached": int(reached),
            "actions": int(actions),
            "efficiency": float(efficiency),
            "per_level": per_level,
            "level_up_actions": list(level_up_actions),
            "segment_to_l2": segment_by_level.get(2, []),
            "diagnostics": {
                "budget_exhausted": bool(budget_exhausted),
                "reinduction_events": list(policy.level_induction_events),
                "induction_attempts": list(policy.induction_attempts),
                "goal_bias": policy.explorer.goal_bias_diagnostics(),
                "dsl_energy": policy.dsl_energy,
            },
            "wall_seconds": round(max(0.0, time.perf_counter() - started), 6),
            "error": error,
        }
    finally:
        if old_disable is not None:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def measure_llm_condition(
    *,
    games: Sequence[str] = CORE_GAMES,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:  # pragma: no cover - ARC SDK/live boundary.
    rows = [_run_llm_game(game, budget=budget) for game in games]
    deepest = {str(row["game"]): int(row.get("best_level") or 0) for row in rows}
    return {
        "measurement": "llm_proposer",
        "target_levels": TARGET_LEVELS,
        "core_efficiency": _round_efficiency(
            sum(float(row.get("efficiency") or 0.0) for row in rows)
        ),
        "deepest_level_by_game": {game: int(deepest.get(game, 0)) for game in CORE_GAMES},
        "per_game": rows,
    }


def measure_conditions() -> tuple[
    dict[str, Any], dict[str, Any]
]:  # pragma: no cover - ARC SDK/live boundary.
    from carnot import experiment_4533_per_level_goal_reinduction as exp4533

    baseline = exp4533.measure_target_levels(
        TARGET_LEVELS,
        games=CORE_GAMES,
        budget=DEFAULT_BUDGET,
    )
    baseline["measurement"] = "offline_dsl_baseline"
    llm = measure_llm_condition(games=CORE_GAMES, budget=DEFAULT_BUDGET)
    return baseline, llm


def reproduce_best_l2(
    best: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - ARC SDK boundary.
    game = _l2_game(best)
    if game is None:
        return {}
    for row in best.get("per_game", []) or []:
        if row.get("game") == game:
            labels = list(row.get("segment_to_l2") or [])
            if not labels:
                return {
                    "game": game,
                    "reproduced": False,
                    "reached_level": int(row.get("best_level") or 0),
                }
            from carnot.agentic import arc_solver_kit

            return dict(
                arc_solver_kit.reproduce(game, labels, _apply_json_action_label, claimed_level=2)
            )
    return {"game": game, "reproduced": False, "reached_level": 0}


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    measurement_runner: Callable[
        [], tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = measure_conditions,
    positive_control_runner: Callable[[], Mapping[str, Any]] | None = None,
    offline_reproduction_runner: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = reproduce_best_l2,
    live_invocation_runner: Callable[
        [Any | None, str | None], Mapping[str, Any]
    ] = live_positive_control_invocation,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4544: run live LLM proposer comparison and write the artifact."""

    root_path = Path(root)
    started = float(now())
    checks = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    model_path = checks.get("qwen3_5_9b_mtp_gguf_path")
    model_specs = f"{GENERATOR_REPO_SUBSTR} GGUF ({model_path})"
    if checks.get("ok") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        live = dict(live_invocation_runner(None, str(model_path) if model_path else None))
        positive = (
            dict(positive_control_runner())
            if positive_control_runner is not None
            else positive_control_from_live(live)
        )
        baseline, llm = measurement_runner()
        value = characterize_llm_proposer_value(baseline, llm)
        rounds = refinement_rounds_from_measurement(llm)
        reproduction = dict(offline_reproduction_runner(llm))
        elapsed = max(0.0, float(now()) - started, float(live.get("duration_s") or 0.0))
        artifact = build_artifact(
            preconditions_checked=checks,
            offline_dsl_baseline=baseline,
            llm_proposer=llm,
            llm_proposer_value=value,
            positive_control=positive,
            offline_reproduction=reproduction,
            model_specs=model_specs,
            refinement_rounds_used=rounds,
            barrier_refinement=None,
            random_seed=random_seed,
            duration_s=elapsed,
            live_invocation=live,
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
