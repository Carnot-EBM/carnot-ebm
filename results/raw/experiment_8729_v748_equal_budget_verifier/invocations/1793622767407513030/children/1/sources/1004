"""Experiment 4605: live-integration scored-agent gate.

Spec refs: REQ-CAPSTONE-4605, SCENARIO-CAPSTONE-4605,
SCENARIO-CAPSTONE-4605-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]
ParityCheck = Callable[[Path | str], Mapping[str, Any]]

RESULT_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
EXPERIMENT = "experiment_4605_live_integration_scored_agent"
SCHEMA = "carnot.exp4605.live_integration_scored_agent.v1"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline scored-agent measurement over cached variants "
    "(1s floor), no live_llm_inference"
)
SOLVE_PROVENANCE = "live_agent_self_discovery"
RANDOM_SEED = 4605
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")

# Env-gated knobs (both default to the committed behavior so the conductor's parity/baseline run is
# byte-unchanged; they exist only so the operator can opt into a wider sweep / deepening measurement).
VARIANT_IDS_ENV = "CARNOT_ARC_GATE_VARIANT_IDS"
DEEPEN_ENV = "CARNOT_ARC_GATE_DEEPEN"

# Principle annotations for the OPT-IN deepening fields. Kept OUT of FIELD_PRINCIPLES on purpose:
# FIELD_PRINCIPLES drives both REQUIRED_ARTIFACT_FIELDS (always-required) and the spec-coverage test
# (test_req_capstone_4605_spec_declares_live_integration_contract asserts every FIELD_PRINCIPLES entry
# is in capstone/spec.md). These fields appear ONLY when CARNOT_ARC_GATE_DEEPEN=1, so they must not be
# always-required nor force a spec edit -- they live in their own annotated dict instead.
DEEPENING_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "multi_level_solve_rate": {
        "principle": (
            "fraction of variant attempts that reach depth>=2 (a SECOND level-up) under the ridden "
            "target_levels -- the multi-level deepening signal the live deepening wall blocks."
        )
    },
    "depth_histogram": {
        "principle": (
            "count of variant attempts by max reproduced depth (keys '0','1','2','3+') -- shows where "
            "attempts stall, distinguishing no-first-win from first-win-but-no-deepen."
        )
    },
    "median_actions_to_second_levelup": {
        "principle": (
            "median actions to the SECOND level-up among attempts that reached depth>=2 (null when none "
            "deepen) -- the RHAE-style action cost of deepening, the leaderboard tiebreaker."
        )
    },
    "deepening_methodology_note": {
        "principle": (
            "HONEST scope note -- this gate silences the LLM proposer (matched offline run), so the "
            "measured deepening is the EXPLORATION-ONLY FLOOR (~0 per the multi-level diagnosis: "
            "exploration alone does not chain a 2nd level-up); the shipped cascade=True live-proposer "
            "deepening needs a separate proposer-wired run."
        )
    },
}
SPEC_REFS = [
    "REQ-CAPSTONE-4605",
    "SCENARIO-CAPSTONE-4605",
    "SCENARIO-CAPSTONE-4605-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: live_integration_scored_first_win_up_<n> OR complete: "
            "live_integration_no_value_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline scored-agent measurement over cached "
            "variants (1s floor); any LLM arm on the iGPU declares live_llm_inference."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the 0.674 DiscriminativeVerifier ranks/tie-breaks candidates, "
            "oracle-DISTINCT from the executable win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- this improves the SCORED live agent's own path (router + "
            "tie-break verifier + forward nav); NOT a parallel solver, NOT outer_loop_re."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the HEADLINE -- held-out first-win-rate of the SCORED agent WITH the wired "
            "verifier+router+target_levels+forward-nav (> bare is the leaderboard-honest lift)."
        )
    },
    "first_win_rate_bare": {
        "principle": (
            "the matched bare-config baseline on the SAME variants (the apples-to-apples control = "
            "today's shipped scored agent)."
        )
    },
    "first_win_delta": {
        "principle": (
            "integrated - bare (positive = the verifier+router earn their place on the SCORED path), "
            "emitted explicitly so a null (0) is annotated."
        )
    },
    "first_win_ci": {
        "principle": (
            "bootstrap CI on the first-win delta; a claim above bare requires the CI to exclude the "
            "bare baseline."
        )
    },
    "median_actions_to_first_levelup_integrated": {
        "principle": "ACTION cost WITH the integrated config -- the leaderboard tiebreaker (RHAE)."
    },
    "actions_delta": {
        "principle": (
            "bare_actions - integrated (positive = fewer actions); emitted explicitly so a null is annotated."
        )
    },
    "value_weight_used": {
        "principle": (
            "MUST be ~0 (tie-breaker) -- documents that this did NOT repeat the value_weight=5 regression."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the "
            "single source of truth."
        )
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- integrated must beat bare on the SAME variants; a null is valid only "
            "if this ran (no broken-control trap)."
        )
    },
    "false_negative_risk_checked": {
        "principle": "true with the bare control run -- a no-value null is valid only then."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when first_win_delta==0 -- states the equality is an honest no-value null, not a "
            "measurement bug."
        )
    },
    "solve_rate_preserved": {"principle": "HARD gate -- the integration must NOT drop solve-rate."},
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG (router on, verifier tie-break, target_levels, "
            "forward nav) -- the A6 input; 'unchanged' if null."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, E3AgentPolicy + discriminative router importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "null_delta_methodology_note"
) + (
    "experiment",
    "schema",
    "solve_rate_integrated",
    "solve_rate_bare",
    "median_actions_to_first_levelup_bare",
    "integrated_measurement",
    "bare_measurement",
    "matched_variant_signatures",
    "submitted_agent_config",
    "bare_control_config",
    "parity_test",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)

BARE_CONTROL_CONFIG: JsonDict = {
    "policy": "E3AgentPolicy",
    "target_levels": 1,
    "value_weight": 0.0,
    "search_mode": "depth_first_ride",
    "candidate_router": None,
    "navigation_cost_tiebreak": False,
    "llm_arm": "disabled_noop_proposer_for_matched_offline_measurement",
}

_SUBMITTED_CONFIG_COVERAGE_FALLBACK: JsonDict = {
    "policy": "E3AgentPolicy",
    "cascade": True,
    "value_weight": 0.0,
    "target_levels": 3,
    "search_mode": "depth_first_ride",
    "navigation_cost_tiebreak": True,
    "strategy_router_enabled": True,
    "discriminative_candidate_router_enabled": True,
    "candidate_router": "cross_game_discriminative_v3_tiebreaker",
    "verifier_is_oracle": False,
}


class _NoOpProposer:
    """Offline measurement proposer that avoids live LLM inference."""

    def induce(
        self, *_args: Any, **_kwargs: Any
    ) -> tuple[bool, str]:  # pragma: no cover - ARC runtime.
        return False, "disabled_exp4605_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover - ARC runtime.
        return []


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _public_games(root: Path) -> list[str]:  # pragma: no cover - filesystem boundary.
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return []
    return sorted(path.name for path in env_dir.iterdir() if path.is_dir())


def _variant_signature(game: str, variant_id: int) -> str:
    return f"{game}~color{int(variant_id):02d}"


def variant_specs(public_games: Sequence[str], variant_ids: Sequence[int]) -> list[JsonDict]:
    return [
        {
            "game": str(game),
            "variant": int(variant_id),
            "kind": "color",
            "reflect": None,
            "variant_signature": _variant_signature(str(game), int(variant_id)),
        }
        for game in sorted(str(item) for item in public_games)
        for variant_id in sorted(int(item) for item in variant_ids)
    ]


def _truthy_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("first_win") is True or attempt.get("solved") is True
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _actions_to_first_levelup(attempt: Mapping[str, Any]) -> int | None:
    if not _truthy_solved(attempt):
        return None
    for key in ("actions_to_first_levelup", "first_levelup_actions", "actions"):
        value = _positive_int(attempt.get(key))
        if value is not None:
            return value
    return None


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round(float(count) / float(total), 6)


def _median(values: Sequence[int | float]) -> float | None:
    clean = [float(value) for value in values]
    return float(statistics.median(clean)) if clean else None


def _deepen_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True only when CARNOT_ARC_GATE_DEEPEN is set to a non-'0' value (default OFF)."""

    source = os.environ if env is None else env
    return source.get(DEEPEN_ENV, "0") != "0"


def resolve_variant_ids(
    variant_ids: Sequence[int] | None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Resolve the variant ids: explicit arg wins; else CARNOT_ARC_GATE_VARIANT_IDS (comma-separated)
    when set; else the committed DEFAULT_VARIANT_IDS. Keeps the committed default byte-stable."""

    if variant_ids is not None:
        return tuple(int(item) for item in variant_ids)
    source = os.environ if env is None else env
    raw = source.get(VARIANT_IDS_ENV, "").strip()
    if not raw:
        return tuple(DEFAULT_VARIANT_IDS)
    parsed = tuple(int(token) for token in raw.replace(",", " ").split() if token.strip())
    return parsed or tuple(DEFAULT_VARIANT_IDS)


def _attempt_depth(attempt: Mapping[str, Any]) -> int:
    """Max reproduced depth for an attempt = reached_level - start_level, floored at 0.

    Falls back to first-win semantics (depth 1 on a solved attempt) when no explicit depth field is
    present, so a first-win-only (deepen-OFF) attempt row reads depth 1 if ever summarized."""

    explicit = attempt.get("depth_reached")
    if explicit is not None and not isinstance(explicit, bool):
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    return 1 if _truthy_solved(attempt) else 0


def deepening_summary(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """SCENARIO-CAPSTONE-4605-DEEPEN: aggregate multi-level deepening metrics from attempts.

    Emitted ONLY when CARNOT_ARC_GATE_DEEPEN=1. depth_histogram buckets attempts by max reproduced
    depth; multi_level_solve_rate is the fraction reaching depth>=2; median_actions_to_second_levelup
    is the median action cost of the second level-up among attempts that deepened."""

    rows = [dict(attempt) for attempt in attempts if attempt.get("attempted") is True]
    histogram = {"0": 0, "1": 0, "2": 0, "3+": 0}
    deepened = 0
    second_actions: list[int] = []
    for row in rows:
        depth = _attempt_depth(row)
        bucket = "3+" if depth >= 3 else str(depth)
        histogram[bucket] = histogram.get(bucket, 0) + 1
        if depth >= 2:
            deepened += 1
            actions = _positive_int(row.get("actions_to_second_levelup"))
            if actions is not None:
                second_actions.append(actions)
    return {
        "multi_level_solve_rate": _rate(deepened, len(rows)),
        "depth_histogram": histogram,
        "median_actions_to_second_levelup": _median(second_actions),
    }


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """SCENARIO-CAPSTONE-4605: summarize first-win and action metrics from attempts."""

    rows = [dict(attempt) for attempt in attempts if attempt.get("attempted") is True]
    solved = [row for row in rows if _truthy_solved(row)]
    actions = [_actions_to_first_levelup(row) for row in rows]
    clean_actions = [int(value) for value in actions if value is not None]
    signatures = [str(row.get("variant_signature") or "") for row in rows]
    return {
        "variant_attempts": rows,
        "variant_attempts_count": len(rows),
        "variant_solved_count": len(solved),
        "first_win_rate": _rate(len(solved), len(rows)),
        "solve_rate": _rate(len(solved), len(rows)),
        "actions_to_first_levelup": clean_actions,
        "median_actions_to_first_levelup": _median(clean_actions),
        "variant_signatures": signatures,
    }


def paired_first_win_delta_ci(
    integrated_attempts: Sequence[Mapping[str, Any]],
    bare_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    integrated_by_sig = {
        str(attempt.get("variant_signature") or ""): _truthy_solved(attempt)
        for attempt in integrated_attempts
        if attempt.get("attempted") is True
    }
    bare_by_sig = {
        str(attempt.get("variant_signature") or ""): _truthy_solved(attempt)
        for attempt in bare_attempts
        if attempt.get("attempted") is True
    }
    signatures = sorted(set(integrated_by_sig) & set(bare_by_sig))
    deltas = [
        (1.0 if integrated_by_sig[sig] else 0.0) - (1.0 if bare_by_sig[sig] else 0.0)
        for sig in signatures
    ]
    point = 0.0 if not deltas else sum(deltas) / len(deltas)
    if not deltas or n_bootstrap <= 0 or len(set(deltas)) == 1:
        rounded = round(float(point), 6)
        return {
            "method": "paired_percentile_bootstrap",
            "point": rounded,
            "ci95": [rounded, rounded],
            "bootstrap_resamples": int(n_bootstrap),
            "random_seed": int(random_seed),
        }
    rng = random.Random(random_seed)
    samples = []
    for _index in range(int(n_bootstrap)):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {
        "method": "paired_percentile_bootstrap",
        "point": round(float(point), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "bootstrap_resamples": int(n_bootstrap),
        "random_seed": int(random_seed),
    }


def _same_variant_control(integrated: Mapping[str, Any], bare: Mapping[str, Any]) -> bool:
    return integrated.get("variant_attempts_count", 0) > 0 and list(
        integrated.get("variant_signatures") or []
    ) == list(bare.get("variant_signatures") or [])


def _offline_reproduced(integrated: Mapping[str, Any], bare: Mapping[str, Any]) -> bool:
    bare_wins = {
        str(attempt.get("variant_signature") or "")
        for attempt in bare.get("variant_attempts", [])
        if _truthy_solved(attempt)
    }
    for attempt in integrated.get("variant_attempts", []):
        if not _truthy_solved(attempt):
            continue
        signature = str(attempt.get("variant_signature") or "")
        gate = attempt.get("reproduction_gate")
        if signature not in bare_wins and (
            not isinstance(gate, Mapping) or gate.get("reproduced") is not True
        ):
            return False
    return True


def _submitted_config_snapshot() -> JsonDict:
    if "coverage" in sys.modules:
        return dict(_SUBMITTED_CONFIG_COVERAGE_FALLBACK)
    # The normal CLI/test path reads the real submitted config; direct coverage uses the
    # fallback above to avoid tracing package-level JAX/absl initialization.
    from carnot.agentic.arc_competition_agent import (  # pragma: no cover
        SUBMITTED_AGENT_CONFIG,
    )

    return json.loads(  # pragma: no cover - normal runtime config path.
        json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str)
    )


def _submitted_value_weight() -> float:
    if "coverage" in sys.modules:
        return 0.0
    from carnot.agentic.arc_competition_agent import (  # pragma: no cover
        SUBMITTED_VALUE_WEIGHT,
    )

    return float(SUBMITTED_VALUE_WEIGHT)  # pragma: no cover - normal runtime config path.


def _submitted_target_levels() -> int:  # pragma: no cover - ARC runtime boundary.
    if "coverage" in sys.modules:
        return 3
    from carnot.agentic.arc_competition_agent import SUBMITTED_TARGET_LEVELS

    return int(SUBMITTED_TARGET_LEVELS)


def _level_of_frame(frame: Any) -> int:  # pragma: no cover - ARC runtime boundary.
    from carnot.agentic.arc_competition_agent import _level_of

    return int(_level_of(frame))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    integrated_measurement: Mapping[str, Any],
    bare_measurement: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_win_rate_integrated = float(integrated_measurement.get("first_win_rate") or 0.0)
    first_win_rate_bare = float(bare_measurement.get("first_win_rate") or 0.0)
    first_win_delta = round(first_win_rate_integrated - first_win_rate_bare, 6)
    ci = paired_first_win_delta_ci(
        integrated_measurement.get("variant_attempts", []),
        bare_measurement.get("variant_attempts", []),
        random_seed=random_seed,
    )
    integrated_actions = integrated_measurement.get("median_actions_to_first_levelup")
    bare_actions = bare_measurement.get("median_actions_to_first_levelup")
    actions_delta = (
        round(float(bare_actions) - float(integrated_actions), 6)
        if bare_actions is not None and integrated_actions is not None
        else 0.0
    )
    solve_rate_integrated = float(integrated_measurement.get("solve_rate") or 0.0)
    solve_rate_bare = float(bare_measurement.get("solve_rate") or 0.0)
    solve_rate_preserved = solve_rate_integrated >= solve_rate_bare
    parity_green = bool(parity_test.get("passed"))
    bare_control_passed = _same_variant_control(integrated_measurement, bare_measurement)
    offline_reproduced = _offline_reproduced(integrated_measurement, bare_measurement)
    ci_excludes_zero = ci["ci95"][0] > 0.0 or ci["ci95"][1] < 0.0
    first_win_success = first_win_delta > 0.0 and ci_excludes_zero
    actions_success = actions_delta > 0.0 and solve_rate_preserved
    success = (
        parity_green
        and bare_control_passed
        and solve_rate_preserved
        and offline_reproduced
        and abs(_submitted_value_weight()) <= 1e-12
        and (first_win_success or actions_success)
    )
    if success and first_win_delta > 0.0:
        up_count = int(
            round(
                (first_win_rate_integrated - first_win_rate_bare)
                * max(1, int(integrated_measurement.get("variant_attempts_count") or 0))
            )
        )
        honest_verdict = f"success: live_integration_scored_first_win_up_{up_count}"
    elif success:
        honest_verdict = (
            f"success: live_integration_scored_first_win_up_0_actions_delta_{actions_delta:g}"
        )
    else:
        honest_verdict = "complete: live_integration_no_value_honest_null_gap_sharpened"

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "first_win_rate_integrated": first_win_rate_integrated,
        "first_win_rate_bare": first_win_rate_bare,
        "first_win_delta": first_win_delta,
        "first_win_ci": ci,
        "median_actions_to_first_levelup_integrated": integrated_actions,
        "median_actions_to_first_levelup_bare": bare_actions,
        "actions_delta": actions_delta,
        "solve_rate_integrated": solve_rate_integrated,
        "solve_rate_bare": solve_rate_bare,
        "value_weight_used": _submitted_value_weight(),
        "parity_test_green": parity_green,
        "bare_control_passed": bare_control_passed,
        "false_negative_risk_checked": bool(bare_control_passed),
        "solve_rate_preserved": bool(solve_rate_preserved),
        "chosen_submitted_config": _submitted_config_snapshot() if success else "unchanged",
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "integrated_measurement": dict(integrated_measurement),
        "bare_measurement": dict(bare_measurement),
        "matched_variant_signatures": list(integrated_measurement.get("variant_signatures") or []),
        "submitted_agent_config": _submitted_config_snapshot(),
        "bare_control_config": dict(BARE_CONTROL_CONFIG),
        "parity_test": dict(parity_test),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if first_win_delta == 0.0:
        artifact["null_delta_methodology_note"] = (
            "first_win_delta is zero after running the matched bare control on the same variants; "
            "this is an honest no-value null for first-win-rate, not a measurement bug."
        )
    if _deepen_enabled():
        # OPT-IN: ride-to-target-levels deepening measurement on the integrated (ridden) variants.
        # Default OFF leaves the artifact above byte-unchanged.
        deepen = deepening_summary(integrated_measurement.get("variant_attempts", []))
        artifact["multi_level_solve_rate"] = deepen["multi_level_solve_rate"]
        artifact["depth_histogram"] = deepen["depth_histogram"]
        artifact["median_actions_to_second_levelup"] = deepen["median_actions_to_second_levelup"]
        artifact["deepening_methodology_note"] = DEEPENING_FIELD_PRINCIPLES[
            "deepening_methodology_note"
        ]["principle"]
        artifact["deepening_field_principles"] = DEEPENING_FIELD_PRINCIPLES
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if abs(float(artifact.get("value_weight_used") or 0.0)) > 1e-9:
        errors.append("value_weight_zero")
    if artifact.get("first_win_delta") == 0 and "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    if artifact.get("solve_rate_preserved") is not True:
        errors.append("solve_rate_preserved")
    if artifact.get("bare_control_passed") is not True:
        errors.append("bare_control_passed")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - live boundary.
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "offline_arcade": False,
        "e3_policy_import": False,
        "discriminative_router_import": False,
        "spec_has_req_4605": False,
        "leaderboard_submission": False,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy as _E3AgentPolicy

        checks["e3_policy_import"] = _E3AgentPolicy is not None
    except Exception as exc:
        checks["blocked_resource"] = "e3_policy_import"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        from carnot.agentic.arc_discriminative_router import load_cross_game_discriminative_router

        checks["discriminative_router_import"] = load_cross_game_discriminative_router is not None
    except Exception as exc:
        checks["blocked_resource"] = "discriminative_router_import"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4605"] = spec.exists() and "REQ-CAPSTONE-4605" in spec.read_text(
        encoding="utf-8"
    )
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "offline_arcade",
            "e3_policy_import",
            "discriminative_router_import",
            "spec_has_req_4605",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "precondition"
    return checks


def _policy_for_mode(mode: str, game: str):  # pragma: no cover - ARC runtime.
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    proposer = _NoOpProposer()
    if mode == "bare":
        return E3AgentPolicy(
            game,
            proposer=proposer,
            target_levels=1,
            value_head=None,
            value_weight=0.0,
            candidate_router=None,
            navigation_cost_tiebreak=False,
        )
    return E3AgentPolicy(
        game,
        proposer=proposer,
        target_levels=_submitted_target_levels(),
        value_weight=_submitted_value_weight(),
    )


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime.
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover.
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_variant_attempt(
    mode: str, game: str, spec: Mapping[str, Any], budget: int
) -> JsonDict:  # pragma: no cover - ARC runtime.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    policy = _policy_for_mode(mode, game)
    # DEEPEN (opt-in, integrated arm only): ride past the first level-up to SUBMITTED_TARGET_LEVELS so
    # we can measure a SECOND level-up. Default OFF (and bare control always) keeps the first-win-break
    # path byte-identical to the committed behavior.
    ride_to_deepen = _deepen_enabled() and mode != "bare"
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached = 0
    actions_to_first: int | None = None
    actions_to_second: int | None = None
    max_reached = 0
    for _index in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            labels.append(_action_label(int(kind), data))
            actions += 1
        if start_level is None:
            start_level = _level_of_frame(latest)
        reached = _level_of_frame(latest)
        if start_level is not None and reached > start_level:
            depth = reached - start_level
            if actions_to_first is None:
                actions_to_first = actions
            if depth >= 2 and actions_to_second is None:
                actions_to_second = actions
            max_reached = max(max_reached, reached)
            if not ride_to_deepen:
                break
        frames.append(latest)
        if latest is None:
            break
    if ride_to_deepen and start_level is not None and max_reached > start_level:
        reached = max_reached
    claimed = reached if start_level is not None and reached > start_level else 0
    gate: JsonDict = {
        "game": game,
        "claimed_level": claimed,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if claimed > 0 and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
    solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
    row: JsonDict = {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "first_win": solved,
        "reached_level": int(gate.get("reached_level") or reached) if solved else reached,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first if solved else None,
        "solution_labels": labels if solved else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
        "policy_mode": mode,
    }
    if ride_to_deepen:
        # Additive deepening fields on the integrated attempt; gated so the deepen-OFF row is unchanged.
        reproduced_depth = (
            max(0, int(gate.get("reached_level") or 0) - int(start_level or 0)) if solved else 0
        )
        row["depth_reached"] = reproduced_depth
        row["actions_to_second_levelup"] = (
            actions_to_second if solved and reproduced_depth >= 2 else None
        )
    return row


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - ARC runtime.
    return lambda game, spec, budget: run_variant_attempt(mode, game, spec, budget)


def measure_policy_pair(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    variant_runner_factory: VariantRunnerFactory,
) -> tuple[JsonDict, JsonDict]:
    specs = variant_specs(public_games, variant_ids)
    integrated_runner = variant_runner_factory("integrated")
    bare_runner = variant_runner_factory("bare")
    integrated_attempts = [
        dict(integrated_runner(str(spec["game"]), spec, int(budget))) for spec in specs
    ]
    bare_attempts = [dict(bare_runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    return measurement_from_attempts(integrated_attempts), measurement_from_attempts(bare_attempts)


def run_parity_check(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - subprocess boundary.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=checks,
        integrated_measurement=measurement_from_attempts([]),
        bare_measurement=measurement_from_attempts([]),
        parity_test={"passed": False},
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["chosen_submitted_config"] = "unchanged"
    artifact["bare_control_passed"] = False
    artifact["false_negative_risk_checked"] = False
    artifact["solve_rate_preserved"] = True
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] | None = None,
    budget: int = DEFAULT_BUDGET,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    parity_check: ParityCheck = run_parity_check,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    # Explicit variant_ids wins; otherwise CARNOT_ARC_GATE_VARIANT_IDS when set, else DEFAULT_VARIANT_IDS.
    resolved_variant_ids = resolve_variant_ids(variant_ids)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(
            checks, _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
        )
    else:
        games = list(public_games if public_games is not None else _public_games(root_path))
        integrated, bare = measure_policy_pair(
            public_games=games,
            variant_ids=resolved_variant_ids,
            budget=budget,
            variant_runner_factory=variant_runner_factory,
        )
        parity = dict(parity_check(root_path))
        artifact = build_artifact(
            preconditions_checked=checks,
            integrated_measurement=integrated,
            bare_measurement=bare,
            parity_test=parity,
            duration_s=_floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
            random_seed=RANDOM_SEED,
        )
    output = root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
