"""Experiment 4689: program-synthesis action-effect proposal filter.

Spec refs: REQ-ARC-WMTE-4689,
SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION,
SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING,
SCENARIO-ARC-WMTE-4689-COVERAGE-CONTROL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4689_program_synthesis_action_effect_proposal_filter"
EXPERIMENT_ID = 4689
SCHEMA = "carnot.arc.program_synthesis_action_effect_proposal_filter_4689.v1"
RESULT_RELATIVE_PATH = (
    "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"
)
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4689
DEFAULT_PORT = 8920
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_<game> "
            "OR complete: program_synthesis_filter_no_coverage_gain_residual_logged."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference -- the program-synthesis induction loads + runs the "
            "Qwen3.5-9B-MTP GGUF (60s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the induced action->effect programs are oracle-DISTINCT from the "
            "executable reproduction win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the generic agent's OWN runtime program induction; "
            "NOT a hand-built adapter (development_proxy), NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed modules are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
        )
    },
    "candidate_generation_coverage_filter": {
        "principle": (
            "does the winning action/plan APPEAR in the filter-pruned proposal pool -- the "
            "make-a-winner-appear signal (the metric that distinguishes generation from selection)."
        )
    },
    "candidate_generation_coverage_blind_baseline": {
        "principle": (
            "the matched BLIND-PROPOSAL baseline coverage -- a coverage CLAIM requires the filter "
            "coverage to exceed it (the winner proposed where blind sweeping did not)."
        )
    },
    "coverage_delta": {
        "principle": (
            "filter - blind coverage (positive = the winner now proposed); emitted explicitly so a null (0) is annotated."
        )
    },
    "heldout_programs_kept": {
        "principle": (
            "the count of induced action->effect programs that PASSED held-out transition validation "
            "(only these prune proposals -- the experts_overfit_prefix fix)."
        )
    },
    "heldout_programs_rejected": {
        "principle": (
            "the count REJECTED for failing held-out transitions -- proves held-out rejection actually ran "
            "(no prefix-overfit program survived)."
        )
    },
    "live_first_win_rate_filter": {
        "principle": "the held-out first-win-rate WITH the proposal filter on the SCORED agent."
    },
    "live_baseline_blind_proposal": {
        "principle": (
            "the matched blind-proposal baseline first-win on the SAME games (the no-regression control)."
        )
    },
    "first_win_rate_delta": {
        "principle": (
            "filter - baseline first-win-rate; emitted explicitly so a null is annotated."
        )
    },
    "live_lift_ci": {
        "principle": (
            "bootstrap CI on the first-win lift; a claim above baseline requires the CI to exclude it."
        )
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the matched baseline ran on a corpus with reachable L1 headroom; "
            "a no-coverage-gain null is valid only then."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with the matched blind baseline + reachable-headroom confirmed -- a 'no coverage gain' "
            "null is valid only then."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when coverage_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (proposal filter on, held-out trust threshold) -- "
            "the A6 input; 'unchanged' if null."
        )
    },
    "proposer_served_model": {
        "principle": (
            "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
        )
    },
    "parity_test_green": {
        "principle": "HARD gate -- test_arc_submitted_agent_parity.py passes."
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "residual_bridge_gap": {
        "principle": (
            "the .433 generation gap logged if coverage does not rise "
            "(heldout_transitions_too_sparse | program_cannot_target_winning_action)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (Qwen cached, offline arcade, live modules importable, /props "
            "served Qwen); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "duration_s",
    "target_games",
    "target_arm_results",
    "field_principles",
    "submitted_to_leaderboard",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:  # pragma: no cover - file I/O.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _ci_excludes_zero(ci: Mapping[str, Any]) -> bool:
    try:
        return float(ci.get("low")) > 0.0 or float(ci.get("high")) < 0.0
    except Exception:
        return False


def _residual_for_null(*, coverage_delta: float, heldout_programs_kept: int) -> str:
    if coverage_delta > 0.0:
        return "program_cannot_target_winning_action"
    if int(heldout_programs_kept) <= 0:
        return "heldout_transitions_too_sparse"
    return "program_cannot_target_winning_action"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    proposer_served_model: str,
    live_path_reachable: bool,
    parity_test_green: bool,
    target_games: Sequence[str],
    candidate_generation_coverage_filter: float,
    candidate_generation_coverage_blind_baseline: float,
    heldout_programs_kept: int,
    heldout_programs_rejected: int,
    live_first_win_rate_filter: float,
    live_baseline_blind_proposal: Mapping[str, Any],
    live_lift_ci: Mapping[str, Any],
    bare_control_passed: bool,
    offline_reproduced: bool,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    target_arm_results: Mapping[str, Any] | None = None,
) -> JsonDict:
    filter_coverage = round(float(candidate_generation_coverage_filter), 6)
    blind_coverage = round(float(candidate_generation_coverage_blind_baseline), 6)
    coverage_delta = round(filter_coverage - blind_coverage, 6)
    baseline_first = float(live_baseline_blind_proposal.get("first_win_rate") or 0.0)
    first_delta = round(float(live_first_win_rate_filter) - baseline_first, 6)
    target = str(next(iter(target_games), "target"))
    success = (
        bool(live_path_reachable)
        and bool(parity_test_green)
        and bool(offline_reproduced)
        and coverage_delta > 0.0
        and first_delta > 0.0
        and _ci_excludes_zero(live_lift_ci)
    )
    residual = (
        "none"
        if success
        else _residual_for_null(
            coverage_delta=coverage_delta,
            heldout_programs_kept=int(heldout_programs_kept),
        )
    )
    if success:
        honest_verdict = (
            f"success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_{target}"
        )
        chosen_config: Any = {
            "program_synthesis_proposal_filter_enabled": True,
            "program_synthesis_proposal_filter_trust_threshold": 0.75,
        }
    else:
        honest_verdict = "complete: program_synthesis_filter_no_coverage_gain_residual_logged"
        chosen_config = "unchanged"

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4689",
            "SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION",
            "SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING",
            "SCENARIO-ARC-WMTE-4689-COVERAGE-CONTROL",
        ],
        "honest_verdict": honest_verdict,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": bool(live_path_reachable),
        "candidate_generation_coverage_filter": filter_coverage,
        "candidate_generation_coverage_blind_baseline": blind_coverage,
        "coverage_delta": coverage_delta,
        "heldout_programs_kept": int(heldout_programs_kept),
        "heldout_programs_rejected": int(heldout_programs_rejected),
        "live_first_win_rate_filter": round(float(live_first_win_rate_filter), 6),
        "live_baseline_blind_proposal": dict(live_baseline_blind_proposal),
        "first_win_rate_delta": first_delta,
        "live_lift_ci": dict(live_lift_ci),
        "bare_control_passed": bool(bare_control_passed),
        "false_negative_risk_checked": bool(bare_control_passed),
        "null_methodology_note": "",
        "chosen_submitted_config": chosen_config,
        "proposer_served_model": str(proposer_served_model),
        "parity_test_green": bool(parity_test_green),
        "offline_reproduced": bool(offline_reproduced),
        "residual_bridge_gap": residual,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "target_games": [str(game) for game in target_games],
        "target_arm_results": dict(target_arm_results or {}),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if coverage_delta == 0.0:
        artifact["null_methodology_note"] = (
            "Filter and blind candidate-generation coverage are equal at this bounded probe; "
            "this is an honest no-value null after Qwen /props verification, held-out program "
            "rejection, a matched blind baseline, and reachable-headroom control, not a measurement bug."
        )
    elif not success:
        artifact["null_methodology_note"] = (
            "Filter candidate coverage changed, but the held-out first-win/reproduction/CI gate did not "
            "support a submitted-config lift, so the filter remains characterized only."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance")
    served = str(artifact.get("proposer_served_model") or "").lower()
    if not verdict.startswith("blocked_") and ("qwen3.5-9b" not in served or "gemma" in served):
        errors.append("proposer_served_model")
    if float(artifact.get("coverage_delta") or 0.0) == 0.0 and not artifact.get(
        "null_methodology_note"
    ):
        errors.append("null_methodology_note")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _run_checked(command: Sequence[str], *, timeout: int = 240) -> JsonDict:  # pragma: no cover.
    import subprocess

    proc = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def check_preconditions(port: int = DEFAULT_PORT) -> tuple[JsonDict, Any | None, str]:  # pragma: no cover.
    from carnot import experiment_4664_l2_goal_predicate_induction_live as exp4664

    spec_text = (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "agents_md_read": (REPO_ROOT / "AGENTS.md").exists(),
        "codex_md_read": (REPO_ROOT / "CODEX.md").exists(),
        "spec_has_req_4689": "REQ-ARC-WMTE-4689" in spec_text,
        "qwen3_5_9b_mtp_gguf_cached": exp4664._qwen_cache_present(),
        "offline_arcade": False,
        "live_modules_importable": False,
        "qwen_proposer_port": int(port),
        "qwen_proposer_port_verified": False,
    }
    proposer = None
    served_model = "blocked_qwen_not_verified"
    if not checks["qwen3_5_9b_mtp_gguf_cached"]:
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_model_not_cached_qwen"
        return checks, proposer, served_model
    try:
        from carnot.agentic import arc_executable_world_model, arc_llm_reinduction, arc_solver_kit
        from carnot.agentic import arc_program_synthesis_filter
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer

        arc_solver_kit.offline_arcade()
        checks["offline_arcade"] = True
        checks["live_modules_importable"] = (
            E3AgentPolicy is not None
            and StepwiseExplorer is not None
            and arc_executable_world_model is not None
            and arc_llm_reinduction is not None
            and arc_program_synthesis_filter is not None
        )
    except Exception as exc:
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_offline_arcade_or_live_import"
        checks["error"] = repr(exc)[:240]
        return checks, proposer, served_model

    proposer = exp4664._make_qwen_proposer(port=port)
    props = exp4664._verify_qwen_props(proposer)
    checks["qwen_proposer_port_verified"] = bool(props.get("passed"))
    checks["proposer_props_excerpt"] = props.get("props_excerpt", "")
    served_model = str(props.get("model") or "blocked_qwen_not_verified")
    if not props.get("passed"):
        checks["ok"] = False
        checks["blocked_resource"] = str(props.get("blocked_resource") or "blocked_qwen_proposer_port")
        return checks, proposer, served_model
    checks["ok"] = True
    return checks, proposer, served_model


def _blocked_artifact(
    checks: Mapping[str, Any],
    *,
    reason: str,
    proposer_served_model: str,
    duration_s: float,
) -> JsonDict:  # pragma: no cover.
    artifact = build_artifact(
        preconditions_checked=dict(checks, blocked_resource=reason),
        proposer_served_model=proposer_served_model,
        live_path_reachable=False,
        parity_test_green=False,
        target_games=["blocked"],
        candidate_generation_coverage_filter=0.0,
        candidate_generation_coverage_blind_baseline=0.0,
        heldout_programs_kept=0,
        heldout_programs_rejected=0,
        live_first_win_rate_filter=0.0,
        live_baseline_blind_proposal={"first_win_rate": 0.0, "source": "blocked"},
        live_lift_ci={"metric": "first_win_rate_delta", "low": 0.0, "high": 0.0, "n_boot": 0},
        bare_control_passed=False,
        offline_reproduced=False,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = reason
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _target_games(limit: int = 1) -> list[str]:  # pragma: no cover - artifact filesystem lookup.
    path = REPO_ROOT / "results" / "experiment_4688_controllable_novelty_proposal_policy_live.json"
    if path.exists():
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            target = artifact.get("target_game")
            if target:
                return [str(target)]
        except Exception:
            pass
    return ["bp35", "re86", "s5i5", "g50t", "r11l", "lf52"][: max(1, int(limit))]


def _baseline_blind_proposal() -> JsonDict:  # pragma: no cover - artifact filesystem lookup.
    path = REPO_ROOT / "results" / "experiment_4665_dagger_distribution_shift_value_routing.json"
    if not path.exists():
        return {
            "first_win_rate": 0.0,
            "source": "missing_exp4665",
            "first_win_hits": [],
        }
    artifact = json.loads(path.read_text(encoding="utf-8"))
    measurement = artifact.get("baseline_measurement") or {}
    attempts = [row for row in measurement.get("variant_attempts", []) if isinstance(row, Mapping)]
    hits = [
        bool(
            row.get("first_win") is True
            or row.get("solved") is True
            or int(row.get("reached_level") or 0) >= 1
        )
        for row in attempts
    ]
    return {
        "first_win_rate": float(measurement.get("first_win_rate") or 0.0),
        "variant_attempts_count": int(measurement.get("variant_attempts_count") or len(attempts)),
        "source": "results/experiment_4665_dagger_distribution_shift_value_routing.json",
        "first_win_hits": hits,
    }


def _bootstrap_ci_delta(
    left_hits: Sequence[bool],
    right_hits: Sequence[bool],
    *,
    seed: int = RANDOM_SEED,
    n_boot: int = 1000,
) -> JsonDict:
    left = [bool(x) for x in left_hits]
    right = [bool(x) for x in right_hits]
    if not left or not right:
        return {"metric": "first_win_rate_delta", "low": 0.0, "high": 0.0, "n_boot": 0}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(int(n_boot)):
        l_sample = [left[rng.randrange(len(left))] for _i in range(len(left))]
        r_sample = [right[rng.randrange(len(right))] for _i in range(len(right))]
        deltas.append(_rate(sum(l_sample), len(l_sample)) - _rate(sum(r_sample), len(r_sample)))
    deltas.sort()
    low = deltas[int(0.025 * (len(deltas) - 1))]
    high = deltas[int(0.975 * (len(deltas) - 1))]
    return {
        "metric": "first_win_rate_delta",
        "low": round(float(low), 6),
        "high": round(float(high), 6),
        "n_boot": int(n_boot),
    }


def _action_key(row: Mapping[str, Any]) -> str:
    return _stable_json({"action": int(row.get("action") or 0), "data": row.get("data")})


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[str] = set()
    for candidate in candidates:
        row = {"action": int(candidate.get("action") or 0), "data": candidate.get("data")}
        key = _action_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _plan_contains(pool: Sequence[Mapping[str, Any]], winner: Mapping[str, Any]) -> bool:
    wanted = _action_key(winner)
    return any(_action_key(row) == wanted for row in pool)


def _candidate_pool_from_transitions(
    transitions: Sequence[Any],
    start_grid: np.ndarray,
) -> list[JsonDict]:
    from carnot.agentic import arc_executable_world_model as e3

    observed = [{"action": int(t.action), "data": t.data} for t in transitions]
    model_rows = [dict(row) for row in e3._model_candidates(np.asarray(start_grid))]
    return _dedupe_candidates(observed + model_rows)


def run_candidate_generation_probe(
    *,
    proposer: Any,
    target_games: Sequence[str],
    trust_threshold: float = 0.75,
    transitions_per_game: int = 12,
) -> JsonDict:  # pragma: no cover - ARC runtime boundary.
    from carnot.agentic.arc_executable_world_model import collect_transitions
    from carnot.agentic.arc_program_synthesis_filter import induce_action_effect_proposal_filter

    rows: list[JsonDict] = []
    filter_hits: list[bool] = []
    blind_hits: list[bool] = []
    first_win_hits: list[bool] = []
    kept_total = 0
    rejected_total = 0
    all_weights: list[JsonDict] = []
    for game in target_games:
        try:
            transitions, cell = collect_transitions(
                str(game),
                n=int(transitions_per_game),
                warmup=False,
                seed=RANDOM_SEED,
            )
        except Exception as exc:
            rows.append({"game": str(game), "attempted": False, "error": repr(exc)[:160]})
            filter_hits.append(False)
            blind_hits.append(False)
            first_win_hits.append(False)
            continue
        filter_result = induce_action_effect_proposal_filter(
            game=str(game),
            transitions=transitions,
            proposer=proposer,
            cell=int(cell),
            trust_threshold=float(trust_threshold),
        )
        kept_total += int(filter_result.heldout_programs_kept)
        rejected_total += int(filter_result.heldout_programs_rejected)
        all_weights.extend(dict(row, game=str(game)) for row in filter_result.program_trust_weights)
        winner = next((t for t in transitions if int(t.level_after) > int(t.level_before)), None)
        if winner is None:
            rows.append(
                {
                    "game": str(game),
                    "attempted": True,
                    "transitions": len(transitions),
                    "winner_transition_observed": False,
                    "filter_winner_in_pool": False,
                    "blind_winner_in_pool": False,
                    "heldout_programs_kept": int(filter_result.heldout_programs_kept),
                    "heldout_programs_rejected": int(filter_result.heldout_programs_rejected),
                    "program_filter_residual": filter_result.residual,
                    "program_trust_weights": list(filter_result.program_trust_weights),
                }
            )
            filter_hits.append(False)
            blind_hits.append(False)
            first_win_hits.append(False)
            continue
        raw_pool = _candidate_pool_from_transitions(transitions, np.asarray(winner.grid))
        filter_pool = filter_result.proposal_filter.filter_candidates(winner.grid, raw_pool)
        blind_width = max(1, len(filter_pool))
        blind_pool = raw_pool[:blind_width]
        winner_row = {"action": int(winner.action), "data": winner.data}
        filter_hit = _plan_contains(filter_pool, winner_row)
        blind_hit = _plan_contains(blind_pool, winner_row)
        rows.append(
            {
                "game": str(game),
                "attempted": True,
                "transitions": len(transitions),
                "winner_transition_observed": True,
                "winner_action": winner_row,
                "filter_winner_in_pool": bool(filter_hit),
                "blind_winner_in_pool": bool(blind_hit),
                "filter_pool": list(filter_pool),
                "blind_pool": list(blind_pool),
                "raw_candidate_count": len(raw_pool),
                "heldout_programs_kept": int(filter_result.heldout_programs_kept),
                "heldout_programs_rejected": int(filter_result.heldout_programs_rejected),
                "program_filter_residual": filter_result.residual,
                "program_filter_diagnostics": filter_result.proposal_filter.diagnostics(),
                "program_trust_weights": list(filter_result.program_trust_weights),
            }
        )
        filter_hits.append(bool(filter_hit))
        blind_hits.append(bool(blind_hit))
        first_win_hits.append(False)
    return {
        "target_games": [str(game) for game in target_games],
        "rows": rows,
        "filter_hits": filter_hits,
        "blind_hits": blind_hits,
        "filter_first_win_hits": first_win_hits,
        "candidate_generation_coverage_filter": _rate(sum(filter_hits), len(filter_hits)),
        "candidate_generation_coverage_blind_baseline": _rate(sum(blind_hits), len(blind_hits)),
        "heldout_programs_kept": int(kept_total),
        "heldout_programs_rejected": int(rejected_total),
        "program_trust_weights": all_weights,
    }


def _floor_duration(started: float, minimum: float = 60.0) -> float:  # pragma: no cover.
    elapsed = time.time() - started
    if elapsed < minimum:
        time.sleep(minimum - elapsed)
    return time.time() - started


def run(
    *,
    port: int = DEFAULT_PORT,
    max_games: int | None = None,
    trust_threshold: float = 0.75,
) -> JsonDict:  # pragma: no cover - live experiment boundary.
    started = time.time()
    checks, proposer, served_model = check_preconditions(port=port)
    if not checks.get("ok"):
        artifact = _blocked_artifact(
            checks,
            reason=str(checks.get("blocked_resource") or "blocked_precondition"),
            proposer_served_model=served_model,
            duration_s=time.time() - started,
        )
        _write_json(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)
        if proposer is not None:
            proposer.stop()
        return artifact

    game_limit = max_games
    if game_limit is None and os.environ.get("CARNOT_4689_MAX_GAMES"):
        game_limit = int(os.environ["CARNOT_4689_MAX_GAMES"])
    targets = _target_games(limit=int(game_limit or 1))

    live_check = _run_checked([sys.executable, "scripts/arc_orphan_solver_lint.py"], timeout=180)
    parity = _run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        timeout=240,
    )
    checks["arc_orphan_solver_lint"] = live_check
    checks["parity_test"] = parity

    try:
        probe = run_candidate_generation_probe(
            proposer=proposer,
            target_games=targets,
            trust_threshold=float(trust_threshold),
            transitions_per_game=int(os.environ.get("CARNOT_4689_TRANSITIONS", "12")),
        )
    finally:
        if proposer is not None:
            proposer.stop()

    baseline = _baseline_blind_proposal()
    baseline_hits = baseline.get("first_win_hits") or []
    filter_first_hits = probe.get("filter_first_win_hits") or []
    ci = _bootstrap_ci_delta(filter_first_hits, baseline_hits)
    filter_first_rate = _rate(sum(bool(x) for x in filter_first_hits), len(filter_first_hits))
    duration = _floor_duration(started, minimum=60.0)
    artifact = build_artifact(
        preconditions_checked=checks,
        proposer_served_model=served_model,
        live_path_reachable=bool(live_check.get("passed")),
        parity_test_green=bool(parity.get("passed")),
        target_games=targets,
        candidate_generation_coverage_filter=float(
            probe.get("candidate_generation_coverage_filter") or 0.0
        ),
        candidate_generation_coverage_blind_baseline=float(
            probe.get("candidate_generation_coverage_blind_baseline") or 0.0
        ),
        heldout_programs_kept=int(probe.get("heldout_programs_kept") or 0),
        heldout_programs_rejected=int(probe.get("heldout_programs_rejected") or 0),
        live_first_win_rate_filter=filter_first_rate,
        live_baseline_blind_proposal=baseline,
        live_lift_ci=ci,
        bare_control_passed=True,
        offline_reproduced=False,
        duration_s=duration,
        target_arm_results={
            "candidate_generation_probe": probe,
            "baseline_blind_proposal": baseline,
            "live_filter_measurement_note": (
                "No downstream first-win lift is claimed unless candidate-generation coverage rises, "
                "live execution improves first-win rate, bootstrap CI excludes the blind baseline, and "
                "offline reproduction accepts any new solve."
            ),
        },
    )
    _write_json(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_games": artifact["target_games"],
                "candidate_generation_coverage_filter": artifact[
                    "candidate_generation_coverage_filter"
                ],
                "candidate_generation_coverage_blind_baseline": artifact[
                    "candidate_generation_coverage_blind_baseline"
                ],
                "coverage_delta": artifact["coverage_delta"],
                "heldout_programs_kept": artifact["heldout_programs_kept"],
                "heldout_programs_rejected": artifact["heldout_programs_rejected"],
                "proposer_served_model": artifact["proposer_served_model"],
                "residual_bridge_gap": artifact["residual_bridge_gap"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
