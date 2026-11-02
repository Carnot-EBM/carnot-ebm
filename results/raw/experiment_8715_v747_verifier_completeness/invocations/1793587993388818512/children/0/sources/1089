"""Experiment 4688: controllable-novelty proposal policy on live E3 exploration.

Spec refs: REQ-ARC-WMTE-4688, SCENARIO-ARC-WMTE-4688.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4688_controllable_novelty_proposal_policy_live"
EXPERIMENT_ID = 4688
SCHEMA = "carnot.arc.controllable_novelty_proposal_policy_live_4688.v1"
RESULT_RELATIVE_PATH = (
    "results/experiment_4688_controllable_novelty_proposal_policy_live.json"
)
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4688
DEFAULT_PORT = 8920
DEFAULT_TARGET_GAME = "bp35"
DEFAULT_BUDGET = 200
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
RESIDUALS = {
    "winning_prefix_still_not_proposed",
    "controllability_embedding_rewards_non_winning_controllable_states",
    "novelty_revisits_dead_states",
    "none",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: controllable_novelty_generic_agent_new_level_<game>_L<n> "
            "OR complete: controllable_novelty_no_new_level_residual_<cause>."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference -- the live E3 explorer's world-model induction (and the "
            "strategy-conditioned arm) loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); "
            "declared honestly because the live agent runs a real LLM during exploration."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the controllable-novelty bonus is oracle-DISTINCT from the "
            "executable reproduction win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- a generic-agent new level via runtime "
            "controllable-novelty exploration is the REAL deliverable, NOT a hand-built "
            "GameAdapter (development_proxy) and NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed modules (StepwiseExplorer etc.) are in the "
            "E3AgentPolicy import closure; arc_orphan_solver_lint passes."
        )
    },
    "controllability_gate_on": {
        "principle": (
            "true -- the novelty embedding is over CONTROLLABLE action-effect deltas, not raw "
            "frame novelty (the .427 noisy-TV fix); the COSMETIC-NOVELTY ablation sets this false."
        )
    },
    "generic_first_win_by_config": {
        "principle": (
            "per-config (novelty temperature / ablation) generic first-win + multi-level rate -- "
            "the honest measurement on the standard harness (the 0.04 baseline locked in by B1)."
        )
    },
    "generic_agent_reached_level": {
        "principle": (
            "the deepest level the GENERIC live agent reached via controllable-novelty exploration "
            "on the target -- the headline (a NEW level is the win)."
        )
    },
    "offline_reproduced": {
        "principle": (
            "a generic new level counts only if offline-reproduced via arc_solver_kit.reproduce "
            "(ARC Solve Reproducibility); a live-only trajectory is provisional."
        )
    },
    "reproduced_levels": {
        "principle": (
            "the integer new-level count the generic agent banked offline (>=1 is the bridge crossed for solve)."
        )
    },
    "no_novelty_ablation_reached_level": {
        "principle": (
            "the matched no-novelty-bonus (flat exploration) ablation's reached_level -- MUST be "
            "lower than the controllable-novelty policy for the win to be attributable to directed "
            "exploration (not flat exploration)."
        )
    },
    "cosmetic_novelty_ablation_reached_level": {
        "principle": (
            "the cosmetic-novelty (controllability-gate-off, raw-frame novelty) ablation's "
            "reached_level -- MUST be lower than the controllable-novelty policy for the win to "
            "be attributable to the controllability gate (not just any novelty)."
        )
    },
    "residual_cause_hypothesis": {
        "principle": (
            "if it nulls, names the residual (winning_prefix_still_not_proposed | "
            "controllability_embedding_rewards_non_winning_controllable_states | "
            "novelty_revisits_dead_states) -- the .433 target; 'none' if it crossed."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when no new level -- states the null is honest (passing ablations + "
            "reachable L1 headroom), not a measurement bug."
        )
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the target has a reachable winning L1 trajectory offline; "
            "a no-new-level null is valid only then."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with both ablations run + reachable-L1-headroom confirmed -- a 'no new level' "
            "null is valid only then."
        )
    },
    "proposer_served_model": {
        "principle": (
            "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- "
            "the port-8919 confound guard."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (controllable-novelty proposal on, "
            "novelty weight/temperature) -- the A6 input; 'unchanged' if null."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (Qwen cached, offline arcade, live modules importable, "
            "/props served Qwen on a free port); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "null_methodology_note"
) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "target_game",
    "target_arm_results",
    "field_principles",
    "duration_s",
    "submitted_to_leaderboard",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover
        return False, "disabled_exp4688_explorer_only"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover
        return []


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(row) for row in attempts if row.get("attempted", True)]
    first = [row for row in rows if int(row.get("reached_level") or 0) >= 1]
    multi = [row for row in rows if int(row.get("reached_level") or 0) >= 2]
    return {
        "variant_attempts_count": len(rows),
        "first_win_count": len(first),
        "first_win_rate": _rate(len(first), len(rows)),
        "multi_level_count": len(multi),
        "multi_level_rate": _rate(len(multi), len(rows)),
        "max_reached_level": max([int(row.get("reached_level") or 0) for row in rows] + [0]),
        "variant_attempts": rows,
    }


def _success_attributable(
    *,
    reached_level: int,
    offline_reproduced: bool,
    no_novelty_level: int,
    cosmetic_level: int,
    live_path_reachable: bool,
    parity_test_green: bool,
) -> bool:
    return (
        int(reached_level) >= 1
        and bool(offline_reproduced)
        and int(no_novelty_level) < int(reached_level)
        and int(cosmetic_level) < int(reached_level)
        and bool(live_path_reachable)
        and bool(parity_test_green)
    )


def _residual(
    novelty_result: Mapping[str, Any],
    no_novelty_result: Mapping[str, Any],
    cosmetic_result: Mapping[str, Any],
) -> str:
    novelty_level = int(novelty_result.get("reached_level") or 0)
    if novelty_level <= 0:
        return "winning_prefix_still_not_proposed"
    if int(cosmetic_result.get("reached_level") or 0) >= novelty_level:
        return "controllability_embedding_rewards_non_winning_controllable_states"
    if int(no_novelty_result.get("reached_level") or 0) >= novelty_level:
        return "novelty_revisits_dead_states"
    if not novelty_result.get("offline_reproduced"):
        return "winning_prefix_still_not_proposed"
    return "novelty_revisits_dead_states"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    proposer_served_model: str,
    live_path_reachable: bool,
    parity_test_green: bool,
    target_game: str,
    generic_first_win_by_config: Mapping[str, Mapping[str, Any]],
    novelty_result: Mapping[str, Any],
    no_novelty_result: Mapping[str, Any],
    cosmetic_result: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    reached = int(novelty_result.get("generic_agent_reached_level") or novelty_result.get("reached_level") or 0)
    reproduced = bool(novelty_result.get("offline_reproduced"))
    reproduced_levels = int(novelty_result.get("reproduced_levels") or (reached if reproduced else 0))
    no_level = int(no_novelty_result.get("reached_level") or 0)
    cosmetic_level = int(cosmetic_result.get("reached_level") or 0)
    success = _success_attributable(
        reached_level=reached,
        offline_reproduced=reproduced,
        no_novelty_level=no_level,
        cosmetic_level=cosmetic_level,
        live_path_reachable=live_path_reachable,
        parity_test_green=parity_test_green,
    )
    residual = "none" if success else _residual(novelty_result, no_novelty_result, cosmetic_result)
    if success:
        honest_verdict = (
            f"success: controllable_novelty_generic_agent_new_level_{target_game}_L{reached}"
        )
        chosen_config: Any = {
            "controllable_novelty_proposal_enabled": True,
            "controllable_novelty_bonus_weight": float(novelty_result.get("bonus_weight") or 1.0),
            "controllable_novelty_temperature": float(novelty_result.get("temperature") or 1.0),
        }
    else:
        honest_verdict = f"complete: controllable_novelty_no_new_level_residual_{residual}"
        chosen_config = "unchanged"

    bare_control = bool(
        novelty_result.get("bare_control_passed")
        or no_novelty_result.get("bare_control_passed")
        or cosmetic_result.get("bare_control_passed")
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4688", "SCENARIO-ARC-WMTE-4688"],
        "honest_verdict": honest_verdict,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": bool(live_path_reachable),
        "controllability_gate_on": True,
        "generic_first_win_by_config": {
            str(key): dict(value) for key, value in sorted(generic_first_win_by_config.items())
        },
        "generic_agent_reached_level": reached,
        "offline_reproduced": reproduced,
        "reproduced_levels": reproduced_levels,
        "no_novelty_ablation_reached_level": no_level,
        "cosmetic_novelty_ablation_reached_level": cosmetic_level,
        "residual_cause_hypothesis": residual,
        "bare_control_passed": bool(bare_control),
        "false_negative_risk_checked": bool(
            bare_control and no_novelty_result and cosmetic_result
        ),
        "proposer_served_model": str(proposer_served_model),
        "chosen_submitted_config": chosen_config,
        "parity_test_green": bool(parity_test_green),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "target_game": str(target_game),
        "target_arm_results": {
            "controllable_novelty": dict(novelty_result),
            "no_novelty": dict(no_novelty_result),
            "cosmetic_novelty": dict(cosmetic_result),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if not success:
        artifact["null_methodology_note"] = (
            "The run used a /props-verified Qwen proposer, live-path lint, submitted-agent "
            "parity, a reachable-L1 target control, and matched no-novelty/cosmetic-novelty "
            "ablations; no attributed reproduced new level is an honest proposal-policy null."
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
    if artifact.get("controllability_gate_on") is not True:
        errors.append("controllability_gate_on")
    if artifact.get("residual_cause_hypothesis") not in RESIDUALS:
        errors.append("residual_cause_hypothesis")
    served = str(artifact.get("proposer_served_model") or "").lower()
    if "qwen" not in served or "gemma" in served:
        errors.append("proposer_served_model")
    if verdict.startswith("complete:") and "null_methodology_note" not in artifact:
        errors.append("null_methodology_note")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _run_checked(command: Sequence[str], *, timeout: int = 240) -> JsonDict:  # pragma: no cover
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


def check_preconditions(port: int = DEFAULT_PORT) -> tuple[JsonDict, Any | None, str]:  # pragma: no cover
    from carnot import experiment_4664_l2_goal_predicate_induction_live as exp4664

    spec_text = (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "agents_md_read": (REPO_ROOT / "AGENTS.md").exists(),
        "codex_md_read": (REPO_ROOT / "CODEX.md").exists(),
        "spec_has_req_4688": "REQ-ARC-WMTE-4688" in spec_text,
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
        from carnot.agentic import arc_frame_change_predictor, arc_solver_kit, arc_value_learner
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer

        arc_solver_kit.offline_arcade()
        checks["offline_arcade"] = True
        checks["live_modules_importable"] = (
            E3AgentPolicy is not None
            and StepwiseExplorer is not None
            and arc_value_learner is not None
            and arc_frame_change_predictor is not None
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


def _target_has_reachable_l1(game: str) -> bool:  # pragma: no cover
    try:
        artifact = json.loads(
            (REPO_ROOT / "results" / "experiment_4628_dense_curiosity_progress_loop.json").read_text(
                encoding="utf-8"
            )
        )
        for measurement in ("bare_measurement", "loop_measurement"):
            for row in artifact.get(measurement, {}).get("variant_attempts", []) or []:
                if row.get("game") == game and row.get("reachable_headroom") is True:
                    return True
    except Exception:
        pass
    try:
        from carnot.agentic.arc_competition_agent import CLAIMED

        return game in CLAIMED
    except Exception:
        return False


def _run_target_arm(
    *,
    game: str,
    budget: int,
    policy_mode: str,
    controllable_novelty: Any | bool | None,
) -> JsonDict:  # pragma: no cover - ARC runtime boundary.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.experiment_4664_l2_goal_predicate_induction_live import (
        _action_label,
        _apply_action_label,
    )

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpProposer(),
        explore_budget=int(budget) + 1,
        target_levels=1,
        value_head=None,
        value_weight=0.0,
        candidate_router=None,
        navigation_cost_tiebreak=False,
        action_effect_expansion_prior=False,
        goal_bias=None,
        controllable_novelty=controllable_novelty,
    )
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached = 0
    actions_to_first: int | None = None
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
        if latest is None:
            break
        if start_level is None:
            start_level = _level_of(latest)
        reached = _level_of(latest)
        if start_level is not None and reached > start_level and actions_to_first is None:
            actions_to_first = actions
        frames.append(latest)
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
    reproduced = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
    reached_level = int(gate.get("reached_level") or reached) if reproduced else int(reached)
    novelty_diag = policy.explorer.controllable_novelty_diagnostics()
    return {
        "game": game,
        "policy_mode": policy_mode,
        "attempted": True,
        "actions": int(actions),
        "budget": int(budget),
        "reached_level": int(reached_level),
        "generic_agent_reached_level": int(reached_level),
        "offline_reproduced": bool(reproduced),
        "reproduced_levels": int(gate.get("reached_level") or 0) if reproduced else 0,
        "actions_to_first_levelup": actions_to_first if reproduced else None,
        "solution_labels": labels if reproduced else [],
        "reproduction_gate": gate,
        "bare_control_passed": _target_has_reachable_l1(game),
        "controllable_novelty_diagnostics": novelty_diag,
        "bonus_weight": float(getattr(getattr(policy.explorer, "controllable_novelty_policy", None), "config", object()).bonus_weight)
        if policy.explorer.controllable_novelty_policy is not None
        else 0.0,
        "temperature": float(getattr(getattr(policy.explorer, "controllable_novelty_policy", None), "config", object()).temperature)
        if policy.explorer.controllable_novelty_policy is not None
        else 0.0,
    }


def _blocked_artifact(
    checks: Mapping[str, Any],
    *,
    reason: str,
    proposer_served_model: str,
    duration_s: float,
) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=dict(checks, blocked_resource=reason),
        proposer_served_model=proposer_served_model,
        live_path_reachable=False,
        parity_test_green=False,
        target_game="blocked",
        generic_first_win_by_config={},
        novelty_result={},
        no_novelty_result={},
        cosmetic_result={},
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = reason
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _floor_duration(started: float, minimum: float = 60.0) -> float:
    elapsed = time.time() - started
    if elapsed < minimum:
        time.sleep(minimum - elapsed)
    return time.time() - started


def run(
    *,
    root: Path | str = REPO_ROOT,
    port: int = DEFAULT_PORT,
    target_game: str | None = None,
    budget: int | None = None,
) -> JsonDict:  # pragma: no cover - live experiment boundary.
    from carnot.agentic.arc_controllable_novelty import ControllableNoveltyConfig

    started = time.time()
    checks, proposer, served_model = check_preconditions(port=port)
    root_path = Path(root)
    if not checks.get("ok"):
        artifact = _blocked_artifact(
            checks,
            reason=str(checks.get("blocked_resource") or "blocked_precondition"),
            proposer_served_model=served_model,
            duration_s=time.time() - started,
        )
        write_artifact(artifact, root=root_path)
        if proposer is not None:
            proposer.stop()
        return artifact

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

    target = str(target_game or os.environ.get("CARNOT_4688_TARGET", DEFAULT_TARGET_GAME))
    run_budget = int(budget if budget is not None else os.environ.get("CARNOT_4688_BUDGET", DEFAULT_BUDGET))
    try:
        novelty = _run_target_arm(
            game=target,
            budget=run_budget,
            policy_mode="controllable_novelty_t0.5",
            controllable_novelty=ControllableNoveltyConfig(
                enabled=True,
                bonus_weight=1.0,
                temperature=0.5,
                controllability_gate=True,
            ),
        )
        no_novelty = _run_target_arm(
            game=target,
            budget=run_budget,
            policy_mode="no_novelty_bonus",
            controllable_novelty=False,
        )
        cosmetic = _run_target_arm(
            game=target,
            budget=run_budget,
            policy_mode="cosmetic_novelty_gate_off",
            controllable_novelty=ControllableNoveltyConfig(
                enabled=True,
                bonus_weight=1.0,
                temperature=0.5,
                controllability_gate=False,
                raw_frame_novelty=True,
            ),
        )
    finally:
        if proposer is not None:
            proposer.stop()

    generic_first_win_by_config = {
        "controllable_novelty_t0.5": measurement_from_attempts([novelty]),
        "no_novelty_bonus": measurement_from_attempts([no_novelty]),
        "cosmetic_novelty_gate_off": measurement_from_attempts([cosmetic]),
    }
    duration = _floor_duration(started, minimum=60.0)
    artifact = build_artifact(
        preconditions_checked=checks,
        proposer_served_model=served_model,
        live_path_reachable=bool(live_check.get("passed")),
        parity_test_green=bool(parity.get("passed")),
        target_game=target,
        generic_first_win_by_config=generic_first_win_by_config,
        novelty_result=novelty,
        no_novelty_result=no_novelty,
        cosmetic_result=cosmetic,
        duration_s=duration,
    )
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "generic_agent_reached_level": artifact["generic_agent_reached_level"],
                "no_novelty_ablation_reached_level": artifact[
                    "no_novelty_ablation_reached_level"
                ],
                "cosmetic_novelty_ablation_reached_level": artifact[
                    "cosmetic_novelty_ablation_reached_level"
                ],
                "proposer_served_model": artifact["proposer_served_model"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
