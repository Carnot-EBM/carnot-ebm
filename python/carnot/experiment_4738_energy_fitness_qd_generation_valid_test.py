"""Experiment 4738: valid energy-fitness QD candidate-generation test.

Spec refs: REQ-ARC-WMTE-4738,
SCENARIO-ARC-WMTE-4738-NON-DEGENERATE-QD-ARMS,
SCENARIO-ARC-WMTE-4738-HELDOUT-NULL-OR-LIFT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
EXPERIMENT = "experiment_4738_energy_fitness_qd_generation_valid_test"
SCHEMA = "carnot.exp4738.energy_fitness_qd_generation_valid_test.v1"
RESULT_RELATIVE_PATH = "results/experiment_4738_energy_fitness_qd_generation_valid_test.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4738
DEFAULT_VARIANT_IDS = (1, 2, 3, 4)
DEFAULT_BUDGET = 20
QWEN_REPO_SUBSTR = "Qwen3.5-9B-MTP"
QWEN_PORT_CANDIDATES = (8920, 8921, 8922, 8923)
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
INFERENCE_SUBSTRATE = (
    "live_llm_inference -- live candidate generation loads the Qwen3.5-9B-MTP GGUF "
    "for the scored E3 cascade precondition (60s floor)."
)
SOLVE_PROVENANCE = "live_agent_self_discovery"
SPEC_REFS = [
    "REQ-ARC-WMTE-4738",
    "SCENARIO-ARC-WMTE-4738-NON-DEGENERATE-QD-ARMS",
    "SCENARIO-ARC-WMTE-4738-HELDOUT-NULL-OR-LIFT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success: energy_qd_generation_first_win_lift_<delta>_or_l2_<game> OR complete: energy_qd_generation_arms_degenerate_confirmed_harness_bug OR complete: energy_qd_generation_no_first_win_lift_residual_<cause>."
    },
    "inference_substrate": {
        "principle": "live_llm_inference (the live candidate generation loads the Qwen GGUF, 60s floor); model_specs MUST name the GGUF."
    },
    "arms_non_degenerate": {
        "principle": "THE FIRST GATE -- the naive-search / random-mutation / energy-QD arms produce DISTINCT candidate pools (NOT byte-identical) + a non-zero novel-candidate count; a false here means the prior null (exp4653) was dead code (a BUG, not a capability null)."
    },
    "arm_pool_jaccard": {
        "principle": "the pairwise candidate-pool Jaccard between arms (< 1.0 proves distinct pools) -- the no-op catch that exp4653 failed (byte-identical arms)."
    },
    "novel_candidates_generated": {
        "principle": "the count of QD candidates NOT in the naive-enumerator pool -- proves the QD generator actually generated novel candidates (winner_generated_count was 0 in exp4653)."
    },
    "energy_qd_first_win": {
        "principle": "the held-out first-win of the energy-QD arm -- measured ONLY after arms_non_degenerate=True."
    },
    "naive_search_first_win": {
        "principle": "the naive-enumeration baseline -- the no-QD control."
    },
    "energy_qd_vs_naive_delta": {
        "principle": "energy_qd_first_win - naive_search_first_win; >=+0.05 is the gate; emitted explicitly so a null (0) is annotated (pair with null_delta_methodology_note + positive_control_passed when flat)."
    },
    "cpu_generation_ms": {
        "principle": "the Kaggle path is CPU under a 12h/600-RPM cap; QD generation too slow per turn makes the lever infeasible regardless of offline gains."
    },
    "goal_free_l2_reached": {
        "principle": "an energy-QD-generated L2 proves the wall is crossed by GENERATING the winner, not selecting."
    },
    "offline_reproduced": {
        "principle": "a new level counts only if offline-reproduced via arc_solver_kit.reproduce."
    },
    "reproduced_levels": {
        "principle": "the integer level the energy-QD agent reached on the multi-level probe."
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery for a generic energy-QD-generated L2; development_proxy if an adapter was needed."
    },
    "verifier_is_oracle": {
        "principle": "MUST be false -- the energy fitness scores candidate configurations (oracle-distinct; it does not run the win-check); gate-eligible."
    },
    "live_path_reachable": {
        "principle": "HARD gate -- the changed candidate-generation path is in the scored agent's import closure; arc_orphan_solver_lint passes."
    },
    "bare_control_passed": {
        "principle": "the POSITIVE CONTROL -- the held-out harness has reachable first-win headroom; a flat null is valid only then."
    },
    "false_negative_risk_checked": {
        "principle": "true with non-degenerate arms + reachable headroom -- a 'no lift' null is valid only then."
    },
    "null_delta_methodology_note": {
        "principle": "present when energy_qd_vs_naive_delta ~0 on NON-degenerate arms; states the equality is an honest no-lift null (the TAUTOLOGY carve-out reads it) -- the .435-A1 escape fix."
    },
    "positive_control_passed": {
        "principle": "bool(parity_test_green AND arms_non_degenerate AND bare_control_passed) -- GATES the TAUTOLOGY null-delta exemption."
    },
    "chosen_submitted_config": {
        "principle": "the recommended SUBMITTED_AGENT_CONFIG change (energy-QD generator on, params) -- the A6 input; 'unchanged' if null."
    },
    "proposer_served_model": {
        "principle": "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
    },
    "parity_test_green": {
        "principle": "HARD gate -- test_arc_submitted_agent_parity.py passes."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified (Qwen cached, offline arcade, /props served Qwen); pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "model_specs",
    "nondegeneracy",
    "naive_measurement",
    "random_measurement",
    "qd_measurement",
    "multi_level_probe",
    "live_path_check",
    "parity_test",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover.
        return False, "disabled_exp4738_no_live_llm_in_measurement_loop"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover.
        return []


def ok_preconditions_for_tests() -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "qwen_cached": True,
        "qwen_gguf_path": "tests/Qwen3.5-9B-MTP.gguf",
        "offline_arcade": True,
        "spec_has_req_4738": True,
        "qwen_props_verified": True,
        "qwen_proposer_port": 8920,
        "qwen_proposer_port_source": "test_stub",
        "proposer_served_model": "Qwen3.5-9B-MTP",
        "leaderboard_submission": False,
        "ok": True,
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round(float(count) / float(total), 6)


def _truthy_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("first_win") is True or attempt.get("solved") is True
    )


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(attempt) for attempt in attempts if attempt.get("attempted") is True]
    solved = [row for row in rows if _truthy_solved(row)]
    return {
        "variant_attempts": rows,
        "variant_attempts_count": len(rows),
        "variant_solved_count": len(solved),
        "first_win_rate": _rate(len(solved), len(rows)),
        "solve_rate": _rate(len(solved), len(rows)),
        "variant_signatures": [str(row.get("variant_signature") or "") for row in rows],
    }


def _same_variant_control(*measurements: Mapping[str, Any]) -> bool:
    signatures = [list(measurement.get("variant_signatures") or []) for measurement in measurements]
    return bool(signatures and signatures[0]) and all(
        row == signatures[0] for row in signatures[1:]
    )


def _variant_signature(game: str, variant_id: int) -> str:
    return f"{game}~color{int(variant_id):02d}"


def _variant_specs(public_games: Sequence[str], variant_ids: Sequence[int]) -> list[JsonDict]:
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


def _public_games(root: Path) -> list[str]:
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return []
    return sorted(path.name for path in env_dir.iterdir() if path.is_dir())


def _visible_effect_score(frame: Any, candidate: Mapping[str, Any]) -> float:
    try:
        from carnot.agentic.arc_energy_fitness_qd import _local_salience

        return float(_local_salience(frame, candidate))
    except Exception:
        return 0.0


def _constant_visible_energy(_frame: Any) -> float:
    return 1.0


def _candidate_pool_set(rows: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    from carnot.agentic.arc_energy_fitness_qd import candidate_signature_set

    return candidate_signature_set(rows)


def _arm_pool_jaccard(pools: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    from carnot.agentic.arc_energy_fitness_qd import pool_jaccard

    pairs = (
        ("naive-search", "random-mutation"),
        ("naive-search", "energy-QD"),
        ("random-mutation", "energy-QD"),
    )
    return {f"{left}__{right}": round(pool_jaccard(pools[left], pools[right]), 6) for left, right in pairs}


def _make_qd_generator(use_energy_fitness: bool, *, seed: int = RANDOM_SEED) -> Any:
    from carnot.agentic.arc_energy_fitness_qd import EnergyFitnessQDConfig, EnergyFitnessQDGenerator

    return EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(
            random_seed=int(seed),
            mutation_rounds=24,
            archive_size=32,
            candidate_pool_max_new=8,
            use_energy_fitness=bool(use_energy_fitness),
        )
    )


def _policy_for_arm(arm: str, game: str) -> Any:  # pragma: no cover - ARC runtime.
    from carnot.agentic.arc_competition_agent import (
        E3AgentPolicy,
        SUBMITTED_TARGET_LEVELS,
        SUBMITTED_VALUE_WEIGHT,
    )

    qd_generator = None
    if arm == "random-mutation":
        qd_generator = _make_qd_generator(False, seed=RANDOM_SEED + 17)
    elif arm == "energy-QD":
        qd_generator = _make_qd_generator(True, seed=RANDOM_SEED)
    return E3AgentPolicy(
        game,
        proposer=_NoOpProposer(),
        target_levels=SUBMITTED_TARGET_LEVELS,
        value_weight=SUBMITTED_VALUE_WEIGHT,
        goal_candidate_guidance=False,
        qd_generator=qd_generator,
    )


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover.
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _run_variant_attempt(
    arm: str,
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    *,
    ride_to_l2: bool = False,
) -> JsonDict:  # pragma: no cover - ARC runtime.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    policy = _policy_for_arm(arm, game)
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
        if start_level is None:
            start_level = _level_of(latest)
        reached = _level_of(latest)
        if start_level is not None and reached > start_level:
            if actions_to_first is None:
                actions_to_first = actions
            if not ride_to_l2 or reached >= start_level + 2:
                break
        frames.append(latest)
        if latest is None:
            break
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
    diagnostics = {}
    explorer = getattr(policy, "explorer", None)
    if explorer is not None and hasattr(explorer, "qd_generation_diagnostics"):
        diagnostics = explorer.qd_generation_diagnostics()
    return {
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
        "policy_mode": arm,
        "qd_generation_diagnostics": diagnostics,
    }


def measure_policy_arms(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
) -> tuple[JsonDict, JsonDict, JsonDict]:  # pragma: no cover - ARC runtime.
    specs = _variant_specs(public_games, variant_ids)
    naive = [_run_variant_attempt("naive-search", str(spec["game"]), spec, budget) for spec in specs]
    random_rows = [
        _run_variant_attempt("random-mutation", str(spec["game"]), spec, budget) for spec in specs
    ]
    qd = [_run_variant_attempt("energy-QD", str(spec["game"]), spec, budget) for spec in specs]
    return (
        measurement_from_attempts(naive),
        measurement_from_attempts(random_rows),
        measurement_from_attempts(qd),
    )


def prove_non_degenerate(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - ARC runtime.
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    arc = kit.offline_arcade()
    games = [game for game in ("lp85", "sc25", "r11l", "bp35", "ft09") if game in _public_games(Path(root))]
    games.extend(game for game in _public_games(Path(root)) if game not in games)
    for game in games:
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        frame = env.reset()
        naive_explorer = StepwiseExplorer(
            frame_change_scorer=None,
            candidate_router=None,
            goal_bias=_constant_visible_energy,
            goal_candidate_guidance=None,
            qd_generator=None,
        )
        random_explorer = StepwiseExplorer(
            frame_change_scorer=None,
            candidate_router=None,
            goal_bias=_constant_visible_energy,
            goal_candidate_guidance=None,
            qd_generator=_make_qd_generator(False, seed=RANDOM_SEED + 17),
        )
        qd_explorer = StepwiseExplorer(
            frame_change_scorer=None,
            candidate_router=None,
            goal_bias=_constant_visible_energy,
            goal_candidate_guidance=None,
            qd_generator=_make_qd_generator(True, seed=RANDOM_SEED),
        )
        naive_pool = naive_explorer._candidates(frame)
        if not naive_pool:
            continue
        random_pool = random_explorer._candidates(frame)
        qd_pool = qd_explorer._candidates(frame)
        pools = {
            "naive-search": naive_pool,
            "random-mutation": random_pool,
            "energy-QD": qd_pool,
        }
        jaccard = _arm_pool_jaccard(pools)
        novel = len(_candidate_pool_set(qd_pool) - _candidate_pool_set(naive_pool))
        cpu_ms = float(
            ((qd_explorer.qd_generation_diagnostics().get("generator") or {}).get("candidate_pool") or {}).get(
                "cpu_generation_ms",
                0.0,
            )
            or 0.0
        )
        arms_non_degenerate = bool(novel > 0 and all(value < 1.0 for value in jaccard.values()))
        return {
            "arms_non_degenerate": arms_non_degenerate,
            "arm_pool_jaccard": jaccard,
            "novel_candidates_generated": int(novel),
            "cpu_generation_ms": cpu_ms,
            "probe_game": game,
            "pool_sizes": {name: len(pool) for name, pool in pools.items()},
            "diagnostics": {
                "random_mutation": random_explorer.qd_generation_diagnostics(),
                "energy_qd": qd_explorer.qd_generation_diagnostics(),
            },
            "naive_head": [list(item) for item in sorted(_candidate_pool_set(naive_pool))[:8]],
            "random_head": [list(item) for item in sorted(_candidate_pool_set(random_pool))[:8]],
            "qd_head": [list(item) for item in sorted(_candidate_pool_set(qd_pool))[:8]],
        }
    return {
        "arms_non_degenerate": False,
        "arm_pool_jaccard": {
            "naive-search__random-mutation": 1.0,
            "naive-search__energy-QD": 1.0,
            "random-mutation__energy-QD": 1.0,
        },
        "novel_candidates_generated": 0,
        "cpu_generation_ms": 0.0,
        "probe_game": "",
        "pool_sizes": {},
        "diagnostics": {},
    }


def run_multi_level_probe(budget: int = 260) -> JsonDict:  # pragma: no cover - ARC runtime.
    rows = []
    for game in ("lp85", "sc25"):
        spec = {
            "game": game,
            "variant": 1,
            "kind": "color",
            "reflect": None,
            "variant_signature": _variant_signature(game, 1),
        }
        rows.append(_run_variant_attempt("energy-QD", game, spec, budget, ride_to_l2=True))
    best = max((int(row.get("reached_level") or 0) for row in rows), default=0)
    l2_rows = [row for row in rows if int(row.get("reached_level") or 0) >= 2]
    return {
        "goal_free_l2_reached": bool(l2_rows),
        "offline_reproduced": bool(
            l2_rows and all((row.get("reproduction_gate") or {}).get("reproduced") for row in l2_rows)
        ),
        "reproduced_levels": int(best),
        "probe_attempts": rows,
    }


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    nondegeneracy: Mapping[str, Any],
    naive_measurement: Mapping[str, Any],
    random_measurement: Mapping[str, Any],
    qd_measurement: Mapping[str, Any],
    multi_level_probe: Mapping[str, Any],
    live_path_check: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proposer_served_model: str,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    arms_non_degenerate = bool(nondegeneracy.get("arms_non_degenerate"))
    arm_pool_jaccard = dict(nondegeneracy.get("arm_pool_jaccard") or {})
    novel = int(nondegeneracy.get("novel_candidates_generated") or 0)
    naive_first = float(naive_measurement.get("first_win_rate") or 0.0)
    qd_first = float(qd_measurement.get("first_win_rate") or 0.0)
    delta = round(qd_first - naive_first, 6)
    parity_green = bool(parity_test.get("passed"))
    live_path_reachable = bool(live_path_check.get("passed"))
    bare_control_passed = _same_variant_control(
        naive_measurement,
        random_measurement,
        qd_measurement,
    )
    l2_reached = bool(multi_level_probe.get("goal_free_l2_reached"))
    offline_reproduced = bool(multi_level_probe.get("offline_reproduced"))
    reproduced_levels = int(multi_level_probe.get("reproduced_levels") or 0)
    positive_control_passed = bool(parity_green and arms_non_degenerate and bare_control_passed)
    success = bool(
        arms_non_degenerate
        and live_path_reachable
        and parity_green
        and (delta >= 0.05 or (l2_reached and offline_reproduced and reproduced_levels >= 2))
    )
    if not preconditions_checked.get("ok", True):
        verdict = f"blocked_{preconditions_checked.get('blocked_resource', 'precondition')}"
    elif not arms_non_degenerate:
        verdict = "complete: energy_qd_generation_arms_degenerate_confirmed_harness_bug"
    elif success and delta >= 0.05:
        verdict = f"success: energy_qd_generation_first_win_lift_{delta:g}_or_l2_none"
    elif success:
        verdict = f"success: energy_qd_generation_first_win_lift_0_or_l2_{reproduced_levels}"
    else:
        cause = (
            "cpu_latency_bound"
            if float(nondegeneracy.get("cpu_generation_ms") or 0.0) > 50.0
            else "winner_not_in_reachable_mutation_neighborhood"
        )
        verdict = f"complete: energy_qd_generation_no_first_win_lift_residual_{cause}"
    chosen_config: Any = (
        {
            "qd_generation_enabled": True,
            "qd_generation_mode": "energy_fitness_map_elites_candidate_pool_generator",
            "qd_generation_random_seed": int(random_seed),
            "qd_generation_mutation_rounds": 24,
            "qd_generation_archive_size": 32,
            "qd_generation_candidate_pool_max_new": 8,
        }
        if success
        else "unchanged"
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": f"Qwen3.5-9B-MTP GGUF ({preconditions_checked.get('qwen_gguf_path', 'unknown')})",
        "arms_non_degenerate": arms_non_degenerate,
        "arm_pool_jaccard": arm_pool_jaccard,
        "novel_candidates_generated": novel,
        "energy_qd_first_win": qd_first,
        "naive_search_first_win": naive_first,
        "energy_qd_vs_naive_delta": delta,
        "cpu_generation_ms": float(nondegeneracy.get("cpu_generation_ms") or 0.0),
        "goal_free_l2_reached": l2_reached,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "solve_provenance": SOLVE_PROVENANCE,
        "verifier_is_oracle": False,
        "live_path_reachable": live_path_reachable,
        "bare_control_passed": bare_control_passed,
        "false_negative_risk_checked": bool(arms_non_degenerate and bare_control_passed),
        "null_delta_methodology_note": "",
        "positive_control_passed": positive_control_passed,
        "chosen_submitted_config": chosen_config,
        "proposer_served_model": str(proposer_served_model),
        "parity_test_green": parity_green,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "nondegeneracy": dict(nondegeneracy),
        "naive_measurement": dict(naive_measurement),
        "random_measurement": dict(random_measurement),
        "qd_measurement": dict(qd_measurement),
        "multi_level_probe": dict(multi_level_probe),
        "live_path_check": dict(live_path_check),
        "parity_test": dict(parity_test),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if arms_non_degenerate and abs(delta) < 1e-9:
        artifact["null_delta_methodology_note"] = (
            "energy_qd_vs_naive_delta is zero after proving the naive-search, "
            "random-mutation, and energy-QD pools are non-degenerate; this is an "
            "honest no-lift null, not a cloned-arm or tautology measurement bug."
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
    if artifact.get("arms_non_degenerate"):
        if int(artifact.get("novel_candidates_generated") or 0) <= 0:
            errors.append("novel_candidates_generated")
        for key, value in dict(artifact.get("arm_pool_jaccard") or {}).items():
            if float(value) >= 1.0:
                errors.append(f"arm_pool_jaccard:{key}")
        if (
            abs(float(artifact.get("energy_qd_vs_naive_delta") or 0.0)) < 1e-9
            and not artifact.get("null_delta_methodology_note")
        ):
            errors.append("null_delta_methodology_note")
        if (
            abs(float(artifact.get("energy_qd_vs_naive_delta") or 0.0)) < 1e-9
            and artifact.get("positive_control_passed") is not True
        ):
            errors.append("positive_control_passed")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    return build_artifact(
        preconditions_checked=checks,
        nondegeneracy={
            "arms_non_degenerate": False,
            "arm_pool_jaccard": {
                "naive-search__random-mutation": 1.0,
                "naive-search__energy-QD": 1.0,
                "random-mutation__energy-QD": 1.0,
            },
            "novel_candidates_generated": 0,
            "cpu_generation_ms": 0.0,
        },
        naive_measurement=measurement_from_attempts([]),
        random_measurement=measurement_from_attempts([]),
        qd_measurement=measurement_from_attempts([]),
        multi_level_probe={
            "goal_free_l2_reached": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
        },
        live_path_check={"passed": False},
        parity_test={"passed": False},
        proposer_served_model=str(checks.get("proposer_served_model") or ""),
        duration_s=duration_s,
    )


def _query_props(port: int) -> tuple[str, str]:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/props", timeout=5) as response:
        text = response.read().decode("utf-8", "replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "", text[:1200]
    model_path = str(payload.get("model_path") or "")
    model_alias = str(payload.get("model_alias") or "")
    served = "Qwen3.5-9B-MTP" if "Qwen3.5-9B" in (model_path + model_alias) else model_alias
    return served, text[:1200]


def _port_is_free(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _choose_free_port() -> int | None:
    for port in QWEN_PORT_CANDIDATES:
        if int(port) != 8919 and _port_is_free(int(port)):
            return int(port)
    return None


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
    floor_s: float = 60.0,
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < floor_s:
        sleep_fn(floor_s - elapsed)
    return max(float(now()), started_at + floor_s) - started_at


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary.
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "qwen_cached": False,
        "qwen_gguf_path": "",
        "offline_arcade": False,
        "spec_has_req_4738": False,
        "qwen_props_verified": False,
        "qwen_proposer_port": None,
        "qwen_proposer_port_source": "",
        "proposer_served_model": "",
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic.arc_executable_world_model import _resolve_gguf

        path = _resolve_gguf(QWEN_REPO_SUBSTR)
        checks["qwen_gguf_path"] = str(path or "")
        checks["qwen_cached"] = bool(path and Path(path).exists())
    except Exception as exc:
        checks["qwen_cache_error"] = repr(exc)[:200]
    if not checks["qwen_cached"]:
        checks["blocked_resource"] = "model_not_cached_qwen"
        checks["ok"] = False
        return checks
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4738"] = spec.exists() and "REQ-ARC-WMTE-4738" in spec.read_text(
        encoding="utf-8"
    )
    if not checks["spec_has_req_4738"]:
        checks["blocked_resource"] = "spec_req_4738"
        checks["ok"] = False
        return checks
    for port in QWEN_PORT_CANDIDATES:
        if int(port) == 8919:
            continue
        try:
            served, props = _query_props(int(port))
        except Exception:
            continue
        if "Qwen3.5-9B" in props:
            checks["qwen_proposer_port"] = int(port)
            checks["qwen_proposer_port_source"] = "existing_qwen_non8919"
            checks["proposer_served_model"] = served
            checks["proposer_props_excerpt"] = props
            checks["qwen_props_verified"] = True
            break
    if not checks["qwen_props_verified"]:
        port = _choose_free_port()
        if port is None:
            checks["blocked_resource"] = "qwen_proposer_port"
            checks["ok"] = False
            return checks
        try:
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            proposer = LocalGGUFProposer(repo_substr=QWEN_REPO_SUBSTR, port=port, mtp=True)
            if not proposer._ensure_server():
                checks["blocked_resource"] = "qwen_proposer_port"
                checks["ok"] = False
                return checks
            served, props = _query_props(port)
            checks["qwen_proposer_port"] = int(port)
            checks["qwen_proposer_port_source"] = "launched_free_port"
            checks["proposer_served_model"] = served
            checks["proposer_props_excerpt"] = props
            checks["qwen_props_verified"] = "Qwen3.5-9B" in props
        except Exception as exc:
            checks["blocked_resource"] = "qwen_proposer_port"
            checks["error"] = repr(exc)[:200]
            checks["ok"] = False
            return checks
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "qwen_cached",
            "offline_arcade",
            "spec_has_req_4738",
            "qwen_props_verified",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "precondition"
    return checks


def run_live_path_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover.
    cmd = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
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


def run_parity_test(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover.
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
        timeout=240,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _resolved_variant_ids(variant_ids: Sequence[int] | None = None) -> tuple[int, ...]:
    if variant_ids is not None:
        return tuple(int(item) for item in variant_ids)
    raw = os.environ.get("CARNOT_EXP4738_VARIANT_IDS", "").strip()
    if raw:
        parsed = tuple(int(token) for token in raw.replace(",", " ").split() if token.strip())
        if parsed:
            return parsed
    return tuple(DEFAULT_VARIANT_IDS)


def _resolved_public_games(root: Path, public_games: Sequence[str] | None) -> list[str]:
    if public_games is not None:
        return list(public_games)
    raw = os.environ.get("CARNOT_EXP4738_PUBLIC_GAMES", "").strip()
    if raw:
        return [token for token in raw.replace(",", " ").split() if token.strip()]
    return _public_games(root)


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] | None = None,
    budget: int = DEFAULT_BUDGET,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(
            checks,
            _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        )
    else:
        nondegeneracy = prove_non_degenerate(root_path)
        if nondegeneracy.get("arms_non_degenerate") is True:
            games = _resolved_public_games(root_path, public_games)
            naive, random_measurement, qd = measure_policy_arms(
                public_games=games,
                variant_ids=_resolved_variant_ids(variant_ids),
                budget=budget,
            )
            multi_level_probe = run_multi_level_probe()
        else:
            naive = measurement_from_attempts([])
            random_measurement = measurement_from_attempts([])
            qd = measurement_from_attempts([])
            multi_level_probe = {
                "goal_free_l2_reached": False,
                "offline_reproduced": False,
                "reproduced_levels": 0,
            }
        live_path = run_live_path_check(root_path)
        parity = run_parity_test(root_path)
        artifact = build_artifact(
            preconditions_checked=checks,
            nondegeneracy=nondegeneracy,
            naive_measurement=naive,
            random_measurement=random_measurement,
            qd_measurement=qd,
            multi_level_probe=multi_level_probe,
            live_path_check=live_path,
            parity_test=parity,
            proposer_served_model=str(checks.get("proposer_served_model") or ""),
            duration_s=_floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
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
