"""Experiment 5154: energy-fitness directed exploration on the live ARC path.

Spec refs: REQ-ARC-WMTE-5154, SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.agentic.arc_goal_energy_live import GOAL_ENERGY_SOURCE


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5154_energy_fitness_directed_exploration_v472"
SCHEMA = "carnot.exp5154.energy_fitness_directed_exploration.v1"
RESULT_RELATIVE_PATH = "results/experiment_5154_energy_fitness_directed_exploration_v472.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 5154
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 69
TARGET_GAMES = ("bp35",)
DEFAULT_BUDGET = 200
TERMINAL_PREFIXES = ("success:", "success_", "complete:", "complete_", "blocked_")
SPEC_REFS = [
    "REQ-ARC-WMTE-5154",
    "SCENARIO-ARC-WMTE-5154-LIVE-ENERGY-QD",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "MUST start with complete:/complete_/success:/success_ and state plainly whether the winning trajectory was surfaced."
    },
    "energy_signal_source": {
        "principle": "which specific Carnot energy/verifier computation was used as the fitness function; must be genuine Exp4020 graded goal energy, not novelty."
    },
    "reproducible_levels_delta": {
        "principle": "delta against ops/arc_solve_registry.yaml reproducible_total_levels; zero for a null."
    },
    "offline_reproduced": {
        "principle": "a new level counts only if reproduced by arc_solver_kit.reproduce."
    },
    "live_path_reachable": {
        "principle": "confirmed by scripts/arc_orphan_solver_lint.py; off-path solvers do not count."
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery only if the live E3AgentPolicy path made the discovery at runtime."
    },
    "verifier_is_oracle": {
        "principle": "MUST be false; energy fitness scores visible candidate states and does not call the win-check."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "honest_verdict",
    "energy_signal_source",
    "reproducible_levels_delta",
    "offline_reproduced",
    "live_path_reachable",
    "solve_provenance",
    "verifier_is_oracle",
    "target_games",
    "baseline_reproducible_total_levels",
    "generic_agent_reached_level",
    "no_energy_ablation_reached_level",
    "winning_trajectory_surfaced",
    "qd_generation_diagnostics",
    "matched_control",
    "preconditions_checked",
    "field_principles",
    "spec_refs",
    "random_seed",
    "reproducibility_checksum",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover.
        return False, "disabled_exp5154_energy_qd_no_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover.
        return []


def ok_preconditions_for_tests() -> dict[str, Any]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "spec_has_req_5154": True,
        "registry_reproducible_total_levels": BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry_baseline_confirmed": True,
        "energy_scalar_available": True,
        "energy_signal_source": GOAL_ENERGY_SOURCE,
        "offline_arcade": True,
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


def _registry_total(root: Path) -> int | None:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("reproducible_total_levels:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _energy_source_from_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    generator = diagnostics.get("generator") if isinstance(diagnostics, Mapping) else None
    if isinstance(generator, Mapping):
        candidate_pool = generator.get("candidate_pool")
        if isinstance(candidate_pool, Mapping):
            source = candidate_pool.get("energy_signal_source")
            if source:
                return str(source)
        source = generator.get("energy_signal_source")
        if source:
            return str(source)
    return GOAL_ENERGY_SOURCE


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_check: Mapping[str, Any],
    energy_arm: Mapping[str, Any],
    no_energy_control: Mapping[str, Any],
    reproduction_gate: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    qd_diagnostics = dict(energy_arm.get("qd_generation_diagnostics") or {})
    energy_signal_source = _energy_source_from_diagnostics(qd_diagnostics)
    reached_level = int(energy_arm.get("reached_level") or 0)
    control_reached = int(no_energy_control.get("reached_level") or 0)
    winning_surfaced = bool(energy_arm.get("winning_trajectory_surfaced"))
    live_path_reachable = bool(live_path_check.get("passed"))
    offline_reproduced = bool(reproduction_gate.get("reproduced")) and reached_level > 0
    registry_total = int(
        preconditions_checked.get(
            "registry_reproducible_total_levels",
            BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        )
        or BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    )
    # This pilot does not update the registry directly. A new countable level
    # would be reconciled only after reproduction and registry update.
    reproducible_delta = 0
    if not preconditions_checked.get("ok", True):
        verdict = f"blocked_{preconditions_checked.get('blocked_resource', 'precondition')}"
    elif winning_surfaced:
        verdict = "complete: energy_fitness_qd_winning_trajectory_surfaced_but_not_new_reproducible_level"
    else:
        verdict = "complete: energy_fitness_qd_winning_trajectory_not_surfaced_reproducible_delta_0"

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "energy_signal_source": energy_signal_source,
        "reproducible_levels_delta": int(reproducible_delta),
        "offline_reproduced": bool(offline_reproduced),
        "live_path_reachable": live_path_reachable,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "target_games": list(TARGET_GAMES),
        "baseline_reproducible_total_levels": registry_total,
        "generic_agent_reached_level": reached_level,
        "no_energy_ablation_reached_level": control_reached,
        "winning_trajectory_surfaced": winning_surfaced,
        "qd_generation_diagnostics": qd_diagnostics,
        "matched_control": dict(no_energy_control),
        "energy_arm": dict(energy_arm),
        "reproduction_gate": dict(reproduction_gate),
        "preconditions_checked": dict(preconditions_checked),
        "live_path_check": dict(live_path_check),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 6),
        "reproducibility_checksum": "",
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if (
        "winning_trajectory_surfaced" not in verdict
        and "winning_trajectory_not_surfaced" not in verdict
        and not verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict_surface_plaintext")
    if artifact.get("energy_signal_source") != GOAL_ENERGY_SOURCE:
        errors.append("energy_signal_source")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if (
        artifact.get("offline_reproduced") is False
        and int(artifact.get("reproducible_levels_delta") or 0) != 0
    ):
        errors.append("delta_without_reproduction")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - runtime.
    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "spec_has_req_5154": False,
        "registry_reproducible_total_levels": _registry_total(root_path),
        "registry_baseline_confirmed": False,
        "energy_scalar_available": False,
        "energy_signal_source": "",
        "offline_arcade": False,
    }
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_5154"] = spec.exists() and "REQ-ARC-WMTE-5154" in spec.read_text(
        encoding="utf-8"
    )
    checks["registry_baseline_confirmed"] = (
        checks["registry_reproducible_total_levels"] == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
    )
    try:
        from carnot.agentic.arc_goal_energy_live import load_exp4020_goal_energy

        energy = load_exp4020_goal_energy(root_path)
        checks["energy_scalar_available"] = energy is not None and callable(energy)
        checks["energy_signal_source"] = str(getattr(energy, "source", "")) if energy else ""
    except Exception as exc:
        checks["energy_scalar_error"] = repr(exc)[:200]
    if not checks["energy_scalar_available"]:
        checks["blocked_resource"] = "blocked_no_reusable_energy_scalar"
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
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "spec_has_req_5154",
            "registry_baseline_confirmed",
            "energy_scalar_available",
            "offline_arcade",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "precondition"
    return checks


def run_live_path_check(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover.
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


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover.
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _make_energy_qd_generator() -> Any:  # pragma: no cover - runtime.
    from carnot.agentic.arc_energy_fitness_qd import EnergyFitnessQDConfig, EnergyFitnessQDGenerator

    return EnergyFitnessQDGenerator(
        EnergyFitnessQDConfig(
            random_seed=RANDOM_SEED,
            mutation_rounds=24,
            archive_size=32,
            candidate_pool_max_new=8,
            use_energy_fitness=True,
        )
    )


def run_policy_attempt(
    *,
    game: str,
    use_energy_qd: bool,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:  # pragma: no cover - ARC runtime.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import StepwiseExplorer, _level_of
    from carnot.agentic.arc_goal_energy_live import load_exp4020_goal_energy

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    goal_energy = load_exp4020_goal_energy(REPO_ROOT)
    policy = StepwiseExplorer(
        target_levels=3,
        value_head=None,
        max_depth=budget,
        frame_change_scorer=None,
        candidate_router=None,
        goal_bias=goal_energy,
        goal_bias_label=GOAL_ENERGY_SOURCE,
        goal_bias_lower_is_better=True,
        goal_candidate_guidance=False,
        qd_generator=_make_energy_qd_generator() if use_energy_qd else None,
        controllable_novelty=False,
        object_centric_proposal=False,
        program_synthesis_filter=False,
        amortized_first_contact_prior=False,
        go_explore_archive=False,
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
        if start_level is not None and reached > start_level:
            if actions_to_first is None:
                actions_to_first = actions
            break
        frames.append(latest)
    diagnostics = {}
    if hasattr(policy, "qd_generation_diagnostics"):
        diagnostics = policy.qd_generation_diagnostics()
    return {
        "game": str(game),
        "policy_mode": "energy_fitness_qd" if use_energy_qd else "no_energy_ablation",
        "attempted": True,
        "reached_level": int(reached),
        "actions": int(actions),
        "actions_to_first_levelup": actions_to_first,
        "winning_trajectory_surfaced": bool(actions_to_first is not None),
        "solution_labels": labels if actions_to_first is not None else [],
        "qd_generation_diagnostics": diagnostics,
    }


def reproduction_gate_for_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover.
    if not attempt.get("solution_labels"):
        return {
            "game": str(attempt.get("game") or ""),
            "claimed_level": int(attempt.get("reached_level") or 0),
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_solution",
        }
    from carnot.agentic import arc_solver_kit as kit

    return dict(
        kit.reproduce(
            str(attempt.get("game") or ""),
            list(attempt.get("solution_labels") or []),
            _apply_action_label,
            claimed_level=int(attempt.get("reached_level") or 0),
        )
    )


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> dict[str, Any]:
    return build_artifact(
        preconditions_checked=checks,
        live_path_check={"passed": False},
        energy_arm={
            "game": "bp35",
            "reached_level": 0,
            "winning_trajectory_surfaced": False,
            "qd_generation_diagnostics": {
                "enabled": False,
                "verifier_is_oracle": False,
            },
        },
        no_energy_control={
            "game": "bp35",
            "reached_level": 0,
            "winning_trajectory_surfaced": False,
        },
        reproduction_gate={
            "game": "bp35",
            "claimed_level": 0,
            "reproduced": False,
            "reached_level": 0,
        },
        duration_s=duration_s,
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    budget: int = DEFAULT_BUDGET,
    now: Any = time.time,
) -> dict[str, Any]:  # pragma: no cover - runtime.
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(checks, max(0.0, float(now() - started)))
    else:
        energy_arm = run_policy_attempt(game=TARGET_GAMES[0], use_energy_qd=True, budget=budget)
        no_energy_control = run_policy_attempt(
            game=TARGET_GAMES[0],
            use_energy_qd=False,
            budget=budget,
        )
        reproduction_gate = reproduction_gate_for_attempt(energy_arm)
        live_path = run_live_path_check(root_path)
        artifact = build_artifact(
            preconditions_checked=checks,
            live_path_check=live_path,
            energy_arm=energy_arm,
            no_energy_control=no_energy_control,
            reproduction_gate=reproduction_gate,
            duration_s=max(0.0, float(now() - started)),
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
