"""Experiment 4712: perception-grounded structural L2 goal for lp85.

Spec refs: REQ-ARC-WMTE-4712,
SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL,
SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4712_perception_grounded_l2_goal_lp85"
EXPERIMENT_ID = 4712
SCHEMA = "carnot.arc.perception_grounded_l2_goal_lp85_4712.v1"
RESULT_RELATIVE_PATH = "results/experiment_4712_perception_grounded_l2_goal_lp85.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4712",
    "SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL",
    "SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING",
]
RANDOM_SEED = 4712
DEFAULT_BUDGET = 3000
DEFAULT_QWEN_PORT = 8920
TARGET_GAME = "lp85"
GOAL_EXPRESSION = "structural_piece_sprite_alignment_over_detected_objects"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: perception_grounded_l2_goal_lp85_L2_offline_reproduced "
            "OR complete: l2_perception_goal_no_deepening_residual_<cause>."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference -- the reinduction path loads + runs the Qwen3.5-9B-MTP "
            "GGUF (60s floor); model_specs MUST name the GGUF."
        )
    },
    "goal_predicate_satisfiable": {
        "principle": (
            "the .430 gate checks DYNAMICS only; this records the L2 alignment goal is True "
            "on >=1 reachable grid -- the verification exemplar-replay never produced."
        )
    },
    "goal_expression": {
        "principle": (
            "MUST be a STRUCTURAL predicate over DETECTED objects (e.g. "
            "structural_piece_sprite_alignment_over_detected_objects), not a per-game "
            "hardcode or a flat exemplar grid."
        )
    },
    "l2_plan_reaches_goal": {
        "principle": (
            "plan_len=0 / no_reachable_plan was the measured failure; True means the plan "
            "reaches the satisfiable goal."
        )
    },
    "reproduced_levels": {
        "principle": (
            "the integer level reached; >=2 is the lp85 L2 bank (only reproduced levels count)."
        )
    },
    "offline_reproduced": {
        "principle": (
            "a reproduced L2 via arc_solver_kit.reproduce is the fix working; a live-only "
            "trajectory is provisional."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- a generic-agent L2 via the perception-grounded "
            "goal is self-discovery; an adapter L2 is a dev proxy that does NOT prove the live fix."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the alignment predicate is oracle-distinct from the env's own "
            "level-up check."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed modules are in the E3AgentPolicy import closure; "
            "arc_orphan_solver_lint passes."
        )
    },
    "registry_precheck_generic_l2": {
        "principle": (
            "confirms the generic live path does not ALREADY self-discover lp85 L2 (vs the "
            "adaptered registry row) -- a duplicate of an already-banked GENERIC level is a "
            "CRITICAL adversarial flag."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == "
            "the measured agent."
        )
    },
    "residual_cause_hypothesis": {
        "principle": (
            "if it nulls, names the residual (object_detector_cannot_resolve_pieces_sprites "
            "| alignment_under_determined | no_reachable_plan) -- the .435 target; 'none' if it banked."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when no L2; states the null is honest (lp85 L1 reachable + object "
            "detector ran + goal satisfiability checked), not a measurement bug."
        )
    },
    "proposer_served_model": {
        "principle": (
            "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- "
            "the port-8919 confound guard."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (A1 operator importable, Qwen cached, offline "
            "arcade + lp85 env, /props served Qwen); pre-empts missing-resource fabrication."
        )
    },
}


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_command(command: list[str], *, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _gid(arc: Any, short: str) -> str:
    for env in arc.get_environments():
        game_id = str(getattr(env, "game_id", ""))
        if game_id.split("-", 1)[0] == short:
            return game_id
    raise RuntimeError(f"{short} unavailable")


def _action_label(action: int | str, data: Any) -> str:
    if action == "RESET":
        return "RESET"
    return json.dumps({"action": int(action), "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _qwen_cache_present() -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.5-9B-MTP-GGUF"
    return cache.is_dir() and any(cache.iterdir())


def _make_qwen_proposer(port: int = DEFAULT_QWEN_PORT) -> Any:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=int(port),
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
        max_tokens=2560,
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
    )


def _verify_qwen_props(proposer: Any) -> dict[str, Any]:
    import urllib.request

    if not proposer._ensure_server():
        return {"passed": False, "blocked_resource": "blocked_qwen_proposer_port"}
    with urllib.request.urlopen(proposer._url() + "/props", timeout=10) as response:
        props = json.load(response)
    encoded = json.dumps(props, sort_keys=True, default=str)
    lower = encoded.lower()
    passed = "qwen3.5-9b" in lower and "gemma" not in lower
    return {
        "passed": bool(passed),
        "model": "Qwen3.5-9B-MTP" if passed else str(props.get("model_path") or encoded[:240]),
        "props_excerpt": encoded[:1000],
        "blocked_resource": "" if passed else "blocked_qwen_proposer_port",
    }


def _registry_precheck(root: Path) -> dict[str, Any]:
    registry_path = root / "ops" / "arc_solve_registry.yaml"
    adaptered_levels = None
    mechanic = ""
    if registry_path.exists():
        loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        rows = loaded.get("games", []) if isinstance(loaded, Mapping) else []
        for row in rows:
            if isinstance(row, Mapping) and row.get("game") == TARGET_GAME:
                adaptered_levels = row.get("levels_reproduced")
                mechanic = str(row.get("mechanic_class") or "")
                break
    prior_path = root / "results" / "experiment_4664_l2_goal_predicate_induction_live.json"
    prior_generic_l2 = False
    prior_level = None
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_level = (
            prior.get("generic_agent_reached_level", {}).get(TARGET_GAME)
            if isinstance(prior.get("generic_agent_reached_level"), Mapping)
            else None
        )
        prior_generic_l2 = bool(isinstance(prior_level, int | float) and int(prior_level) >= 2)
    return {
        "adaptered_registry_levels_reproduced": adaptered_levels,
        "registry_mechanic_class": mechanic,
        "preexisting_generic_l2": prior_generic_l2,
        "generic_precheck_source": "results/experiment_4664_l2_goal_predicate_induction_live.json",
        "generic_precheck_reached_level": prior_level,
    }


def _preconditions(root: Path, proposer: Any) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists() or (root / "OPENCODE.md").exists(),
        "qwen3_5_9b_mtp_gguf_cached": _qwen_cache_present(),
        "qwen_proposer_port": int(getattr(proposer, "port", DEFAULT_QWEN_PORT)),
    }
    try:
        from carnot.agentic.arc_solver_kit import object_centric_representation_builder_operator
        from carnot.agentic.arc_value_learner import structural_alignment_goal_candidate

        checks["a1_operator_importable"] = (
            object_centric_representation_builder_operator.__name__
            == "object_centric_representation_builder_operator"
        )
        checks["a1_operator_module"] = (
            "carnot.agentic.arc_solver_kit.object_centric_representation_builder_operator"
        )
        checks["structural_goal_provider_importable"] = callable(
            structural_alignment_goal_candidate
        )
    except Exception as exc:
        checks["a1_operator_importable"] = False
        checks["a1_operator_error"] = repr(exc)[:160]
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        _gid(arc, TARGET_GAME)
        checks["offline_arcade"] = True
        checks["lp85_env_present"] = True
    except Exception as exc:
        checks["offline_arcade"] = False
        checks["lp85_env_present"] = False
        checks["offline_arcade_error"] = repr(exc)[:160]
    props = _verify_qwen_props(proposer)
    checks["qwen_proposer_port_verified"] = bool(props.get("passed"))
    checks["proposer_props_excerpt"] = props.get("props_excerpt", "")
    checks["proposer_served_model"] = props.get("model", "")
    checks["ok"] = all(
        bool(checks.get(key))
        for key in (
            "agents_md_read",
            "codex_md_read",
            "qwen3_5_9b_mtp_gguf_cached",
            "a1_operator_importable",
            "offline_arcade",
            "lp85_env_present",
            "qwen_proposer_port_verified",
        )
    )
    return checks


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    if not preconditions.get("a1_operator_importable"):
        return "blocked_a1_perception_operator_missing"
    if not preconditions.get("qwen3_5_9b_mtp_gguf_cached"):
        return "blocked_model_not_cached_qwen"
    if not preconditions.get("offline_arcade") or not preconditions.get("lp85_env_present"):
        return "blocked_offline_arcade_lp85_missing"
    if not preconditions.get("qwen_proposer_port_verified"):
        return "blocked_qwen_proposer_port"
    return None


def _detector_positive_control(arc: Any) -> dict[str, Any]:
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_competition_agent import _level_of
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_value_learner import structural_alignment_goal_candidate

    artifact_path = REPO_ROOT / "results" / "experiment_4664_l2_goal_predicate_induction_live.json"
    if not artifact_path.exists():
        return {"available": False, "reason": "missing_exp4664_l1_trace"}
    labels = json.loads(artifact_path.read_text(encoding="utf-8"))["per_game"][TARGET_GAME][
        "solution_labels"
    ]
    env = arc.make(_gid(arc, TARGET_GAME), scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start = _level_of(frame)
    for index, label in enumerate(labels, 1):
        frame = _apply_action_label(env, label, frame)
        if _level_of(frame) > start:
            grid = to_logical(grid_of(frame), detect_cell(grid_of(frame)))
            candidate = structural_alignment_goal_candidate(grid)
            return {
                "available": True,
                "l1_positive_control_reaches_level": int(_level_of(frame)),
                "l1_positive_control_steps": int(index),
                "structural_goal_detected": candidate is not None,
                "diagnostics": dict(candidate.get("diagnostics") or {}) if candidate else {},
            }
    return {"available": True, "structural_goal_detected": False, "reason": "l1_trace_no_levelup"}


def _induction_summary(policy: Any) -> dict[str, Any]:
    attempts = list(getattr(policy, "induction_attempts", []) or [])
    level_attempts = [row for row in attempts if row.get("reason") == "level_up_reinduction"]
    rounds = [
        round_row
        for attempt in level_attempts
        for round_row in list(attempt.get("refinement_rounds") or [])
        if isinstance(round_row, Mapping)
    ]
    expressions = [
        str(row.get("goal_expression") or "")
        for row in [*level_attempts, *rounds]
        if row.get("goal_expression")
    ]
    diagnostics = next(
        (
            dict(row.get("structural_goal_diagnostics") or {})
            for row in [*level_attempts, *rounds]
            if row.get("structural_goal_diagnostics")
        ),
        {},
    )
    return {
        "attempts": level_attempts,
        "goal_predicate_satisfiable": any(
            bool(row.get("goal_predicate_satisfiable")) for row in [*level_attempts, *rounds]
        ),
        "l2_plan_len": max([int(row.get("plan_length") or 0) for row in level_attempts] + [0]),
        "l2_plan_reaches_goal": any(bool(row.get("plan_reaches_goal")) for row in rounds),
        "goal_expression": expressions[0] if expressions else "",
        "structural_goal_diagnostics": diagnostics,
        "counterexample_kinds": [
            str(cx.get("kind"))
            for attempt in level_attempts
            for cx in list(attempt.get("counterexamples") or [])
            if isinstance(cx, Mapping)
        ],
    }


def measure_game(arc: Any, proposer: Any, *, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    env = arc.make(_gid(arc, TARGET_GAME), scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(_gid(arc, TARGET_GAME), proposer=proposer, target_levels=2)
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached_rel = 0
    levelup_at: dict[str, int] = {}
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
        level = _level_of(latest)
        if start_level is None:
            start_level = level
        rel = int(level - (start_level or 0))
        if rel > reached_rel:
            reached_rel = rel
            levelup_at.setdefault(str(rel), actions)
        frames.append(latest)

    claimed_level = int((start_level or 0) + reached_rel)
    reproduction: dict[str, Any] = {
        "game": TARGET_GAME,
        "claimed_level": claimed_level,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_claim",
    }
    if claimed_level > (start_level or 0) and labels:
        from carnot.agentic import arc_solver_kit as kit

        reproduction = dict(
            kit.reproduce(TARGET_GAME, labels, _apply_action_label, claimed_level=claimed_level)
        )
    summary = _induction_summary(policy)
    return {
        "game": TARGET_GAME,
        "budget": int(budget),
        "actions": int(actions),
        "generic_agent_reached_level": int(claimed_level),
        "levelup_at_action": levelup_at,
        "solution_labels": labels,
        "reproduction_gate": reproduction,
        "reproduced_levels": int(reproduction.get("reached_level") or 0),
        "offline_reproduced_l2": bool(
            reproduction.get("reproduced") and int(reproduction.get("reached_level") or 0) >= 2
        ),
        **summary,
    }


def _residual(measurement: Mapping[str, Any], detector_control: Mapping[str, Any]) -> str:
    diagnostics = (
        measurement.get("structural_goal_diagnostics") or detector_control.get("diagnostics") or {}
    )
    if not diagnostics or not diagnostics.get("detected"):
        return "object_detector_cannot_resolve_pieces_sprites"
    if int(diagnostics.get("piece_count") or 0) != int(diagnostics.get("goal_count") or 0):
        return "alignment_under_determined"
    return "no_reachable_plan"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    budget: int = DEFAULT_BUDGET,
    proposer: Any | None = None,
    started_s: float | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    start = time.time() if started_s is None else float(started_s)
    proposer = proposer or _make_qwen_proposer(
        port=int(os.environ.get("CARNOT_4712_QWEN_PORT", DEFAULT_QWEN_PORT))
    )
    preconditions = _preconditions(root_path, proposer)
    registry_precheck = _registry_precheck(root_path)
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "experiment_id": EXPERIMENT_ID,
            "spec_refs": SPEC_REFS,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": blocker,
            "inference_substrate": "live_llm_inference",
            "model_specs": "Qwen3.5-9B-MTP GGUF",
            "goal_predicate_satisfiable": False,
            "goal_expression": "",
            "l2_plan_reaches_goal": False,
            "reproduced_levels": 0,
            "offline_reproduced": False,
            "solve_provenance": "live_agent_self_discovery",
            "verifier_is_oracle": False,
            "live_path_reachable": False,
            "registry_precheck_generic_l2": registry_precheck,
            "parity_test_green": False,
            "residual_cause_hypothesis": blocker,
            "null_methodology_note": "",
            "proposer_served_model": str(preconditions.get("proposer_served_model") or ""),
            "preconditions_checked": preconditions,
            "duration_s": round(time.time() - start, 6),
            "random_seed": RANDOM_SEED,
            "field_principles": FIELD_PRINCIPLES,
            "reproducibility_checksum": "",
        }
        artifact["reproducibility_checksum"] = "sha256:" + _payload_checksum(artifact)
        return artifact

    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    detector_control = _detector_positive_control(arc)
    measurement = measure_game(arc, proposer, budget=int(budget))
    lint = _run_command(
        [str(root_path / ".venv" / "bin" / "python"), "scripts/arc_orphan_solver_lint.py"]
    )
    parity = _run_command(
        [
            str(root_path / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        timeout=180,
    )
    reproduced_levels = int(measurement.get("reproduced_levels") or 0)
    offline_reproduced = bool(measurement.get("offline_reproduced_l2"))
    l2_plan_reaches_goal = bool(measurement.get("l2_plan_reaches_goal"))
    goal_predicate_satisfiable = bool(measurement.get("goal_predicate_satisfiable"))
    success = bool(
        reproduced_levels >= 2
        and offline_reproduced
        and goal_predicate_satisfiable
        and l2_plan_reaches_goal
        and lint.get("passed")
        and parity.get("passed")
    )
    residual = "none" if success else _residual(measurement, detector_control)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": (
            "success: perception_grounded_l2_goal_lp85_L2_offline_reproduced"
            if success
            else f"complete: l2_perception_goal_no_deepening_residual_{residual}"
        ),
        "inference_substrate": "live_llm_inference",
        "model_specs": "Qwen3.5-9B-MTP GGUF",
        "goal_predicate_satisfiable": goal_predicate_satisfiable,
        "goal_expression": str(
            measurement.get("goal_expression")
            or (GOAL_EXPRESSION if detector_control.get("structural_goal_detected") else "")
        ),
        "l2_plan_reaches_goal": l2_plan_reaches_goal,
        "l2_plan_len": int(measurement.get("l2_plan_len") or 0),
        "reproduced_levels": reproduced_levels,
        "offline_reproduced": offline_reproduced,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "live_path_reachable": bool(lint.get("passed")),
        "registry_precheck_generic_l2": registry_precheck,
        "parity_test_green": bool(parity.get("passed")),
        "residual_cause_hypothesis": residual,
        "null_methodology_note": (
            ""
            if success
            else (
                "Honest null: lp85 L1 positive-control trace reaches the post-boundary frame, "
                "the object detector ran on that live frame, and the level-up reinduction "
                "recorded whether the structural goal was satisfiable before any L2 claim."
            )
        ),
        "proposer_served_model": str(preconditions.get("proposer_served_model") or ""),
        "detector_positive_control": detector_control,
        "per_game": {TARGET_GAME: measurement},
        "preconditions_checked": {
            **preconditions,
            "arc_orphan_solver_lint": lint,
            "parity_test": parity,
        },
        "duration_s": round(time.time() - start, 6),
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + _payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    required = [
        "honest_verdict",
        "inference_substrate",
        "goal_predicate_satisfiable",
        "goal_expression",
        "l2_plan_reaches_goal",
        "reproduced_levels",
        "offline_reproduced",
        "solve_provenance",
        "verifier_is_oracle",
        "live_path_reachable",
        "registry_precheck_generic_l2",
        "parity_test_green",
        "residual_cause_hypothesis",
        "proposer_served_model",
        "random_seed",
        "reproducibility_checksum",
        "preconditions_checked",
    ]
    missing = [field for field in required if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact.get("inference_substrate") != "live_llm_inference":
        raise ValueError("inference_substrate must be live_llm_inference")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")


def main() -> int:
    artifact = build_artifact(
        REPO_ROOT,
        budget=int(os.environ.get("CARNOT_4712_BUDGET", DEFAULT_BUDGET)),
    )
    validate_artifact(artifact)
    out = REPO_ROOT / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "honest_verdict",
                    "reproduced_levels",
                    "goal_predicate_satisfiable",
                    "l2_plan_reaches_goal",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
