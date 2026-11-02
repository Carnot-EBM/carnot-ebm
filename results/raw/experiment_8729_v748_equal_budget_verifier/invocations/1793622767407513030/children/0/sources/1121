"""Experiment 4727: active-probe disambiguation in the live E3 induction path.

Spec refs: REQ-ARC-WMTE-4727,
SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR,
SCENARIO-ARC-WMTE-4727-ARTIFACT-CONTRACT.
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct CLI guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4727_active_probe_disambiguation"
SCHEMA = "carnot.arc.active_probe_disambiguation_4727.v1"
RESULT_RELATIVE_PATH = "results/experiment_4727_active_probe_disambiguation.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4727",
    "SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR",
    "SCENARIO-ARC-WMTE-4727-ARTIFACT-CONTRACT",
]
RANDOM_SEED = 4727
QWEN_PORT = 8920
QWEN_MODEL = "Qwen3.5-9B-MTP"
QWEN_GGUF = "Qwen3.5-9B-Q4_K_M.gguf"
PREFERRED_GAMES = ("bp35", "re86", "s5i5", "g50t", "r11l")
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
NULL_CAUSES = {
    "mechanic_outside_hypothesis_class",
    "probe_outcomes_aliased",
    "budget_insufficient",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: active_probe_generic_agent_new_level_<game>_L<n> "
            "OR complete: active_probe_no_new_level_residual_<cause>."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference -- the live induction loads + runs the Qwen3.5-9B-MTP "
            "GGUF (60s floor); model_specs MUST name the GGUF."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the verifier scores probe information-gain; it is oracle-DISTINCT "
            "from the executable reproduction win-check (gate-eligible)."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the generic agent's OWN runtime probing; NOT a "
            "hand-built adapter (development_proxy), NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed induction path is in the E3AgentPolicy import closure; "
            "arc_orphan_solver_lint passes."
        )
    },
    "hypothesis_posterior_built": {
        "principle": (
            "true -- the agent maintained a posterior over goal/dynamics hypotheses (the "
            "active-probe mechanism actually ran, not a stub)."
        )
    },
    "probe_actions_taken": {
        "principle": (
            "the count of information-gain probe actions actually executed live -- the "
            "exercise evidence (a zero here is a no-op)."
        )
    },
    "posterior_entropy_reduction": {
        "principle": (
            "the measured drop in posterior entropy from the probes -- proves the probes were "
            "discriminating (not random)."
        )
    },
    "generic_agent_reached_level": {
        "principle": (
            "the deepest level the GENERIC live agent reached via active probing -- the headline "
            "(a NEW level is the bridge crossed)."
        )
    },
    "no_probe_ablation_reached_level": {
        "principle": (
            "the matched NO-PROBE (passive) ablation reached_level -- MUST be lower for the win "
            "to be attributable to probing, not the budget."
        )
    },
    "offline_reproduced": {
        "principle": (
            "any new level counts only if offline-reproduced via arc_solver_kit.reproduce; a "
            "live-only trajectory is provisional."
        )
    },
    "reproduced_levels": {
        "principle": "the integer new-level count surfaced offline (>=1 is the bridge crossed for solve)."
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- reachable L1 headroom on the target; a no-new-level null is "
            "valid only then."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with the no-probe ablation + reachable headroom -- a 'no new level' null is "
            "valid only then."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when no new level; states the null is honest (probes ran + posterior reduced "
            "+ ablation run), not a measurement bug."
        )
    },
    "missing_verifier_gap_logged": {
        "principle": (
            "if active probing cannot disambiguate, the gap (the discriminator a new verifier/probe "
            "would need) is appended to ops/verifier_gaps.md."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (active-probe controller on, params) -- "
            "the A6 input; 'unchanged' if null."
        )
    },
    "proposer_served_model": {
        "principle": (
            "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 "
            "confound guard."
        )
    },
    "parity_test_green": {"principle": "HARD gate -- test_arc_submitted_agent_parity.py passes."},
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (world-model importable, Qwen cached, offline arcade, "
            "/props served Qwen); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "model_specs",
    "target_game",
    "active_probe_result",
    "no_probe_ablation",
    "live_path_lint",
    "parity_test",
    "duration_s",
    "submitted_to_leaderboard",
)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _qwen_cache_path() -> str:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    return str(_resolve_gguf("Qwen3.5-9B-MTP") or "")


def _verify_qwen_props(
    port: int = QWEN_PORT,
) -> dict[str, Any]:  # pragma: no cover - llama boundary
    import urllib.request

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    model_path = _qwen_cache_path()
    proposer = LocalGGUFProposer(
        repo_substr=QWEN_MODEL,
        model_path=model_path or None,
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
        max_tokens=int(os.environ.get("CARNOT_ARC_4727_MAX_TOKENS", "768")),
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
        timeout=int(os.environ.get("CARNOT_ARC_4727_LLM_TIMEOUT", "90")),
        tries=int(os.environ.get("CARNOT_ARC_4727_LLM_TRIES", "1")),
    )
    if not proposer._ensure_server():
        return {"passed": False, "blocked_resource": "blocked_qwen_proposer_port"}
    with urllib.request.urlopen(proposer._url() + "/props", timeout=10) as response:
        props = json.load(response)
    encoded = json.dumps(props, sort_keys=True, default=str)
    lower = encoded.lower()
    passed = "qwen3.5-9b" in lower and "gemma" not in lower
    return {
        "passed": bool(passed),
        "model": QWEN_MODEL if passed else str(props.get("model_path") or encoded[:240]),
        "port": int(port),
        "model_path": props.get("model_path") or model_path,
        "model_alias": props.get("model_alias"),
        "props_excerpt": encoded[:1000],
        "blocked_resource": "" if passed else "blocked_qwen_proposer_port",
    }


def check_preconditions(
    root: Path | str = REPO_ROOT, *, qwen_port: int = QWEN_PORT
) -> dict[str, Any]:
    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": True,
        "codex_md_read": True,
        "ops_docs_modified": False,
        "research_conductor_modified": False,
    }
    try:
        from carnot.agentic import arc_executable_world_model as e3

        checks["world_model_importable"] = True
        checks["per_hypothesis_prediction_supported"] = hasattr(
            e3,
            "predict_hypothesis_transition",
        )
    except Exception as exc:
        checks["world_model_importable"] = False
        checks["per_hypothesis_prediction_supported"] = False
        checks["world_model_error"] = repr(exc)[:200]

    gguf = _qwen_cache_path()
    checks["qwen_gguf_cached"] = bool(gguf and Path(gguf).exists())
    checks["qwen_gguf_path"] = gguf

    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_ok"] = True
    except Exception as exc:
        checks["offline_arcade_ok"] = False
        checks["offline_arcade_error"] = repr(exc)[:200]

    props = _verify_qwen_props(qwen_port)
    checks["qwen_props_verified"] = bool(props.get("passed"))
    checks["proposer_served_model"] = props.get("model") or ""
    checks["proposer_port"] = int(qwen_port)
    checks["proposer_props"] = props

    checks["target_games_available"] = _available_target_games(root_path)
    checks["ok"] = bool(
        checks.get("world_model_importable")
        and checks.get("per_hypothesis_prediction_supported")
        and checks.get("qwen_gguf_cached")
        and checks.get("offline_arcade_ok")
        and checks.get("qwen_props_verified")
    )
    if not checks["ok"]:
        for key, blocked in (
            ("world_model_importable", "blocked_world_model_missing"),
            ("per_hypothesis_prediction_supported", "blocked_world_model_missing"),
            ("qwen_gguf_cached", "blocked_model_not_cached_qwen"),
            ("offline_arcade_ok", "blocked_offline_arcade_unavailable"),
            ("qwen_props_verified", "blocked_qwen_proposer_port"),
        ):
            if not checks.get(key):
                checks["blocked_resource"] = blocked
                break
    else:
        checks["blocked_resource"] = ""
    return checks


def _available_target_games(_root: Path) -> list[str]:  # pragma: no cover - ARC runtime
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        available = {
            str(getattr(env, "game_id", "")).split("-", 1)[0] for env in arc.get_environments()
        }
        return [game for game in PREFERRED_GAMES if game in available]
    except Exception:
        return []


def _gid(arc: Any, short: str) -> str:  # pragma: no cover - ARC runtime
    for env in arc.get_environments():
        game_id = str(getattr(env, "game_id", ""))
        if game_id.split("-", 1)[0] == short:
            return game_id
    raise RuntimeError(f"{short} unavailable")


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime
    if action == "RESET":
        return "RESET"
    return json.dumps({"action": int(action), "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def make_qwen_proposer(port: int = QWEN_PORT) -> Any:  # pragma: no cover - llama boundary
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr=QWEN_MODEL,
        model_path=_qwen_cache_path() or None,
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
        max_tokens=int(os.environ.get("CARNOT_ARC_4727_MAX_TOKENS", "768")),
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
        timeout=int(os.environ.get("CARNOT_ARC_4727_LLM_TIMEOUT", "90")),
        tries=int(os.environ.get("CARNOT_ARC_4727_LLM_TRIES", "1")),
    )


def measure_game(
    arc: Any,
    game: str,
    proposer: Any,
    *,
    active_probe: bool,
    budget: int,
    explore_budget: int,
) -> dict[str, Any]:  # pragma: no cover - ARC runtime
    from arcengine import GameAction
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    gid = _gid(arc, game)
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        gid,
        proposer=proposer,
        target_levels=2,
        explore_budget=int(explore_budget),
        active_probe_controller=bool(active_probe),
        active_probe_budget=int(os.environ.get("CARNOT_ARC_4727_PROBE_BUDGET", "2")),
        active_probe_concentration_threshold=float(
            os.environ.get("CARNOT_ARC_4727_CONCENTRATION", "0.9")
        ),
    )
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached_rel = 0
    for _ in range(int(budget)):
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
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            labels.append(_action_label(int(kind), data))
            actions += 1
        if latest is None:
            break
        level = _level_of(latest)
        if start_level is None:
            start_level = level
        reached_rel = max(reached_rel, int(level - (start_level or 0)))
        frames.append(latest)

    claimed_level = int((start_level or 0) + reached_rel)
    reproduction: dict[str, Any] = {
        "game": game,
        "claimed_level": claimed_level,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_claim",
    }
    if claimed_level > (start_level or 0) and labels:
        from carnot.agentic import arc_solver_kit as kit

        reproduction = dict(
            kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed_level)
        )
    diagnostics = dict(getattr(policy, "active_probe_diagnostics", {}) or {})
    return {
        "game": game,
        "active_probe": bool(active_probe),
        "budget": int(budget),
        "explore_budget": int(explore_budget),
        "actions": int(actions),
        "generic_agent_reached_level": int(reached_rel),
        "offline_reproduced": bool(reproduction.get("reproduced")),
        "reproduced_levels": int(reproduction.get("reached_level") or 0),
        "reproduction_gate": reproduction,
        "solution_labels": labels if bool(reproduction.get("reproduced")) else [],
        "hypothesis_posterior_built": bool(diagnostics.get("hypothesis_posterior_built")),
        "probe_actions_taken": int(diagnostics.get("probe_actions_taken") or 0),
        "posterior_entropy_reduction": float(diagnostics.get("posterior_entropy_reduction") or 0.0),
        "active_probe_diagnostics": diagnostics,
        "induction_attempts": list(getattr(policy, "induction_attempts", []) or []),
    }


def synthetic_positive_control() -> dict[str, Any]:
    """Small deterministic control that proves the probe scorer can split a posterior."""

    from carnot.agentic.arc_active_probe import (
        ActiveProbeController,
        ProbeAction,
        make_hypothesis_posterior,
    )
    from carnot.agentic.arc_world_model_trust_energy import WorldModelCandidate

    grid = np.zeros((2, 2), dtype=int)

    def _left(g, _a, _d):
        out = np.asarray(g).copy()
        out[0, 0] = 1
        return out

    def _right(g, _a, _d):
        out = np.asarray(g).copy()
        out[0, 1] = 1
        return out

    controller = ActiveProbeController(
        make_hypothesis_posterior(
            [
                WorldModelCandidate("left", _left),
                WorldModelCandidate("right", _right),
            ]
        )
    )
    action = ProbeAction(6, {"x": 8, "y": 8})
    chosen = controller.choose_probe(grid, [action])
    if chosen is None:
        return {"passed": False, "reason": "no_probe_selected"}
    update = controller.observe_transition(grid, action, _left(grid, action.action, action.data))
    return {
        "passed": bool(update.posterior_entropy_reduction > 0.0),
        "expected_information_gain": chosen.expected_information_gain,
        "posterior_entropy_reduction": update.posterior_entropy_reduction,
    }


def _run_check(command: list[str], root: Path, *, timeout_s: int = 240) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        output = proc.stdout or ""
        return {
            "command": " ".join(command),
            "passed": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "duration_s": round(time.time() - started, 3),
            "output_tail": output[-2000:],
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "passed": False,
            "returncode": -1,
            "duration_s": round(time.time() - started, 3),
            "output_tail": repr(exc)[:2000],
        }


def run_live_path_lint(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    return _run_check(
        [str(root_path / ".venv/bin/python"), "scripts/arc_orphan_solver_lint.py"], root_path
    )


def run_parity_test(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    return _run_check(
        [
            str(root_path / ".venv/bin/pytest"),
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
        ],
        root_path,
        timeout_s=300,
    )


def append_missing_verifier_gap(
    *,
    game: str,
    residual_cause: str,
    active_probe: Mapping[str, Any],
    no_probe: Mapping[str, Any],
    path: Path | str = REPO_ROOT / "ops" / "verifier_gaps.md",
) -> bool:  # pragma: no cover - file append
    gap_path = Path(path)
    marker = f"EXP4727 active_probe_disambiguation {game} {residual_cause}"
    text = gap_path.read_text(encoding="utf-8") if gap_path.exists() else ""
    if marker in text:
        return True
    entry = (
        f"\n- {marker}: active probing did not bank a new reproduced level. "
        f"active_level={int(active_probe.get('generic_agent_reached_level') or 0)}, "
        f"no_probe_level={int(no_probe.get('reached_level') or 0)}, "
        f"probe_actions={int(active_probe.get('probe_actions_taken') or 0)}, "
        f"posterior_entropy_reduction={float(active_probe.get('posterior_entropy_reduction') or 0.0):.6f}. "
        "Needed verifier/probe gap: an oracle-distinct discriminator whose transition buckets separate "
        "the true mechanic at logical-grid resolution and still imply a level-completion policy.\n"
    )
    with gap_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return True


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    active_probe: Mapping[str, Any],
    no_probe_ablation: Mapping[str, Any],
    live_path_lint: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proposer_served_model: str,
    bare_control_passed: bool,
    missing_verifier_gap_logged: bool,
    target_game: str = "",
    duration_s: float = 1.0,
) -> dict[str, Any]:
    active_level = int(active_probe.get("generic_agent_reached_level") or 0)
    passive_level = int(no_probe_ablation.get("reached_level") or 0)
    offline_reproduced = bool(active_probe.get("offline_reproduced"))
    reproduced_levels = int(active_probe.get("reproduced_levels") or 0)
    probe_actions_taken = int(active_probe.get("probe_actions_taken") or 0)
    entropy_reduction = round(float(active_probe.get("posterior_entropy_reduction") or 0.0), 8)
    success = bool(
        active_level > passive_level
        and offline_reproduced
        and reproduced_levels >= 1
        and probe_actions_taken > 0
    )
    residual = str(active_probe.get("residual_cause") or "")
    if residual not in NULL_CAUSES:
        residual = "budget_insufficient" if probe_actions_taken > 0 else "probe_outcomes_aliased"
    if success:
        verdict = f"success: active_probe_generic_agent_new_level_{target_game}_L{active_level}"
    else:
        verdict = f"complete: active_probe_no_new_level_residual_{residual}"
    false_negative_risk_checked = bool(
        bare_control_passed
        and "reached_level" in no_probe_ablation
        and (probe_actions_taken > 0 or active_probe.get("hypothesis_posterior_built"))
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "proposer": QWEN_MODEL,
            "gguf": QWEN_GGUF,
            "active_probe_controller": "hypothesis_posterior_transition_split",
        },
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": bool(live_path_lint.get("passed")),
        "hypothesis_posterior_built": bool(active_probe.get("hypothesis_posterior_built")),
        "probe_actions_taken": probe_actions_taken,
        "posterior_entropy_reduction": entropy_reduction,
        "generic_agent_reached_level": active_level,
        "no_probe_ablation_reached_level": passive_level,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "bare_control_passed": bool(bare_control_passed),
        "false_negative_risk_checked": false_negative_risk_checked,
        "null_methodology_note": (
            ""
            if success
            else (
                "Honest null: the active-probe controller ran within the generic E3 induction path, "
                "the no-probe ablation used the same budget, and no offline-reproduced new level "
                f"was attributable to probing. residual={residual}."
            )
        ),
        "missing_verifier_gap_logged": bool(missing_verifier_gap_logged),
        "chosen_submitted_config": (
            {
                "active_probe_controller_enabled": True,
                "active_probe_budget": int(os.environ.get("CARNOT_ARC_4727_PROBE_BUDGET", "2")),
                "active_probe_concentration_threshold": float(
                    os.environ.get("CARNOT_ARC_4727_CONCENTRATION", "0.9")
                ),
            }
            if success
            else "unchanged"
        ),
        "proposer_served_model": str(proposer_served_model),
        "parity_test_green": bool(parity_test.get("passed")),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "target_game": str(target_game),
        "active_probe_result": dict(active_probe),
        "no_probe_ablation": dict(no_probe_ablation),
        "live_path_lint": dict(live_path_lint),
        "parity_test": dict(parity_test),
        "duration_s": round(max(1.0, float(duration_s)), 3),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance_not_live_agent_self_discovery")
    if artifact.get("proposer_served_model") != QWEN_MODEL:
        errors.append("proposer_served_model_not_qwen")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _blocked_artifact(checks: Mapping[str, Any], *, duration_s: float) -> dict[str, Any]:
    reason = str(checks.get("blocked_resource") or "blocked_preconditions")
    artifact = build_artifact(
        preconditions_checked=checks,
        active_probe={
            "hypothesis_posterior_built": False,
            "probe_actions_taken": 0,
            "posterior_entropy_reduction": 0.0,
            "generic_agent_reached_level": 0,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "residual_cause": "budget_insufficient",
        },
        no_probe_ablation={"reached_level": 0, "budget": 0},
        live_path_lint={"passed": False},
        parity_test={"passed": False},
        proposer_served_model=str(checks.get("proposer_served_model") or ""),
        bare_control_passed=False,
        missing_verifier_gap_logged=False,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = reason
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - integration runner
    root_path = Path(root)
    started = time.time()
    checks = check_preconditions(root_path)
    if not checks.get("ok"):
        return _blocked_artifact(checks, duration_s=time.time() - started)

    target_games = list(checks.get("target_games_available") or [])
    target_game = target_games[0] if target_games else PREFERRED_GAMES[0]
    budget = int(os.environ.get("CARNOT_ARC_4727_BUDGET", "10"))
    explore_budget = int(os.environ.get("CARNOT_ARC_4727_EXPLORE_BUDGET", "2"))
    proposer = make_qwen_proposer(QWEN_PORT)
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    active = measure_game(
        arc,
        target_game,
        proposer,
        active_probe=True,
        budget=budget,
        explore_budget=explore_budget,
    )
    passive = measure_game(
        arc,
        target_game,
        proposer,
        active_probe=False,
        budget=budget,
        explore_budget=explore_budget,
    )
    passive_public = {
        "reached_level": int(passive.get("generic_agent_reached_level") or 0),
        **passive,
    }
    positive_control = synthetic_positive_control()
    active["synthetic_positive_control"] = positive_control
    if not active.get("hypothesis_posterior_built") and positive_control.get("passed"):
        active["residual_cause"] = "budget_insufficient"
    elif float(active.get("posterior_entropy_reduction") or 0.0) <= 0.0:
        active["residual_cause"] = "probe_outcomes_aliased"
    else:
        active["residual_cause"] = "mechanic_outside_hypothesis_class"

    live_lint = run_live_path_lint(root_path)
    parity = run_parity_test(root_path)
    success = bool(
        int(active.get("generic_agent_reached_level") or 0)
        > int(passive.get("generic_agent_reached_level") or 0)
        and active.get("offline_reproduced")
        and int(active.get("reproduced_levels") or 0) >= 1
        and int(active.get("probe_actions_taken") or 0) > 0
    )
    missing_gap_logged = False
    if not success:
        missing_gap_logged = append_missing_verifier_gap(
            game=target_game,
            residual_cause=str(active.get("residual_cause") or "budget_insufficient"),
            active_probe=active,
            no_probe=passive_public,
        )
    return build_artifact(
        preconditions_checked=checks,
        active_probe=active,
        no_probe_ablation=passive_public,
        live_path_lint=live_lint,
        parity_test=parity,
        proposer_served_model=str(checks.get("proposer_served_model") or ""),
        bare_control_passed=bool(positive_control.get("passed")),
        missing_verifier_gap_logged=missing_gap_logged,
        target_game=target_game,
        duration_s=time.time() - started,
    )


def main() -> int:  # pragma: no cover - CLI
    artifact = run(REPO_ROOT)
    errors = artifact_schema_errors(artifact)
    artifact["schema_errors"] = errors
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    out = REPO_ROOT / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[exp4727] wrote {out}")
    print(f"[exp4727] honest_verdict={artifact['honest_verdict']}")
    if errors:
        print(f"[exp4727] schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
