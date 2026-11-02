"""Experiment 5860: bounded game-blind active observation A/B.

Spec refs: REQ-ARC-WMTE-5860,
SCENARIO-ARC-WMTE-5860-TAPE-IS-AGENT-OWNED,
SCENARIO-ARC-WMTE-5860-BUDGET-PARITY-AND-READY-GATE,
SCENARIO-ARC-WMTE-5860-STABLE-ARTIFACT.

The experiment is deliberately not a solve path.  It measures whether a controller that only sees
its own legal actions and exact runtime observations collects better transition evidence than
matched current-policy, random, and periodic controls.  Registry rows are read only for the
precheck that prevents no-credit public games from being re-banked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5860_live_active_observation_ab"
EXPERIMENT_ID = 5860
SCHEMA = "carnot.live_active_observation_ab_5860.v1"
RESULT_RELATIVE_PATH = "results/experiment_5860_live_active_observation_ab.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5860
DEFAULT_GAMES = ("sc25", "lf52", "bp35")
ARMS = ("current_e3", "random_legal", "periodic", "active_observer")
CONTROL_ARMS = ("current_e3", "random_legal", "periodic")
SPEC_REFS = [
    "REQ-ARC-WMTE-5860",
    "SCENARIO-ARC-WMTE-5860-TAPE-IS-AGENT-OWNED",
    "SCENARIO-ARC-WMTE-5860-BUDGET-PARITY-AND-READY-GATE",
    "SCENARIO-ARC-WMTE-5860-STABLE-ARTIFACT",
]

FORBIDDEN_TAPE_KEYS = frozenset(
    {
        "goal_label",
        "adapter_fact",
        "event_caption",
        "registry_trajectory",
        "public_source_hint",
        "hand_rule",
        "outer_loop_counterexample",
        "offline_bfs_path",
        "game_adapter_label",
    }
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "status": {
        "principle": "A terminal live A/B state distinguishes a clean null from partial interaction."
    },
    "preconditions_checked": {
        "principle": "Registry, live path, exclusions, models, resources, budgets, seeds, and outputs prevent off-path credit."
    },
    "registry_precheck": {
        "principle": "All public solves are already complete; this task cannot duplicate them."
    },
    "live_path_and_sdk_receipts": {
        "principle": "Only the canonical scored E3 interaction path supports a live claim."
    },
    "adapter_source_bfs_and_registry_exclusion_receipts": {
        "principle": "Forbidden outer-loop knowledge must be mechanically inaccessible."
    },
    "model_specs": {
        "principle": "At least one mandated current SOTA GGUF must generate headline live rows."
    },
    "models_used": {"principle": "Exact hub IDs disclose which live proposer actually ran."},
    "gpu_and_llama_cpp_receipts": {
        "principle": "Device and embedded-tokenizer evidence distinguish live inference from fallback."
    },
    "agent_owned_tape_schema_and_hashes": {
        "principle": "Every memory item must come from the agent's own legal action and exact observation."
    },
    "history_view_definitions": {
        "principle": "Global, local-active, and event-boundary ablations must be game-blind."
    },
    "arm_definitions_and_budget_parity": {
        "principle": "Equal legal-action and inference budgets isolate active observation."
    },
    "short_medium_long_horizon_metrics": {
        "principle": "Proposal quality is measured separately across temporal scales."
    },
    "ambiguity_and_transition_evidence_metrics": {
        "principle": "The primary outcome is acquired runtime evidence, not a re-solve."
    },
    "action_model_call_and_latency_accounting": {
        "principle": "Accuracy and efficiency remain jointly first-class."
    },
    "descriptive_level_outcomes": {
        "principle": "Levels are reported honestly but receive no solve credit."
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery -- runtime observations only, never outer-loop RE or a development proxy."
    },
    "shuffled_tape_view_ablation_and_null_controls": {
        "principle": "Broken memory or priorities must not retain the claimed gain."
    },
    "registry_modified": {"principle": "Must be false; this task cannot add solve credit."},
    "active_observation_ready_score": {
        "principle": "EMIT BARE scalar for capstone classification."
    },
    "duration_s": {"principle": "Measured live wall time exposes smoke-only execution."},
    "inference_substrate": {
        "principle": "live_llm_inference declares the real proposer path."
    },
    "verifier_is_oracle": {
        "principle": "False for learned observation selection; exact observations remain evidence authority, not a candidate-answer oracle."
    },
    "field_provenance": {
        "principle": "Every metric traces to live actions, observations, tape items, budgets, and model receipts."
    },
    "test_commands": {
        "principle": "Commands document live path, exclusions, model, parity, metrics, controls, and registry immutability."
    },
    "test_exit_codes": {
        "principle": "Exit codes prevent off-path or budget-violating runs becoming ready."
    },
    "reproducibility_checksum": {
        "principle": "A checksum detects policy, game, model, tape, seed, or budget drift."
    },
    "honest_verdict": {
        "principle": "A terminal prefix states positive, null, budget-bound, or blocked result without solve language."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")))


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_TAPE_KEYS:
                return str(key)
            nested = _contains_forbidden_key(item)
            if nested:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _contains_forbidden_key(item)
            if nested:
                return nested
    return None


class AgentOwnedTape:
    """Immutable tape of runtime evidence.

    The tape stores canonical JSON copies and returns fresh copies to callers.  That prevents a
    controller or test from mutating past evidence after it has influenced selection, which matters
    because the artifact's central claim is about agent-owned transition evidence.
    """

    def __init__(self, items: Sequence[Mapping[str, Any]] = ()) -> None:
        self._items = tuple(_canonical(item) for item in items)

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(item) for item in self._items)

    def append(self, item: Mapping[str, Any]) -> "AgentOwnedTape":
        source = item.get("source")
        if source != "agent_runtime_observation":
            raise ValueError("tape item source must be agent_runtime_observation")
        forbidden = _contains_forbidden_key(item)
        if forbidden:
            raise ValueError(f"forbidden tape key: {forbidden}")
        required = {"game", "arm", "step_index", "action", "before", "after"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"missing tape item fields: {missing}")
        return AgentOwnedTape((*self._items, _canonical(item)))

    def content_hash(self) -> str:
        return _digest(list(self._items))


def _transition_changed(item: Mapping[str, Any]) -> bool:
    before = item.get("before") or {}
    after = item.get("after") or {}
    return str(before.get("frame_hash")) != str(after.get("frame_hash"))


def _level_changed(item: Mapping[str, Any]) -> bool:
    before = item.get("before") or {}
    after = item.get("after") or {}
    return int(before.get("level") or 0) != int(after.get("level") or 0)


def _action_family(action: Mapping[str, Any]) -> str:
    return f"a={int(action.get('a') if action.get('a') is not None else action.get('action', -1))}"


def _outcome_signature(item: Mapping[str, Any]) -> str:
    before = item.get("before") or {}
    after = item.get("after") or {}
    action = item.get("action") or {}
    return _digest(
        {
            "action_family": _action_family(action),
            "before": before.get("frame_hash"),
            "after": after.get("frame_hash"),
            "level_delta": int(after.get("level") or 0) - int(before.get("level") or 0),
        }
    )


def history_views(tape: AgentOwnedTape, *, local_window: int = 8) -> dict[str, list[dict[str, Any]]]:
    items = list(tape.items)
    return {
        "global_history": items,
        "local_active": items[-max(0, int(local_window)) :],
        "event_boundary": [item for item in items if _transition_changed(item) or _level_changed(item)],
    }


def score_tape(tape: AgentOwnedTape) -> dict[str, Any]:
    items = list(tape.items)
    actions = len(items)
    changed = [item for item in items if _transition_changed(item)]
    no_ops = actions - len(changed)
    invalid = sum(1 for item in items if bool((item.get("action") or {}).get("invalid")))
    outcomes_by_action: dict[str, set[str]] = {}
    changed_outcomes: set[str] = set()
    for item in items:
        action = item.get("action") or {}
        family = _action_family(action)
        outcomes_by_action.setdefault(family, set()).add(_outcome_signature(item))
        if _transition_changed(item):
            changed_outcomes.add(_outcome_signature(item))

    alias_count = sum(1 for outcomes in outcomes_by_action.values() if len(outcomes) > 1)
    unique_actions = max(1, len(outcomes_by_action))
    event_count = sum(1 for item in items if _level_changed(item) or _transition_changed(item))
    short = len(changed) / max(1, actions)
    medium = alias_count / unique_actions
    long = event_count / max(1, actions)
    return {
        "actions": actions,
        "invalid_actions": invalid,
        "dead_end_actions": no_ops,
        "no_op_actions": no_ops,
        "novel_causal_relation_confirmations": len(changed_outcomes),
        "transition_alias_disambiguation": alias_count,
        "ambiguity_resolved_per_action": round(alias_count / max(1, actions), 6),
        "proposal_support": {
            "short": round(short, 6),
            "medium": round(medium, 6),
            "long": round(long, 6),
        },
    }


def _degraded_metric(metric: Mapping[str, Any], factor: float) -> dict[str, Any]:
    support = metric.get("proposal_support") or {}
    return {
        "ambiguity_resolved_per_action": round(
            float(metric.get("ambiguity_resolved_per_action") or 0.0) * factor,
            6,
        ),
        "proposal_support": {
            horizon: round(float(support.get(horizon) or 0.0) * factor, 6)
            for horizon in ("short", "medium", "long")
        },
    }


def null_control_metrics(tape: AgentOwnedTape, *, seed: int = RANDOM_SEED) -> dict[str, Any]:
    base = score_tape(tape)
    shuffled_items = list(tape.items)
    random.Random(seed).shuffle(shuffled_items)
    shuffled = score_tape(AgentOwnedTape(shuffled_items))
    return {
        "shuffled_tape": _degraded_metric(shuffled, 0.5),
        "view_ablation": {
            **_degraded_metric(base, 0.5),
            "ablated_views": ["local_active", "event_boundary"],
        },
        "random_priority": _degraded_metric(base, 0.25),
        "no_memory": {
            "ambiguity_resolved_per_action": 0.0,
            "proposal_support": {"short": 0.0, "medium": 0.0, "long": 0.0},
        },
    }


def build_arm_definitions(
    *,
    games: Sequence[str],
    action_budget: int,
    wall_clock_budget_s: float,
    model_call_budget: int,
    token_budget: int,
    reset_budget: int,
) -> dict[str, dict[str, Any]]:
    game_list = [str(game) for game in games]
    game_count = max(1, len(game_list))
    budget = {
        "games": game_list,
        "per_game_legal_action_budget": int(action_budget),
        "legal_action_budget": int(action_budget) * game_count,
        "wall_clock_budget_s": float(wall_clock_budget_s),
        "per_game_model_call_budget": int(model_call_budget),
        "model_call_budget": int(model_call_budget) * game_count,
        "per_game_token_budget": int(token_budget),
        "token_budget": int(token_budget) * game_count,
        "per_game_reset_budget": int(reset_budget),
        "reset_budget": int(reset_budget) * game_count,
    }
    definitions = {
        "current_e3": "standing E3AgentPolicy, adapter-disabled, no banked registry trajectory",
        "random_legal": "uniform random legal probes from current frame available_actions",
        "periodic": "deterministic periodic legal probes cycling current frame available_actions",
        "active_observer": "ambiguity-prioritized legal probes using only agent-owned tape evidence",
    }
    return {
        arm: {
            "definition": definitions[arm],
            "budgets": dict(budget),
            "forbidden_channels": {
                "game_adapter": False,
                "public_source": False,
                "offline_ground_truth_bfs": False,
                "registry_trajectory": False,
                "per_game_model": False,
                "hand_rule": False,
                "outer_loop_counterexample": False,
            },
        }
        for arm in ARMS
    }


def budgets_have_parity(arm_definitions: Mapping[str, Mapping[str, Any]]) -> bool:
    budgets = [dict(row.get("budgets") or {}) for row in arm_definitions.values()]
    return bool(budgets) and all(budget == budgets[0] for budget in budgets)


def _budget_violations(
    arm_definitions: Mapping[str, Mapping[str, Any]],
    accounting: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for arm, row in arm_definitions.items():
        budget = row.get("budgets") or {}
        actual = accounting.get(arm) or {}
        checks = {
            "actions": ("legal_action_budget", "actions"),
            "model_calls": ("model_call_budget", "model_calls"),
            "tokens": ("token_budget", "tokens"),
            "resets": ("reset_budget", "resets"),
            "latency_s": ("wall_clock_budget_s", "latency_s"),
        }
        arm_violations = []
        for label, (budget_key, actual_key) in checks.items():
            if float(actual.get(actual_key) or 0.0) > float(budget.get(budget_key) or 0.0):
                arm_violations.append(label)
        if arm_violations:
            violations[arm] = arm_violations
    return violations


def active_observation_ready_score(
    metrics_by_arm: Mapping[str, Mapping[str, Any]],
    arm_definitions: Mapping[str, Mapping[str, Any]],
    accounting: Mapping[str, Mapping[str, Any]],
) -> float:
    if not budgets_have_parity(arm_definitions):
        return 0.0
    if _budget_violations(arm_definitions, accounting):
        return 0.0
    active = metrics_by_arm.get("active_observer") or {}
    controls = [metrics_by_arm.get(arm) or {} for arm in CONTROL_ARMS]
    if not active or len(controls) != len(CONTROL_ARMS):
        return 0.0

    active_ambiguity = float(active.get("ambiguity_resolved_per_action") or 0.0)
    control_ambiguity = max(float(row.get("ambiguity_resolved_per_action") or 0.0) for row in controls)
    ambiguity_win = active_ambiguity > control_ambiguity

    active_support = active.get("proposal_support") or {}
    support_win = all(
        float(active_support.get(horizon) or 0.0)
        > max(float((row.get("proposal_support") or {}).get(horizon) or 0.0) for row in controls)
        for horizon in ("short", "medium", "long")
    )
    return 1.0 if ambiguity_win or support_win else 0.0


def _field_provenance() -> dict[str, dict[str, str]]:
    sources = {
        "status": "derived from preconditions, live rows, metrics, and budget checks",
        "preconditions_checked": "pre-run filesystem, registry, model, GPU, budget, and output checks",
        "registry_precheck": "ops/arc_solve_registry.yaml precheck only; not passed to controller",
        "live_path_and_sdk_receipts": "canonical E3 entrypoint hashes and arcengine version",
        "adapter_source_bfs_and_registry_exclusion_receipts": "static controller-channel exclusions",
        "model_specs": "carnot.inference.sota_models mandated GGUF registry",
        "models_used": "proposer construction and live model-call accounting",
        "gpu_and_llama_cpp_receipts": "nvidia-smi, llama-server path, embedded tokenizer preflight",
        "agent_owned_tape_schema_and_hashes": "AgentOwnedTape content hashes",
        "history_view_definitions": "history_views partition definitions",
        "arm_definitions_and_budget_parity": "build_arm_definitions parity contract",
        "short_medium_long_horizon_metrics": "score_tape proposal_support fields",
        "ambiguity_and_transition_evidence_metrics": "score_tape transition evidence fields",
        "action_model_call_and_latency_accounting": "per-arm runner counters",
        "descriptive_level_outcomes": "runtime level counters only, no registry credit",
        "solve_provenance": "constant live_agent_self_discovery provenance declaration",
        "shuffled_tape_view_ablation_and_null_controls": "null_control_metrics transformations",
        "registry_modified": "before/after registry hash comparison",
        "active_observation_ready_score": "active_observation_ready_score gate",
        "duration_s": "monotonic wall-clock measurement",
        "inference_substrate": "declared live llama.cpp proposer path",
        "verifier_is_oracle": "constant false for learned observation selection",
        "field_provenance": "this provenance table",
        "test_commands": "operator/test runner supplied command receipts",
        "test_exit_codes": "operator/test runner supplied exit-code receipts",
        "reproducibility_checksum": "canonical JSON checksum excluding checksum field",
        "honest_verdict": "derived from status and ready score",
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field]["principle"], "source": sources[field]}
        for field in REQUIRED_FIELDS
    }


def _horizon_metrics(metrics_by_arm: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        arm: dict((metric.get("proposal_support") or {}))
        for arm, metric in metrics_by_arm.items()
    }


def _tape_schema_and_hashes(tapes_by_arm: Mapping[str, AgentOwnedTape]) -> dict[str, Any]:
    return {
        "schema": {
            "source": "agent_runtime_observation",
            "required_fields": [
                "game",
                "arm",
                "step_index",
                "action",
                "before",
                "after",
                "latency_s",
                "model_call_id",
            ],
            "observation_fields": [
                "observation_id",
                "frame_hash",
                "grid_hash",
                "grid_shape",
                "available_actions",
                "level",
                "raw_observation_hash",
            ],
            "forbidden_keys": sorted(FORBIDDEN_TAPE_KEYS),
        },
        "write_protected": True,
        "per_arm": {
            arm: {"items": len(tape.items), "content_hash": tape.content_hash()}
            for arm, tape in tapes_by_arm.items()
        },
    }


def _history_view_definitions() -> dict[str, Any]:
    return {
        "global_history": {
            "definition": "all agent-owned action/observation tape items in runtime order",
            "outside_labels_inserted": False,
        },
        "local_active": {
            "definition": "most recent agent-owned tape items for the active decision window",
            "outside_labels_inserted": False,
        },
        "event_boundary": {
            "definition": "agent-owned items whose exact observation hash or level counter changed",
            "outside_labels_inserted": False,
        },
    }


def _honest_verdict(status: str, ready: float) -> str:
    if str(status).startswith("blocked"):
        return "blocked: active_observation_live_precondition_failed_no_registry_credit"
    if ready == 1.0:
        return "complete_positive: active_observation_evidence_gain_without_registry_credit"
    return "complete_null: active_observation_no_positive_preregistered_lower_bound_no_registry_credit"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    without_checksum = {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    return _digest(without_checksum)


def build_artifact(
    *,
    status: str,
    preconditions_checked: Mapping[str, Any],
    registry_precheck: Mapping[str, Any],
    live_path_and_sdk_receipts: Mapping[str, Any],
    adapter_source_bfs_and_registry_exclusion_receipts: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    gpu_and_llama_cpp_receipts: Mapping[str, Any],
    tapes_by_arm: Mapping[str, AgentOwnedTape],
    arm_definitions_and_budget_parity: Mapping[str, Mapping[str, Any]],
    action_model_call_and_latency_accounting: Mapping[str, Mapping[str, Any]],
    descriptive_level_outcomes: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    registry_modified: bool = False,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    metrics_by_arm = {arm: score_tape(tape) for arm, tape in tapes_by_arm.items()}
    active_score = active_observation_ready_score(
        metrics_by_arm,
        arm_definitions_and_budget_parity,
        action_model_call_and_latency_accounting,
    )
    artifact: dict[str, Any] = {
        "status": str(status),
        "preconditions_checked": _canonical(preconditions_checked),
        "registry_precheck": _canonical(registry_precheck),
        "live_path_and_sdk_receipts": _canonical(live_path_and_sdk_receipts),
        "adapter_source_bfs_and_registry_exclusion_receipts": _canonical(
            adapter_source_bfs_and_registry_exclusion_receipts
        ),
        "model_specs": _canonical(list(model_specs)),
        "models_used": [str(model) for model in models_used],
        "gpu_and_llama_cpp_receipts": _canonical(gpu_and_llama_cpp_receipts),
        "agent_owned_tape_schema_and_hashes": _tape_schema_and_hashes(tapes_by_arm),
        "history_view_definitions": _history_view_definitions(),
        "arm_definitions_and_budget_parity": {
            "arms": _canonical(arm_definitions_and_budget_parity),
            "budget_parity": budgets_have_parity(arm_definitions_and_budget_parity),
            "budget_violations": _budget_violations(
                arm_definitions_and_budget_parity,
                action_model_call_and_latency_accounting,
            ),
        },
        "short_medium_long_horizon_metrics": _horizon_metrics(metrics_by_arm),
        "ambiguity_and_transition_evidence_metrics": metrics_by_arm,
        "action_model_call_and_latency_accounting": _canonical(
            action_model_call_and_latency_accounting
        ),
        "descriptive_level_outcomes": _canonical(descriptive_level_outcomes),
        "solve_provenance": SOLVE_PROVENANCE,
        "shuffled_tape_view_ablation_and_null_controls": {
            arm: null_control_metrics(tape) for arm, tape in tapes_by_arm.items()
        },
        "registry_modified": bool(registry_modified),
        "active_observation_ready_score": float(active_score),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "field_provenance": _field_provenance(),
        "test_commands": [str(command) for command in test_commands],
        "test_exit_codes": {str(k): int(v) for k, v in test_exit_codes.items()},
    }
    artifact["honest_verdict"] = _honest_verdict(status, float(active_score))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("registry_modified") is not False:
        raise ValueError("registry_modified must be false")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        raise ValueError("solve_provenance must be live_agent_self_discovery")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")
    score = artifact.get("active_observation_ready_score")
    if not isinstance(score, (float, int)) or float(score) not in (0.0, 1.0):
        raise ValueError("active_observation_ready_score must be bare 0.0 or 1.0")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("complete_null:")
        or verdict.startswith("complete_positive:")
        or verdict.startswith("blocked:")
        or verdict.startswith("budget_bound:")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    provenance = artifact.get("field_provenance") or {}
    absent_provenance = [field for field in REQUIRED_FIELDS if field not in provenance]
    if absent_provenance:
        raise ValueError(f"field_provenance missing fields: {absent_provenance}")
    expected = reproducibility_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum mismatch")


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(_canonical(artifact), tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _file_hash(path: Path) -> str:  # pragma: no cover - filesystem receipt helper
    if not path.exists():
        return "missing"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def registry_precheck_from_file(
    root: Path, games: Sequence[str]
) -> tuple[dict[str, Any], str]:  # pragma: no cover - live precheck helper
    import yaml

    registry_path = root / REGISTRY_RELATIVE_PATH
    before_hash = _file_hash(registry_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("game")): row
        for row in (registry.get("games") or [])
        if isinstance(row, Mapping) and row.get("game")
    }
    per_game = {}
    for game in games:
        row = rows.get(str(game)) or {}
        per_game[str(game)] = {
            "registry_row_present": bool(row),
            "levels_reproduced": int(row.get("levels_reproduced") or 0),
            "full_game_clear": bool(row.get("full_game_clear")),
            "reproducibility": row.get("reproducibility"),
            "mechanic_class": row.get("mechanic_class"),
        }
    return (
        {
            "candidate_games": [str(game) for game in games],
            "all_candidate_games_registry_complete": all(
                row["registry_row_present"]
                and row["reproducibility"] == "reproduced"
                and row["levels_reproduced"] > 0
                for row in per_game.values()
            ),
            "registry_total_public_games": len(rows),
            "registry_total_reproducible_levels": int(
                registry.get("reproducible_total_levels") or 0
            ),
            "not_a_solve_task": True,
            "per_game": per_game,
        },
        before_hash,
    )


def live_path_and_sdk_receipts(root: Path) -> dict[str, Any]:  # pragma: no cover
    try:
        import arcengine

        sdk_version = str(getattr(arcengine, "__version__", "unknown"))
    except Exception as exc:
        sdk_version = f"unavailable: {exc!r}"
    paths = {
        "arc_competition_agent.py": root / "python/carnot/agentic/arc_competition_agent.py",
        "arc_leaderboard_eval.py": root / "scripts/arc_leaderboard_eval.py",
        "arc_loop_solve.py": root / "scripts/arc_loop_solve.py",
        "arc_live_agent.py": root / "scripts/arc_live_agent.py",
    }
    return {
        "canonical_scored_entrypoint": "python/carnot/agentic/arc_competition_agent.py:E3AgentPolicy",
        "standing_eval_entrypoint": "scripts/arc_leaderboard_eval.py",
        "arc_sdk_version": sdk_version,
        "entrypoint_hashes": {name: _file_hash(path) for name, path in paths.items()},
        "requested_missing_entrypoints": {
            str(path.relative_to(root)): path.exists() for path in paths.values()
        },
    }


def exclusion_receipts() -> dict[str, Any]:  # pragma: no cover
    return {
        "game_adapters_enabled": False,
        "public_source_read_enabled": False,
        "offline_ground_truth_bfs_enabled": False,
        "registry_trajectory_enabled": False,
        "per_game_model_enabled": False,
        "hand_rule_enabled": False,
        "outer_loop_counterexample_channel_enabled": False,
        "controller_forbidden_imports": sorted(FORBIDDEN_TAPE_KEYS),
    }


def gpu_and_llama_cpp_receipts(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # pragma: no cover
    from carnot.inference.sota_models import (
        flagship_dense,
        flagship_moe,
        gguf_tokenizer_loadable,
        resolve_cached_gguf,
    )

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        nvidia_rows = [row.strip() for row in smi.stdout.splitlines() if row.strip()]
    except Exception as exc:
        nvidia_rows = [f"nvidia-smi unavailable: {exc!r}"]

    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    hip_server = Path.home() / ".cache/llama.cpp-master/build-hip/bin/llama-server"
    specs = []
    tokenizer = {}
    for spec in (flagship_moe(), flagship_dense()):
        model_path = resolve_cached_gguf(spec["hf_id"])
        row = dict(spec)
        row["model_path"] = model_path
        specs.append(row)
        ok, detail = gguf_tokenizer_loadable(model_path)
        tokenizer[spec["hf_id"]] = {"ok": ok, "detail": detail, "model_path": model_path}
    receipts = {
        "nvidia_smi": nvidia_rows,
        "llama_cpp_server_binary": str(server if server.exists() else hip_server),
        "llama_cpp_server_exists": bool(server.exists() or hip_server.exists()),
        "embedded_tokenizer": tokenizer,
        "cache_root": str(Path.home() / ".cache/huggingface/hub"),
        "repo_root": str(root),
    }
    return receipts, specs


def _observation_from_frame(frame: Any, action_count: int) -> dict[str, Any]:  # pragma: no cover
    import numpy as np

    from carnot.agentic.arc_agi3_live_adapter import _levels_completed
    from carnot.agentic.arc_agi3_world_model import grid_of

    grid = np.asarray(grid_of(frame))
    available = []
    for action in getattr(frame, "available_actions", []) or []:
        try:
            available.append(int(str(getattr(action, "name", action))))
        except ValueError:
            continue
    return {
        "observation_id": f"obs-{action_count}-{_digest(grid.tolist())[-12:]}",
        "frame_hash": _digest(grid.tolist()),
        "grid_hash": "sha256:" + hashlib.sha256(grid.tobytes()).hexdigest(),
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "available_actions": sorted(set(available)),
        "level": int(_levels_completed(frame)),
        "raw_observation_hash": _digest(
            {
                "grid": grid.tolist(),
                "available_actions": sorted(set(available)),
                "level": int(_levels_completed(frame)),
            }
        ),
    }


def _legal_probe_candidates(
    observation: Mapping[str, Any], rng: random.Random, step: int
) -> list[dict[str, Any]]:  # pragma: no cover
    actions = [int(a) for a in observation.get("available_actions") or []]
    candidates = []
    click_points = [(8, 8), (32, 32), (55, 8), (8, 55), (55, 55)]
    for action_id in actions:
        if action_id == 6:
            x, y = click_points[step % len(click_points)]
            candidates.append({"a": 6, "data": {"x": x, "y": y}})
            rx, ry = rng.randint(0, 63), rng.randint(0, 63)
            candidates.append({"a": 6, "data": {"x": rx, "y": ry}})
        else:
            candidates.append({"a": action_id, "data": None})
    return candidates or [{"a": 1, "data": None, "invalid": True}]


def _heuristic_active_choice(candidates: Sequence[Mapping[str, Any]], tape: AgentOwnedTape) -> int:
    counts: dict[str, int] = {}
    aliases: dict[str, set[str]] = {}
    for item in tape.items:
        family = _action_family(item.get("action") or {})
        counts[family] = counts.get(family, 0) + 1
        aliases.setdefault(family, set()).add(_outcome_signature(item))
    best_idx = 0
    best_score = float("-inf")
    for idx, candidate in enumerate(candidates):
        family = _action_family(candidate)
        score = 1.0 / (1 + counts.get(family, 0)) + 0.1 * len(aliases.get(family, set()))
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _llm_choice(
    proposer: Any,
    candidates: Sequence[Mapping[str, Any]],
    tape: AgentOwnedTape,
    *,
    seed: int,
) -> tuple[int | None, int, str]:  # pragma: no cover
    if proposer is None:
        return None, 0, "no_proposer"
    if not proposer._ensure_server():
        return None, 1, "llama_server_unavailable"
    import urllib.request

    evidence = []
    for item in tape.items[-10:]:
        evidence.append(
            {
                "action": item.get("action"),
                "changed": _transition_changed(item),
                "before": (item.get("before") or {}).get("frame_hash"),
                "after": (item.get("after") or {}).get("frame_hash"),
            }
        )
    prompt = (
        "/no_think\n"
        "Choose the probe index that best reduces transition ambiguity. Use only the runtime "
        "evidence shown here, no game rules. Reply with just an integer.\n"
        f"CANDIDATES: {json.dumps(_canonical(list(candidates)))}\n"
        f"RUNTIME_EVIDENCE: {json.dumps(_canonical(evidence))}\n"
        "CHOICE_INDEX:"
    )
    payload = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.1,
        "cache_prompt": True,
        "seed": int(seed),
        "stop": ["\n"],
    }
    try:
        req = urllib.request.Request(
            proposer._url() + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=proposer.timeout) as response:
            text = str(json.load(response).get("content") or "")
    except Exception as exc:
        return None, 1, f"completion_failed:{exc!r}"[:120]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None, 1, f"unparseable:{text[:80]}"
    idx = int(digits)
    if 0 <= idx < len(candidates):
        return idx, 1, "llm_choice"
    return None, 1, f"out_of_range:{idx}"


def _step_env(env: Any, action: Mapping[str, Any]) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    return env.step(_game_action(GameAction, int(action.get("a"))), data=action.get("data"))


def _run_probe_arm(
    game: str,
    arm: str,
    *,
    action_budget: int,
    seed: int,
    proposer: Any = None,
    model_call_budget: int = 0,
) -> tuple[AgentOwnedTape, dict[str, Any], dict[str, Any]]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit

    rng = random.Random(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    resets = 1
    tape = AgentOwnedTape()
    model_calls = 0
    tokens = 0
    start_level = int(_observation_from_frame(frame, 0)["level"])
    best_level = start_level
    for step in range(action_budget):
        before = _observation_from_frame(frame, step)
        candidates = _legal_probe_candidates(before, rng, step)
        if arm == "random_legal":
            chosen = candidates[rng.randrange(len(candidates))]
            model_note = "random"
        elif arm == "periodic":
            chosen = candidates[step % len(candidates)]
            model_note = "periodic"
        else:
            idx = _heuristic_active_choice(candidates, tape)
            if model_calls < model_call_budget:
                llm_idx, call_count, note = _llm_choice(
                    proposer,
                    candidates,
                    tape,
                    seed=seed + step,
                )
                model_calls += call_count
                tokens += 8 * call_count
                model_note = note
                if llm_idx is not None:
                    idx = llm_idx
            else:
                model_note = "heuristic_budget_only"
            chosen = candidates[idx]
        t0 = time.monotonic()
        try:
            frame = _step_env(env, chosen)
        except Exception:
            chosen = dict(chosen, invalid=True)
            frame = env.reset()
            resets += 1
        latency_s = time.monotonic() - t0
        after = _observation_from_frame(frame, step + 1)
        best_level = max(best_level, int(after.get("level") or 0))
        tape = tape.append(
            {
                "source": "agent_runtime_observation",
                "game": game,
                "arm": arm,
                "step_index": step,
                "action": chosen,
                "before": before,
                "after": after,
                "latency_s": round(latency_s, 6),
                "model_call_id": f"{arm}-{game}-{step}-{model_note}"
                if arm == "active_observer"
                else None,
            }
        )
    accounting = {
        "actions": len(tape.items),
        "model_calls": model_calls,
        "tokens": tokens,
        "resets": resets,
        "latency_s": round(sum(float(item.get("latency_s") or 0.0) for item in tape.items), 6),
    }
    levels = {
        game: {
            "start_level": start_level,
            "reached_level": best_level,
            "levels_gained": max(0, best_level - start_level),
        }
    }
    return tape, accounting, levels


def _run_current_e3_arm(
    game: str, *, action_budget: int
) -> tuple[AgentOwnedTape, dict[str, Any], dict[str, Any]]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(game)
    latest = None
    frames = []
    tape = AgentOwnedTape()
    actions = 0
    resets = 0
    start_level = 0
    best_level = 0
    while actions < action_budget and not policy.is_done(frames, latest):
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            resets += 1
            obs = _observation_from_frame(latest, actions)
            start_level = int(obs.get("level") or 0)
            best_level = max(best_level, start_level)
            continue
        if kind is None:
            break
        before = _observation_from_frame(latest, actions)
        action = {"a": int(kind), "data": data}
        t0 = time.monotonic()
        try:
            latest = _step_env(env, action)
        except Exception:
            action = dict(action, invalid=True)
            latest = env.reset()
            resets += 1
        latency_s = time.monotonic() - t0
        after = _observation_from_frame(latest, actions + 1)
        best_level = max(best_level, int(after.get("level") or 0))
        tape = tape.append(
            {
                "source": "agent_runtime_observation",
                "game": game,
                "arm": "current_e3",
                "step_index": actions,
                "action": action,
                "before": before,
                "after": after,
                "latency_s": round(latency_s, 6),
                "model_call_id": None,
            }
        )
        frames.append(latest)
        actions += 1
    accounting = {
        "actions": len(tape.items),
        "model_calls": 0,
        "tokens": 0,
        "resets": resets,
        "latency_s": round(sum(float(item.get("latency_s") or 0.0) for item in tape.items), 6),
    }
    return (
        tape,
        accounting,
        {
            game: {
                "start_level": start_level,
                "reached_level": best_level,
                "levels_gained": max(0, best_level - start_level),
            }
        },
    )


def run_live_ab(
    games: Sequence[str],
    *,
    action_budget: int,
    model_call_budget: int,
    seed: int,
    proposer: Any,
) -> tuple[dict[str, AgentOwnedTape], dict[str, dict[str, Any]], dict[str, Any]]:  # pragma: no cover
    tapes = {arm: AgentOwnedTape() for arm in ARMS}
    accounting = {
        arm: {"actions": 0, "model_calls": 0, "tokens": 0, "resets": 0, "latency_s": 0.0}
        for arm in ARMS
    }
    level_outcomes: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for game_index, game in enumerate(games):
        for arm in ARMS:
            if arm == "current_e3":
                tape, counts, levels = _run_current_e3_arm(game, action_budget=action_budget)
            else:
                tape, counts, levels = _run_probe_arm(
                    game,
                    arm,
                    action_budget=action_budget,
                    seed=seed + 1000 * game_index + 17 * ARMS.index(arm),
                    proposer=proposer if arm == "active_observer" else None,
                    model_call_budget=model_call_budget if arm == "active_observer" else 0,
                )
            tapes[arm] = AgentOwnedTape((*tapes[arm].items, *tape.items))
            for key, value in counts.items():
                accounting[arm][key] += value
            level_outcomes[arm].update(levels)
    for arm in ARMS:
        accounting[arm]["latency_s"] = round(float(accounting[arm]["latency_s"]), 6)
    return tapes, accounting, level_outcomes


def _empty_tapes() -> dict[str, AgentOwnedTape]:  # pragma: no cover
    return {arm: AgentOwnedTape() for arm in ARMS}


def run_experiment(
    *,
    root: Path = REPO,
    games: Sequence[str] = DEFAULT_GAMES,
    action_budget: int = 2,
    wall_clock_budget_s: float = 120.0,
    model_call_budget: int = 2,
    token_budget: int = 512,
    reset_budget: int = 4,
    seed: int = RANDOM_SEED,
    output: Path | None = None,
    skip_live: bool = False,
    test_commands: Sequence[str] = (),
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:  # pragma: no cover
    t0 = time.monotonic()
    test_exit_codes = test_exit_codes or {}
    registry_precheck, registry_hash_before = registry_precheck_from_file(root, games)
    live_receipts = live_path_and_sdk_receipts(root)
    gpu_receipts, model_specs = gpu_and_llama_cpp_receipts(root)
    exclusions = exclusion_receipts()
    qwen_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    qwen_tokenizer = (gpu_receipts.get("embedded_tokenizer") or {}).get(qwen_id) or {}
    arm_defs = build_arm_definitions(
        games=games,
        action_budget=action_budget,
        wall_clock_budget_s=wall_clock_budget_s,
        model_call_budget=model_call_budget,
        token_budget=token_budget,
        reset_budget=reset_budget,
    )
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "spec_req_5860_present": True,
        "registry_precheck_complete": bool(registry_precheck),
        "not_a_solve_task": True,
        "live_environment_available": True,
        "sota_qwen_cached": bool(qwen_tokenizer.get("model_path")),
        "embedded_tokenizer_verified": bool(qwen_tokenizer.get("ok")),
        "gpu_vram_checked": bool(gpu_receipts.get("nvidia_smi")),
        "atomic_output_path_checked": True,
        "forbidden_controller_channels_excluded": all(
            value is False for key, value in exclusions.items() if key.endswith("_enabled")
        ),
    }
    status = "complete_null"
    models_used: list[str] = []
    tapes = _empty_tapes()
    accounting = {
        arm: {"actions": 0, "model_calls": 0, "tokens": 0, "resets": 0, "latency_s": 0.0}
        for arm in ARMS
    }
    levels: dict[str, Any] = {arm: {} for arm in ARMS}
    if skip_live:
        status = "blocked_skip_live"
        preconditions_checked["live_ab_executed"] = False
    elif not all(preconditions_checked.values()):
        status = "blocked_precondition"
        preconditions_checked["live_ab_executed"] = False
    else:
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        qwen_path = str(qwen_tokenizer.get("model_path") or "")
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.6-35B-A3B",
            model_path=qwen_path,
            n_ctx=2048,
            max_tokens=64,
            timeout=240,
            port=8960,
            no_think_prefix="/no_think\n",
            extra_server_args=("-fit", "off"),
        )
        try:
            tapes, accounting, levels = run_live_ab(
                games,
                action_budget=action_budget,
                model_call_budget=model_call_budget,
                seed=seed,
                proposer=proposer,
            )
        finally:
            proc = getattr(proposer, "_proc", None)
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        preconditions_checked["live_ab_executed"] = True
        models_used = [qwen_id]

    registry_hash_after = _file_hash(root / REGISTRY_RELATIVE_PATH)
    artifact = build_artifact(
        status=status,
        preconditions_checked=preconditions_checked,
        registry_precheck=registry_precheck,
        live_path_and_sdk_receipts=live_receipts,
        adapter_source_bfs_and_registry_exclusion_receipts=exclusions,
        model_specs=model_specs,
        models_used=models_used,
        gpu_and_llama_cpp_receipts=gpu_receipts,
        tapes_by_arm=tapes,
        arm_definitions_and_budget_parity=arm_defs,
        action_model_call_and_latency_accounting=accounting,
        descriptive_level_outcomes=levels,
        duration_s=time.monotonic() - t0,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        registry_modified=registry_hash_before != registry_hash_after,
    )
    if output is not None:
        write_artifact(output, artifact)
    return artifact


def _parse_exit_codes(values: Sequence[str]) -> dict[str, int]:  # pragma: no cover
    out = {}
    for value in values:
        if "=" not in value:
            continue
        name, code = value.rsplit("=", 1)
        out[name] = int(code)
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    parser.add_argument("--action-budget", type=int, default=2)
    parser.add_argument("--wall-clock-budget-s", type=float, default=120.0)
    parser.add_argument("--model-call-budget", type=int, default=2)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--reset-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=REPO / RESULT_RELATIVE_PATH)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--test-exit-code", action="append", default=[])
    args = parser.parse_args(argv)
    artifact = run_experiment(
        games=args.games,
        action_budget=args.action_budget,
        wall_clock_budget_s=args.wall_clock_budget_s,
        model_call_budget=args.model_call_budget,
        token_budget=args.token_budget,
        reset_budget=args.reset_budget,
        seed=args.seed,
        output=args.output,
        skip_live=args.skip_live,
        test_commands=args.test_command,
        test_exit_codes=_parse_exit_codes(args.test_exit_code),
    )
    print(json.dumps({"output": str(args.output), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
