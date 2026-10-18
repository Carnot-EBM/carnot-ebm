"""Build the Exp 1303 QueryBandits + NGC online memory-policy artifact."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1303_querybandits_ngc_online_memory_policy.json"

EXPERIMENT = "1303_querybandits_ngc_online_memory_policy"
SCHEMA = "querybandits_ngc_online_memory_policy_v1"
RUN_DATE = "20260505"
SEED = 1303

EXP1302_FILE = "experiment_1302_skill_graph_promotion_demotion_v2.json"
EXP1288_FILE = "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
SOURCE_ARTIFACTS = [f"results/{EXP1302_FILE}", f"results/{EXP1288_FILE}"]

ACTION_REPLAY_MEMORY = "replay_memory"
ACTION_REWRITE_REPAIR_PROMPT = "rewrite_repair_prompt"
ACTION_ABSTAIN_ESCALATE = "abstain_escalate"
ACTION_DEMOTE_EXPIRE_MEMORY = "demote_expire_memory"
ACTIONS = (
    ACTION_REPLAY_MEMORY,
    ACTION_REWRITE_REPAIR_PROMPT,
    ACTION_ABSTAIN_ESCALATE,
    ACTION_DEMOTE_EXPIRE_MEMORY,
)

PROMOTED_ROUTING = "promote"
STALE_ROUTINGS = {"demote", "expire"}
PROMOTE_MIN_SUPPORT = 5
SUPPORTED_VERDICTS = {
    "online_memory_policy_improved_non_headline",
    "online_memory_policy_neutral_non_headline",
    "online_memory_policy_regressed_non_headline",
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _metadata(project_root: str | Path, run_date: str, seed: int) -> dict[str, Any]:
    return {"project_root": str(project_root), "run_date": run_date, "seed": seed}


def _zero_distribution() -> dict[str, float]:
    return {action: 0.0 for action in ACTIONS}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    seed: int = SEED,
) -> dict[str, Any]:
    """REQ-LEARN-1303-1: write the bootstrap artifact before simulation."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date, seed),
            "run_date": run_date,
            "source_artifacts": SOURCE_ARTIFACTS,
            "status": "in_progress",
            "action_space": list(ACTIONS),
            "n_examples": 0,
            "self_learning_delta_overall": 0.0,
            "accepted_violation_delta": 0.0,
            "bandit_regret": 0.0,
            "selected_policy_counts": {action: 0 for action in ACTIONS},
            "selected_policy_distribution": _zero_distribution(),
            "memory_demotion_count": 0,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
        },
    )


def _support(candidate: dict[str, Any] | None) -> int:
    if not candidate:
        return 0
    replay_evidence = candidate.get("replay_evidence") or {}
    return int(replay_evidence.get("support") or candidate.get("support") or 0)


def _target_decision(row: dict[str, Any]) -> str:
    target = str(row.get("target_decision") or "")
    if target:
        return target
    return "repair" if row.get("verifier_result") == "failed" else "accept"


def _fallback_pattern(
    records: list[dict[str, Any]],
    target_decision: str,
    verifier_result: str,
    position: int,
) -> str:
    matching = [
        record
        for record in records
        if record.get("selected_decision") == target_decision
        and record.get("verifier_result") == verifier_result
    ]
    source = matching[position % len(matching)] if matching else {}
    return str(source.get("constraint_pattern") or "unknown")


def _best_candidate(
    candidates: list[dict[str, Any]],
    pattern: str,
    target_decision: str,
    verifier_result: str,
) -> dict[str, Any] | None:
    matching = [
        candidate
        for candidate in candidates
        if candidate.get("constraint_pattern") == pattern
        and candidate.get("selected_decision") == target_decision
        and candidate.get("verifier_result") == verifier_result
    ]
    return max(matching, key=_support) if matching else None


def build_feedback_examples(
    exp1302_payload: dict[str, Any],
    exp1288_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1303-2: join skill candidates to verifier-feedback examples."""

    candidates = list(exp1302_payload.get("skill_graph_candidates") or [])
    records = list(exp1288_payload.get("clause_prediction_records") or [])
    examples: list[dict[str, Any]] = []
    for position, row in enumerate(exp1288_payload.get("replay_slices") or []):
        target_decision = _target_decision(row)
        verifier_result = str(
            row.get("verifier_result") or ("failed" if target_decision == "repair" else "passed")
        )
        pattern = str(
            row.get("constraint_pattern")
            or _fallback_pattern(records, target_decision, verifier_result, position)
        )
        candidate = _best_candidate(candidates, pattern, target_decision, verifier_result)
        examples.append(
            {
                "case_id": str(row.get("case_id") or f"case-{position}"),
                "chronological_index": int(row.get("chronological_index") or position),
                "constraint_pattern": pattern,
                "target_decision": target_decision,
                "verifier_result": verifier_result,
                "memory_available": candidate is not None,
                "memory_selected_decision": str(
                    candidate.get("selected_decision") if candidate else "accept"
                ),
                "memory_routing_decision": str(
                    candidate.get("memory_routing_decision") if candidate else "missing"
                ),
                "memory_support": _support(candidate),
                "memory_skill_id": str(candidate.get("skill_id") if candidate else "missing"),
            }
        )
    return examples


def _action_decision(action: str, example: dict[str, Any]) -> str:
    if action == ACTION_REPLAY_MEMORY:
        return (
            str(example["memory_selected_decision"])
            if example["memory_available"] and example.get("memory_active", True)
            else "accept"
        )
    if action == ACTION_REWRITE_REPAIR_PROMPT:
        return "repair"
    if action == ACTION_ABSTAIN_ESCALATE:
        return "abstain"
    return "demote"


def _score_action(action: str, example: dict[str, Any]) -> tuple[float, int]:
    decision = _action_decision(action, example)
    target = str(example["target_decision"])
    routing = str(example["memory_routing_decision"])
    support = int(example["memory_support"])
    accepted_violation = int(decision == "accept" and target == "repair")
    if accepted_violation:
        return -1.0, accepted_violation
    if action == ACTION_ABSTAIN_ESCALATE:
        return 0.55, 0
    if action == ACTION_DEMOTE_EXPIRE_MEMORY:
        return (0.6 if routing in STALE_ROUTINGS else 0.2), 0
    if decision == target:
        reward = 1.0
        if action == ACTION_REPLAY_MEMORY:
            reward += 0.1 if routing == PROMOTED_ROUTING and support >= PROMOTE_MIN_SUPPORT else -0.7
        if action == ACTION_REWRITE_REPAIR_PROMPT:
            reward -= 0.1
        return round(reward, 6), 0
    return 0.2, 0


def _baseline_score(example: dict[str, Any]) -> tuple[float, int]:
    accepted_violation = int(example["target_decision"] == "repair")
    return (-1.0, accepted_violation) if accepted_violation else (1.0, 0)


def _initial_action_values(example: dict[str, Any]) -> dict[str, float]:
    routing = str(example["memory_routing_decision"])
    support = int(example["memory_support"])
    replay_prior = 0.8 if routing == PROMOTED_ROUTING and support >= PROMOTE_MIN_SUPPORT else 0.25
    demote_prior = 0.7 if routing in STALE_ROUTINGS else 0.15
    rewrite_prior = 0.65 if example["target_decision"] == "repair" else 0.2
    return {
        ACTION_REPLAY_MEMORY: replay_prior,
        ACTION_REWRITE_REPAIR_PROMPT: rewrite_prior,
        ACTION_ABSTAIN_ESCALATE: 0.55,
        ACTION_DEMOTE_EXPIRE_MEMORY: demote_prior,
    }


def _distribution(counts: Counter[str], n_examples: int) -> dict[str, float]:
    remaining = 1.0
    distribution: dict[str, float] = {}
    for action in ACTIONS[:-1]:
        value = round(counts[action] / n_examples, 6) if n_examples else 0.0
        distribution[action] = value
        remaining -= value
    distribution[ACTIONS[-1]] = round(max(0.0, remaining), 6) if n_examples else 0.0
    return distribution


def simulate_online_policy(
    examples: list[dict[str, Any]],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """REQ-LEARN-1303-3/4: run the fixed-seed bandit-style policy simulation."""

    rng = random.Random(seed)
    counts: Counter[str] = Counter({action: 0 for action in ACTIONS})
    q_values: dict[tuple[str, str, str], dict[str, float]] = {}
    q_counts: dict[tuple[str, str, str], Counter[str]] = {}
    demoted_patterns: set[str] = set()
    policy_reward = 0.0
    baseline_reward = 0.0
    policy_violations = 0
    baseline_violations = 0
    bandit_regret = 0.0

    for example in examples:
        active_example = dict(
            example,
            memory_active=example["constraint_pattern"] not in demoted_patterns,
        )
        context = (
            str(active_example["verifier_result"]),
            str(active_example["constraint_pattern"]),
            str(active_example["memory_routing_decision"]),
        )
        if context not in q_values:
            q_values[context] = _initial_action_values(active_example)
            q_counts[context] = Counter({action: 0 for action in ACTIONS})
        action_order = list(ACTIONS)
        rng.shuffle(action_order)
        selected = max(action_order, key=lambda action: q_values[context][action])
        action_scores = {action: _score_action(action, active_example) for action in ACTIONS}
        selected_reward, selected_violation = action_scores[selected]
        best_reward = max(reward for reward, _violation in action_scores.values())
        base_reward, base_violation = _baseline_score(active_example)

        counts[selected] += 1
        q_counts[context][selected] += 1
        learning_rate = 1.0 / q_counts[context][selected]
        q_values[context][selected] += learning_rate * (
            selected_reward - q_values[context][selected]
        )
        policy_reward += selected_reward
        baseline_reward += base_reward
        policy_violations += selected_violation
        baseline_violations += base_violation
        bandit_regret += best_reward - selected_reward
        if selected == ACTION_DEMOTE_EXPIRE_MEMORY:
            demoted_patterns.add(str(active_example["constraint_pattern"]))

    n_examples = len(examples)
    denominator = n_examples or 1
    return {
        "n_examples": n_examples,
        "memory_policy_reward": round(policy_reward / denominator, 6),
        "no_memory_baseline_reward": round(baseline_reward / denominator, 6),
        "self_learning_delta_overall": round(
            (policy_reward - baseline_reward) / denominator,
            6,
        ),
        "memory_accepted_violation_count": policy_violations,
        "baseline_accepted_violation_count": baseline_violations,
        "accepted_violation_delta": round(
            (policy_violations - baseline_violations) / denominator,
            6,
        ),
        "bandit_regret": round(bandit_regret, 6),
        "selected_policy_counts": {action: counts[action] for action in ACTIONS},
        "selected_policy_distribution": _distribution(counts, n_examples),
        "memory_demotion_count": counts[ACTION_DEMOTE_EXPIRE_MEMORY],
    }


def derive_honest_verdict(self_learning_delta_overall: float) -> str:
    """REQ-LEARN-1303-6: report improvement without headline overclaiming."""

    if self_learning_delta_overall > 0.0:
        return "online_memory_policy_improved_non_headline"
    if self_learning_delta_overall < 0.0:
        return "online_memory_policy_regressed_non_headline"
    return "online_memory_policy_neutral_non_headline"


def build_artifact(
    exp1302_payload: dict[str, Any],
    exp1288_payload: dict[str, Any],
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    seed: int = SEED,
) -> dict[str, Any]:
    """REQ-LEARN-1303-5: build the final online memory-policy artifact."""

    examples = build_feedback_examples(exp1302_payload, exp1288_payload)
    metrics = simulate_online_policy(examples, seed=seed)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date, seed),
        "run_date": run_date,
        "source_artifacts": SOURCE_ARTIFACTS,
        "status": "complete",
        "action_space": list(ACTIONS),
        **metrics,
        "headline_result_allowed": False,
        "honest_verdict": derive_honest_verdict(metrics["self_learning_delta_overall"]),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required Exp 1303 schema fields."""

    required = {
        "status",
        "self_learning_delta_overall",
        "accepted_violation_delta",
        "bandit_regret",
        "selected_policy_distribution",
        "memory_demotion_count",
        "headline_result_allowed",
        "honest_verdict",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] != "complete":
        raise AssertionError("status must be complete")
    if float(artifact["bandit_regret"]) < 0.0:
        raise AssertionError("bandit_regret must be non-negative")
    if set(artifact["selected_policy_distribution"]) != set(ACTIONS):
        raise AssertionError("selected_policy_distribution must cover actions")
    if int(artifact["memory_demotion_count"]) < 0:
        raise AssertionError("memory_demotion_count must be non-negative")
    if not isinstance(artifact["headline_result_allowed"], bool):
        raise AssertionError("headline_result_allowed must be boolean")
    if artifact["honest_verdict"] not in SUPPORTED_VERDICTS:
        raise AssertionError("honest_verdict is unsupported")


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    seed: int = SEED,
) -> dict[str, Any]:
    """REQ-LEARN-1303-1/2: load source artifacts and write the final result."""

    results_path = Path(results_dir)
    output_path = Path(out_path)
    write_in_progress_artifact(output_path, project_root=project_root, run_date=run_date, seed=seed)
    exp1302_payload = json.loads((results_path / EXP1302_FILE).read_text(encoding="utf-8"))
    exp1288_payload = json.loads((results_path / EXP1288_FILE).read_text(encoding="utf-8"))
    artifact = build_artifact(
        exp1302_payload,
        exp1288_payload,
        project_root=project_root,
        run_date=run_date,
        seed=seed,
    )
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


__all__ = [
    "ACTIONS",
    "ACTION_ABSTAIN_ESCALATE",
    "ACTION_DEMOTE_EXPIRE_MEMORY",
    "ACTION_REPLAY_MEMORY",
    "ACTION_REWRITE_REPAIR_PROMPT",
    "build_artifact",
    "build_feedback_examples",
    "derive_honest_verdict",
    "run",
    "simulate_online_policy",
    "validate_artifact",
    "write_in_progress_artifact",
]
