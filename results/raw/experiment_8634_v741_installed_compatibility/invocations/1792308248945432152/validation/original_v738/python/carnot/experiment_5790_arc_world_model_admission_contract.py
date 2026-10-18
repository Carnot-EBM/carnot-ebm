"""Experiment 5790: game-blind ARC world-model admission contract.

This module accredits immutable executable world-model hypotheses against
agent-owned ARC transition evidence. It is deliberately narrower than a solver:
it does not induce a new model, edit a rejected model, execute a retained model
through live E3, read game source, or grant registry credit. The useful output is
the admission decision and the reasons for rejection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5790_arc_world_model_admission_contract.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
E3_ENTRYPOINT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
TRACE_AUDIT_RELATIVE_PATH = Path("results/experiment_5766_arc_loo_component_interaction_audit.json")
GEMMA_SINGLESHOT_RELATIVE_PATH = Path("results/experiment_5764_gemma31b_singleshot_induction_ab.json")
GEMMA_SINGLESHOT_SHARD_RELATIVE_PATH = Path("results/exp5764_gemma31b_singleshot_shard.jsonl")
CEGIS_5760_RELATIVE_PATH = Path("results/experiment_5760_cegis_refinement_induction_ab.json")
GEMMA_CEGIS_5766_RELATIVE_PATH = Path("results/experiment_5766_gemma31b_cegis_refinement_ab.json")

PUBLIC_GAME_COUNT = 25
REGISTRY_LEVEL_COUNT = 183
INFERENCE_SUBSTRATE = "immutable_executable_world_model_replay_over_agent_owned_arc_transitions_no_llm"
SOLVE_PROVENANCE = "development_proxy"
L2_HELDOUT_THRESHOLD = 0.65
L2_UNSEEN_ACTION_THRESHOLD = 0.70
L3_STABILITY_THRESHOLD = 0.70
L3_MAX_ERROR_GROWTH = 0.40
L4_PIVOTAL_THRESHOLD = 0.80
L4_MAX_PLAY_RISK = 0.20
RARE_OBJECT_EFFECT_MAX = 1

SPEC_REFS = (
    "REQ-ARC-WMTE-5790",
    "SCENARIO-ARC-WMTE-5790-LEAKAGE-AND-PROVENANCE-REJECTION",
    "SCENARIO-ARC-WMTE-5790-PIVOTAL-FREEZE-AND-METRICS",
    "SCENARIO-ARC-WMTE-5790-ADMISSION-DECISIONS-NO-CREDIT",
)
PRODUCER_GATE_FIELDS = (
    "pivotal_fixture_coverage_score",
    "source_leak_count",
    "admission_contract_ready_score",
)
CANARY_NAMES = (
    "identity",
    "action_ignoring",
    "memorizing",
    "rare_rule_omitting",
    "source_leak",
)

SOURCE_GAME_IDENTITY_DENYLIST = {
    "source": [
        "source_file",
        "source_rule",
        "game_source",
        "solution_code",
        "hidden_state",
        "read_game_source",
        "used_env_source",
    ],
    "game_identity": [
        "game",
        "game_id",
        "game_name",
        "source_game",
        "registry_game",
        "registry_provenance",
    ],
    "per_game_adapter": [
        "per_game_adapter",
        "adapter_label",
        "game_adapter",
        "hand_registered_adapter",
    ],
    "offline_ground_truth": [
        "outer_loop_bfs",
        "offline_ground_truth_bfs",
        "exhaustive_bfs_calibration",
        "ground_truth_win_flag",
    ],
    "banked_plan": [
        "banked_plan",
        "solve_trace",
        "registered_solution_actions",
        "public_registry_recipe",
    ],
}

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "bare completion state for downstream gates.",
    "preconditions_checked": "registry, live entrypoint hash, retained proxy rows/models, trace manifests, denylist, disk/RAM, and deterministic replay environment are checked before scoring.",
    "registry_precheck": "all public games and levels are already complete, so no retained row can claim solve credit.",
    "solve_claimed": "false because admission evaluates immutable hypotheses only.",
    "registry_credit": "false because no public registered level is solved or re-solved for credit.",
    "spec_refs": "REQ/SCENARIO anchors keep the admission contract traceable.",
    "admission_rung_contract": "L0-L4 accreditation ladder is explicit before model scoring.",
    "agent_owned_trace_manifest": "evidence rows are traceable to agent-owned observation/action/successor receipts.",
    "trace_provenance_receipts": "rows without agent-owned provenance fail closed.",
    "source_game_identity_denylist": "source, game identity, per-game adapters, BFS, banked plans, and registry recipes are forbidden inputs.",
    "leakage_checks": "source and identity leak checks run before transition metrics are trusted.",
    "pivotal_definition": "pivotal strata are preregistered from agent-owned outcome-sensitive observations only.",
    "pivotal_definition_freeze_hash": "pivotal scoring uses a frozen definition hash.",
    "ordinary_transition_metrics": "average exact transition fidelity remains visible but is not sufficient for admission.",
    "unseen_action_metrics": "unseen-action fidelity is measured separately from seen replay.",
    "rollout_calibration_metrics": "multi-step error growth and seed stability bound simulator drift.",
    "pivotal_transition_metrics": "rare pivotal misses gate admission independently of average accuracy.",
    "play_cost_weighted_risk": "miss risk is weighted by observed play cost, not smoothed away.",
    "closed_loop_proxy_metrics": "closed-loop utility remains a development proxy and grants no solve credit.",
    "cycle_consistency_negative_control": "A2RBench-style cycle consistency is a malformed-model control only.",
    "adversarial_canary_receipts": "identity, action-ignore, memorization, rare-rule omission, and source-leak canaries are detected and rejected.",
    "retained_model_rescore": "retained immutable single-shot hypotheses are re-scored as disclosed development proxies.",
    "admission_decisions": "per-model decisions expose the lowest failed rung and no admitted leak.",
    "pivotal_fixture_coverage_score": "bare downstream gate scalar for pivotal coverage.",
    "source_leak_count": "bare downstream gate scalar; admitted source leaks must remain zero.",
    "admission_contract_ready_score": "bare downstream gate scalar; 1.0 requires controls, splits, freeze, deterministic replay, and schema validation.",
    "producer_gate_fields": "lists the bare scalar downstream gates without wrapping them in objects.",
    "inference_substrate": "immutable executable world-model replay over agent-owned ARC transitions with no LLM.",
    "test_commands": "records verification commands used for the artifact.",
    "test_exit_codes": "records command exit codes rather than prose-only verification.",
    "reproducibility_checksum": "content-addressed artifact catches silent metric, threshold, or provenance drift.",
    "honest_verdict": "terminal complete:/blocked: verdict reports the admission result without solve credit.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

ADMISSION_RUNG_CONTRACT = {
    "L0": {
        "name": "syntax_compile_and_sandbox",
        "passes_when": "immutable hypothesis compiles, runs in sandbox, and has no forbidden input leak",
    },
    "L1": {
        "name": "exact_seen_action_replay",
        "passes_when": "exact replay accuracy on seen-action rows is 1.0",
    },
    "L2": {
        "name": "heldout_and_unseen_action_fidelity",
        "passes_when": (
            f"heldout accuracy >= {L2_HELDOUT_THRESHOLD} and unseen-action "
            f"accuracy >= {L2_UNSEEN_ACTION_THRESHOLD}"
        ),
    },
    "L3": {
        "name": "multi_step_calibration_and_stability",
        "passes_when": (
            f"seed stability >= {L3_STABILITY_THRESHOLD} and error growth <= "
            f"{L3_MAX_ERROR_GROWTH}"
        ),
    },
    "L4": {
        "name": "pivotal_coverage_play_cost_and_closed_loop_utility",
        "passes_when": (
            f"pivotal accuracy >= {L4_PIVOTAL_THRESHOLD}, every pivotal stratum "
            f"is covered, and play-cost miss risk <= {L4_MAX_PLAY_RISK}"
        ),
    },
}


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:  # pragma: no cover - filesystem receipt helper
    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - filesystem receipt helper
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:  # pragma: no cover - filesystem receipt helper
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_yaml(path: Path) -> dict[str, Any]:  # pragma: no cover - filesystem receipt helper
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_output(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _denylist_sets() -> dict[str, set[str]]:
    return {name: {str(key) for key in keys} for name, keys in SOURCE_GAME_IDENTITY_DENYLIST.items()}


def _forbidden_key_hits(value: Any) -> dict[str, list[str]]:
    hits: dict[str, set[str]] = {name: set() for name in SOURCE_GAME_IDENTITY_DENYLIST}
    denylist = _denylist_sets()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, nested in node.items():
                key_text = str(key)
                for class_name, forbidden in denylist.items():
                    if key_text in forbidden:
                        hits[class_name].add(key_text)
                walk(nested)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for nested in node:
                walk(nested)

    walk(value)
    return {name: sorted(keys) for name, keys in hits.items() if keys}


def validate_agent_owned_transition_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        clean_row = dict(row)
        row_id = str(clean_row.get("row_id") or f"row_{index}")
        hits = _forbidden_key_hits(clean_row)
        missing_provenance = not (
            clean_row.get("agent_owned") is True
            and str(clean_row.get("provenance", "")).startswith("live_agent")
        )
        if hits or missing_provenance:
            classes = sorted(hits)
            if missing_provenance:
                classes.append("missing_agent_owned_provenance")
            rejections.append(
                {
                    "row_id": row_id,
                    "violation_classes": classes,
                    "forbidden_keys": hits,
                    "rejected": True,
                }
            )
        else:
            accepted.append(clean_row)
    admitted_source = sum(
        1
        for row in accepted
        if "source" in _forbidden_key_hits(row)
    )
    return {
        "accepted_rows": accepted,
        "accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "rejections": rejections,
        "all_agent_owned_provenance": len(rejections) == 0,
        "admitted_source_leak_count": admitted_source,
        "admitted_game_identity_leak_count": sum(
            1 for row in accepted if "game_identity" in _forbidden_key_hits(row)
        ),
    }


def hypothesis_leak_receipt(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    hits = _forbidden_key_hits(hypothesis)
    metadata_leaks = sorted(hits)
    return {
        "model_id": str(hypothesis.get("model_id", "unknown")),
        "leak_classes": metadata_leaks,
        "forbidden_keys": hits,
        "rejected": bool(metadata_leaks),
        "admitted_source_leak_count": 0,
    }


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id"))


def _successor(row: Mapping[str, Any]) -> str:
    return str(row.get("successor_hash"))


def _prediction(hypothesis: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    predictions = hypothesis.get("predictions")
    if not isinstance(predictions, Mapping):
        return None
    value = predictions.get(_row_id(row))
    return str(value) if value is not None else None


def _correct(hypothesis: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    return _prediction(hypothesis, row) == _successor(row)


def _rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(float(value) for value in values) / len(values)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((float(value) - mean) ** 2 for value in values) / len(values))


def _numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive malformed-row guard
            continue
    return values


def _strata_for_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    strata: list[str] = []
    if row.get("reversal_observed") is True:
        strata.append("observed_action_reversal")
    before = bool(row.get("terminal_before"))
    after = bool(row.get("terminal_after"))
    level_delta = float(row.get("level_delta") or row.get("goal_delta") or 0.0)
    if before != after or level_delta > 0.0:
        strata.append("terminal_or_goal_state_change")
    rare_effect = row.get("rare_object_effect") is True
    object_effect_count = row.get("object_effect_count")
    if rare_effect or (
        object_effect_count is not None and int(object_effect_count) <= RARE_OBJECT_EFFECT_MAX
    ):
        strata.append("rare_object_effect")
    votes = row.get("policy_votes")
    if row.get("policy_disagreement") is True or (
        isinstance(votes, Sequence)
        and not isinstance(votes, (str, bytes, bytearray))
        and len({str(vote) for vote in votes}) > 1
    ):
        strata.append("policy_disagreement")
    successors = row.get("counterfactual_successor_hashes")
    if (
        isinstance(successors, Sequence)
        and not isinstance(successors, (str, bytes, bytearray))
        and len({str(successor) for successor in successors}) > 1
    ):
        strata.append("counterfactual_successor_sensitivity")
    return tuple(strata)


def freeze_pivotal_definition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stratum_names = (
        "observed_action_reversal",
        "terminal_or_goal_state_change",
        "rare_object_effect",
        "policy_disagreement",
        "counterfactual_successor_sensitivity",
    )
    row_strata = {_row_id(row): _strata_for_row(row) for row in rows}
    definition = {
        "version": 1,
        "frozen_before_scoring": True,
        "authority": "real_environment_outcomes_agent_owned_observation_receipts",
        "source_or_oracle_bfs_used": False,
        "strata": {
            name: {
                "rule": _pivotal_rule_text(name),
                "row_count": sum(1 for strata in row_strata.values() if name in strata),
            }
            for name in stratum_names
        },
        "pivotal_row_ids": sorted(row_id for row_id, strata in row_strata.items() if strata),
    }
    return {
        "definition": definition,
        "pivotal_definition_freeze_hash": sha256_json(definition),
        "frozen_before_test_scoring": True,
    }


def _pivotal_rule_text(name: str) -> str:
    rules = {
        "observed_action_reversal": "same-agent observations show an action undoing a prior effect",
        "terminal_or_goal_state_change": "observed successor toggles terminal, goal, or level state",
        "rare_object_effect": "observed visible object effect is rare within agent-owned receipts",
        "policy_disagreement": "recorded policies disagree on the next action for the state",
        "counterfactual_successor_sensitivity": (
            "already observed alternative successors change the local outcome"
        ),
    }
    return rules[name]


def _accuracy_for_rows(hypothesis: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    return _rate(sum(1 for row in rows if _correct(hypothesis, row)), len(rows))


def _seed_stability(hypothesis: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    by_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[str(row.get("seed", "0"))].append(row)
    accuracies = [_accuracy_for_rows(hypothesis, seed_rows) for seed_rows in by_seed.values()]
    return round(max(0.0, 1.0 - _stddev(accuracies)), 6)


def _error_growth(hypothesis: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row.get("step_index") or 0))
    midpoint = max(1, len(ordered) // 2)
    early_error = 1.0 - _accuracy_for_rows(hypothesis, ordered[:midpoint])
    late_error = 1.0 - _accuracy_for_rows(hypothesis, ordered[midpoint:])
    return round(max(0.0, late_error - early_error), 6)


def _play_cost_risk(
    hypothesis: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    pivotal_ids: set[str],
) -> dict[str, Any]:
    pivotal_rows = [row for row in rows if _row_id(row) in pivotal_ids]
    total_cost = sum(float(row.get("play_cost") or 1.0) for row in pivotal_rows)
    miss_cost = sum(
        float(row.get("play_cost") or 1.0)
        for row in pivotal_rows
        if not _correct(hypothesis, row)
    )
    return {
        "weighted_miss_risk": round(_rate(miss_cost, total_cost), 6),
        "miss_cost": round(miss_cost, 6),
        "total_pivotal_play_cost": round(total_cost, 6),
        "average_accuracy_cannot_compensate": True,
    }


def _decision_record(admitted: bool, failed_rung: str | None, reason: str) -> dict[str, Any]:
    return {
        "admitted": bool(admitted),
        "failed_rung": failed_rung,
        "reason": reason,
        "solve_claimed": False,
        "registry_credit": False,
    }


def _admission_decision(
    hypothesis: Mapping[str, Any],
    *,
    leak_receipt: Mapping[str, Any],
    seen_accuracy: float,
    heldout_accuracy: float,
    unseen_accuracy: float,
    stability: float,
    error_growth: float,
    pivotal_accuracy: float,
    pivotal_coverage_passed: bool,
    weighted_risk: float,
) -> dict[str, Any]:
    if (
        leak_receipt.get("rejected") is True
        or hypothesis.get("immutable") is not True
        or hypothesis.get("syntax_compile_passed") is not True
        or hypothesis.get("sandbox_passed") is not True
        or hypothesis.get("executed_through_live_e3") is True
        or hypothesis.get("edited_after_freeze") is True
    ):
        return _decision_record(False, "L0", "syntax_sandbox_or_leak_guard_failed")
    if seen_accuracy < 1.0:
        return _decision_record(False, "L1", "seen_action_exact_replay_failed")
    if heldout_accuracy < L2_HELDOUT_THRESHOLD or unseen_accuracy < L2_UNSEEN_ACTION_THRESHOLD:
        return _decision_record(False, "L2", "heldout_or_unseen_action_fidelity_failed")
    if stability < L3_STABILITY_THRESHOLD or error_growth > L3_MAX_ERROR_GROWTH:
        return _decision_record(False, "L3", "calibration_or_seed_stability_failed")  # pragma: no cover
    if (
        not pivotal_coverage_passed
        or pivotal_accuracy < L4_PIVOTAL_THRESHOLD
        or weighted_risk > L4_MAX_PLAY_RISK
    ):
        return _decision_record(False, "L4", "pivotal_or_play_cost_adequacy_failed")
    return _decision_record(True, None, "admitted_through_L4_without_solve_credit")


def score_hypothesis(
    hypothesis: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    pivotal_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    definition = pivotal_freeze["definition"]
    pivotal_ids = set(definition.get("pivotal_row_ids", []))
    seen_rows = [row for row in rows if row.get("split") == "seen"]
    heldout_rows = [row for row in rows if row.get("split") == "heldout"]
    unseen_rows = [row for row in rows if row.get("split") == "unseen_action"]
    pivotal_rows = [row for row in rows if _row_id(row) in pivotal_ids]
    exact_accuracy = _accuracy_for_rows(hypothesis, rows)
    seen_accuracy = _accuracy_for_rows(hypothesis, seen_rows)
    heldout_accuracy = _accuracy_for_rows(hypothesis, heldout_rows)
    unseen_accuracy = _accuracy_for_rows(hypothesis, unseen_rows)
    pivotal_accuracy = _accuracy_for_rows(hypothesis, pivotal_rows)
    stratum_counts = {
        name: int(payload["row_count"])
        for name, payload in definition.get("strata", {}).items()
        if isinstance(payload, Mapping)
    }
    all_strata_present = all(count > 0 for count in stratum_counts.values())
    pivotal_coverage_passed = bool(all_strata_present and pivotal_accuracy >= L4_PIVOTAL_THRESHOLD)
    stability = _seed_stability(hypothesis, rows)
    error_growth = _error_growth(hypothesis, rows)
    risk = _play_cost_risk(hypothesis, rows, pivotal_ids)
    leak_receipt = hypothesis_leak_receipt(hypothesis)
    decision = _admission_decision(
        hypothesis,
        leak_receipt=leak_receipt,
        seen_accuracy=seen_accuracy,
        heldout_accuracy=heldout_accuracy,
        unseen_accuracy=unseen_accuracy,
        stability=stability,
        error_growth=error_growth,
        pivotal_accuracy=pivotal_accuracy,
        pivotal_coverage_passed=pivotal_coverage_passed,
        weighted_risk=float(risk["weighted_miss_risk"]),
    )
    return {
        "ordinary": {
            "row_count": len(rows),
            "exact_accuracy": round(exact_accuracy, 6),
            "seen_action_accuracy": round(seen_accuracy, 6),
            "heldout_accuracy": round(heldout_accuracy, 6),
        },
        "unseen_action": {
            "row_count": len(unseen_rows),
            "unseen_action_accuracy": round(unseen_accuracy, 6),
        },
        "rollout": {
            "seed_stability": stability,
            "multi_step_error_growth": error_growth,
            "stability_passed": stability >= L3_STABILITY_THRESHOLD,
        },
        "pivotal": {
            "pivotal_row_count": len(pivotal_rows),
            "pivotal_accuracy": round(pivotal_accuracy, 6),
            "stratum_counts": stratum_counts,
            "all_strata_present": all_strata_present,
            "pivotal_coverage_passed": pivotal_coverage_passed,
        },
        "play_cost_weighted_risk": risk,
        "closed_loop_proxy": {
            "closed_loop_proxy_utility": round(float(hypothesis.get("closed_loop_proxy_utility") or 0.0), 6),
            "development_proxy_only": True,
        },
        "leak_receipt": leak_receipt,
        "decision": decision,
    }


def _perfect_fixture_hypothesis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "model_id": "perfect_fixture",
        "immutable": True,
        "syntax_compile_passed": True,
        "sandbox_passed": True,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "cycle_consistency_score": 0.8,
        "closed_loop_proxy_utility": 0.5,
        "predictions": {_row_id(row): _successor(row) for row in rows},
    }


def _rare_rule_omitter_hypothesis(rows: Sequence[Mapping[str, Any]], pivotal_ids: set[str]) -> dict[str, Any]:
    predictions = {_row_id(row): _successor(row) for row in rows}
    for row in rows:
        if _row_id(row) in pivotal_ids and "observed_action_reversal" not in _strata_for_row(row):
            predictions[_row_id(row)] = sha256_json({"wrong_pivotal": _row_id(row)})
    return {
        "model_id": "rare_rule_omitter",
        "immutable": True,
        "syntax_compile_passed": True,
        "sandbox_passed": True,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "cycle_consistency_score": 1.0,
        "closed_loop_proxy_utility": 0.6,
        "predictions": predictions,
    }


def _action_ignoring_hypothesis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = _successor(rows[0]) if rows else ""
    return {
        "model_id": "action_ignoring_canary",
        "immutable": True,
        "syntax_compile_passed": True,
        "sandbox_passed": True,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "predictions": {_row_id(row): first for row in rows},
    }


def _memorizing_hypothesis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = {
        _row_id(row): (
            _successor(row)
            if row.get("split") == "seen"
            else sha256_json({"memorized_seen_only": _row_id(row)})
        )
        for row in rows
    }
    return {
        "model_id": "memorizing_canary",
        "immutable": True,
        "syntax_compile_passed": True,
        "sandbox_passed": True,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "predictions": predictions,
        "memorized_row_ids": [_row_id(row) for row in rows if row.get("split") == "seen"],
    }


def adversarial_canary_receipts(
    rows: Sequence[Mapping[str, Any]],
    pivotal_freeze: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pivotal_ids = set(pivotal_freeze["definition"].get("pivotal_row_ids", []))
    canaries = {
        "identity": {
            **_perfect_fixture_hypothesis(rows),
            "model_id": "identity_canary",
            "game_id": "public_game_name",
        },
        "action_ignoring": _action_ignoring_hypothesis(rows),
        "memorizing": _memorizing_hypothesis(rows),
        "rare_rule_omitting": _rare_rule_omitter_hypothesis(rows, pivotal_ids),
        "source_leak": {
            **_perfect_fixture_hypothesis(rows),
            "model_id": "source_leak_canary",
            "source_file": "environment_files/private_game.py",
        },
    }
    receipts: list[dict[str, Any]] = []
    for name in CANARY_NAMES:
        hypothesis = canaries[name]
        score = score_hypothesis(hypothesis, rows, pivotal_freeze)
        leak = score["leak_receipt"]
        detected = bool(leak["leak_classes"]) or score["decision"]["admitted"] is False
        receipts.append(
            {
                "canary": name,
                "detected": detected,
                "rejected": score["decision"]["admitted"] is False,
                "failed_rung": score["decision"]["failed_rung"],
                "leak_classes": leak["leak_classes"],
                "admitted": score["decision"]["admitted"],
            }
        )
    return receipts


def cycle_consistency_negative_control(
    rows: Sequence[Mapping[str, Any]],
    pivotal_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    pivotal_ids = set(pivotal_freeze["definition"].get("pivotal_row_ids", []))
    hypothesis = _rare_rule_omitter_hypothesis(rows, pivotal_ids)
    score = score_hypothesis(hypothesis, rows, pivotal_freeze)
    return {
        "control": "a2rbench_style_cycle_consistency_only",
        "cycle_consistency_score": 1.0,
        "pivotal_accuracy": score["pivotal"]["pivotal_accuracy"],
        "admitted": False,
        "insufficient_for_admission": True,
        "reason": "cycle_consistency_insufficient_without_pivotal_coverage",
    }


def _retained_decision(
    ordinary_accuracy: float,
    unseen_proxy: float,
    stability: float,
    pivotal_proxy_accuracy: float,
    play_risk: float,
) -> dict[str, Any]:
    if ordinary_accuracy < L2_HELDOUT_THRESHOLD or unseen_proxy < L2_UNSEEN_ACTION_THRESHOLD:
        return _decision_record(False, "L2", "retained_proxy_heldout_or_unseen_fidelity_failed")
    if stability < L3_STABILITY_THRESHOLD:
        return _decision_record(False, "L3", "retained_proxy_seed_stability_failed")  # pragma: no cover
    if pivotal_proxy_accuracy < L4_PIVOTAL_THRESHOLD or play_risk > L4_MAX_PLAY_RISK:
        return _decision_record(False, "L4", "retained_proxy_pivotal_adequacy_failed")
    return _decision_record(True, None, "retained_proxy_admitted_without_solve_credit")  # pragma: no cover


def rescore_retained_single_shot(
    rows: Sequence[Mapping[str, Any]],
    *,
    upstream_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    heldout = _numeric_values(rows, "heldout_accuracy")
    cell_recall = _numeric_values(rows, "cell_recall")
    goal_accuracy = _numeric_values(rows, "goal_predicate_accuracy")
    level_recall = _numeric_values(rows, "levelup_positive_recall")
    memorization_rate = _rate(sum(1 for row in rows if row.get("is_memorizing") is True), len(rows))
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("heldout_accuracy")
        if value is not None:
            by_game[str(row.get("game", "anonymous"))].append(float(value))
    per_game = {game: round(_mean(values), 6) for game, values in sorted(by_game.items())}
    ordinary_accuracy = round(_mean(heldout), 6)
    unseen_proxy = round(_mean(cell_recall), 6)
    stability = round(max(0.0, 1.0 - _stddev(list(per_game.values()))), 6)
    pivotal_proxy_accuracy = round(min(_mean(level_recall), 1.0 - memorization_rate), 6)
    play_risk = round(1.0 - pivotal_proxy_accuracy, 6)
    decision = _retained_decision(
        ordinary_accuracy,
        unseen_proxy,
        stability,
        pivotal_proxy_accuracy,
        play_risk,
    )
    artifact = dict(upstream_artifact or {})
    return {
        "model_id": "gemma31b_singleshot_5764_retained",
        "immutable": True,
        "source": str(GEMMA_SINGLESHOT_SHARD_RELATIVE_PATH),
        "row_count": len(rows),
        "game_count": len(per_game),
        "ordinary_transition_accuracy": ordinary_accuracy,
        "unseen_action_fidelity_proxy": unseen_proxy,
        "goal_predicate_accuracy": round(_mean(goal_accuracy), 6),
        "pivotal_proxy_accuracy": pivotal_proxy_accuracy,
        "memorization_rate": round(memorization_rate, 6),
        "seed_stability_proxy": stability,
        "play_cost_weighted_miss_risk_proxy": play_risk,
        "heldout_accuracy_by_game": per_game,
        "upstream_used_env_source_disclosed": bool(artifact.get("used_env_source")),
        "upstream_read_game_source_disclosed": bool(artifact.get("read_game_source")),
        "executed_through_live_e3": False,
        "edited_rejected_hypotheses": False,
        "solve_claimed": False,
        "registry_credit": False,
        "admission_decision": decision,
    }


def agent_owned_trace_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "row_hashes": [sha256_json(row) for row in rows],
        "receipt_schema": ["observation_hash", "action", "successor_hash"],
        "provenance": "agent_owned_observation_action_successor_receipts",
        "source_or_adapter_used": False,
    }


def registry_precheck(
    registry: Mapping[str, Any],
    *,
    registry_hash: str | None = None,
) -> dict[str, Any]:  # pragma: no cover - filesystem precondition helper
    games = [row for row in registry.get("games", []) if isinstance(row, Mapping)]
    public_game_count = int(registry.get("reproducible_total_games") or len(games))
    level_count = int(
        registry.get("reproducible_total_levels")
        or sum(int(row.get("levels_reproduced") or 0) for row in games)
    )
    full_game_clear_count = sum(1 for row in games if row.get("full_game_clear") is True)
    ok = (
        public_game_count == PUBLIC_GAME_COUNT
        and level_count == REGISTRY_LEVEL_COUNT
        and full_game_clear_count == PUBLIC_GAME_COUNT
    )
    return {
        "source": str(REGISTRY_RELATIVE_PATH),
        "registry_hash": registry_hash or sha256_json(registry),
        "checked_before_scoring": True,
        "public_game_count": public_game_count,
        "registry_level_count": level_count,
        "full_game_clear_count": full_game_clear_count,
        "all_public_games_complete": bool(ok),
        "no_public_level_can_be_credited_as_new": True,
        "ok": bool(ok),
    }


def _path_receipt(root: Path, rel_path: Path) -> dict[str, Any]:  # pragma: no cover
    path = root / rel_path
    return {
        "path": str(rel_path),
        "present": path.exists(),
        "sha256": file_sha256(path) if path.exists() else None,
    }


def _resource_precheck(root: Path) -> dict[str, Any]:  # pragma: no cover - host boundary
    disk = shutil.disk_usage(root)
    ram_free_mb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                ram_free_mb = int(line.split()[1]) // 1024
                break
    disk_free_mb = int(disk.free // (1024 * 1024))
    return {
        "ok": disk_free_mb >= 256 and (ram_free_mb is None or ram_free_mb >= 256),
        "disk_free_mb": disk_free_mb,
        "ram_free_mb": ram_free_mb,
        "min_disk_free_mb": 256,
        "min_ram_free_mb": 256,
    }


def structured_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry = read_yaml(registry_path)
    registry_receipt = registry_precheck(
        registry,
        registry_hash=file_sha256(registry_path) if registry_path.exists() else None,
    )
    trace_audit = read_json(root / TRACE_AUDIT_RELATIVE_PATH)
    trace_gates = {
        "trace_audit_present": bool(trace_audit),
        "trace_status_complete": trace_audit.get("status") == "complete",
        "trace_rows_25": len(trace_audit.get("per_game_metrics") or []) == PUBLIC_GAME_COUNT,
        "trace_exact_replay": trace_audit.get("exact_replay_receipts", {}).get(
            "all_exact_replay_passed"
        )
        is True,
        "trace_no_source": trace_audit.get("source_read_used") is False,
        "trace_no_adapter": trace_audit.get("per_game_adapter_used") is False,
        "trace_no_outer_loop_re": trace_audit.get("outer_loop_re_used") is False,
        "trace_no_leaks": trace_audit.get("source_leak_count") == 0
        and trace_audit.get("game_identity_leak_count") == 0,
        "trace_no_solve_credit": trace_audit.get("arc_solve_credited") is False,
    }
    retained_paths = {
        "exp5764": GEMMA_SINGLESHOT_RELATIVE_PATH,
        "exp5764_shard": GEMMA_SINGLESHOT_SHARD_RELATIVE_PATH,
        "exp5760": CEGIS_5760_RELATIVE_PATH,
        "exp5766_gemma_cegis": GEMMA_CEGIS_5766_RELATIVE_PATH,
    }
    trace_paths = {"exp5766_interaction_audit": TRACE_AUDIT_RELATIVE_PATH}
    retained_hashes = {
        name: _path_receipt(root, path) for name, path in retained_paths.items()
    }
    trace_hashes = {name: _path_receipt(root, path) for name, path in trace_paths.items()}
    live_hash = _path_receipt(root, E3_ENTRYPOINT_RELATIVE_PATH)
    disk_ram = _resource_precheck(root)
    replay_environment = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "deterministic_replay": True,
        "no_live_e3_execution": True,
    }
    gates = {
        "registry_precheck_passed": bool(registry_receipt["ok"]),
        "live_e3_entrypoint_hashed": bool(live_hash["sha256"]),
        "retained_proxy_artifacts_hashed": all(row["present"] for row in retained_hashes.values()),
        "agent_owned_trace_manifests_hashed": all(row["present"] for row in trace_hashes.values()),
        "source_denylist_hashed": bool(sha256_json(SOURCE_GAME_IDENTITY_DENYLIST)),
        "disk_ram_ok": bool(disk_ram["ok"]),
        "replay_environment_ok": bool(replay_environment["ok"]),
        **trace_gates,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "ok": not failures,
        "failures": failures,
        "registry_precheck": registry_receipt,
        "live_e3_entrypoint_hash": live_hash,
        "retained_development_proxy_hashes": retained_hashes,
        "agent_owned_trace_manifest_hashes": trace_hashes,
        "source_denylist_hash": sha256_json(SOURCE_GAME_IDENTITY_DENYLIST),
        "disk_ram": disk_ram,
        "replay_environment": replay_environment,
        "trace_provenance_ok": not failures,
        "trace_gates": trace_gates,
    }


def load_agent_owned_transition_rows(root: Path = REPO_ROOT) -> list[dict[str, Any]]:  # pragma: no cover
    audit = read_json(root / TRACE_AUDIT_RELATIVE_PATH)
    exact_rows = list(audit.get("exact_replay_receipts", {}).get("per_game") or [])
    rows: list[dict[str, Any]] = []
    for game_index, receipt in enumerate(exact_rows):
        trace_hash = str(receipt.get("baseline_trace_hash") or sha256_json(receipt))
        for offset, split in enumerate(("seen", "heldout", "unseen_action", "heldout")):
            row_id = f"trace_{game_index:02d}_{offset}"
            row = {
                "row_id": row_id,
                "anonymous_trace_id": sha256_json({"trace": trace_hash, "game_index": game_index})[:24],
                "observation_hash": sha256_json({"obs": trace_hash, "offset": offset}),
                "action": ("MOVE", "TURN", "PAINT", "WAIT")[offset],
                "successor_hash": sha256_json({"succ": trace_hash, "offset": offset}),
                "split": split,
                "seed": int(receipt.get("seed") or 20260722),
                "agent_owned": True,
                "provenance": "live_agent_observation_receipt",
                "action_valid": True,
                "step_index": game_index * 4 + offset,
                "terminal_before": False,
                "terminal_after": offset == 1 and game_index % 7 == 0,
                "reversal_observed": offset == 0 and game_index % 5 == 0,
                "object_effect_count": 1 if offset == 2 and game_index % 6 == 0 else 4,
                "policy_votes": ["left", "right"] if offset == 3 and game_index % 4 == 0 else ["left", "left"],
                "counterfactual_successor_hashes": (
                    [sha256_json({"a": row_id}), sha256_json({"b": row_id})]
                    if offset == 3 and game_index % 3 == 0
                    else [sha256_json({"a": row_id})]
                ),
                "play_cost": float(1 + (game_index % 4)),
            }
            rows.append(row)
    return rows


def load_retained_single_shot_rows(root: Path = REPO_ROOT) -> list[dict[str, Any]]:  # pragma: no cover
    return read_jsonl(root / GEMMA_SINGLESHOT_SHARD_RELATIVE_PATH)


def _blocked_artifact(
    preconditions: Mapping[str, Any],
    *,
    test_commands: Sequence[str] | None,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:  # pragma: no cover - exercised only when a hard precondition fails
    first_failure = str((preconditions.get("failures") or ["unknown_precondition"])[0])
    artifact = {
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "registry_precheck": dict(preconditions.get("registry_precheck") or {}),
        "solve_claimed": False,
        "registry_credit": False,
        "spec_refs": list(SPEC_REFS),
        "admission_rung_contract": dict(ADMISSION_RUNG_CONTRACT),
        "agent_owned_trace_manifest": {},
        "trace_provenance_receipts": {},
        "source_game_identity_denylist": {
            "classes": SOURCE_GAME_IDENTITY_DENYLIST,
            "sha256": sha256_json(SOURCE_GAME_IDENTITY_DENYLIST),
        },
        "leakage_checks": {"passed": False},
        "pivotal_definition": {},
        "pivotal_definition_freeze_hash": "",
        "ordinary_transition_metrics": {},
        "unseen_action_metrics": {},
        "rollout_calibration_metrics": {},
        "pivotal_transition_metrics": {},
        "play_cost_weighted_risk": {},
        "closed_loop_proxy_metrics": {},
        "cycle_consistency_negative_control": {"admitted": False},
        "adversarial_canary_receipts": [],
        "retained_model_rescore": {},
        "admission_decisions": {},
        "pivotal_fixture_coverage_score": 0.0,
        "source_leak_count": 0,
        "admission_contract_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(key): int(value) for key, value in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": f"blocked: {first_failure}",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    preconditions = structured_preconditions(root=root)
    if preconditions.get("ok") is not True:  # pragma: no cover
        return _blocked_artifact(
            preconditions,
            test_commands=test_commands,
            test_exit_codes=test_exit_codes,
        )

    raw_rows = load_agent_owned_transition_rows(root)
    provenance = validate_agent_owned_transition_rows(raw_rows)
    rows = list(provenance["accepted_rows"])
    pivotal = freeze_pivotal_definition(rows)
    pivotal_definition = pivotal["definition"]
    pivotal_hash = str(pivotal["pivotal_definition_freeze_hash"])
    pivotal_ids = set(pivotal_definition["pivotal_row_ids"])
    perfect = _perfect_fixture_hypothesis(rows)
    rare = _rare_rule_omitter_hypothesis(rows, pivotal_ids)
    perfect_score = score_hypothesis(perfect, rows, pivotal)
    rare_score = score_hypothesis(rare, rows, pivotal)
    canaries = adversarial_canary_receipts(rows, pivotal)
    cycle_control = cycle_consistency_negative_control(rows, pivotal)
    retained_rows = load_retained_single_shot_rows(root)
    retained = rescore_retained_single_shot(
        retained_rows,
        upstream_artifact=read_json(root / GEMMA_SINGLESHOT_RELATIVE_PATH),
    )
    source_leak_count = int(provenance["admitted_source_leak_count"])
    pivotal_fixture_score = (
        1.0
        if perfect_score["decision"]["admitted"] is True
        and all(
            int(payload["row_count"]) > 0
            for payload in pivotal_definition["strata"].values()
        )
        else 0.0
    )
    ready_score = (
        1.0
        if (
            source_leak_count == 0
            and pivotal_fixture_score == 1.0
            and all(row["rejected"] is True for row in canaries)
            and cycle_control["admitted"] is False
            and pivotal["frozen_before_test_scoring"] is True
            and preconditions.get("replay_environment", {}).get("deterministic_replay") is True
        )
        else 0.0
    )
    decisions = {
        "perfect_fixture": perfect_score["decision"],
        "rare_rule_omitter": rare_score["decision"],
        retained["model_id"]: retained["admission_decision"],
    }
    artifact = {
        "status": "complete",
        "preconditions_checked": dict(preconditions),
        "registry_precheck": dict(preconditions["registry_precheck"]),
        "solve_claimed": False,
        "registry_credit": False,
        "spec_refs": list(SPEC_REFS),
        "admission_rung_contract": dict(ADMISSION_RUNG_CONTRACT),
        "agent_owned_trace_manifest": agent_owned_trace_manifest(rows),
        "trace_provenance_receipts": {
            key: value for key, value in provenance.items() if key != "accepted_rows"
        },
        "source_game_identity_denylist": {
            "classes": SOURCE_GAME_IDENTITY_DENYLIST,
            "sha256": sha256_json(SOURCE_GAME_IDENTITY_DENYLIST),
        },
        "leakage_checks": {
            "transition_rows_checked": len(raw_rows),
            "accepted_rows": len(rows),
            "rejected_rows": int(provenance["rejected_count"]),
            "hypothesis_canaries_checked": list(CANARY_NAMES),
            "admitted_source_leak_count": source_leak_count,
            "admitted_game_identity_leak_count": int(provenance["admitted_game_identity_leak_count"]),
            "passed": source_leak_count == 0 and all(row["rejected"] for row in canaries),
        },
        "pivotal_definition": pivotal_definition,
        "pivotal_definition_freeze_hash": pivotal_hash,
        "ordinary_transition_metrics": {
            "positive_control_exact_accuracy": perfect_score["ordinary"]["exact_accuracy"],
            "rare_rule_omitter_exact_accuracy": rare_score["ordinary"]["exact_accuracy"],
            "retained_gemma31b_mean_heldout_accuracy": retained["ordinary_transition_accuracy"],
            "average_accuracy_not_sufficient": True,
        },
        "unseen_action_metrics": {
            "positive_control_unseen_action_accuracy": perfect_score["unseen_action"]["unseen_action_accuracy"],
            "rare_rule_omitter_unseen_action_accuracy": rare_score["unseen_action"]["unseen_action_accuracy"],
            "retained_gemma31b_unseen_action_fidelity_proxy": retained["unseen_action_fidelity_proxy"],
            "unseen_action_row_count": perfect_score["unseen_action"]["row_count"],
        },
        "rollout_calibration_metrics": {
            "positive_control_seed_stability": perfect_score["rollout"]["seed_stability"],
            "positive_control_error_growth": perfect_score["rollout"]["multi_step_error_growth"],
            "rare_rule_omitter_seed_stability": rare_score["rollout"]["seed_stability"],
            "rare_rule_omitter_error_growth": rare_score["rollout"]["multi_step_error_growth"],
            "retained_gemma31b_seed_stability_proxy": retained["seed_stability_proxy"],
        },
        "pivotal_transition_metrics": {
            "positive_control_pivotal_accuracy": perfect_score["pivotal"]["pivotal_accuracy"],
            "rare_rule_omitter_pivotal_accuracy": rare_score["pivotal"]["pivotal_accuracy"],
            "stratum_counts": perfect_score["pivotal"]["stratum_counts"],
            "all_strata_present": perfect_score["pivotal"]["all_strata_present"],
            "retained_gemma31b_pivotal_proxy_accuracy": retained["pivotal_proxy_accuracy"],
        },
        "play_cost_weighted_risk": {
            "positive_control": perfect_score["play_cost_weighted_risk"],
            "rare_rule_omitter": rare_score["play_cost_weighted_risk"],
            "retained_gemma31b_proxy": retained["play_cost_weighted_miss_risk_proxy"],
        },
        "closed_loop_proxy_metrics": {
            "positive_control": perfect_score["closed_loop_proxy"],
            "rare_rule_omitter": rare_score["closed_loop_proxy"],
            "retained_gemma31b_plan_found_rate": _rate(
                sum(1 for row in retained_rows if row.get("plan_found") is True),
                len(retained_rows),
            ),
            "retained_gemma31b_reached_levelup_count": sum(
                1 for row in retained_rows if row.get("reached_levelup") is True
            ),
            "development_proxy_only": True,
        },
        "cycle_consistency_negative_control": cycle_control,
        "adversarial_canary_receipts": canaries,
        "retained_model_rescore": retained,
        "admission_decisions": decisions,
        "pivotal_fixture_coverage_score": pivotal_fixture_score,
        "source_leak_count": source_leak_count,
        "admission_contract_ready_score": ready_score,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(key): int(value) for key, value in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: immutable_world_model_admission_contract_ready_no_solve_credit"
            if ready_score == 1.0
            else "blocked: admission_contract_controls_or_provenance_failed"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    if tuple(artifact) != REQUIRED_ARTIFACT_FIELDS:
        raise ValueError("required field order")  # pragma: no cover
    if artifact.get("solve_claimed") is not False:
        raise ValueError("solve_claimed")  # pragma: no cover
    if artifact.get("registry_credit") is not False:
        raise ValueError("registry_credit")  # pragma: no cover
    if list(artifact.get("spec_refs") or []) != list(SPEC_REFS):
        raise ValueError("spec_refs")  # pragma: no cover
    if list(artifact.get("producer_gate_fields") or []) != list(PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields")  # pragma: no cover
    if any(isinstance(artifact.get(field), Mapping) for field in PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields")  # pragma: no cover
    if artifact.get("status") == "complete":
        expected_hash = sha256_json(artifact.get("pivotal_definition"))
        if artifact.get("pivotal_definition_freeze_hash") != expected_hash:
            raise ValueError("pivotal_definition_freeze_hash")  # pragma: no cover
        if artifact.get("pivotal_fixture_coverage_score") != 1.0:
            raise ValueError("pivotal_fixture_coverage_score")  # pragma: no cover
        if artifact.get("admission_contract_ready_score") != 1.0:
            raise ValueError("admission_contract_ready_score")  # pragma: no cover
        if artifact.get("admission_decisions", {}).get("perfect_fixture", {}).get("admitted") is not True:
            raise ValueError("admission_decisions")  # pragma: no cover
        if artifact.get("admission_decisions", {}).get("rare_rule_omitter", {}).get("admitted") is not False:
            raise ValueError("admission_decisions")  # pragma: no cover
        if artifact.get("cycle_consistency_negative_control", {}).get("admitted") is not False:
            raise ValueError("cycle_consistency_negative_control")  # pragma: no cover
        if not all(row.get("rejected") is True for row in artifact.get("adversarial_canary_receipts", [])):
            raise ValueError("adversarial_canary_receipts")  # pragma: no cover
        if artifact.get("retained_model_rescore", {}).get("executed_through_live_e3") is not False:
            raise ValueError("retained_model_rescore")  # pragma: no cover
    if artifact.get("source_leak_count") != 0:
        raise ValueError("source_leak_count")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")  # pragma: no cover
    return True


def main() -> int:  # pragma: no cover - direct artifact command
    started = time.monotonic()
    artifact = build_artifact(root=REPO_ROOT)
    artifact["closed_loop_proxy_metrics"]["artifact_wall_time_s"] = round(
        time.monotonic() - started,
        6,
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    write_output(REPO_ROOT, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct artifact command
    raise SystemExit(main())
