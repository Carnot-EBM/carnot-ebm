"""Use the bounded belief query as an optional E3 candidate ranker.

The selector reads only fields that exist before an action. It never reads a
game source, adapter, registry, or outcome label. Unsafe evidence causes one
decision-level abstention and preserves the incoming candidate order.

Spec refs: REQ-ARC-WMTE-7024 and SCENARIO-ARC-WMTE-7024-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.agentic import arc_belief_ledger as ledger_mod
from carnot.agentic import arc_belief_query as query_mod


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7024
SCHEMA = "carnot.exp7024.belief_aware_e3_selector.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_242_026_090_5
INFERENCE_SUBSTRATE = "deterministic_live_policy_wiring_fixtures_no_llm"
BELIEF_SELECTOR_ENV_FLAG = "CARNOT_ARC_BELIEF_AWARE_E3_SELECTOR"
SUBMITTED_BELIEF_AWARE_SELECTOR_ENABLED = False
DEFAULT_BELIEF_WEIGHT = 0.25
FROZEN_OFF_TRACE_SHA256 = "sha256:0d91eb1c095d103ceedba01caa26905812d872907a723f49fa4c523611d98820"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path("results/experiment_7024_belief_aware_e3_selector.json")
WRAPPER_PATH = Path("scripts/experiments/experiment_7024_belief_aware_e3_selector.py")
TEST_PATH = Path("tests/python/test_experiment_7024_belief_aware_e3_selector.py")
MODULE_PATH = Path("python/carnot/agentic/arc_belief_aware_e3_selector.py")
EXP7023_PATH = Path("results/experiment_7023_belief_query_api.json")
EXPECTED_EXP7023_HASH = "sha256:f1890eb3dffba8caed3dfeccbbe3cff3219693755f1fd0d99d059b50f0612d9d"

SOURCE_PATHS = {
    "exp7023": EXP7023_PATH,
    "belief_query_module": Path("python/carnot/agentic/arc_belief_query.py"),
    "belief_ledger_module": Path("python/carnot/agentic/arc_belief_ledger.py"),
    "competition_policy_module": Path("python/carnot/agentic/arc_competition_agent.py"),
    "action_provenance_module": Path("python/carnot/agentic/arc_action_provenance.py"),
    "eval_provenance_module": Path("python/carnot/agentic/arc_eval_provenance.py"),
    "live_runner_binding_module": Path(
        "python/carnot/agentic/arc_live_runner_execution_binding.py"
    ),
    "selector_module": MODULE_PATH,
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "cited_upstream_artifacts",
    "source_artifact_hashes",
    "rows",
    "factory_wiring_rows",
    "constructor_wiring_rows",
    "flag_rows",
    "flag_off_equivalence_rows",
    "query_firing_rows",
    "candidate_score_rows",
    "ranking_influence_rows",
    "abstention_rows",
    "malformed_evidence_rows",
    "action_provenance_rows",
    "live_path_reachability_rows",
    "shipped_default_unchanged",
    "belief_selector_live_path_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

ROW_TABLES = (
    "rows",
    "factory_wiring_rows",
    "constructor_wiring_rows",
    "flag_rows",
    "flag_off_equivalence_rows",
    "query_firing_rows",
    "candidate_score_rows",
    "ranking_influence_rows",
    "abstention_rows",
    "malformed_evidence_rows",
    "action_provenance_rows",
    "live_path_reachability_rows",
)
ACCEPTANCE_ROW_TABLES = ROW_TABLES[1:]
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the live selector claim reviewable.",
    "preconditions_checked": "Exact gates stop changed upstream evidence before policy wiring runs.",
    "inference_substrate": "The substrate separates deterministic policy fixtures from model inference.",
    "duration_s": "Measured wall time proves that the wiring fixtures executed.",
    "cited_upstream_artifacts": "Field citations identify the bounded query evidence used by this result.",
    "source_artifact_hashes": "Content hashes bind the result to exact query and live-policy code.",
    "rows": "Gate summaries keep each acceptance family independently falsifiable.",
    "factory_wiring_rows": "Factory rows prove the submitted construction seam passes the ledger.",
    "constructor_wiring_rows": "Constructor rows prove explicit overrides control one selector instance.",
    "flag_rows": "Flag rows prove exact parsing and explicit-value precedence.",
    "flag_off_equivalence_rows": "Frozen trace hashes detect any disabled-path action drift.",
    "query_firing_rows": "Firing rows distinguish an enabled mechanism from a silent no-op.",
    "candidate_score_rows": "Per-candidate terms expose every contribution to the final order.",
    "ranking_influence_rows": "Influence rows state whether belief changed a candidate ranking.",
    "abstention_rows": "Rejection rows prove uncertain evidence preserves the base order.",
    "malformed_evidence_rows": "Malformed rows prove invalid inputs fail closed before influence.",
    "action_provenance_rows": "Action rows bind query evidence and influence to the emitted action.",
    "live_path_reachability_rows": "Reachability rows prove the scored factory can call the selector.",
    "shipped_default_unchanged": "A true value requires both the disabled config and frozen trace match.",
    "belief_selector_live_path_ready_score": "One requires accepted firing and every rejection to fail closed.",
    "random_seed": "A fixed seed makes the action-trace fixture reproducible.",
    "reproducibility_checksum": "A timing-free digest detects later artifact drift.",
    "gate_check_summary": "The first exact failure makes a blocked or partial result actionable.",
    "verifier_is_oracle": "False states that prior belief is evidence rather than correctness authority.",
    "verdict_class": "A closed class separates readiness from blocked and partial execution.",
    "honest_verdict": "A class-consistent prefix gives downstream tools one terminal meaning.",
}

FORBIDDEN_ACCESS_COUNTS = {
    "game_source_reads": 0,
    "adapter_reads": 0,
    "solve_registry_reads": 0,
    "current_outcome_reads": 0,
    "future_label_reads": 0,
}


def sha256_bytes(value: bytes) -> str:
    """Label the digest so callers cannot mistake it for another hash type."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def belief_selector_enabled(explicit: bool | None = None) -> bool:
    """Resolve the explicit override before the exact default-off environment flag."""

    if explicit is not None:
        return bool(explicit)
    return os.environ.get(BELIEF_SELECTOR_ENV_FLAG) == "1"


def _action(candidate: Mapping[str, Any]) -> JsonDict:
    action = candidate.get("action")
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError("candidate action must be an integer")
    data = candidate.get("data")
    if data is not None and not isinstance(data, Mapping):
        raise ValueError("candidate data must be a mapping or null")
    return {"action": int(action), "data": deepcopy(dict(data)) if data is not None else None}


def _score(candidate: Mapping[str, Any], field: str, default: float) -> float:
    value = candidate.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _candidate_evidence(candidate: Mapping[str, Any]) -> tuple[Any, str, int]:
    required = (
        "belief_mechanic_signature",
        "belief_outcome_key",
        "belief_observation_index",
    )
    if any(candidate.get(field) is None for field in required):
        raise KeyError("missing_candidate_evidence")
    signature = candidate.get("belief_mechanic_signature")
    if isinstance(signature, ledger_mod.MechanicKey):
        key = signature
    elif isinstance(signature, Mapping):
        key = ledger_mod.MechanicKey(**dict(signature))
    else:
        raise ValueError("belief mechanic signature must be typed")
    outcome_key = candidate.get("belief_outcome_key")
    if not ledger_mod._is_sha256(outcome_key):
        raise ValueError("belief outcome key must be a SHA-256 digest")
    observation_index = candidate.get("belief_observation_index")
    if (
        isinstance(observation_index, bool)
        or not isinstance(observation_index, int)
        or observation_index < 0
    ):
        raise ValueError("belief observation index must be nonnegative")
    return key, str(outcome_key), int(observation_index)


class BeliefAwareE3Selector:
    """Rank one unchanged candidate set with safe bounded belief evidence."""

    def __init__(
        self,
        ledger: ledger_mod.BeliefLedger | None,
        *,
        query_api: Any | None = None,
        belief_weight: float = DEFAULT_BELIEF_WEIGHT,
    ) -> None:
        if isinstance(belief_weight, bool) or not isinstance(belief_weight, int | float):
            raise ValueError("belief_weight must be a nonnegative finite number")
        self.belief_weight = float(belief_weight)
        if not math.isfinite(self.belief_weight) or self.belief_weight < 0.0:
            raise ValueError("belief_weight must be a nonnegative finite number")
        self.ledger = ledger
        self.query_api = query_api or query_mod.BoundedBeliefQuery()
        self.decisions: list[JsonDict] = []
        self.last_decision: JsonDict = {}
        self.forbidden_access_counts = dict(FORBIDDEN_ACCESS_COUNTS)

    def _finish(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        query_fired: bool,
        query_count: int,
        evidence_hashes: Sequence[str],
        abstention_reason: str | None,
        parsed: Sequence[tuple[Any, str, int, float, float, JsonDict]] = (),
        contributions: Sequence[float] = (),
    ) -> list[Mapping[str, Any]]:
        base = list(candidates)
        ranked = base
        if abstention_reason is None and base:
            ranked = [
                row
                for _, row in sorted(
                    enumerate(base),
                    key=lambda item: (
                        -(parsed[item[0]][3] + parsed[item[0]][4] + float(contributions[item[0]])),
                        item[0],
                    ),
                )
            ]

        def safe_action(row: Mapping[str, Any]) -> JsonDict:
            try:
                return _action(row)
            except (TypeError, ValueError):
                return {"action": None, "data": None}

        base_actions = [safe_action(row) for row in base]
        ranked_actions = [safe_action(row) for row in ranked]
        selected_action = ranked_actions[0] if ranked_actions else None
        score_rows = []
        for index, candidate in enumerate(base):
            action = parsed[index][5] if index < len(parsed) else safe_action(candidate)
            base_score = parsed[index][3] if index < len(parsed) else float(len(base) - index)
            simulation = parsed[index][4] if index < len(parsed) else 0.0
            belief = float(contributions[index]) if index < len(contributions) else 0.0
            score_rows.append(
                {
                    "decision_index": len(self.decisions),
                    "candidate_index": index,
                    "candidate_action": action,
                    "candidate_hash": sha256_bytes(ledger_mod.canonical_bytes(action)),
                    "base_score": base_score,
                    "simulation_contribution": simulation,
                    "belief_contribution": belief,
                    "final_score": base_score + simulation + belief,
                    "selected": action == selected_action,
                    "selected_action": deepcopy(selected_action),
                    "query_fired": query_fired,
                    "evidence_hashes": list(evidence_hashes),
                    "terminal": True,
                }
            )
        decision = {
            "decision_index": len(self.decisions),
            "candidate_count": len(base),
            "base_actions": base_actions,
            "ranked_actions": ranked_actions,
            "selected_action": deepcopy(selected_action),
            "query_fired": bool(query_fired),
            "query_count": int(query_count),
            "evidence_hashes": list(evidence_hashes),
            "belief_influence": any(value != 0.0 for value in contributions),
            "ranking_changed": ranked_actions != base_actions,
            "abstained": abstention_reason is not None,
            "abstention_reason": abstention_reason,
            "candidate_score_rows": score_rows,
            "base_candidate_set_preserved": len(ranked) == len(base)
            and sorted(sha256_bytes(ledger_mod.canonical_bytes(row)) for row in ranked_actions)
            == sorted(sha256_bytes(ledger_mod.canonical_bytes(row)) for row in base_actions),
            "terminal": True,
        }
        self.decisions.append(decision)
        self.last_decision = decision
        return ranked

    def rank_candidates(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """Return a reordered view or the exact base order after one safe abstention."""

        del frame
        base = list(candidates)
        if not base:
            return self._finish(
                base,
                query_fired=False,
                query_count=0,
                evidence_hashes=(),
                abstention_reason="no_legal_candidates",
            )
        if self.ledger is None:
            return self._finish(
                base,
                query_fired=False,
                query_count=0,
                evidence_hashes=(),
                abstention_reason="missing_ledger",
            )

        parsed = []
        try:
            for index, candidate in enumerate(base):
                key, outcome_key, observation_index = _candidate_evidence(candidate)
                parsed.append(
                    (
                        key,
                        outcome_key,
                        observation_index,
                        _score(candidate, "base_score", float(len(base) - index)),
                        _score(candidate, "simulation_contribution", 0.0),
                        _action(candidate),
                    )
                )
        except KeyError:
            return self._finish(
                base,
                query_fired=False,
                query_count=0,
                evidence_hashes=(),
                abstention_reason="missing_candidate_evidence",
            )
        except (TypeError, ValueError):
            return self._finish(
                base,
                query_fired=False,
                query_count=0,
                evidence_hashes=(),
                abstention_reason="malformed_evidence",
            )

        evidence_hashes: set[str] = set()
        contributions: list[float] = []
        query_count = 0
        for key, outcome_key, observation_index, *_ in parsed:
            query_count += 1
            try:
                result = self.query_api.query(
                    self.ledger,
                    query_mod.BeliefQueryRequest(key, observation_index),
                )
                if not isinstance(result, query_mod.BeliefQueryResult):
                    raise ValueError("bounded query returned a foreign result")
                query_mod.canonical_prompt_bytes(result)
            except Exception:
                return self._finish(
                    base,
                    query_fired=True,
                    query_count=query_count,
                    evidence_hashes=sorted(evidence_hashes),
                    abstention_reason="malformed_evidence",
                    parsed=parsed,
                )
            evidence_hashes.update(result.source_event_hashes)
            if result.abstained or result.truncated:
                return self._finish(
                    base,
                    query_fired=True,
                    query_count=query_count,
                    evidence_hashes=sorted(evidence_hashes),
                    abstention_reason=result.abstention_reason or "budget_truncated",
                    parsed=parsed,
                )
            try:
                rank = result.ranked_outcome_keys.index(outcome_key)
            except ValueError:
                contributions.append(0.0)
            else:
                contributions.append(self.belief_weight / float(rank + 1))
        return self._finish(
            base,
            query_fired=True,
            query_count=query_count,
            evidence_hashes=sorted(evidence_hashes),
            abstention_reason=None,
            parsed=parsed,
            contributions=contributions,
        )


def _terminal(row: Mapping[str, Any]) -> JsonDict:
    return {**deepcopy(dict(row)), "terminal": True}


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    return _terminal(
        {
            "check": check,
            "expected_value": expected,
            "observed_value": observed,
            "passed": (expected == observed) if passed is None else bool(passed),
        }
    )


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "checks": [deepcopy(dict(row)) for row in checks],
        "expected_value": "all checks pass" if failed is None else failed.get("expected_value"),
        "failed_check": None if failed is None else failed.get("check"),
        "observed_value": "all checks pass" if failed is None else failed.get("observed_value"),
        "passed": failed is None,
    }


def _path_is_writable(path: Path) -> bool:
    current = path if path.exists() and path.is_dir() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current.is_dir() and os.access(current, os.W_OK)


def _read_json(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_preconditions(
    repo_root: Path | str,
    output_path: Path | str,
) -> tuple[list[JsonDict], JsonDict]:
    """Check frozen evidence, import contracts, and every requested write target."""

    root = Path(repo_root).resolve()
    artifact = _read_json(root / EXP7023_PATH)
    source_hashes = {
        name: ledger_mod.sha256_path(root / relative) for name, relative in SOURCE_PATHS.items()
    }
    checks = [
        _gate("exp7023_artifact_hash", EXPECTED_EXP7023_HASH, source_hashes["exp7023"]),
        _gate(
            "exp7023_belief_query_api_ready_score",
            1,
            None if artifact is None else artifact.get("belief_query_api_ready_score"),
            passed=artifact is not None
            and type(artifact.get("belief_query_api_ready_score")) is int
            and artifact.get("belief_query_api_ready_score") == 1,
        ),
    ]
    module_contracts = {
        "query_module_importable": (
            "carnot.agentic.arc_belief_query",
            ("BoundedBeliefQuery", "BeliefQueryRequest", "BeliefQueryResult"),
        ),
        "live_policy_module_importable": (
            "carnot.agentic.arc_competition_agent",
            ("E3AgentPolicy", "make_carnot_agent", "SUBMITTED_AGENT_CONFIG"),
        ),
        "action_provenance_contract": (
            "carnot.agentic.arc_action_provenance",
            ("ActionProvenanceRecorder", "maybe_make_recorder"),
        ),
        "eval_provenance_contract": (
            "carnot.agentic.arc_eval_provenance",
            ("validate_arc_eval_provenance", "build_arc_eval_provenance_for_policy"),
        ),
        "live_runner_binding_contract": (
            "carnot.agentic.arc_live_runner_execution_binding",
            ("ProcessBindingContext", "preconditions_checked"),
        ),
    }
    for check, (module_name, symbols) in module_contracts.items():
        try:
            module = importlib.import_module(module_name)
            observed = all(hasattr(module, symbol) for symbol in symbols)
        except Exception:
            observed = False
        checks.append(_gate(check, True, observed))
    for label, path in (
        ("module_path_writable", root / MODULE_PATH),
        ("test_path_writable", root / TEST_PATH),
        ("wrapper_path_writable", root / WRAPPER_PATH),
        ("artifact_path_writable", Path(output_path)),
    ):
        checks.append(_gate(label, True, _path_is_writable(path)))
    return checks, source_hashes


def _event(stream_index: int, outcome: str, mechanic: str) -> JsonDict:
    return ledger_mod._fixture_event(stream_index, mechanic=mechanic, outcome=outcome)


def _known_ledger(root: Path, mechanic: str) -> tuple[ledger_mod.BeliefLedger, JsonDict]:
    first = _event(0, "left", mechanic)
    ledger = ledger_mod.BeliefLedger(root, capacity=16, min_support=2)
    ledger.observe(first)
    ledger.observe(_event(1, "left", mechanic))
    return ledger, first


def _candidate(
    action: int,
    event: Mapping[str, Any],
    outcome: str,
    observation_index: int,
    base_score: float,
    simulation: float,
) -> JsonDict:
    return {
        "action": action,
        "data": None,
        "base_score": base_score,
        "simulation_contribution": simulation,
        "belief_mechanic_signature": ledger_mod.mechanic_key_from_event(event).to_dict(),
        "belief_outcome_key": ledger_mod._sha256_json({"outcome": outcome}),
        "belief_observation_index": observation_index,
    }


def _candidates(event: Mapping[str, Any], observation_index: int) -> list[JsonDict]:
    return [
        _candidate(1, event, "right", observation_index, 0.60, 0.05),
        _candidate(2, event, "left", observation_index, 0.55, 0.00),
    ]


def _policy_kwargs() -> JsonDict:
    return {
        "proposer": None,
        "explore_budget": 50,
        "value_head": None,
        "candidate_router": None,
        "frame_change_scorer": None,
        "action_effect_expansion_prior": False,
        "goal_bias": None,
        "goal_candidate_guidance": False,
        "epistemic_ledger": False,
        "structured_evidence_memory": False,
    }


def _with_isolated_loaders(module: Any, callback: Any) -> Any:
    names = (
        "load_cross_game_value_head",
        "_load_submitted_candidate_router",
        "_load_submitted_frame_change_scorer",
        "_load_submitted_goal_energy_bias",
    )
    old = {name: getattr(module, name) for name in names}
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    try:
        module.load_cross_game_value_head = lambda: None
        module._load_submitted_candidate_router = lambda game_id: None
        module._load_submitted_frame_change_scorer = lambda: None
        module._load_submitted_goal_energy_bias = lambda: None
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        return callback()
    finally:
        for name, value in old.items():
            setattr(module, name, value)
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _off_trace(policy_module: Any) -> bytes:
    import numpy as np

    class Frame:
        def __init__(self, grid: Any) -> None:
            self.frame = [grid.tolist()]
            self.levels_completed = 0
            self.state = "NOT_FINISHED"
            self.score = 0
            self.available_actions = [1, 2, 3, 4, 5, 6]

    policy = policy_module.E3AgentPolicy(
        "belief-fixture",
        belief_aware_selector=False,
        **_policy_kwargs(),
    )
    frames = []
    latest = None
    actions = []
    for index in range(16):
        action, data = policy.next_move(frames, latest)
        actions.append({"action": action, "data": data})
        if action is None:
            break
        grid = np.random.RandomState(index + 101).randint(0, 4, size=(6, 6)).astype(int)
        latest = Frame(grid)
        frames.append(latest)
    return json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()


def _fixture_rows(root: Path) -> JsonDict:
    policy_module = importlib.import_module("carnot.agentic.arc_competition_agent")
    action_module = importlib.import_module("carnot.agentic.arc_action_provenance")
    known, event = _known_ledger(root / "known", "known")
    selector = BeliefAwareE3Selector(known)
    base = _candidates(event, 1)
    ranked = selector.rank_candidates(None, base)
    decision = deepcopy(selector.last_decision)

    constructor_rows: list[JsonDict] = []
    factory_rows: list[JsonDict] = []
    provenance_rows: list[JsonDict] = []
    reachability_rows: list[JsonDict] = []

    def wiring() -> None:
        old_provenance = os.environ.get(action_module.PROVENANCE_ENV_FLAG)
        os.environ[action_module.PROVENANCE_ENV_FLAG] = "1"
        try:
            off = policy_module.E3AgentPolicy(
                "constructor-off",
                belief_ledger=known,
                belief_aware_selector=False,
                **_policy_kwargs(),
            )
            on = policy_module.E3AgentPolicy(
                "constructor-on",
                belief_ledger=known,
                belief_aware_selector=True,
                **_policy_kwargs(),
            )
            constructor_rows.extend(
                [
                    _terminal(
                        {
                            "fixture": "explicit_off",
                            "same_ledger": off.belief_ledger is known,
                            "selector_active": off.belief_candidate_selector is not None,
                            "passed": off.belief_ledger is known
                            and off.belief_candidate_selector is None,
                        }
                    ),
                    _terminal(
                        {
                            "fixture": "explicit_on",
                            "same_ledger": on.belief_ledger is known,
                            "selector_active": on.belief_candidate_selector is not None,
                            "explorer_same_selector": on.explorer.belief_candidate_selector
                            is on.belief_candidate_selector,
                            "passed": on.belief_ledger is known
                            and on.belief_candidate_selector is not None
                            and on.explorer.belief_candidate_selector
                            is on.belief_candidate_selector,
                        }
                    ),
                ]
            )
            on.belief_candidate_selector.rank_candidates(None, _candidates(event, 1))
            selected = on.belief_candidate_selector.last_decision["selected_action"]
            move = (selected["action"], selected["data"])
            on._prov_top = "explore.explorer"
            row = on._provenance_post_state(
                on._provenance_pre_state([], None), move, on.plan, [], None
            )
            provenance_passed = (
                row.get("belief_query_fired") is True
                and row.get("belief_evidence_hashes")
                == on.belief_candidate_selector.last_decision["evidence_hashes"]
                and row.get("belief_influence") is True
                and row.get("belief_abstained") is False
                and row.get("belief_ranking_changed") is True
                and row.get("belief_final_action") == selected
            )
            provenance_rows.append(
                _terminal(
                    {
                        "fixture": "enabled_action_row",
                        "action_row": row,
                        "passed": provenance_passed,
                    }
                )
            )

            class Base:
                def __init__(self) -> None:
                    self.game_id = "factory-fixture"

            agent_type = policy_module.make_carnot_agent(
                Base,
                belief_ledger=known,
                belief_aware_selector=True,
            )
            agent = agent_type()
            factory_passed = (
                isinstance(agent._policy, policy_module.E3AgentPolicy)
                and agent._policy.belief_ledger is known
                and agent._policy.belief_candidate_selector is not None
                and agent._policy.belief_candidate_selector.ledger is known
            )
            factory_rows.append(
                _terminal(
                    {
                        "fixture": "make_carnot_agent_enabled",
                        "policy_type": type(agent._policy).__name__,
                        "same_ledger": agent._policy.belief_ledger is known,
                        "selector_active": agent._policy.belief_candidate_selector is not None,
                        "passed": factory_passed,
                    }
                )
            )
            known_outcome = ledger_mod._sha256_json({"outcome": "left"})
            other_outcome = ledger_mod._sha256_json({"outcome": "right"})
            mechanic = ledger_mod.mechanic_key_from_event(event).to_dict()

            class PriorEvidenceAnnotator:
                def rank_candidates(
                    self,
                    _frame: Any,
                    rows: Sequence[Mapping[str, Any]],
                    **_kwargs: Any,
                ) -> list[JsonDict]:
                    return [
                        {
                            **dict(candidate),
                            "base_score": (
                                0.60 if index == 0 else (0.55 if index == 1 else -100.0)
                            ),
                            "simulation_contribution": 0.05 if index == 0 else 0.0,
                            "belief_mechanic_signature": mechanic,
                            "belief_outcome_key": (known_outcome if index == 1 else other_outcome),
                            "belief_observation_index": 1,
                        }
                        for index, candidate in enumerate(rows)
                    ]

            class Frame:
                def __init__(self) -> None:
                    self.frame = [[[0] * 6 for _ in range(6)]]
                    self.levels_completed = 0
                    self.state = "NOT_FINISHED"
                    self.score = 0
                    self.available_actions = [1, 2, 3, 4, 5]

            agent._policy.explorer.structured_evidence_memory = PriorEvidenceAnnotator()
            live_candidates = agent._policy.explorer._candidates(Frame())
            factory_decision = agent._policy.belief_candidate_selector.last_decision
            reachability_rows.append(
                _terminal(
                    {
                        "fixture": "factory_to_selector_query",
                        "policy_candidate_seam": "StepwiseExplorer._candidates",
                        "query_fired": factory_decision["query_fired"],
                        "ranking_changed": factory_decision["ranking_changed"],
                        "same_ledger": agent._policy.belief_ledger is known,
                        "selected_action": factory_decision["selected_action"],
                        "candidate_actions": [row["action"] for row in live_candidates],
                        "passed": factory_decision["query_fired"] is True
                        and factory_decision["ranking_changed"] is True
                        and agent._policy.belief_ledger is known,
                    }
                )
            )
        finally:
            if old_provenance is None:
                os.environ.pop(action_module.PROVENANCE_ENV_FLAG, None)
            else:
                os.environ[action_module.PROVENANCE_ENV_FLAG] = old_provenance

    _with_isolated_loaders(policy_module, wiring)

    old_flag = os.environ.get(BELIEF_SELECTOR_ENV_FLAG)
    flag_rows = []
    try:
        for raw, expected in (
            (None, False),
            ("", False),
            ("0", False),
            ("true", False),
            ("1", True),
        ):
            if raw is None:
                os.environ.pop(BELIEF_SELECTOR_ENV_FLAG, None)
            else:
                os.environ[BELIEF_SELECTOR_ENV_FLAG] = raw
            observed = belief_selector_enabled()
            flag_rows.append(
                _terminal(
                    {
                        "raw": raw,
                        "expected": expected,
                        "observed": observed,
                        "passed": observed is expected,
                    }
                )
            )
        os.environ[BELIEF_SELECTOR_ENV_FLAG] = "1"
        override_off = belief_selector_enabled(False)
        os.environ[BELIEF_SELECTOR_ENV_FLAG] = "0"
        override_on = belief_selector_enabled(True)
        flag_rows.append(
            _terminal(
                {
                    "raw": "explicit_override",
                    "expected": [False, True],
                    "observed": [override_off, override_on],
                    "passed": override_off is False and override_on is True,
                }
            )
        )
    finally:
        if old_flag is None:
            os.environ.pop(BELIEF_SELECTOR_ENV_FLAG, None)
        else:
            os.environ[BELIEF_SELECTOR_ENV_FLAG] = old_flag

    trace = _with_isolated_loaders(policy_module, lambda: _off_trace(policy_module))
    trace_hash = sha256_bytes(trace)
    off_rows = [
        _terminal(
            {
                "fixture": "frozen_disabled_trace",
                "expected_trace_sha256": FROZEN_OFF_TRACE_SHA256,
                "observed_trace_sha256": trace_hash,
                "passed": trace_hash == FROZEN_OFF_TRACE_SHA256,
            }
        )
    ]

    query_rows = [
        _terminal(
            {
                "fixture": "supported",
                "query_fired": decision["query_fired"],
                "query_count": decision["query_count"],
                "evidence_hashes": decision["evidence_hashes"],
                "passed": decision["query_fired"] is True
                and decision["query_count"] == len(base)
                and bool(decision["evidence_hashes"]),
            }
        )
    ]
    candidate_rows = [
        _terminal(
            {
                "fixture": "supported",
                **{key: value for key, value in row.items() if key != "terminal"},
                "passed": all(
                    key in row
                    for key in (
                        "base_score",
                        "simulation_contribution",
                        "belief_contribution",
                        "final_score",
                        "selected_action",
                    )
                ),
            }
        )
        for row in decision["candidate_score_rows"]
    ]
    influence_rows = [
        _terminal(
            {
                "fixture": "supported",
                "base_actions": decision["base_actions"],
                "ranked_actions": decision["ranked_actions"],
                "ranking_changed": decision["ranking_changed"],
                "base_candidate_set_preserved": decision["base_candidate_set_preserved"],
                "passed": decision["ranking_changed"] is True
                and decision["base_candidate_set_preserved"] is True
                and ranked[0]["action"] == 2,
            }
        )
    ]

    abstention_rows = []
    rejection_specs = []
    weak_event = _event(0, "left", "low_support")
    weak = ledger_mod.BeliefLedger(root / "low_support", capacity=16, min_support=2)
    weak.observe(weak_event)
    rejection_specs.append(("low_support", weak, weak_event, 0, "low_support"))
    conflict, conflict_event = _known_ledger(root / "conflict", "conflict")
    conflict.observe(_event(2, "right", "conflict"))
    rejection_specs.append(("conflict", conflict, conflict_event, 2, "conflicting_evidence"))
    stale, stale_event = _known_ledger(root / "stale", "stale")
    rejection_specs.append(("stale", stale, stale_event, 40, "stale_evidence"))
    trunc_event = _event(0, "left", "truncated")
    truncated = ledger_mod.BeliefLedger(root / "truncated", capacity=16, min_support=2)
    for index in range(9):
        truncated.observe(_event(index, "left", "truncated"))
    rejection_specs.append(("truncated", truncated, trunc_event, 8, "budget_truncated"))
    rejection_specs.append(("missing_ledger", None, event, 1, "missing_ledger"))
    for fixture, fixture_ledger, fixture_event, index, reason in rejection_specs:
        fixture_candidates = _candidates(fixture_event, index)
        fixture_selector = BeliefAwareE3Selector(fixture_ledger)
        fixture_ranked = fixture_selector.rank_candidates(None, fixture_candidates)
        fixture_decision = fixture_selector.last_decision
        abstention_rows.append(
            _terminal(
                {
                    "fixture": fixture,
                    "query_fired": fixture_decision["query_fired"],
                    "abstention_reason": fixture_decision["abstention_reason"],
                    "ranking_changed": fixture_decision["ranking_changed"],
                    "base_candidate_set_preserved": fixture_decision[
                        "base_candidate_set_preserved"
                    ],
                    "legal_candidate_count": len(fixture_ranked),
                    "passed": fixture_decision["abstained"] is True
                    and fixture_decision["abstention_reason"] == reason
                    and fixture_decision["ranking_changed"] is False
                    and fixture_decision["base_candidate_set_preserved"] is True
                    and len(fixture_ranked) == len(fixture_candidates) > 0,
                }
            )
        )

    malformed_rows = []
    malformed_candidates = _candidates(event, 1)
    malformed_candidates[0]["belief_outcome_key"] = "not-a-hash"
    malformed = BeliefAwareE3Selector(known)
    malformed_ranked = malformed.rank_candidates(None, malformed_candidates)
    malformed_rows.append(
        _terminal(
            {
                "fixture": "invalid_outcome_hash",
                "query_fired": malformed.last_decision["query_fired"],
                "abstention_reason": malformed.last_decision["abstention_reason"],
                "ranking_changed": malformed.last_decision["ranking_changed"],
                "passed": malformed.last_decision["abstention_reason"] == "malformed_evidence"
                and malformed.last_decision["query_fired"] is False
                and malformed_ranked == malformed_candidates,
            }
        )
    )

    class RaisingQuery:
        def query(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("deterministic malformed query fixture")

    raising = BeliefAwareE3Selector(known, query_api=RaisingQuery())
    raising_candidates = _candidates(event, 1)
    raising_ranked = raising.rank_candidates(None, raising_candidates)
    malformed_rows.append(
        _terminal(
            {
                "fixture": "query_exception",
                "query_fired": raising.last_decision["query_fired"],
                "abstention_reason": raising.last_decision["abstention_reason"],
                "ranking_changed": raising.last_decision["ranking_changed"],
                "passed": raising.last_decision["abstention_reason"] == "malformed_evidence"
                and raising.last_decision["query_fired"] is True
                and raising_ranked == raising_candidates,
            }
        )
    )

    no_access = selector.forbidden_access_counts
    reachability_rows.append(
        _terminal(
            {
                "fixture": "no_source_or_adapter_access",
                "forbidden_access_counts": no_access,
                "passed": no_access == FORBIDDEN_ACCESS_COUNTS,
            }
        )
    )
    return {
        "factory_wiring_rows": factory_rows,
        "constructor_wiring_rows": constructor_rows,
        "flag_rows": flag_rows,
        "flag_off_equivalence_rows": off_rows,
        "query_firing_rows": query_rows,
        "candidate_score_rows": candidate_rows,
        "ranking_influence_rows": influence_rows,
        "abstention_rows": abstention_rows,
        "malformed_evidence_rows": malformed_rows,
        "action_provenance_rows": provenance_rows,
        "live_path_reachability_rows": reachability_rows,
    }


def _artifact_projection(artifact: Mapping[str, Any]) -> JsonDict:
    projected = deepcopy(dict(artifact))
    projected.pop("duration_s", None)
    projected.pop("reproducibility_checksum", None)
    return projected


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic fields without wall time or the self-hash."""

    return sha256_bytes(ledger_mod.canonical_bytes(_artifact_projection(artifact)))


def _empty_artifact(
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "cited_upstream_artifacts": [],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        **{field: [] for field in ROW_TABLES},
        "shipped_default_unchanged": False,
        "belief_selector_live_path_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_belief_aware_e3_selector",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | str | None = None,
) -> JsonDict:
    """Run deterministic constructor, factory, selector, and provenance fixtures."""

    started = time.perf_counter()
    root = Path(repo_root).resolve()
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    preconditions, source_hashes = collect_preconditions(root, output)
    if not _gate_summary(preconditions)["passed"]:
        return _empty_artifact(
            run_date,
            time.perf_counter() - started,
            preconditions,
            source_hashes,
        )
    with tempfile.TemporaryDirectory(prefix="exp7024-") as directory:
        evidence = _fixture_rows(Path(directory))
    shipped_default_unchanged = bool(
        SUBMITTED_BELIEF_AWARE_SELECTOR_ENABLED is False
        and importlib.import_module(
            "carnot.agentic.arc_competition_agent"
        ).SUBMITTED_AGENT_CONFIG.get("belief_aware_selector_enabled")
        is False
        and all(row.get("passed") is True for row in evidence["flag_off_equivalence_rows"])
    )
    table_checks = [
        _gate(
            field,
            True,
            bool(evidence[field]) and all(row.get("passed") is True for row in evidence[field]),
        )
        for field in ACCEPTANCE_ROW_TABLES
    ]
    checks = [
        _gate("preconditions", True, True),
        *table_checks,
        _gate("shipped_default_unchanged", True, shipped_default_unchanged),
    ]
    gate_summary = _gate_summary(checks)
    ready = int(gate_summary["passed"])
    rows = [
        _terminal(
            {
                "check": row["check"],
                "row_count": len(evidence.get(str(row["check"]), [])),
                "passed": row["passed"],
            }
        )
        for row in table_checks
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "cited_upstream_artifacts": [
            {
                "experiment_id": 7023,
                "fields_imported": [
                    "belief_query_api_ready_score",
                    "query_result_rows",
                    "abstention_rows",
                ],
                "sha256": source_hashes["exp7023"],
            }
        ],
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        **evidence,
        "shipped_default_unchanged": shipped_default_unchanged,
        "belief_selector_live_path_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "partial",
        "honest_verdict": (
            "complete_positive_belief_aware_e3_selector_live_path_ready"
            if ready
            else "partial_belief_aware_e3_selector_fixture_failed"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Return every structural and row-consistency error without repairing data."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    score = artifact.get("belief_selector_live_path_ready_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("ready_score_not_bare_integer")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    prefix = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }.get(verdict_class)
    verdict = artifact.get("honest_verdict")
    if prefix is None or not isinstance(verdict, str) or not verdict.startswith(prefix):
        errors.append("verdict_prefix_mismatch")
    for field in ROW_TABLES:
        rows = artifact.get(field, [])
        if not isinstance(rows, list):
            errors.append(f"{field}_not_list")
        elif any(not isinstance(row, Mapping) or row.get("terminal") is not True for row in rows):
            errors.append(f"nonterminal_row:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping) or "failed_check" not in summary:
        errors.append("gate_check_summary_invalid")
    if score == 1:
        if (
            verdict_class != "positive"
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not True
            or artifact.get("shipped_default_unchanged") is not True
        ):
            errors.append("ready_gate_inconsistent")
        for field in ACCEPTANCE_ROW_TABLES:
            rows = artifact.get(field, [])
            if not rows or any(row.get("passed") is not True for row in rows):
                errors.append(f"ready_rows_failed:{field}")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish one complete result document."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7024 artifact: " + ";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic Exp7024 policy fixtures")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, and write the required Exp7024 result artifact."""

    args = _parser().parse_args(argv)
    artifact = build_artifact(run_date=args.date, output_path=args.output)
    write_artifact(args.output, artifact)
    print(
        f"wrote {args.output} belief_selector_live_path_ready_score="
        f"{artifact['belief_selector_live_path_ready_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
