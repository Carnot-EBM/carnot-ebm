"""Expose compact game-blind evidence from the durable ARC belief ledger.

The query reads one published snapshot. It does not decide whether an action
is correct. It returns small, typed evidence so a later policy can rank or
abstain without receiving game identity or future outcomes.

Spec ref: REQ-ARC-WMTE-7023 and SCENARIO-ARC-WMTE-7023-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.agentic import arc_belief_ledger as ledger_mod


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7023
SCHEMA = "carnot.exp7023.belief_query_api.v1"
QUERY_SCHEMA = "carnot.arc.belief_query.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_232_026_090_5
INFERENCE_SUBSTRATE = "deterministic_bounded_belief_query_no_llm"
REPO_ROOT = Path(__file__).resolve().parents[3]

WRAPPER_PATH = Path("scripts/experiments/experiment_7023_belief_query_api.py")
OUTPUT_PATH = Path("results/experiment_7023_belief_query_api.json")
CHECKPOINT_ROOT = Path("results/checkpoints/experiment_7023_belief_query_api")
FIXTURE_PATH = Path("results/raw/experiment_7019_arc_belief_stream_fixture/transition_events.jsonl")

MAX_QUERY_EVENTS = 8
MIN_QUERY_BYTES = 1024
MAX_QUERY_BYTES = 4096
DEFAULT_MAX_AGE = 32
ALLOWED_CHANGE_SCALES = frozenset({"none", "single", "local", "regional", "global"})

EXACT_RESULT_KEYS = frozenset(
    {
        "schema",
        "mechanic_signature",
        "known",
        "possible",
        "contradicted",
        "uncertain",
        "support",
        "contradiction_count",
        "age",
        "source_event_hashes",
        "ranked_outcome_keys",
        "abstained",
        "abstention_reason",
        "truncated",
        "omitted_belief_count",
        "omitted_event_count",
        "omitted_contradiction_event_count",
        "event_budget",
        "byte_budget",
    }
)
EXACT_BELIEF_KEYS = frozenset({"outcome_key", "support", "contradiction_count", "age"})

PROHIBITED_QUERY_KEYS = frozenset(
    {
        "adapter",
        "adapter_id",
        "adapter_identity",
        "adapter_name",
        "current_outcome",
        "current_outcome_key",
        "future_label",
        "future_outcome",
        "future_outcome_key",
        "game",
        "game_id",
        "held_future",
        "hidden_rule",
        "later_outcome",
        "registry_label",
        "registry_result",
        "sealed_future",
        "solve_registry",
        "solve_registry_label",
        "source_game",
        "source_path",
    }
)

SOURCE_PATHS = {
    "exp7022": Path("results/experiment_7022_belief_ledger_cold_audit.json"),
    "exp7020": Path("results/experiment_7020_counterexample_belief_ledger.json"),
    "exp7019": Path("results/experiment_7019_arc_belief_stream_fixture.json"),
    "construction_fixture": FIXTURE_PATH,
    "ledger_module": Path("python/carnot/agentic/arc_belief_ledger.py"),
    "competition_policy_module": Path("python/carnot/agentic/arc_competition_agent.py"),
    "trust_energy_module": Path("python/carnot/agentic/arc_world_model_trust_energy.py"),
    "action_provenance_module": Path("python/carnot/agentic/arc_action_provenance.py"),
}
EXPECTED_SOURCE_HASHES = {
    "exp7022": "sha256:74187ae153b5207553da75867c02c0323d9eb06a756840635bc79d7475534e92",
    "exp7020": "sha256:dc253079e62f68f2577b03d2b29ff6e89f47e4046917c0c89d92ed09da27ec25",
    "exp7019": "sha256:597bcc1f2f479e816c86caec731c6443233a47244031c97b0f1514ea97349d44",
    "construction_fixture": "sha256:0f7f0e75c6fc8612f2685bb3ff6d1a014795f15aa0cb361b33415343987d98ea",
    "ledger_module": "sha256:9a3a8c9da50862be71c0749dac9758660ba775fdaec58edf178db5d368f4e61d",
    "competition_policy_module": "sha256:6021f5ebbca0b2b5f8cb85d2d4e69f391f09c4b8069137ca5084f92cd8728810",
    "trust_energy_module": "sha256:b92c5c5eb630b4061dee73fddfdc874ec6b77a94fb118b7563761525872f8d05",
    "action_provenance_module": "sha256:44a92dbad3c5047de09f0c1d4b5653cb5dcc212c3d9bd7c5bdad76b9ca2c0010",
}
EXPECTED_LEDGER_STATE_HASH = (
    "sha256:f488e0f07ce7fcf70064930dd85d6c690faba1f5a82a5cb0457278359fd95343"
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "cited_upstream_artifacts",
    "source_artifact_hashes",
    "rows",
    "query_fixture_rows",
    "query_result_rows",
    "budget_rows",
    "truncation_rows",
    "abstention_rows",
    "contradiction_preservation_rows",
    "prohibited_field_rows",
    "mutation_rejection_rows",
    "restart_stability_rows",
    "serialization_hashes",
    "belief_query_api_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
ROW_TABLES = (
    "rows",
    "query_fixture_rows",
    "query_result_rows",
    "budget_rows",
    "truncation_rows",
    "abstention_rows",
    "contradiction_preservation_rows",
    "prohibited_field_rows",
    "mutation_rejection_rows",
    "restart_stability_rows",
    "serialization_hashes",
)
ACCEPTANCE_ROW_TABLES = ROW_TABLES[1:]
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for each field makes the query contract reviewable.",
    "preconditions_checked": "Exact gates stop changed evidence before the query runs.",
    "inference_substrate": "The substrate distinguishes deterministic queries from model inference.",
    "duration_s": "Measured wall time proves that the query fixtures executed.",
    "cited_upstream_artifacts": "Field citations identify the evidence inherited from earlier runs.",
    "source_artifact_hashes": "Content hashes bind the result to exact code and ledger evidence.",
    "rows": "Gate summaries keep each acceptance family independently falsifiable.",
    "query_fixture_rows": "Named fixtures expose the evidence conditions exercised by the API.",
    "query_result_rows": "Structured results let auditors recompute every query decision.",
    "budget_rows": "Per-query counts prove both hard budgets instead of asserting them.",
    "truncation_rows": "Truncation rows make omitted evidence visible.",
    "abstention_rows": "Abstention rows prevent uncertainty from becoming an exact action oracle.",
    "contradiction_preservation_rows": "Conflict rows prove counterevidence survives bounded output.",
    "prohibited_field_rows": "Denied-field rows test game identity and future leakage directly.",
    "mutation_rejection_rows": "Rejection rows prove invalid queries leave durable bytes unchanged.",
    "restart_stability_rows": "Restart rows bind query bytes to one immutable ledger state.",
    "serialization_hashes": "Content hashes detect prompt serialization drift.",
    "belief_query_api_ready_score": "One permits interface use only after every fixture passes.",
    "random_seed": "A fixed seed makes the fixture protocol addressable.",
    "reproducibility_checksum": "A timing-free digest detects later artifact drift.",
    "gate_check_summary": "The first exact failure makes a block or partial run actionable.",
    "verifier_is_oracle": "False states that the query reports evidence rather than correctness.",
    "verdict_class": "A closed class separates interface readiness from policy value.",
    "honest_verdict": "A class-consistent prefix gives downstream tools one terminal meaning.",
}


def find_prohibited_query_paths(value: Any) -> list[str]:
    """Find denied field names at every nested query path."""

    found: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).casefold() in PROHIBITED_QUERY_KEYS:
                    found.append(child)
                visit(nested, child)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "")
    return sorted(set(found))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalized_mechanic_group(key: ledger_mod.MechanicKey) -> str:
    level = "boundary" if key.level_boundary else "same_level"
    contact = "at_action" if key.touches_action_coordinate else "away_from_action"
    return f"action_{key.action_type}:{key.change_scale}:{level}:{contact}"


def _validate_mechanic_key(key: ledger_mod.MechanicKey) -> None:
    if not isinstance(key, ledger_mod.MechanicKey):
        raise TypeError("mechanic_signature must be a typed MechanicKey")
    if key.change_scale not in ALLOWED_CHANGE_SCALES:
        raise ValueError("mechanic_signature has an unsupported change scale")
    if key.mechanic_group != _normalized_mechanic_group(key):
        raise ValueError("mechanic_signature must use the normalized game-blind mechanic group")


@dataclass(frozen=True)
class BeliefQueryLimits:
    """Bound one query so later prompts cannot grow with ledger history."""

    max_events: int = MAX_QUERY_EVENTS
    max_bytes: int = MAX_QUERY_BYTES
    max_age: int = DEFAULT_MAX_AGE

    def __post_init__(self) -> None:
        if not _is_int(self.max_events) or not 1 <= self.max_events <= MAX_QUERY_EVENTS:
            raise ValueError("max_events is outside the fixed query bound")
        if not _is_int(self.max_bytes) or not MIN_QUERY_BYTES <= self.max_bytes <= MAX_QUERY_BYTES:
            raise ValueError("max_bytes is outside the fixed query bound")
        if not _is_int(self.max_age) or self.max_age < 0:
            raise ValueError("max_age must be a nonnegative integer")


@dataclass(frozen=True)
class BeliefQueryRequest:
    """Name an observable mechanic and the current observation index."""

    mechanic_signature: ledger_mod.MechanicKey
    observation_index: int

    def __post_init__(self) -> None:
        _validate_mechanic_key(self.mechanic_signature)
        if not _is_int(self.observation_index) or self.observation_index < 0:
            raise ValueError("observation_index must be a nonnegative integer")


@dataclass(frozen=True)
class BeliefSummary:
    """Summarize one observed outcome without exposing event identity."""

    outcome_key: str
    support: int
    contradiction_count: int
    age: int

    def __post_init__(self) -> None:
        if not ledger_mod._is_sha256(self.outcome_key):
            raise ValueError("outcome_key must be a SHA-256 digest")
        if any(
            not _is_int(value) or value < 0
            for value in (self.support, self.contradiction_count, self.age)
        ):
            raise ValueError("belief counts and age must be nonnegative integers")

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class BeliefQueryResult:
    """Return evidence and abstention only; no field can approve an action."""

    schema: str
    mechanic_signature: JsonDict
    known: tuple[BeliefSummary, ...]
    possible: tuple[BeliefSummary, ...]
    contradicted: tuple[BeliefSummary, ...]
    uncertain: tuple[BeliefSummary, ...]
    support: int
    contradiction_count: int
    age: int | None
    source_event_hashes: tuple[str, ...]
    ranked_outcome_keys: tuple[str, ...]
    abstained: bool
    abstention_reason: str | None
    truncated: bool
    omitted_belief_count: int
    omitted_event_count: int
    omitted_contradiction_event_count: int
    event_budget: int
    byte_budget: int

    def to_dict(self) -> JsonDict:
        value: JsonDict = {
            "schema": self.schema,
            "mechanic_signature": deepcopy(self.mechanic_signature),
            "known": [row.to_dict() for row in self.known],
            "possible": [row.to_dict() for row in self.possible],
            "contradicted": [row.to_dict() for row in self.contradicted],
            "uncertain": [row.to_dict() for row in self.uncertain],
            "support": self.support,
            "contradiction_count": self.contradiction_count,
            "age": self.age,
            "source_event_hashes": list(self.source_event_hashes),
            "ranked_outcome_keys": list(self.ranked_outcome_keys),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "truncated": self.truncated,
            "omitted_belief_count": self.omitted_belief_count,
            "omitted_event_count": self.omitted_event_count,
            "omitted_contradiction_event_count": self.omitted_contradiction_event_count,
            "event_budget": self.event_budget,
            "byte_budget": self.byte_budget,
        }
        if set(value) != EXACT_RESULT_KEYS:
            raise ValueError("belief query result keys changed")
        return value


def _raw_prompt_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_prompt_bytes(result: BeliefQueryResult) -> bytes:
    """Serialize canonical JSON after checking prompt and budget safety."""

    if not isinstance(result, BeliefQueryResult):
        raise TypeError("canonical_prompt_bytes requires a BeliefQueryResult")
    value = result.to_dict()
    prohibited = find_prohibited_query_paths(value)
    if prohibited:
        raise ValueError(f"belief query result has prohibited fields:{prohibited}")
    key = ledger_mod.MechanicKey(**value["mechanic_signature"])
    _validate_mechanic_key(key)
    payload = _raw_prompt_bytes(value)
    if len(payload) > result.byte_budget:
        raise ValueError("belief query result exceeds its byte budget")
    if len(result.source_event_hashes) > result.event_budget:
        raise ValueError("belief query result exceeds its event budget")
    if any(not ledger_mod._is_sha256(value) for value in result.source_event_hashes):
        raise ValueError("belief query source provenance must contain SHA-256 digests")
    return payload


def query_serialization_hash(result: BeliefQueryResult) -> str:
    """Hash exact prompt bytes so restart comparisons need no text parsing."""

    return ledger_mod.sha256_bytes(canonical_prompt_bytes(result))


def _state_rank(state: str) -> int:
    return {"known": 0, "possible": 1, "uncertain": 2, "contradicted": 3}[state]


def _belief_sort(item: tuple[str, BeliefSummary]) -> tuple[int, int, int, str]:
    state, summary = item
    return (_state_rank(state), -summary.support, summary.age, summary.outcome_key)


def _semantic_abstention(
    beliefs: Sequence[tuple[str, BeliefSummary]],
    contradiction_count: int,
    max_age: int,
) -> str | None:
    active = sorted((row for row in beliefs if row[0] != "contradicted"), key=_belief_sort)
    if not active:
        return "no_evidence"
    if contradiction_count > 0 or any(state == "contradicted" for state, _ in beliefs):
        return "conflicting_evidence"
    if len(active) > 1 and _belief_sort(active[0])[:3] == _belief_sort(active[1])[:3]:
        return "tied_active_evidence"
    if active[0][0] != "known":
        return "low_support"
    if active[0][1].age > max_age:
        return "stale_evidence"
    return None


def _make_result(
    *,
    key: ledger_mod.MechanicKey,
    included_beliefs: Sequence[tuple[str, BeliefSummary]],
    all_belief_count: int,
    source_hashes: Sequence[str],
    total_event_count: int,
    contradiction_hashes: set[str],
    total_support: int,
    contradiction_count: int,
    age: int | None,
    semantic_reason: str | None,
    limits: BeliefQueryLimits,
) -> BeliefQueryResult:
    ordered = sorted(included_beliefs, key=_belief_sort)
    grouped = {
        state: tuple(summary for row_state, summary in ordered if row_state == state)
        for state in ("known", "possible", "contradicted", "uncertain")
    }
    active = [summary for state, summary in ordered if state != "contradicted"]
    omitted_beliefs = all_belief_count - len(ordered)
    omitted_events = total_event_count - len(source_hashes)
    omitted_contradictions = len(contradiction_hashes - set(source_hashes))
    truncated = omitted_beliefs > 0 or omitted_events > 0
    reason = semantic_reason or ("budget_truncated" if truncated else None)
    return BeliefQueryResult(
        schema=QUERY_SCHEMA,
        mechanic_signature=key.to_dict(),
        known=grouped["known"],
        possible=grouped["possible"],
        contradicted=grouped["contradicted"],
        uncertain=grouped["uncertain"],
        support=total_support,
        contradiction_count=contradiction_count,
        age=age,
        source_event_hashes=tuple(source_hashes),
        ranked_outcome_keys=tuple(summary.outcome_key for summary in active),
        abstained=reason is not None,
        abstention_reason=reason,
        truncated=truncated,
        omitted_belief_count=omitted_beliefs,
        omitted_event_count=omitted_events,
        omitted_contradiction_event_count=omitted_contradictions,
        event_budget=limits.max_events,
        byte_budget=limits.max_bytes,
    )


class BoundedBeliefQuery:
    """Read one ledger snapshot and return deterministic bounded evidence."""

    def __init__(self, limits: BeliefQueryLimits | None = None) -> None:
        self.limits = limits or BeliefQueryLimits()

    def query(
        self,
        ledger: ledger_mod.BeliefLedger,
        request: BeliefQueryRequest,
    ) -> BeliefQueryResult:
        if not isinstance(ledger, ledger_mod.BeliefLedger):
            raise TypeError("query requires a BeliefLedger")
        if not isinstance(request, BeliefQueryRequest):
            raise TypeError("query requires a BeliefQueryRequest")
        _validate_mechanic_key(request.mechanic_signature)
        state_before = ledger.snapshot()
        journal_before = ledger.journal_path.read_bytes()
        state = json.loads(state_before)
        if not isinstance(state, dict) or state.get("schema") != ledger_mod.STATE_SCHEMA:
            raise ValueError("ledger snapshot has an invalid schema")
        last_index = state.get("last_stream_index")
        if not _is_int(last_index):
            raise ValueError("ledger snapshot has an invalid stream index")
        if request.observation_index < last_index:
            raise ValueError("observation_index precedes published ledger evidence")

        key_dict = request.mechanic_signature.to_dict()
        facts = [
            row
            for row in state.get("facts", [])
            if isinstance(row, Mapping) and row.get("mechanic_key") == key_dict
        ]
        events = state.get("events", {})
        clusters = [
            row
            for row in state.get("clusters", {}).values()
            if isinstance(row, Mapping) and row.get("mechanic_key") == key_dict
        ]
        beliefs: list[tuple[str, BeliefSummary]] = []
        event_order: dict[str, tuple[int, str]] = {}
        contradiction_hashes: set[str] = set()
        for fact in facts:
            state_name = str(fact.get("state"))
            if state_name not in {"known", "possible", "contradicted", "uncertain"}:
                raise ValueError("ledger fact has an invalid belief state")
            fact_last = fact.get("last_stream_index")
            if not _is_int(fact_last) or fact_last > request.observation_index:
                raise ValueError("ledger fact is newer than the query observation")
            summary = BeliefSummary(
                outcome_key=fact.get("outcome_key"),
                support=fact.get("support"),
                contradiction_count=fact.get("contradiction_count"),
                age=request.observation_index - fact_last,
            )
            beliefs.append((state_name, summary))
            event_ids = list(fact.get("event_ids", []))
            tombstone_id = fact.get("tombstoned_by")
            if isinstance(tombstone_id, str):
                event_ids.append(tombstone_id)
            for event_id in event_ids:
                event = events.get(str(event_id)) if isinstance(events, Mapping) else None
                if not isinstance(event, Mapping) or not ledger_mod._is_sha256(
                    event.get("row_hash")
                ):
                    raise ValueError("ledger fact has missing source event provenance")
                event_hash = str(event["row_hash"])
                stream_index = event.get("stream_index")
                if not _is_int(stream_index):
                    raise ValueError("ledger event has an invalid stream index")
                event_order[event_hash] = (stream_index, event_hash)
                if event_id == tombstone_id:
                    contradiction_hashes.add(event_hash)

        contradiction_count = max(
            [sum(summary.contradiction_count for _, summary in beliefs)]
            + [int(row.get("counterexample_count", 0)) for row in clusters]
        )
        total_support = sum(summary.support for _, summary in beliefs)
        active_ages = [
            summary.age for state_name, summary in beliefs if state_name != "contradicted"
        ]
        all_ages = [summary.age for _, summary in beliefs]
        age = min(active_ages or all_ages) if all_ages else None
        semantic_reason = _semantic_abstention(
            beliefs,
            contradiction_count,
            self.limits.max_age,
        )

        ordered_hashes = sorted(
            event_order,
            key=lambda digest: (
                digest not in contradiction_hashes,
                event_order[digest][0],
                digest,
            ),
        )
        included_hashes = ordered_hashes[: self.limits.max_events]
        inclusion_order = sorted(
            beliefs,
            key=lambda row: (
                row[0] != "contradicted",
                *_belief_sort(row),
            ),
        )
        included_beliefs: list[tuple[str, BeliefSummary]] = []

        def candidate() -> BeliefQueryResult:
            return _make_result(
                key=request.mechanic_signature,
                included_beliefs=included_beliefs,
                all_belief_count=len(beliefs),
                source_hashes=included_hashes,
                total_event_count=len(ordered_hashes),
                contradiction_hashes=contradiction_hashes,
                total_support=total_support,
                contradiction_count=contradiction_count,
                age=age,
                semantic_reason=semantic_reason,
                limits=self.limits,
            )

        while len(_raw_prompt_bytes(candidate().to_dict())) > self.limits.max_bytes:
            if not included_hashes:
                raise ValueError("byte budget cannot hold the fixed query contract")
            included_hashes.pop()
        for belief in inclusion_order:
            included_beliefs.append(belief)
            if len(_raw_prompt_bytes(candidate().to_dict())) > self.limits.max_bytes:
                included_beliefs.pop()

        result = candidate()
        canonical_prompt_bytes(result)
        if ledger.snapshot() != state_before or ledger.journal_path.read_bytes() != journal_before:
            raise RuntimeError("belief query mutated the ledger")
        return result


def _terminal(row: Mapping[str, Any]) -> JsonDict:
    return {**deepcopy(dict(row)), "terminal": True}


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    return _terminal(
        {
            "check": check,
            "expected_value": deepcopy(expected),
            "observed_value": deepcopy(observed),
            "passed": bool(expected == observed if passed is None else passed),
        }
    )


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _path_is_writable(path: Path) -> bool:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return target.exists() and os.access(target, os.W_OK)


def collect_preconditions(
    repo_root: Path,
    output_path: Path,
    checkpoint_root: Path,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict] | None]:
    """Verify exact frozen evidence before any query result is trusted."""

    hashes = {name: ledger_mod.sha256_path(repo_root / path) for name, path in SOURCE_PATHS.items()}
    checks = [
        _gate(f"source_hash:{name}", EXPECTED_SOURCE_HASHES[name], hashes[name])
        for name in SOURCE_PATHS
    ]
    values: dict[str, JsonDict] = {}
    events: list[JsonDict] | None = None
    if all(row["passed"] for row in checks):
        for name in ("exp7022", "exp7020", "exp7019"):
            value = json.loads((repo_root / SOURCE_PATHS[name]).read_text(encoding="utf-8"))
            values[name] = value if isinstance(value, dict) else {}
        events = ledger_mod.load_updater_events(repo_root / FIXTURE_PATH)
    exp7022_score = values.get("exp7022", {}).get("belief_shadow_safe_score")
    exp7020_score = values.get("exp7020", {}).get("belief_ledger_ready_score")
    checks.extend(
        [
            _gate(
                "exp7022_belief_shadow_safe_score",
                1,
                exp7022_score,
                passed=type(exp7022_score) is int and exp7022_score == 1,
            ),
            _gate(
                "exp7020_belief_ledger_ready_score",
                1,
                exp7020_score,
                passed=type(exp7020_score) is int and exp7020_score == 1,
            ),
            _gate("construction_event_count", 27, len(events or [])),
            _gate(
                "current_policy_symbols",
                True,
                all(
                    marker in (repo_root / SOURCE_PATHS[name]).read_text(encoding="utf-8")
                    if (repo_root / SOURCE_PATHS[name]).is_file()
                    else False
                    for name, marker in (
                        ("competition_policy_module", "class E3AgentPolicy"),
                        ("trust_energy_module", "def select_trusted_world_model"),
                        ("action_provenance_module", "class ActionProvenanceRecorder"),
                    )
                ),
            ),
            _gate(
                "module_path_writable",
                True,
                _path_is_writable(repo_root / Path(__file__).relative_to(REPO_ROOT)),
            ),
            _gate("wrapper_path_writable", True, _path_is_writable(repo_root / WRAPPER_PATH)),
            _gate(
                "test_path_writable",
                True,
                _path_is_writable(
                    repo_root / "tests/python/test_experiment_7023_belief_query_api.py"
                ),
            ),
            _gate("checkpoint_path_writable", True, _path_is_writable(checkpoint_root)),
            _gate("artifact_path_writable", True, _path_is_writable(output_path)),
        ]
    )
    if events is not None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp7023-precondition-") as directory:
            replay = ledger_mod.replay_events(events, Path(directory) / "ledger")
            observed_ledger_hash = replay["ledger"].state_hash
    else:
        observed_ledger_hash = None
    checks.append(
        _gate("exp7020_exact_ledger_state_hash", EXPECTED_LEDGER_STATE_HASH, observed_ledger_hash)
    )
    return checks, hashes, events


def _fixture_query_digest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay and hash the frozen query set without retaining temporary state."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7023-fixture-") as directory:
        replay = ledger_mod.replay_events(events, Path(directory) / "ledger")
        ledger = replay["ledger"]
        keys = sorted(
            {
                ledger_mod.canonical_bytes(ledger_mod.mechanic_key_from_event(row).to_dict()): row
                for row in events
            }.values(),
            key=lambda row: ledger_mod.canonical_bytes(
                ledger_mod.mechanic_key_from_event(row).to_dict()
            ),
        )
        observation_index = max((int(row["stream_index"]) for row in events), default=0)
        before = ledger.state_hash
        query_hashes = [
            query_serialization_hash(
                BoundedBeliefQuery().query(
                    ledger,
                    BeliefQueryRequest(ledger_mod.mechanic_key_from_event(row), observation_index),
                )
            )
            for row in keys
        ]
        restarted = ledger.restart()
        restart_hashes = [
            query_serialization_hash(
                BoundedBeliefQuery().query(
                    restarted,
                    BeliefQueryRequest(ledger_mod.mechanic_key_from_event(row), observation_index),
                )
            )
            for row in keys
        ]
        projection = {
            "ledger_state_hash": before,
            "query_hashes": query_hashes,
            "query_count": len(query_hashes),
        }
        return {
            **projection,
            "restart_ledger_state_hash": restarted.state_hash,
            "restart_query_hashes": restart_hashes,
            "digest": ledger_mod.sha256_bytes(ledger_mod.canonical_bytes(projection)),
            "ledger_unchanged": before == ledger.state_hash == restarted.state_hash,
        }


def _fresh_process_fixture_digest(repo_root: Path, fixture_path: Path) -> JsonDict:
    env = os.environ.copy()
    python_root = str(repo_root / "python")
    env["PYTHONPATH"] = python_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.agentic.arc_belief_query",
            "--fresh-fixture-digest",
            str(fixture_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or completed.stdout.strip()}
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "fresh process returned invalid JSON"}
    return value if isinstance(value, dict) else {"error": "fresh process returned non-object"}


def _query_fixture_matrix(root: Path, source_events: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixture_rows: list[JsonDict] = []
    result_rows: list[JsonDict] = []
    budget_rows: list[JsonDict] = []
    truncation_rows: list[JsonDict] = []
    abstention_rows: list[JsonDict] = []
    contradiction_rows: list[JsonDict] = []
    serialization_hashes: list[JsonDict] = []
    results: dict[str, BeliefQueryResult] = {}

    def record(name: str, result: BeliefQueryResult, expected_reason: str | None) -> None:
        payload = canonical_prompt_bytes(result)
        value = result.to_dict()
        passed = result.abstention_reason == expected_reason
        results[name] = result
        fixture_rows.append(
            _terminal(
                {
                    "fixture": name,
                    "expected_abstention_reason": expected_reason,
                    "observed_abstention_reason": result.abstention_reason,
                    "passed": passed,
                }
            )
        )
        result_rows.append(
            _terminal(
                {
                    "fixture": name,
                    "result": value,
                    "exact_keys": sorted(value),
                    "passed": set(value) == EXACT_RESULT_KEYS and passed,
                }
            )
        )
        budget_rows.append(
            _terminal(
                {
                    "fixture": name,
                    "source_event_count": len(result.source_event_hashes),
                    "event_budget": result.event_budget,
                    "serialized_bytes": len(payload),
                    "byte_budget": result.byte_budget,
                    "passed": len(result.source_event_hashes) <= result.event_budget
                    and len(payload) <= result.byte_budget,
                }
            )
        )
        abstention_rows.append(
            _terminal(
                {
                    "fixture": name,
                    "abstained": result.abstained,
                    "reason": result.abstention_reason,
                    "exact_action_decision_present": bool(
                        {"accepted", "rejected", "allowed", "denied", "correct"} & set(value)
                    ),
                    "passed": result.abstained == (expected_reason is not None)
                    and result.abstention_reason == expected_reason,
                }
            )
        )
        serialization_hashes.append(
            _terminal(
                {
                    "fixture": name,
                    "sha256": query_serialization_hash(result),
                    "serialized_bytes": len(payload),
                    "passed": ledger_mod._is_sha256(query_serialization_hash(result)),
                }
            )
        )

    known = ledger_mod.BeliefLedger(root / "known", capacity=8, min_support=2)
    known_first = ledger_mod._fixture_event(0, mechanic="known", outcome="left")
    known.observe(known_first)
    known.observe(ledger_mod._fixture_event(1, mechanic="known", outcome="left"))
    known_request = BeliefQueryRequest(ledger_mod.mechanic_key_from_event(known_first), 1)
    record("known", BoundedBeliefQuery().query(known, known_request), None)

    weak = ledger_mod.BeliefLedger(root / "possible", capacity=8, min_support=2)
    weak_event = ledger_mod._fixture_event(0, mechanic="possible", outcome="weak")
    weak.observe(weak_event)
    record(
        "possible_low_support",
        BoundedBeliefQuery().query(
            weak,
            BeliefQueryRequest(ledger_mod.mechanic_key_from_event(weak_event), 0),
        ),
        "low_support",
    )

    empty = ledger_mod.BeliefLedger(root / "empty", capacity=8, min_support=2)
    record(
        "empty",
        BoundedBeliefQuery().query(
            empty,
            BeliefQueryRequest(ledger_mod.mechanic_key_from_event(weak_event), 0),
        ),
        "no_evidence",
    )

    conflict = ledger_mod.BeliefLedger(root / "conflict", capacity=8, min_support=2)
    conflict_first = ledger_mod._fixture_event(0, mechanic="conflict", outcome="left")
    conflict.observe(conflict_first)
    conflict.observe(ledger_mod._fixture_event(1, mechanic="conflict", outcome="left"))
    conflict.observe(ledger_mod._fixture_event(2, mechanic="conflict", outcome="right"))
    conflict_request = BeliefQueryRequest(ledger_mod.mechanic_key_from_event(conflict_first), 2)
    conflict_result = BoundedBeliefQuery().query(conflict, conflict_request)
    record("conflict_tombstone", conflict_result, "conflicting_evidence")

    stale = BoundedBeliefQuery(BeliefQueryLimits(max_age=2)).query(
        known,
        BeliefQueryRequest(ledger_mod.mechanic_key_from_event(known_first), 9),
    )
    record("stale", stale, "stale_evidence")

    tied = ledger_mod.BeliefLedger(root / "tie", capacity=8, min_support=2)
    tie_first = ledger_mod._fixture_event(0, mechanic="tie", outcome="left")
    tied.observe(tie_first)
    tie_state = json.loads(tied.snapshot())
    tie_event = ledger_mod._fixture_event(0, mechanic="tie", outcome="right")
    tie_fact = deepcopy(tie_state["facts"][0])
    tie_fact["outcome_key"] = tie_event["mechanic_signature"]["observed_outcome_key"]
    tie_fact["fact_id"] = ledger_mod.sha256_bytes(
        ledger_mod.canonical_bytes({"tie": tie_fact["outcome_key"]})
    )
    tie_fact["event_ids"] = [tie_event["event_id"]]
    tie_state["facts"].append(tie_fact)
    tie_state["facts"] = sorted(tie_state["facts"], key=lambda row: row["fact_id"])
    tie_state["events"][tie_event["event_id"]] = {
        "row_hash": tie_event["row_hash"],
        "stream_index": 0,
        "mechanic_key": tie_fact["mechanic_key"],
        "outcome_key": tie_fact["outcome_key"],
    }
    tied.state_path.write_bytes(ledger_mod.canonical_bytes(tie_state))
    tied = tied.restart()
    record(
        "semantic_tie",
        BoundedBeliefQuery().query(
            tied,
            BeliefQueryRequest(ledger_mod.mechanic_key_from_event(tie_first), 0),
        ),
        "tied_active_evidence",
    )

    truncated = BoundedBeliefQuery(
        BeliefQueryLimits(max_events=2, max_bytes=MIN_QUERY_BYTES, max_age=32)
    ).query(conflict, conflict_request)
    record("truncated_conflict", truncated, "conflicting_evidence")

    exp_replay = ledger_mod.replay_events(source_events, root / "exp7019")
    source_key = ledger_mod.mechanic_key_from_event(source_events[0])
    exp_result = BoundedBeliefQuery().query(
        exp_replay["ledger"],
        BeliefQueryRequest(source_key, len(source_events) - 1),
    )
    record("exp7019_exp7020_replay", exp_result, "conflicting_evidence")

    for name in ("conflict_tombstone", "truncated_conflict", "exp7019_exp7020_replay"):
        result = results[name]
        row = _terminal(
            {
                "fixture": name,
                "contradiction_count": result.contradiction_count,
                "contradicted_belief_count": len(result.contradicted),
                "omitted_contradiction_event_count": result.omitted_contradiction_event_count,
                "abstained": result.abstained,
                "passed": result.contradiction_count > 0
                and result.abstained
                and (
                    bool(result.contradicted)
                    or result.omitted_belief_count > 0
                    or result.omitted_contradiction_event_count > 0
                ),
            }
        )
        contradiction_rows.append(row)
    truncation_rows.append(
        _terminal(
            {
                "fixture": "truncated_conflict",
                "truncated": truncated.truncated,
                "omitted_belief_count": truncated.omitted_belief_count,
                "omitted_event_count": truncated.omitted_event_count,
                "omitted_contradiction_event_count": truncated.omitted_contradiction_event_count,
                "passed": truncated.truncated
                and (truncated.omitted_belief_count + truncated.omitted_event_count > 0),
            }
        )
    )
    return {
        "query_fixture_rows": fixture_rows,
        "query_result_rows": result_rows,
        "budget_rows": budget_rows,
        "truncation_rows": truncation_rows,
        "abstention_rows": abstention_rows,
        "contradiction_preservation_rows": contradiction_rows,
        "serialization_hashes": serialization_hashes,
        "known_ledger": known,
        "known_request": known_request,
        "known_result": results["known"],
    }


def _rejection_rows(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    ledger = ledger_mod.BeliefLedger(root / "rejections", capacity=8, min_support=2)
    event = ledger_mod._fixture_event(0, mechanic="reject", outcome="left")
    ledger.observe(event)
    request = BeliefQueryRequest(ledger_mod.mechanic_key_from_event(event), 0)
    prohibited_rows = [
        _terminal(
            {
                "field": field,
                "detected_paths": find_prohibited_query_paths({"nested": {field: "x"}}),
                "passed": find_prohibited_query_paths({"nested": {field: "x"}})
                == [f"nested.{field}"],
            }
        )
        for field in sorted(PROHIBITED_QUERY_KEYS)
    ]
    result = BoundedBeliefQuery().query(ledger, request)
    prohibited_rows.append(
        _terminal(
            {
                "field": "complete_query_result",
                "detected_paths": find_prohibited_query_paths(result.to_dict()),
                "passed": find_prohibited_query_paths(result.to_dict()) == [],
            }
        )
    )
    state_before = ledger.snapshot()
    journal_before = ledger.journal_path.read_bytes()
    cases: list[tuple[str, Any]] = []
    bad_key = ledger_mod.MechanicKey(
        **{**request.mechanic_signature.to_dict(), "mechanic_group": "game_id:secret"}
    )
    cases.append(
        (
            "arbitrary_mechanic_text",
            lambda: BoundedBeliefQuery().query(ledger, BeliefQueryRequest(bad_key, 0)),
        )
    )
    cases.append(("untyped_request", lambda: BoundedBeliefQuery().query(ledger, {})))
    cases.append(
        (
            "observation_time_travel",
            lambda: BoundedBeliefQuery().query(
                ledger,
                BeliefQueryRequest(request.mechanic_signature, -1),
            ),
        )
    )
    cases.append(("oversized_event_budget", lambda: BeliefQueryLimits(max_events=9)))
    mutation_rows: list[JsonDict] = []
    for name, operation in cases:
        error: str | None = None
        try:
            operation()
        except (TypeError, ValueError) as exc:
            error = type(exc).__name__
        mutation_rows.append(
            _terminal(
                {
                    "mutation": name,
                    "rejected": error is not None,
                    "error_type": error,
                    "state_unchanged": ledger.snapshot() == state_before,
                    "journal_unchanged": ledger.journal_path.read_bytes() == journal_before,
                    "passed": error is not None
                    and ledger.snapshot() == state_before
                    and ledger.journal_path.read_bytes() == journal_before,
                }
            )
        )
    return prohibited_rows, mutation_rows


def _artifact_projection(artifact: Mapping[str, Any]) -> JsonDict:
    projected = deepcopy(dict(artifact))
    projected.pop("duration_s", None)
    projected.pop("reproducibility_checksum", None)
    return projected


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return ledger_mod.sha256_bytes(ledger_mod.canonical_bytes(_artifact_projection(artifact)))


def _empty_artifact(
    *,
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
        "belief_query_api_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_belief_query_api",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
) -> JsonDict:
    """Run all acceptance and rejection fixtures for the query interface."""

    started = time.perf_counter()
    root = Path(repo_root).resolve()
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    checkpoint = Path(checkpoint_root) if checkpoint_root is not None else root / CHECKPOINT_ROOT
    preconditions, source_hashes, events = collect_preconditions(root, output, checkpoint)
    if not _gate_summary(preconditions)["passed"]:
        return _empty_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
            source_hashes=source_hashes,
        )

    checkpoint.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkpoint, prefix="run-") as directory:
        fixture = _query_fixture_matrix(Path(directory), events or [])
        prohibited_rows, mutation_rows = _rejection_rows(Path(directory))
        known_ledger = fixture.pop("known_ledger")
        known_request = fixture.pop("known_request")
        known_result = fixture.pop("known_result")
        restarted_ledger = known_ledger.restart()
        restarted_result = BoundedBeliefQuery().query(restarted_ledger, known_request)
        known_state_hash = known_ledger.state_hash
        restarted_state_hash = restarted_ledger.state_hash
        known_query_hash = query_serialization_hash(known_result)
        restarted_query_hash = query_serialization_hash(restarted_result)
        local_digest = _fixture_query_digest(events or [])
        fresh_digest = _fresh_process_fixture_digest(root, root / FIXTURE_PATH)

    restart_passed = (
        known_state_hash == restarted_state_hash
        and known_query_hash == restarted_query_hash
        and local_digest.get("digest") == fresh_digest.get("digest")
        and local_digest.get("query_hashes") == fresh_digest.get("query_hashes")
        and local_digest.get("ledger_unchanged") is True
    )
    restart_rows = [
        _terminal(
            {
                "ledger_state_hash": known_state_hash,
                "restart_ledger_state_hash": restarted_state_hash,
                "query_hash": known_query_hash,
                "restart_query_hash": restarted_query_hash,
                "fixture_digest": local_digest.get("digest"),
                "fresh_fixture_digest": fresh_digest.get("digest"),
                "fresh_process_match": local_digest.get("digest") == fresh_digest.get("digest"),
                "fresh_process_error": fresh_digest.get("error"),
                "passed": restart_passed,
            }
        )
    ]
    evidence: JsonDict = {
        **fixture,
        "prohibited_field_rows": prohibited_rows,
        "mutation_rejection_rows": mutation_rows,
        "restart_stability_rows": restart_rows,
    }
    table_checks = [
        _gate(
            field,
            True,
            bool(evidence.get(field)) and all(row.get("passed") is True for row in evidence[field]),
        )
        for field in ACCEPTANCE_ROW_TABLES
    ]
    gate_summary = _gate_summary([_gate("preconditions", True, True), *table_checks])
    ready = int(gate_summary["passed"])
    rows = [
        _terminal(
            {
                "check": row["check"],
                "row_count": len(evidence.get(row["check"], [])),
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
                "experiment_id": 7022,
                "fields_imported": ["belief_shadow_safe_score", "source_artifact_hashes"],
                "sha256": source_hashes["exp7022"],
            },
            {
                "experiment_id": 7020,
                "fields_imported": ["belief_ledger_ready_score", "snapshot_hashes"],
                "sha256": source_hashes["exp7020"],
            },
            {
                "experiment_id": 7019,
                "fields_imported": ["arc_belief_stream_ready_score", "fixture_hash"],
                "sha256": source_hashes["exp7019"],
            },
        ],
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        **evidence,
        "belief_query_api_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "partial",
        "honest_verdict": (
            "complete_positive_bounded_game_blind_belief_query_api_ready"
            if ready
            else "partial_belief_query_api_acceptance_fixture_failed"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Return every structural or row-derived artifact error without repair."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    score = artifact.get("belief_query_api_ready_score")
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
    gate_summary = artifact.get("gate_check_summary")
    if not isinstance(gate_summary, Mapping) or "failed_check" not in gate_summary:
        errors.append("gate_check_summary_invalid")
    for row in artifact.get("query_result_rows", []):
        if isinstance(row, Mapping):
            result = row.get("result")
            if not isinstance(result, Mapping) or set(result) != EXACT_RESULT_KEYS:
                errors.append("query_result_keys_mismatch")
            elif find_prohibited_query_paths(result):
                errors.append("query_result_prohibited_fields")
    if score == 1:
        if (
            verdict_class != "positive"
            or not isinstance(gate_summary, Mapping)
            or not gate_summary.get("passed")
        ):
            errors.append("ready_gate_inconsistent")
        for field in ACCEPTANCE_ROW_TABLES:
            rows = artifact.get(field, [])
            if not rows or any(
                not isinstance(row, Mapping) or row.get("passed") is not True for row in rows
            ):
                errors.append(f"ready_rows_failed:{field}")
        for row in artifact.get("budget_rows", []):
            if isinstance(row, Mapping) and (
                row.get("serialized_bytes", MAX_QUERY_BYTES + 1) > row.get("byte_budget", 0)
                or row.get("source_event_count", MAX_QUERY_EVENTS + 1) > row.get("event_budget", 0)
            ):
                errors.append("ready_budget_exceeded")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> None:
    payload = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    ledger_mod._atomic_write(Path(path), payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--fresh-fixture-digest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fresh_fixture_digest is not None:
        print(
            json.dumps(
                _fixture_query_digest(ledger_mod.load_updater_events(args.fresh_fixture_digest)),
                sort_keys=True,
            )
        )
        return 0
    root = args.repo_root.resolve()
    output = args.output or root / OUTPUT_PATH
    checkpoint = args.checkpoint_root or root / CHECKPOINT_ROOT
    artifact = build_artifact(
        repo_root=root,
        run_date=args.date,
        output_path=output,
        checkpoint_root=checkpoint,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp7023 artifact validation failed:{errors}")
    write_artifact(output, artifact)
    print(f"wrote {output} belief_query_api_ready_score={artifact['belief_query_api_ready_score']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the command test.
    raise SystemExit(main())
