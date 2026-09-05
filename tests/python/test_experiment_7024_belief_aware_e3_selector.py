"""Deterministic tests for REQ-ARC-WMTE-7024 and its named scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot.agentic import arc_action_provenance as action_provenance
from carnot.agentic import arc_belief_aware_e3_selector as selector_mod
from carnot.agentic import arc_belief_ledger as ledger_mod
from carnot.agentic import arc_competition_agent as competition


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_OFF_TRACE_SHA256 = "sha256:0d91eb1c095d103ceedba01caa26905812d872907a723f49fa4c523611d98820"


class _Frame:
    """Give the policy only the visible frame fields used by the deterministic fixture."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


class _ForbiddenAccess(dict):
    """Count any attempt to read a hidden source or adapter field."""

    forbidden = {"game_source", "adapter", "solve_registry", "current_outcome", "future_label"}

    def __init__(self, value: dict) -> None:
        super().__init__(value)
        self.forbidden_access_count = 0

    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key in self.forbidden:
            self.forbidden_access_count += 1
            raise AssertionError(f"forbidden access: {key}")
        return super().get(key, default)

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if key in self.forbidden:
            self.forbidden_access_count += 1
            raise AssertionError(f"forbidden access: {key}")
        return super().__getitem__(key)


def _event(stream_index: int, outcome: str, *, mechanic: str = "selector") -> dict:
    return ledger_mod._fixture_event(stream_index, mechanic=mechanic, outcome=outcome)


def _known_ledger(
    root: Path, *, mechanic: str = "selector"
) -> tuple[ledger_mod.BeliefLedger, dict]:
    first = _event(0, "left", mechanic=mechanic)
    ledger = ledger_mod.BeliefLedger(root, capacity=16, min_support=2)
    assert ledger.observe(first)["accepted"] is True
    assert ledger.observe(_event(1, "left", mechanic=mechanic))["accepted"] is True
    return ledger, first


def _candidate(
    action: int,
    event: dict,
    outcome: str,
    *,
    observation_index: int,
    base_score: float,
    simulation_contribution: float,
) -> dict:
    return {
        "action": action,
        "data": None,
        "base_score": base_score,
        "simulation_contribution": simulation_contribution,
        "belief_mechanic_signature": ledger_mod.mechanic_key_from_event(event).to_dict(),
        "belief_outcome_key": ledger_mod._sha256_json({"outcome": outcome}),
        "belief_observation_index": observation_index,
    }


def _supported_candidates(event: dict, *, observation_index: int = 1) -> list[dict]:
    return [
        _candidate(
            1,
            event,
            "right",
            observation_index=observation_index,
            base_score=0.60,
            simulation_contribution=0.05,
        ),
        _candidate(
            2,
            event,
            "left",
            observation_index=observation_index,
            base_score=0.55,
            simulation_contribution=0.00,
        ),
    ]


def _isolate_policy_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(competition, "load_cross_game_value_head", lambda: None)
    monkeypatch.setattr(competition, "_load_submitted_candidate_router", lambda game_id: None)
    monkeypatch.setattr(competition, "_load_submitted_frame_change_scorer", lambda: None)
    monkeypatch.setattr(competition, "_load_submitted_goal_energy_bias", lambda: None)
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")


def _policy_kwargs() -> dict:
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


def _drive(policy: competition.E3AgentPolicy, count: int = 16) -> bytes:
    frames: list[_Frame] = []
    latest = None
    actions = []
    for index in range(count):
        action, data = policy.next_move(frames, latest)
        actions.append({"action": action, "data": data})
        if action is None:
            break
        grid = np.random.RandomState(index + 101).randint(0, 4, size=(6, 6)).astype(int)
        latest = _Frame(grid)
        frames.append(latest)
    return json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()


def test_req_7024_spec_exists_before_implementation() -> None:
    """REQ-ARC-WMTE-7024 names all behavior tested below."""

    spec = (REPO_ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md").read_text()
    assert "REQ-ARC-WMTE-7024" in spec
    assert {
        "SCENARIO-ARC-WMTE-7024-DEFAULT-OFF-WIRING",
        "SCENARIO-ARC-WMTE-7024-QUERY-INFLUENCE",
        "SCENARIO-ARC-WMTE-7024-SAFE-ABSTENTION",
        "SCENARIO-ARC-WMTE-7024-ACTION-PROVENANCE",
        "SCENARIO-ARC-WMTE-7024-NO-HIDDEN-ACCESS",
        "SCENARIO-ARC-WMTE-7024-READINESS-GATE",
    } <= {line.removeprefix("#### ") for line in spec.splitlines()}


def test_scenario_7024_flag_parsing_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7024-DEFAULT-OFF-WIRING uses exact opt-in parsing."""

    for raw, expected in ((None, False), ("", False), ("0", False), ("true", False), ("1", True)):
        if raw is None:
            monkeypatch.delenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, raising=False)
        else:
            monkeypatch.setenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, raw)
        assert selector_mod.belief_selector_enabled() is expected
    monkeypatch.setenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, "1")
    assert selector_mod.belief_selector_enabled(False) is False
    monkeypatch.setenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, "0")
    assert selector_mod.belief_selector_enabled(True) is True


def test_scenario_7024_supported_query_changes_ranking_and_keeps_candidate_set(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-QUERY-INFLUENCE records each score term."""

    ledger, event = _known_ledger(tmp_path / "known")
    candidates = _supported_candidates(event)
    selector = selector_mod.BeliefAwareE3Selector(ledger, belief_weight=0.25)
    ranked = selector.rank_candidates(None, candidates)
    decision = selector.last_decision

    assert [row["action"] for row in ranked] == [2, 1]
    assert {id(row) for row in ranked} == {id(row) for row in candidates}
    assert decision["query_fired"] is True
    assert decision["query_count"] == 2
    assert decision["ranking_changed"] is True
    assert decision["abstained"] is False
    assert decision["selected_action"] == {"action": 2, "data": None}
    assert decision["evidence_hashes"]
    rows = decision["candidate_score_rows"]
    assert len(rows) == 2
    assert rows[0]["base_score"] == pytest.approx(0.60)
    assert rows[0]["simulation_contribution"] == pytest.approx(0.05)
    assert rows[0]["belief_contribution"] == pytest.approx(0.0)
    assert rows[1]["belief_contribution"] == pytest.approx(0.25)
    assert sum(bool(row["selected"]) for row in rows) == 1
    assert all(row["selected_action"] == {"action": 2, "data": None} for row in rows)


@pytest.mark.parametrize(
    ("fixture", "expected_reason"),
    [
        ("low_support", "low_support"),
        ("conflict", "conflicting_evidence"),
        ("stale", "stale_evidence"),
        ("truncated", "budget_truncated"),
    ],
)
def test_scenario_7024_query_rejections_abstain_without_reordering(
    tmp_path: Path,
    fixture: str,
    expected_reason: str,
) -> None:
    """SCENARIO-ARC-WMTE-7024-SAFE-ABSTENTION fails closed for unsafe query results."""

    root = tmp_path / fixture
    if fixture == "low_support":
        event = _event(0, "left", mechanic=fixture)
        ledger = ledger_mod.BeliefLedger(root, capacity=16, min_support=2)
        ledger.observe(event)
        observation_index = 0
    elif fixture == "conflict":
        ledger, event = _known_ledger(root, mechanic=fixture)
        ledger.observe(_event(2, "right", mechanic=fixture))
        observation_index = 2
    elif fixture == "stale":
        ledger, event = _known_ledger(root, mechanic=fixture)
        observation_index = 40
    else:
        event = _event(0, "left", mechanic=fixture)
        ledger = ledger_mod.BeliefLedger(root, capacity=16, min_support=2)
        for index in range(9):
            ledger.observe(_event(index, "left", mechanic=fixture))
        observation_index = 8
    candidates = _supported_candidates(event, observation_index=observation_index)
    selector = selector_mod.BeliefAwareE3Selector(ledger)

    ranked = selector.rank_candidates(None, candidates)

    assert ranked == candidates
    assert [id(row) for row in ranked] == [id(row) for row in candidates]
    assert selector.last_decision["query_fired"] is True
    assert selector.last_decision["abstained"] is True
    assert selector.last_decision["abstention_reason"] == expected_reason
    assert selector.last_decision["ranking_changed"] is False
    assert len(ranked) == len(candidates) > 0


def test_scenario_7024_missing_ledger_and_malformed_evidence_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7024 records missing and malformed input without a query."""

    ledger, event = _known_ledger(tmp_path / "known")
    candidates = _supported_candidates(event)
    missing = selector_mod.BeliefAwareE3Selector(None)
    assert missing.rank_candidates(None, candidates) == candidates
    assert missing.last_decision["abstention_reason"] == "missing_ledger"
    assert missing.last_decision["query_fired"] is False

    malformed_candidates = deepcopy(candidates)
    malformed_candidates[0]["belief_outcome_key"] = "not-a-hash"
    malformed = selector_mod.BeliefAwareE3Selector(ledger)
    assert malformed.rank_candidates(None, malformed_candidates) == malformed_candidates
    assert malformed.last_decision["abstention_reason"] == "malformed_evidence"
    assert malformed.last_decision["query_fired"] is False

    absent = selector_mod.BeliefAwareE3Selector(ledger)
    assert absent.rank_candidates(None, [{"action": 1, "data": None}]) == [
        {"action": 1, "data": None}
    ]
    assert absent.last_decision["abstention_reason"] == "missing_candidate_evidence"


def test_req_7024_selector_rejects_every_malformed_candidate_shape(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7024 validates scores, actions, signatures, indexes, and query results."""

    ledger, event = _known_ledger(tmp_path / "known")
    valid = _supported_candidates(event)
    typed = deepcopy(valid)
    typed[0]["belief_mechanic_signature"] = ledger_mod.mechanic_key_from_event(event)
    assert selector_mod.BeliefAwareE3Selector(ledger).rank_candidates(None, typed)[0]["action"] == 2

    for field, value in (
        ("action", True),
        ("data", []),
        ("base_score", "bad"),
        ("simulation_contribution", float("inf")),
        ("belief_mechanic_signature", object()),
        ("belief_observation_index", -1),
    ):
        malformed = deepcopy(valid)
        malformed[0][field] = value
        selector = selector_mod.BeliefAwareE3Selector(ledger)
        assert selector.rank_candidates(None, malformed) == malformed
        assert selector.last_decision["abstention_reason"] == "malformed_evidence"
        assert selector.last_decision["ranking_changed"] is False

    for weight in (True, "bad", -0.1, float("nan")):
        with pytest.raises(ValueError, match="belief_weight"):
            selector_mod.BeliefAwareE3Selector(ledger, belief_weight=weight)  # type: ignore[arg-type]

    empty = selector_mod.BeliefAwareE3Selector(ledger)
    assert empty.rank_candidates(None, []) == []
    assert empty.last_decision["abstention_reason"] == "no_legal_candidates"

    class _ForeignQuery:
        def query(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"abstained": False}

    foreign = selector_mod.BeliefAwareE3Selector(ledger, query_api=_ForeignQuery())
    assert foreign.rank_candidates(None, valid) == valid
    assert foreign.last_decision["query_fired"] is True
    assert foreign.last_decision["abstention_reason"] == "malformed_evidence"


def test_scenario_7024_constructor_and_factory_pass_the_same_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-DEFAULT-OFF-WIRING proves live factory propagation."""

    _isolate_policy_loaders(monkeypatch)
    ledger, _ = _known_ledger(tmp_path / "known")
    off = competition.E3AgentPolicy(
        "fixture",
        belief_ledger=ledger,
        belief_aware_selector=False,
        **_policy_kwargs(),
    )
    assert off.belief_ledger is ledger
    assert off.belief_aware_selector_enabled is False
    assert off.belief_candidate_selector is None
    assert off.explorer.belief_candidate_selector is None

    on = competition.E3AgentPolicy(
        "fixture",
        belief_ledger=ledger,
        belief_aware_selector=True,
        **_policy_kwargs(),
    )
    assert on.belief_ledger is ledger
    assert on.belief_candidate_selector.ledger is ledger
    assert on.explorer.belief_candidate_selector is on.belief_candidate_selector

    class _Base:
        def __init__(self) -> None:
            self.game_id = "factory-fixture"

    agent_type = competition.make_carnot_agent(
        _Base,
        belief_ledger=ledger,
        belief_aware_selector=True,
    )
    agent = agent_type()
    assert isinstance(agent._policy, competition.E3AgentPolicy)
    assert agent._policy.belief_ledger is ledger
    assert agent._policy.belief_candidate_selector.ledger is ledger


def test_scenario_7024_factory_reaches_query_through_candidate_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-QUERY-INFLUENCE fires inside `_candidates`."""

    _isolate_policy_loaders(monkeypatch)
    ledger, event = _known_ledger(tmp_path / "known")

    class _Base:
        def __init__(self) -> None:
            self.game_id = "factory-fixture"

    known_outcome = ledger_mod._sha256_json({"outcome": "left"})
    other_outcome = ledger_mod._sha256_json({"outcome": "right"})
    mechanic = ledger_mod.mechanic_key_from_event(event).to_dict()

    class _PriorEvidenceAnnotator:
        def rank_candidates(self, _frame, rows, **_kwargs):  # type: ignore[no-untyped-def]
            self.last_actions = [row["action"] for row in rows]
            return [
                {
                    **row,
                    "base_score": 0.60 if index == 0 else (0.55 if index == 1 else -100.0),
                    "simulation_contribution": 0.05 if index == 0 else 0.0,
                    "belief_mechanic_signature": mechanic,
                    "belief_outcome_key": known_outcome if index == 1 else other_outcome,
                    "belief_observation_index": 1,
                }
                for index, row in enumerate(rows)
            ]

    agent_type = competition.make_carnot_agent(
        _Base,
        belief_ledger=ledger,
        belief_aware_selector=True,
    )
    agent = agent_type()
    agent._policy.explorer.structured_evidence_memory = _PriorEvidenceAnnotator()

    rows = agent._policy.explorer._candidates(_Frame(np.zeros((6, 6), dtype=int)))
    decision = agent._policy.belief_candidate_selector.last_decision

    assert [row["action"] for row in rows[:2]] == [2, 1]
    assert decision["query_fired"] is True
    assert decision["ranking_changed"] is True
    assert decision["base_candidate_set_preserved"] is True

    class _RaisingSelector:
        def rank_candidates(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("deterministic selector failure")

    agent._policy.explorer.belief_candidate_selector = _RaisingSelector()
    fallback = agent._policy.explorer._candidates(_Frame(np.zeros((6, 6), dtype=int)))
    assert [row["action"] for row in fallback] == (
        agent._policy.explorer.structured_evidence_memory.last_actions
    )


def test_scenario_7024_flag_off_action_trace_matches_frozen_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7024-DEFAULT-OFF-WIRING pins disabled action bytes."""

    _isolate_policy_loaders(monkeypatch)
    monkeypatch.delenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, raising=False)
    first = competition.E3AgentPolicy("belief-fixture", **_policy_kwargs())
    second = competition.E3AgentPolicy(
        "belief-fixture",
        belief_aware_selector=False,
        **_policy_kwargs(),
    )
    first_trace = _drive(first)
    second_trace = _drive(second)

    assert first.belief_candidate_selector is None
    assert first_trace == second_trace
    assert selector_mod.sha256_bytes(first_trace) == FROZEN_OFF_TRACE_SHA256


def test_scenario_7024_action_provenance_records_selector_influence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-ACTION-PROVENANCE joins the final action to evidence."""

    _isolate_policy_loaders(monkeypatch)
    monkeypatch.setenv(action_provenance.PROVENANCE_ENV_FLAG, "1")
    ledger, event = _known_ledger(tmp_path / "known")
    policy = competition.E3AgentPolicy(
        "fixture",
        belief_ledger=ledger,
        belief_aware_selector=True,
        **_policy_kwargs(),
    )
    selector = policy.belief_candidate_selector
    ranked = selector.rank_candidates(None, _supported_candidates(event))
    move = (ranked[0]["action"], ranked[0]["data"])
    policy._prov_top = "explore.explorer"
    pre = policy._provenance_pre_state([], None)
    row = policy._provenance_post_state(pre, move, policy.plan, [], None)

    assert row["belief_query_fired"] is True
    assert row["belief_evidence_hashes"] == selector.last_decision["evidence_hashes"]
    assert row["belief_influence"] is True
    assert row["belief_abstained"] is False
    assert row["belief_abstention_reason"] is None
    assert row["belief_ranking_changed"] is True
    assert row["belief_final_action"] == {"action": 2, "data": None}


def test_scenario_7024_selector_does_not_read_source_or_adapter_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-NO-HIDDEN-ACCESS reads only bounded candidate fields."""

    ledger, event = _known_ledger(tmp_path / "known")
    guarded = []
    for candidate in _supported_candidates(event):
        guarded.append(
            _ForbiddenAccess(
                {
                    **candidate,
                    "game_source": object(),
                    "adapter": object(),
                    "solve_registry": object(),
                    "current_outcome": object(),
                    "future_label": object(),
                }
            )
        )
    selector = selector_mod.BeliefAwareE3Selector(ledger)

    ranked = selector.rank_candidates(object(), guarded)

    assert len(ranked) == len(guarded)
    assert sum(row.forbidden_access_count for row in guarded) == 0
    assert selector.forbidden_access_counts == {
        "game_source_reads": 0,
        "adapter_reads": 0,
        "solve_registry_reads": 0,
        "current_outcome_reads": 0,
        "future_label_reads": 0,
    }


def test_scenario_7024_artifact_is_complete_and_reproducible(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7024-READINESS-GATE derives readiness from every row."""

    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        run_date="20260905",
        output_path=tmp_path / "experiment_7024.json",
    )

    assert set(selector_mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(artifact["field_principles"]) == set(selector_mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == ("deterministic_live_policy_wiring_fixtures_no_llm")
    assert artifact["shipped_default_unchanged"] is True
    assert artifact["belief_selector_live_path_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    live_row = next(
        row
        for row in artifact["live_path_reachability_rows"]
        if row["fixture"] == "factory_to_selector_query"
    )
    assert live_row["policy_candidate_seam"] == "StepwiseExplorer._candidates"
    assert live_row["query_fired"] is True
    assert artifact["reproducibility_checksum"] == selector_mod.artifact_checksum(artifact)
    assert selector_mod.validate_artifact(artifact) == []

    rebuilt = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        run_date="20260905",
        output_path=tmp_path / "experiment_7024_second.json",
    )
    assert artifact["reproducibility_checksum"] == rebuilt["reproducibility_checksum"]


def test_req_7024_precondition_helpers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7024 treats unreadable inputs and import failures as blocked gates."""

    assert selector_mod._path_is_writable(tmp_path / "missing" / "nested" / "result.json")
    assert selector_mod._read_json(tmp_path / "missing.json") is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert selector_mod._read_json(malformed) is None
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert selector_mod._read_json(array) is None

    real_import = selector_mod.importlib.import_module

    def _raise_one(name: str):
        if name == "carnot.agentic.arc_eval_provenance":
            raise ImportError("fixture")
        return real_import(name)

    monkeypatch.setattr(selector_mod.importlib, "import_module", _raise_one)
    checks, _ = selector_mod.collect_preconditions(REPO_ROOT, tmp_path / "result.json")
    by_name = {row["check"]: row for row in checks}
    assert by_name["eval_provenance_contract"]["passed"] is False


def test_req_7024_fixture_restores_preexisting_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7024 fixtures do not leak their flags into later policy tests."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "already-set")
    monkeypatch.setenv(action_provenance.PROVENANCE_ENV_FLAG, "0")
    monkeypatch.setenv(selector_mod.BELIEF_SELECTOR_ENV_FLAG, "previous")
    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "environment.json",
    )
    assert artifact["belief_selector_live_path_ready_score"] == 1
    assert selector_mod.os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "already-set"
    assert selector_mod.os.environ[action_provenance.PROVENANCE_ENV_FLAG] == "0"
    assert selector_mod.os.environ[selector_mod.BELIEF_SELECTOR_ENV_FLAG] == "previous"


def test_req_7024_off_trace_stops_on_terminal_policy_action() -> None:
    """REQ-ARC-WMTE-7024 trace serialization keeps a terminal action in its bytes."""

    class _TerminalPolicy:
        def next_move(self, _frames, _latest):  # type: ignore[no-untyped-def]
            return (None, None)

    class _PolicyModule:
        @staticmethod
        def E3AgentPolicy(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return _TerminalPolicy()

    assert json.loads(selector_mod._off_trace(_PolicyModule())) == [{"action": None, "data": None}]


def test_scenario_7024_blocked_precondition_names_exact_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7024-READINESS-GATE writes an exact blocked verdict."""

    monkeypatch.setattr(selector_mod, "EXPECTED_EXP7023_HASH", "sha256:" + "0" * 64)
    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "blocked.json",
    )

    assert artifact["belief_selector_live_path_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_belief_aware_e3_selector"
    assert artifact["gate_check_summary"]["failed_check"] == "exp7023_artifact_hash"
    assert artifact["gate_check_summary"]["expected_value"] == "sha256:" + "0" * 64
    assert artifact["gate_check_summary"]["observed_value"] == (
        "sha256:f1890eb3dffba8caed3dfeccbbe3cff3219693755f1fd0d99d059b50f0612d9d"
    )
    assert selector_mod.validate_artifact(artifact) == []


def test_req_7024_artifact_validator_rejects_inconsistent_ready_rows(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7024 validation does not trust a self-reported ready score."""

    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "valid.json",
    )
    bad = deepcopy(artifact)
    bad["query_firing_rows"][0]["passed"] = False
    bad["reproducibility_checksum"] = selector_mod.artifact_checksum(bad)
    assert "ready_rows_failed:query_firing_rows" in selector_mod.validate_artifact(bad)

    bad_checksum = deepcopy(artifact)
    bad_checksum["shipped_default_unchanged"] = False
    assert "reproducibility_checksum_mismatch" in selector_mod.validate_artifact(bad_checksum)


def test_req_7024_artifact_validator_rejection_matrix(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7024 rejects every required structural contradiction."""

    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "valid.json",
    )
    assert selector_mod.validate_artifact(None) == ["artifact_object_required"]

    def errors_for(change) -> list[str]:  # type: ignore[no-untyped-def]
        candidate = deepcopy(artifact)
        change(candidate)
        candidate["reproducibility_checksum"] = selector_mod.artifact_checksum(candidate)
        return selector_mod.validate_artifact(candidate)

    assert any(
        value.startswith("required_fields_missing:")
        for value in errors_for(lambda row: row.pop("random_seed"))
    )
    assert "field_principles_mismatch" in errors_for(
        lambda row: row.__setitem__("field_principles", {})
    )
    assert "ready_score_not_bare_integer" in errors_for(
        lambda row: row.__setitem__("belief_selector_live_path_ready_score", True)
    )
    assert "verdict_class_invalid" in errors_for(
        lambda row: row.__setitem__("verdict_class", "unknown")
    )
    assert "verdict_prefix_mismatch" in errors_for(
        lambda row: row.__setitem__("honest_verdict", "blocked_wrong_prefix")
    )
    assert "factory_wiring_rows_not_list" in errors_for(
        lambda row: row.__setitem__("factory_wiring_rows", {})
    )
    assert "nonterminal_row:factory_wiring_rows" in errors_for(
        lambda row: row["factory_wiring_rows"][0].pop("terminal")
    )
    assert "inference_substrate_mismatch" in errors_for(
        lambda row: row.__setitem__("inference_substrate", "wrong")
    )
    assert "verifier_is_oracle_mismatch" in errors_for(
        lambda row: row.__setitem__("verifier_is_oracle", True)
    )
    assert "gate_check_summary_invalid" in errors_for(
        lambda row: row.__setitem__("gate_check_summary", {})
    )
    assert "ready_gate_inconsistent" in errors_for(
        lambda row: row.__setitem__("shipped_default_unchanged", False)
    )


def test_req_7024_writer_and_main_publish_only_valid_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-7024 validates before each atomic artifact publication."""

    artifact = selector_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "source.json",
    )
    output = tmp_path / "written.json"
    selector_mod.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    invalid = deepcopy(artifact)
    invalid["verifier_is_oracle"] = True
    with pytest.raises(ValueError, match="invalid Exp7024 artifact"):
        selector_mod.write_artifact(tmp_path / "invalid.json", invalid)

    def _replace_failure(_source, _target):  # type: ignore[no-untyped-def]
        raise OSError("replace fixture")

    monkeypatch.setattr(selector_mod.os, "replace", _replace_failure)
    with pytest.raises(OSError, match="replace fixture"):
        selector_mod.write_artifact(tmp_path / "replace-failure.json", artifact)
    assert not list(tmp_path.glob(".replace-failure.json.*"))
    monkeypatch.undo()

    main_output = tmp_path / "main.json"
    assert selector_mod.main(["--date", "20260905", "--output", str(main_output)]) == 0
    assert (
        json.loads(main_output.read_text(encoding="utf-8"))["belief_selector_live_path_ready_score"]
        == 1
    )
    assert "belief_selector_live_path_ready_score=1" in capsys.readouterr().out


def test_req_7024_module_entrypoint_runs_the_same_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7024 keeps the direct module command executable."""

    output = tmp_path / "module-main.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arc_belief_aware_e3_selector",
            "--date",
            "20260905",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(
            "carnot.agentic.arc_belief_aware_e3_selector",
            run_name="__main__",
        )
    assert stopped.value.code == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["belief_selector_live_path_ready_score"] == 1
    )


def test_req_7024_wrapper_exists_and_does_not_enable_submission_default() -> None:
    """REQ-ARC-WMTE-7024 keeps the experiment wrapper outside the submission kernel."""

    wrapper = REPO_ROOT / "scripts/experiments/experiment_7024_belief_aware_e3_selector.py"
    assert wrapper.is_file()
    assert competition.SUBMITTED_AGENT_CONFIG["belief_aware_selector_enabled"] is False
    kernel = REPO_ROOT / "submission_kernel/main.py"
    if kernel.exists():
        assert selector_mod.BELIEF_SELECTOR_ENV_FLAG not in kernel.read_text(encoding="utf-8")
