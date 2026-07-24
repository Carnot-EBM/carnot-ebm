"""Tests for Exp5900 structured evidence memory.

Spec refs: REQ-ARC-WMTE-5900,
SCENARIO-ARC-WMTE-5900-IDENTICAL-BYTES,
SCENARIO-ARC-WMTE-5900-BOUNDS-DELETE-RESTART-STALE,
SCENARIO-ARC-WMTE-5900-LIVE-E3-REACHABILITY.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from carnot.agentic import arc_competition_agent as competition
from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG
from carnot.agentic.arc_structured_evidence_memory import (
    StructuredEvidenceConfig,
    StructuredEvidenceMemory,
    coerce_structured_evidence_memory,
    registry_precheck,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / "results/experiment_5900_arc_structured_evidence_memory_contract.json"
REQUIRED_RESULT_FIELDS = [
    "status",
    "preconditions_checked",
    "registry_precheck",
    "public_level_solve_claimed",
    "submitted_live_path_receipt",
    "feature_default_off_and_disabled_equivalence",
    "agent_owned_event_schema",
    "raw_tape_and_structured_index_identical_byte_receipt",
    "structured_evidence_index_schema",
    "provenance_and_authority_receipts",
    "bounded_bytes_queries_and_eviction",
    "deletion_restart_stale_evidence_receipts",
    "source_bfs_adapter_and_prior_game_access_count",
    "live_path_reachability_tests",
    "protected_files_unchanged",
    "structured_evidence_memory_contract_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
]


def _frame(grid: list[list[int]], *, actions: tuple[int, ...] = (1, 2), level: int = 0):
    return SimpleNamespace(
        frame=np.asarray(grid, dtype=np.int16),
        available_actions=list(actions),
        levels_completed=level,
    )


def _minimal_e3(*, memory: StructuredEvidenceMemory | bool | None) -> E3AgentPolicy:
    return E3AgentPolicy(
        "zz99",
        proposer=None,
        explore_budget=10,
        target_levels=1,
        value_head=lambda *_args, **_kwargs: 0.0,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        action_prior=None,
        candidate_router=None,
        goal_bias=None,
        goal_candidate_guidance=False,
        qd_generator=False,
        controllable_novelty=False,
        object_centric_proposal=False,
        program_synthesis_filter=False,
        inert_click_pruner=False,
        object_history_salience=False,
        amortized_first_contact_prior=False,
        go_explore_archive=False,
        similarity_retrieval=False,
        epistemic_ledger=False,
        structured_evidence_memory=memory,
    )


def test_req_arc_wmte_5900_spec_and_default_off_contract() -> None:
    """REQ-ARC-WMTE-5900: OpenSpec exists and submitted path keeps the feature off."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5900") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5900-IDENTICAL-BYTES",
        "SCENARIO-ARC-WMTE-5900-BOUNDS-DELETE-RESTART-STALE",
        "SCENARIO-ARC-WMTE-5900-LIVE-E3-REACHABILITY",
        "results/experiment_5900_arc_structured_evidence_memory_contract.json",
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    ):
        assert marker in section

    assert competition.SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED is False
    assert SUBMITTED_AGENT_CONFIG["structured_evidence_memory_enabled"] is False
    assert coerce_structured_evidence_memory(False) is None
    assert coerce_structured_evidence_memory(None) is None
    assert isinstance(coerce_structured_evidence_memory(True), StructuredEvidenceMemory)

    default_policy = _minimal_e3(memory=None)
    disabled_policy = _minimal_e3(memory=False)
    assert default_policy.structured_evidence_memory is None
    assert disabled_policy.structured_evidence_memory is None
    assert default_policy.next_move([], None) == disabled_policy.next_move([], None)


def test_scenario_arc_wmte_5900_identical_tape_and_index_bytes() -> None:
    """SCENARIO-ARC-WMTE-5900-IDENTICAL-BYTES: structure adds no extra evidence."""

    before = _frame([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    after = _frame([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    memory = StructuredEvidenceMemory(
        config=StructuredEvidenceConfig(max_events=32, max_bytes=20000, stale_after_events=8)
    )

    memory.observe_state(before, phase="explore", provenance={"source": "unit"})
    memory.observe_candidates(
        before,
        [{"action": 1, "data": None}, {"action": 2, "data": None}],
        provenance={"source": "unit_candidates"},
    )
    memory.observe_action_candidate(
        1,
        None,
        uncertainty={"score": 0.4, "label": "ambiguous"},
        provenance={"source": "unit_action"},
    )
    memory.observe_action_result(
        before,
        1,
        None,
        before,
        level_before=0,
        level_after=0,
        provenance={"source": "unit_noop"},
    )
    memory.observe_action_result(
        before,
        2,
        None,
        after,
        level_before=0,
        level_after=0,
        provenance={"source": "unit_change"},
    )

    raw = memory.query_raw(event_type="action_result")
    indexed = memory.query_index(event_type="action_result")
    assert raw["source_event_ids"] == indexed["source_event_ids"]
    assert raw["source_event_hashes"] == indexed["source_event_hashes"]
    assert raw["tape_hash"] == indexed["tape_hash"]
    assert indexed["index_hash"] == memory.index_hash()
    assert {entry["action_effect"]["outcome"] for entry in indexed["index_entries"]} == {
        "noop",
        "visible_change",
    }

    ranked = memory.rank_candidates(
        before,
        [{"action": 1, "data": None}, {"action": 2, "data": None}],
        provenance={"source": "unit_rank"},
    )
    assert [row["action"] for row in ranked] == [2, 1]
    diagnostics = memory.diagnostics()
    assert diagnostics["raw_query_count"] >= 1
    assert diagnostics["structured_query_count"] >= 1
    assert diagnostics["rank_consumed_count"] == 1
    assert diagnostics["authority"]["source_bfs_adapter_and_prior_game_access_count"] == 0


def test_scenario_arc_wmte_5900_bounds_delete_restart_and_stale_receipts() -> None:
    """SCENARIO-ARC-WMTE-5900-BOUNDS-DELETE-RESTART-STALE: bounded lifecycle."""

    memory = StructuredEvidenceMemory(
        config=StructuredEvidenceConfig(
            max_events=5,
            max_bytes=16000,
            max_query_events=3,
            max_query_bytes=2000,
            max_queries=8,
            stale_after_events=1,
        )
    )
    for i in range(8):
        memory.observe_state(
            _frame([[i % 3, 0], [0, i % 5]]),
            phase="explore",
            provenance={"source": f"step-{i}"},
        )

    diag = memory.diagnostics()
    assert diag["retained_event_count"] <= 5
    assert diag["retained_byte_count"] <= 16000
    assert diag["loss_receipt_count"] > 0

    raw_fresh = memory.query_raw(include_stale=False)
    indexed_fresh = memory.query_index(include_stale=False)
    assert raw_fresh["source_event_ids"] == indexed_fresh["source_event_ids"]
    assert raw_fresh["stale_event_ids"] == indexed_fresh["stale_event_ids"]
    assert len(raw_fresh["events"]) <= 3
    assert raw_fresh["query_byte_count"] <= 2000

    tape_bytes = memory.tape_bytes()
    restarted = StructuredEvidenceMemory.from_tape_bytes(tape_bytes, config=memory.config)
    assert restarted.tape_hash() == memory.tape_hash()
    assert restarted.index_hash() == memory.index_hash()

    memory.delete()
    assert memory.diagnostics()["retained_event_count"] == 0
    assert memory.query_raw()["source_event_ids"] == []

    limited = StructuredEvidenceMemory(config=StructuredEvidenceConfig(max_queries=1))
    limited.observe_state(_frame([[0]]), phase="explore")
    assert limited.query_raw()["query_limit_exceeded"] is False
    assert limited.query_raw()["query_limit_exceeded"] is True


def test_scenario_arc_wmte_5900_live_e3_constructs_updates_queries_and_consumes(
    monkeypatch,
) -> None:
    """SCENARIO-ARC-WMTE-5900-LIVE-E3-REACHABILITY: enabled path exercises both arms."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    before = _frame([[0, 0], [1, 0]])
    after = _frame([[0, 2], [1, 0]])
    memory = StructuredEvidenceMemory(
        config=StructuredEvidenceConfig(max_events=64, max_bytes=40000, stale_after_events=16)
    )
    memory.observe_action_result(
        before,
        1,
        None,
        before,
        level_before=0,
        level_after=0,
        provenance={"source": "prior_live_noop"},
    )

    policy = _minimal_e3(memory=memory)

    assert policy.structured_evidence_memory is memory
    assert policy.explorer.structured_evidence_memory is memory
    assert policy.next_move([], None) == ("RESET", None)
    first_move = policy.next_move([before], before)
    assert first_move == (2, None)
    policy.next_move([before, after], after)

    diagnostics = memory.diagnostics()
    assert diagnostics["append_count"] >= 4
    assert diagnostics["raw_query_count"] > 0
    assert diagnostics["structured_query_count"] > 0
    assert diagnostics["rank_consumed_count"] > 0
    assert diagnostics["authority"]["source_bfs_adapter_and_prior_game_access_count"] == 0
    assert diagnostics["public_level_solve_claimed"] is False
    assert memory.query_raw()["source_event_ids"] == memory.query_index()["source_event_ids"]


def test_req_arc_wmte_5900_registry_precheck_and_repository_artifact() -> None:
    """REQ-ARC-WMTE-5900: Exp5900 artifact carries the required contract fields."""

    precheck = registry_precheck(REPO)
    assert precheck["ok"] is True
    assert precheck["public_game_count"] == 25
    assert precheck["duplicate_public_solve_target_prohibited"] is True
    assert precheck["no_level_solve_targeted"] is True

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact
    assert artifact["status"] == "ready"
    assert artifact["public_level_solve_claimed"] is False
    assert artifact["structured_evidence_memory_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["source_bfs_adapter_and_prior_game_access_count"] == 0
    assert artifact["protected_files_unchanged"]["scripts/research_conductor.py"] is True
    assert artifact["honest_verdict"].startswith("ready:")
