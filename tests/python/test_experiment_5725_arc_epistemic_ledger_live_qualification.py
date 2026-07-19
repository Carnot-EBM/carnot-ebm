"""Tests for Exp5725 ARC epistemic-ledger live qualification.

Spec refs: REQ-ARC-WMTE-5725,
SCENARIO-ARC-WMTE-5725-LEDGER-STATE-AND-CONTROLS,
SCENARIO-ARC-WMTE-5725-LIVE-E3-REACHABILITY,
SCENARIO-ARC-WMTE-5725-QUALIFICATION-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from carnot import experiment_5725_arc_epistemic_ledger_live_qualification as mod
from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG
from carnot.agentic import arc_epistemic_ledger as ledger_mod
from carnot.agentic.arc_epistemic_ledger import (
    AgentEpistemicLedger,
    LedgerConfig,
    candidate_signature,
    coerce_epistemic_ledger,
    stable_state_hash,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _frame(grid: list[list[int]], *, actions: tuple[int, ...] = (1, 2), level: int = 0):
    return SimpleNamespace(
        frame=np.asarray(grid, dtype=np.int16),
        available_actions=list(actions),
        levels_completed=level,
    )


def _minimal_e3(*, ledger: AgentEpistemicLedger) -> E3AgentPolicy:
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
        epistemic_ledger=ledger,
    )


def test_req_arc_wmte_5725_spec_declares_schema_and_gates() -> None:
    """REQ-ARC-WMTE-5725: OpenSpec anchors the ledger schema and artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5725") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5725-LEDGER-STATE-AND-CONTROLS",
        "SCENARIO-ARC-WMTE-5725-LIVE-E3-REACHABILITY",
        "SCENARIO-ARC-WMTE-5725-QUALIFICATION-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
        "confirmed observation/action facts",
        "active hypotheses ranked by support minus",
        "open discriminating questions",
        "superseded entries",
        "arc_epistemic_ledger_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_5725_ledger_state_updates_rank_and_recover() -> None:
    """SCENARIO-ARC-WMTE-5725-LEDGER-STATE-AND-CONTROLS: exact state transitions."""

    before = _frame([[0, 0], [1, 0]])
    after = _frame([[0, 0], [1, 0]])
    changed = _frame([[0, 2], [1, 0]])
    candidates = [{"action": 1, "data": None}, {"action": 2, "data": None}]
    ledger = AgentEpistemicLedger(
        config=LedgerConfig(min_support_to_commit=2, stale_after_steps=4)
    )

    ledger.observe_state(before, runtime_receipts={"candidate_count": 2})
    assert ledger.rank_candidates(before, candidates) == candidates

    ledger.observe_transition(before, 1, None, after, level_before=0, level_after=0)
    ledger.observe_transition(before, 1, None, after, level_before=0, level_after=0)
    ranked = ledger.rank_candidates(before, candidates)

    assert [row["action"] for row in ranked] == [2, 1]
    snapshot = ledger.snapshot()
    noop_hyp = [
        row for row in snapshot["active_hypotheses"] if row["kind"] == "repeated_noop"
    ][0]
    assert noop_hyp["support_count"] == 2
    assert noop_hyp["contradiction_count"] == 0
    assert snapshot["commitments"][-1]["reason"] == "evidence_sufficient_repeated_noop"
    assert ledger.diagnostics()["open_question_resolution_count"] >= 1

    ledger.observe_transition(before, 1, None, changed, level_before=0, level_after=0)
    recovered = ledger.rank_candidates(before, candidates)
    assert recovered == candidates
    assert ledger.diagnostics()["unsafe_commit_count"] == 0
    assert any(row["reason"] == "contradicted" for row in ledger.snapshot()["superseded_entries"])


def test_scenario_arc_wmte_5725_controls_cover_stale_conflict_and_fallbacks() -> None:
    """REQ-ARC-WMTE-5725: control suite covers stale/conflict/corrupt/disabled cases."""

    controls = mod.run_synthetic_controls()
    by_name = {row["name"]: row for row in controls}

    assert set(by_name) == {
        "exact_positive_visible_change",
        "repeated_noop_demotes",
        "stale_evidence",
        "contradiction_recovery",
        "misleading_hypothesis",
        "shuffled_links",
        "missing_observation",
        "corrupted_hash",
        "always_commit",
        "never_commit",
        "ledger_disabled",
    }
    assert by_name["repeated_noop_demotes"]["candidate_order_changed"] is True
    assert by_name["exact_positive_visible_change"]["action_order_changed"] is True
    assert by_name["stale_evidence"]["safe_fallback"] is True
    assert by_name["contradiction_recovery"]["safe_fallback"] is True
    assert by_name["misleading_hypothesis"]["false_commit_count"] == 0
    assert by_name["corrupted_hash"]["safe_fallback"] is True
    assert by_name["always_commit"]["unsafe_commit_count"] == 1
    assert by_name["never_commit"]["commitment_count"] == 0
    assert by_name["ledger_disabled"]["fallback_equivalence"] is True


def test_scenario_arc_wmte_5725_submitted_e3_reaches_live_hooks(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5725-LIVE-E3-REACHABILITY: E3 reads/writes the ledger."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    before = _frame([[0, 0], [1, 0]])
    after = _frame([[0, 2], [1, 0]])
    ledger = AgentEpistemicLedger(
        config=LedgerConfig(min_support_to_commit=2, stale_after_steps=8)
    )
    ledger.observe_transition(before, 1, None, before, level_before=0, level_after=0)
    ledger.observe_transition(before, 1, None, before, level_before=0, level_after=0)

    policy = _minimal_e3(ledger=ledger)

    assert SUBMITTED_AGENT_CONFIG["policy"] == "E3AgentPolicy"
    assert SUBMITTED_AGENT_CONFIG["epistemic_ledger_enabled"] is True
    assert policy.next_move([], None) == ("RESET", None)
    first_move = policy.next_move([before], before)
    assert first_move == (2, None)
    policy.next_move([before, after], after)

    diagnostics = ledger.diagnostics()
    assert diagnostics["live_read_call_count"] > 0
    assert diagnostics["live_write_call_count"] > 0
    assert diagnostics["candidate_order_change_count"] > 0
    assert diagnostics["action_order_change_count"] > 0
    assert diagnostics["commitment_count"] > 0
    assert diagnostics["false_commit_count"] == 0
    assert diagnostics["unsafe_commit_count"] == 0


def test_req_arc_wmte_5725_artifact_builder_sets_ready_gate() -> None:
    """SCENARIO-ARC-WMTE-5725-QUALIFICATION-ARTIFACT: build emits required fields."""

    artifact = mod.build_artifact(root=REPO)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["openspec_requirement_ids"] == ["REQ-ARC-WMTE-5725"]
    assert artifact["live_read_call_count"] > 0
    assert artifact["live_write_call_count"] > 0
    assert artifact["candidate_order_change_count"] > 0
    assert artifact["action_order_change_count"] > 0
    assert artifact["commitment_count"] > 0
    assert artifact["false_commit_count"] == 0
    assert artifact["unsafe_commit_count"] == 0
    assert artifact["known_level_regression_count"] == 0
    assert artifact["registry_updated"] is False
    assert artifact["new_levels_claimed"] == 0
    assert artifact["game_source_read_count"] == 0
    assert artifact["game_adapter_count"] == 0
    assert artifact["outer_loop_bfs_used"] is False
    assert artifact["per_game_leakage_detected"] is False
    assert artifact["live_path_reachable"] is True
    assert artifact["live_path_reachable_score"] == 1.0
    assert artifact["arc_epistemic_ledger_ready_score"] == 1.0
    assert artifact["inference_substrate"] == "arc_visible_state_epistemic_ledger_no_llm"
    assert len(artifact["reproducibility_checksum"].removeprefix("sha256:")) == 64


def test_req_arc_wmte_5725_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5725: checked-in artifact is the stable qualification receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["new_levels_claimed"] == 0
    assert artifact["registry_updated"] is False
    assert artifact["arc_epistemic_ledger_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_arc_wmte_5725_edge_branches_and_caps(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5725: edge branches stay deterministic and covered."""

    assert stable_state_hash({"grid": [[1, 0], [0, 0]]})
    assert stable_state_hash(SimpleNamespace(grid=np.asarray([[1, 0], [0, 0]])))
    assert stable_state_hash(np.asarray([[[0, 0], [0, 1]], [[0, 1], [0, 1]]]))
    assert candidate_signature(SimpleNamespace(action_id=3, data={"x": 1})) == 'a=3|d={"x":1}'
    assert candidate_signature(SimpleNamespace(action="bad", data=None)) == "a=-1|d=null"

    disabled = AgentEpistemicLedger(enabled=False)
    assert disabled.observe_state(_frame([[0]])) is None
    assert (
        disabled.observe_transition(_frame([[0]]), 1, None, _frame([[0]]), level_before=0)
        is None
    )
    assert coerce_epistemic_ledger(False) is None
    assert coerce_epistemic_ledger(None) is None
    assert isinstance(coerce_epistemic_ledger(True), AgentEpistemicLedger)
    assert coerce_epistemic_ledger(disabled) is disabled

    missing = AgentEpistemicLedger()
    assert missing.observe_state(None) is None
    assert missing.observe_transition(None, 1, None, _frame([[0]])) is None
    assert missing.rank_candidates(_frame([[0]]), []) == []
    assert missing.diagnostics()["fallback_reasons"]["missing_observation"] == 2
    assert missing.diagnostics()["fallback_reasons"]["missing_candidates"] == 1

    corrupt = AgentEpistemicLedger()
    before = _frame([[0, 0], [1, 0]])
    after = _frame([[0, 1], [1, 0]])
    assert (
        corrupt.observe_transition(
            before,
            1,
            None,
            after,
            before_hash_override="sha256:not-the-before-hash",
        )
        is None
    )
    assert (
        corrupt.observe_transition(
            before,
            1,
            None,
            after,
            after_hash_override="sha256:not-the-after-hash",
        )
        is None
    )
    assert corrupt.diagnostics()["fallback_reasons"]["corrupted_hash"] == 2

    progress = AgentEpistemicLedger()
    progress.rank_candidates(before, [{"action": 2, "data": None}])
    progress.observe_transition(before, 2, None, after, level_before=0, level_after=1)
    progress.observe_transition(before, 2, None, after, level_before=0, level_after=1)
    assert progress.rank_candidates(before, [{"action": 1}, {"action": 2}])[0]["action"] == 2
    progress.observe_transition(before, 2, None, before, level_before=0, level_after=0)
    assert progress.diagnostics()["false_commit_count"] >= 1

    shape = AgentEpistemicLedger()
    shape.observe_transition(before, 4, None, _frame([[1, 2, 3]]), level_before=0)
    assert shape.confirmed_facts[-1]["changed_count"] == -1

    capped = AgentEpistemicLedger(config=LedgerConfig(max_facts=1, max_hypotheses=1))
    capped.observe_state(_frame([[0]]))
    capped.observe_state(_frame([[1]]))
    assert len(capped.confirmed_facts) == 1
    capped.observe_transition(before, 1, None, before)
    capped.observe_transition(after, 2, None, after)
    assert any(row["reason"] == "resource_cap" for row in capped.snapshot()["superseded_entries"])

    qcap = AgentEpistemicLedger(config=LedgerConfig(max_questions=1))
    qcap.rank_candidates(before, [{"action": 1}, {"action": 2}])
    assert any(row["reason"] == "question_cap" for row in qcap.snapshot()["superseded_entries"])

    frozen = AgentEpistemicLedger(config=LedgerConfig(allow_reordering=False))
    frozen.observe_transition(before, 2, None, after, level_before=0, level_after=0)
    frozen.observe_transition(before, 2, None, after, level_before=0, level_after=0)
    frozen_rows = [{"action": 1}, {"action": 2}]
    assert frozen.rank_candidates(before, frozen_rows) == frozen_rows

    manual = AgentEpistemicLedger()
    manual.observe_transition(before, 1, None, before)
    key, row = next(iter(manual.active_hypotheses.items()))
    row["support_count"] = 2
    row["contradiction_count"] = 1
    assert key
    assert manual.rank_candidates(before, [{"action": 1}, {"action": 2}]) == [
        {"action": 1},
        {"action": 2},
    ]

    class BadArray:
        def __array__(self, *_args):
            raise RuntimeError("bad array")

    assert ledger_mod._grid(BadArray()) is None
    saved_grid = ledger_mod._grid

    class BadCompare:
        shape = (1, 1)

        def __ne__(self, _other):
            raise RuntimeError("bad comparison")

    monkeypatch.setattr(ledger_mod, "_grid", lambda _value: BadCompare())
    assert AgentEpistemicLedger._changed_count("a", "b") == 0
    monkeypatch.setattr(ledger_mod, "_grid", saved_grid)

    registryless = tmp_path / "empty-root"
    (registryless / "ops").mkdir(parents=True)
    assert mod.registry_precheck(registryless)["ok"] is False
    leak_root = tmp_path / "leak-root"
    live_dir = leak_root / "python" / "carnot" / "agentic"
    live_dir.mkdir(parents=True)
    (live_dir / "arc_epistemic_ledger.py").write_text("tu93\n", encoding="utf-8")
    (live_dir / "arc_competition_agent.py").write_text("", encoding="utf-8")
    assert mod.scan_per_game_constants(leak_root)["clean"] is False
    out_path = tmp_path / "artifact.json"
    mod.write_json(out_path, {"ok": True})
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"ok": True}
