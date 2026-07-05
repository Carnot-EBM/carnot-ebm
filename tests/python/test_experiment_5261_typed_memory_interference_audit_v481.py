"""Tests for Exp 5261 typed-memory interference audit.

Spec refs: REQ-LEARN-5261, SCENARIO-LEARN-5261.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline import typed_memory_interference_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5261_spec_declares_cached_interference_audit_contract() -> None:
    """REQ-LEARN-5261: OpenSpec anchors the no-LLM memory safety audit."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5261") :]

    for marker in (
        "REQ-LEARN-5261",
        "SCENARIO-LEARN-5261",
        "aligned memory",
        "irrelevant memory",
        "conflicting memory",
        "stale memory",
        "shuffled memory",
        "cached_fixture_replay_no_llm",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in section


def test_req_learn_5261_fixtures_cover_schema_invariants_and_memory_kinds() -> None:
    """REQ-LEARN-5261: fixtures cover all required memory-policy cases."""

    fixtures = mod.build_deterministic_fixtures()

    assert set(fixtures.memory_kinds) == {
        "aligned",
        "conflicting",
        "irrelevant",
        "shuffled",
        "stale",
    }
    assert {memory.head for memory in fixtures.memories} == set(mod.TYPED_MEMORY_HEADS)
    assert any(memory.conflicts_with for memory in fixtures.memories)
    assert any(memory.stale for memory in fixtures.memories)
    assert any(memory.harmful for memory in fixtures.memories)
    assert {task.task_kind for task in fixtures.tasks} == {"useful", "unrelated"}

    invariants = mod.typed_memory_schema_invariants()
    assert "fixed typed heads" in " ".join(invariants)
    assert "evidence-gated promotion" in " ".join(invariants)
    assert "invalidation-gated rollback" in " ".join(invariants)
    assert "test-gold leakage rejection" in " ".join(invariants)


def test_req_learn_5261_policy_retains_evicts_and_blocks_interference() -> None:
    """REQ-LEARN-5261-1/2/3/5: policy metrics cover retention and safety."""

    audit = mod.evaluate_audit(mod.build_deterministic_fixtures())

    assert audit["retention_rate"] == 1.0
    assert audit["interference_rate"] == 0.0
    assert audit["harmful_memory_rollback_passed"] is True
    assert audit["aligned_accuracy"] == 1.0
    assert audit["aligned_accuracy"] > audit["shuffled_accuracy"]
    assert audit["eviction_summary"]["evicted_by_reason"] == {
        "conflicting": 1,
        "stale": 1,
    }
    assert audit["promotion_summary"] == {
        "held": 1,
        "promoted": 3,
        "rolled_back": 3,
    }
    assert audit["memory_policy_ready"] is True


def test_req_learn_5261_promotion_eviction_policy_explains_each_decision() -> None:
    """REQ-LEARN-5261-3/4: promotion and eviction decisions are auditable."""

    policy = mod.evaluate_promotion_eviction_policy(
        mod.build_deterministic_fixtures().memories
    )
    by_subject = {decision["subject"]: decision for decision in policy["decisions"]}

    assert by_subject["GAP-1 orientation discriminator memory-only promotion"][
        "effective_state"
    ] == "promoted"
    assert by_subject["GAP-1 contradictory registry promotion stale copy"][
        "eviction_reason"
    ] == "conflicting"
    assert by_subject["Hardware speedup outdated smoke-only shortcut"][
        "eviction_reason"
    ] == "stale"
    assert by_subject["ARC harmful direct patch rollback"][
        "effective_state"
    ] == "rolled_back"
    assert by_subject["ARC harmful direct patch rollback"]["active"] is True
    assert by_subject["Citation-style irrelevant held memory"]["effective_state"] == "held"
    assert policy["active_subjects"] == [
        "ARC harmful direct patch rollback",
        "GAP-1 orientation discriminator memory-only promotion",
        "Hardware speedup claim boundary",
        "MMLU hidden-state verifier path retired",
    ]


def test_req_learn_5261_run_writes_required_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5261: run() writes the required no-LLM artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["memory_policy_ready"] is True
    assert artifact["memory_policy_ready_principle"]
    assert artifact["tests_run"] == tests_run
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "ready" in artifact["honest_verdict"]["value"]

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5261_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5261-6: checked-in artifact is stable under cached replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(
        root=REPO,
        tests_run=result["tests_run"],
    )

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == "cached_fixture_replay_no_llm"
    assert result["retention_rate"]["value"] == 1.0
    assert result["interference_rate"]["value"] == 0.0
    assert result["harmful_memory_rollback_passed"]["value"] is True
    assert result["memory_policy_ready"] is True
    mod.validate_artifact(result)
