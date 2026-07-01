"""Tests for Exp 5064 audited skill-graph self-learning.

Spec refs: REQ-VERIFY-5064, SCENARIO-VERIFY-5064.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5064_audited_skillgraph_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    split = {
        "train_ids": ["q0000", "q0001", "q0002", "q0003"],
        "heldout_ids": ["q0004", "q0005"],
    }
    _write_json(
        root / mod.EXP5051_RESULT_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5051_verifier_trace_self_learning.v1",
            "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_minus_0p500",
            "heldout_delta": -0.5,
            "pre_update_accuracy": 1.0,
            "post_update_accuracy": 0.5,
            "split_ids": split,
            "heldout_evaluation": {
                "selector_decisions": [
                    {
                        "row_id": "q0004",
                        "selector": "tuned_self_consistency",
                        "structural_trigger": "verifier_sc_disagreement",
                    },
                    {
                        "row_id": "q0005",
                        "selector": "pre_update_verifier",
                        "structural_trigger": "none",
                    },
                ]
            },
        },
    )
    _write_json(
        root / mod.EXP5051_MEMORY_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5051.replay_memory.v1",
            "update_type": "replay_memory_insertion",
            "support_row_ids": ["q0000", "q0002"],
            "support_trace_ids": ["5051:5031:q0000", "5051:5045:q0002"],
            "verified_trace_count": 2,
            "verified_traces": [
                {
                    "trace_id": "5051:5031:q0000",
                    "row_id": "q0000",
                    "source_experiment": 5031,
                    "near_miss_reasons": ["verifier_uncertain_disagreement"],
                    "features": {"verifier_sc_disagreement": True},
                    "trace_text": "OBSERVED_SIGNAL: q0000\nREVISION: structural fallback\nVERIFICATION: candidate_set_preserved\nMEMORY_UPDATE: fallback_to_genuine_tuned_sc",
                },
                {
                    "trace_id": "5051:5045:q0002",
                    "row_id": "q0002",
                    "source_experiment": 5045,
                    "near_miss_reasons": ["verifier_wrong_oracle_recoverable"],
                    "features": {"verifier_sc_disagreement": True},
                    "trace_text": "OBSERVED_SIGNAL: q0002\nREVISION: structural fallback\nVERIFICATION: candidate_set_preserved\nMEMORY_UPDATE: fallback_to_genuine_tuned_sc",
                },
            ],
        },
    )
    _write_json(
        root / mod.EXP5059_RESULT_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
            "honest_verdict": "complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
            "accuracy": 0.75,
            "tuned_sc_accuracy": 0.75,
            "delta_vs_tuned_sc": 0.0,
            "legacy_models_smoke_only": True,
            "refreshed_candidate_metrics": {
                "n_questions": 6,
                "accuracy": 0.666667,
                "tuned_sc_accuracy": 0.833333,
                "oracle_at_k": 1.0,
                "headroom_present": True,
                "paired_correct": {
                    "verifier": [0, 1, 0, 1, 1, 1],
                    "tuned_self_consistency": [1, 1, 1, 1, 0, 1],
                    "oracle_at_k": [1, 1, 1, 1, 1, 1],
                },
                "predictions": ["A", "B", "C", "D", "E", "F"],
            },
            "model_specs": {
                "mandated_sota": {
                    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
                    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
                }
            },
        },
    )
    return root


def test_req_verify_5064_spec_declares_audited_no_promote_contract() -> None:
    """REQ-VERIFY-5064: OpenSpec anchors audited FR-11 no-promotion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5064",
        "SCENARIO-VERIFY-5064",
        "experiment_5064_audited_skillgraph_self_learning.py",
        "results/experiment_5064_audited_skillgraph_self_learning.json",
        "continuous_self_learning_task",
        "candidate_skill_count",
        "verified_skill_count",
        "no_promote_reason",
        "nonforgetting_delta",
        "skill_graph_path",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5064_mines_train_near_misses_before_skill_proposal(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5064: near-miss mining excludes frozen held-out IDs."""

    root = _fixture_root(tmp_path)
    exp5051, memory, exp5059 = mod.load_inputs(root)
    split = mod.freeze_split(exp5051)
    near_misses = mod.mine_near_misses(exp5051, memory, exp5059, split)
    skills = mod.build_candidate_skills(near_misses, memory, exp5059, split["heldout_ids"])

    assert split["heldout_ids"] == ["q0004", "q0005"]
    assert near_misses
    assert {row["row_id"] for row in near_misses} <= set(split["train_ids"])
    assert {row["row_id"] for row in near_misses}.isdisjoint(split["heldout_ids"])
    assert {row["source_artifact"] for row in near_misses} == {
        mod.EXP5051_MEMORY_RELATIVE_PATH,
        mod.EXP5059_RESULT_RELATIVE_PATH,
    }
    assert len(skills) == 2
    for skill in skills:
        assert skill["self_audit"]["verdict"] == "pass"
        assert skill["external_verifier_receipt"]["passed"] is True
        assert not (set(skill["source_row_ids"]) & set(split["heldout_ids"]))


def test_scenario_verify_5064_no_promotes_negative_heldout_and_nonforgetting(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5064: harmful proposed skill is audited and no-promoted."""

    root = _fixture_root(tmp_path)
    out = tmp_path / "out.json"

    artifact = mod.run(root=root, artifact_path=out, write=True)

    assert artifact["continuous_self_learning_task"] is True
    assert artifact["self_learning_loop_executed"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["candidate_skill_count"] == 2
    assert artifact["verified_skill_count"] == 2
    assert artifact["promoted"] is False
    assert artifact["no_promote_reason"]
    assert "heldout_delta_nonpositive" in artifact["no_promote_reason"]
    assert "nonforgetting_regressed" in artifact["no_promote_reason"]
    assert artifact["pre_update_accuracy"] == pytest.approx(1.0)
    assert artifact["post_update_accuracy"] == pytest.approx(0.5)
    assert artifact["heldout_delta"] == pytest.approx(-0.5)
    assert artifact["nonforgetting_delta"] == pytest.approx(-0.5)
    assert artifact["contamination_guard_passed"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(out.read_text(encoding="utf-8")) == artifact

    graph_path = Path(artifact["skill_graph_path"])
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["promoted_skill_ids"] == []
    assert graph["promotion_decision"]["promoted"] is False
    assert graph["heldout_evaluation"]["no_promote_fallback_accuracy"] == pytest.approx(1.0)


def test_promotion_gate_and_contamination_guard_edges() -> None:
    """REQ-VERIFY-5064: promotion requires positive utility and clean provenance."""

    clean = mod.contamination_guard(
        train_ids=["q0000"],
        heldout_ids=["q0001"],
        trace_inputs=[{"row_id": "q0000"}],
        candidate_skills=[{"source_row_ids": ["q0000"]}],
        promoted_memory_entries=[],
    )
    leaked = mod.contamination_guard(
        train_ids=["q0000", "q0001"],
        heldout_ids=["q0001"],
        trace_inputs=[{"row_id": "q0001"}],
        candidate_skills=[{"source_row_ids": ["q0001"]}],
        promoted_memory_entries=[{"row_id": "q0001"}],
    )

    assert clean["passed"] is True
    assert clean["violations"] == []
    assert leaked["passed"] is False
    assert "split_overlap:q0001" in leaked["violations"]
    assert any("heldout_id_leak:q0001" in item for item in leaked["violations"])

    promoted = mod.promotion_decision(
        heldout_delta=0.125,
        nonforgetting_delta=0.0,
        contamination_guard_passed=True,
    )
    rejected = mod.promotion_decision(
        heldout_delta=0.0,
        nonforgetting_delta=-0.25,
        contamination_guard_passed=False,
    )

    assert promoted == {"promoted": True, "no_promote_reason": ""}
    assert rejected["promoted"] is False
    assert rejected["no_promote_reason"] == (
        "heldout_delta_nonpositive;nonforgetting_regressed;contamination_guard_failed"
    )


def test_defensive_schema_and_audit_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5064: malformed evidence fails closed before promotion."""

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(list_path)
    with pytest.raises(ValueError, match="split_ids missing"):
        mod.freeze_split({})
    with pytest.raises(ValueError, match="train and heldout"):
        mod.freeze_split({"split_ids": {"train_ids": [], "heldout_ids": ["q0001"]}})
    with pytest.raises(ValueError, match="paired_correct missing"):
        mod._paired_correct({"refreshed_candidate_metrics": {}})

    split = {"train_ids": ["q0000"], "heldout_ids": ["q0001", "q9999"]}
    near_misses = mod.mine_near_misses(
        {},
        {
            "verified_traces": [
                "malformed",
                {"row_id": "q0001", "trace_id": "5051:bad:q0001"},
                {"row_id": "q0000"},
            ]
        },
        {
            "refreshed_candidate_metrics": {
                "paired_correct": {
                    "verifier": [0],
                    "tuned_self_consistency": [1],
                    "oracle_at_k": [1],
                }
            }
        },
        split,
    )
    assert near_misses == [
        {
            "row_id": "q0000",
            "source_artifact": mod.EXP5059_RESULT_RELATIVE_PATH,
            "source_trace_id": "5059:refreshed:q0000",
            "source_experiment": 5059,
            "near_miss_reasons": [
                "verifier_wrong_oracle_recoverable",
                "verifier_wrong_tuned_sc_correct",
            ],
            "proposal_signal": "refreshed_d1_paired_correct_near_miss",
        }
    ]

    bad_skill = {
        "source_trace_ids": ["missing"],
        "source_artifacts": [],
        "source_row_ids": ["q0001"],
        "source_summary": "Gold answer leakage",
    }
    self_audit = mod._self_audit_skill(bad_skill)
    external = mod._external_verify_skill(
        bad_skill,
        known_trace_ids={"known"},
        heldout_ids=["q0001"],
    )
    assert self_audit["verdict"] == "fail"
    assert self_audit["failed_checks"] == [
        "source_artifacts_missing",
        "final_answer_leakage",
    ]
    assert external["passed"] is False
    assert external["failed_checks"] == [
        "unknown_source_trace:missing",
        "heldout_id_leak:q0001",
        "auditable_action_missing",
    ]
    no_source_audit = mod._self_audit_skill({"source_artifacts": ["x"]})
    assert no_source_audit["failed_checks"] == ["source_trace_ids_missing"]

    heldout = mod.evaluate_heldout(
        {"heldout_evaluation": {"selector_decisions": []}},
        {
            "refreshed_candidate_metrics": {
                "paired_correct": {
                    "verifier": [1],
                    "tuned_self_consistency": [1],
                    "oracle_at_k": [1],
                }
            }
        },
        split,
    )
    assert heldout["heldout_n"] == 0
    assert heldout["nonforgetting_delta"] == 0.0

    errors = mod.artifact_schema_errors(
        {
            "schema": "wrong",
            "spec_refs": [],
            "continuous_self_learning_task": "yes",
            "legacy_models_smoke_only": False,
            "promoted": False,
            "contamination_guard_passed": "yes",
            "near_miss_count": -1,
            "candidate_skill_count": "two",
            "verified_skill_count": -1,
            "pre_update_accuracy": True,
            "post_update_accuracy": "bad",
            "heldout_delta": "bad",
            "nonforgetting_delta": "bad",
            "model_specs": {"mandated_sota": {"wrong": "model"}},
            "skill_graph_path": "",
        }
    )
    for expected in (
        "schema",
        "spec_refs",
        "continuous_self_learning_task",
        "continuous_self_learning_task_true",
        "legacy_models_smoke_only_true",
        "candidate_skill_count",
        "post_update_accuracy",
        "no_promote_reason",
        "model_specs",
        "skill_graph_path",
    ):
        assert expected in errors
