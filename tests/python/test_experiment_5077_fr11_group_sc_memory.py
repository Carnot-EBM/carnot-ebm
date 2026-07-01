"""Tests for Exp 5077 guarded FR-11 group-SC memory evolution.

Spec refs: REQ-LEARN-5077, SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5077_fr11_group_sc_memory as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    disagreement_ids = {"q0001", "q0003", "q0005", "q0008", "q0010", "q0011"}
    for index in range(12):
        row_id = f"q{index:04d}"
        sc_answer = "T" if row_id in disagreement_ids else "A"
        energy_answer = "V" if row_id in disagreement_ids else sc_answer
        rows.append(
            {
                "row_id": row_id,
                "sc_answer": sc_answer,
                "energy_answer": energy_answer,
                "energy_pure_answer": energy_answer,
                "answers": ["V", "T"],
                "energy_abstained": False,
            }
        )
    return rows


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    split = {
        "train_ids": [f"q{index:04d}" for index in range(8)],
        "heldout_ids": [f"q{index:04d}" for index in range(8, 12)],
    }
    _write_json(
        root / exp.EXP5051_RESULT_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5051_verifier_trace_self_learning.v1",
            "split_ids": split,
            "heldout_evaluation": {"selector_decisions": []},
            "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_minus_0p250",
        },
    )
    _write_json(
        root / exp.EXP5051_MEMORY_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5051.replay_memory.v1",
            "verified_traces": [
                {
                    "trace_id": f"5051:fixture:q{index:04d}",
                    "row_id": f"q{index:04d}",
                    "source_experiment": 5051,
                    "near_miss_reasons": ["verifier_uncertain_disagreement"],
                    "features": {"verifier_sc_disagreement": index % 2 == 1},
                }
                for index in range(6)
            ],
        },
    )
    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
            "honest_verdict": "complete_fixture",
            "model_specs": {"mandated_sota": dict(exp.MANDATED_MODEL_SPECS)},
            "flagged_adversarial": False,
            "refreshed_candidate_metrics": {
                "paired_correct": {
                    "verifier": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                    "tuned_self_consistency": [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                    "oracle_at_k": [1] * 12,
                },
                "predictions": ["A"] * 12,
            },
        },
    )
    _write_json(
        root / exp.EXP5064_RESULT_RELATIVE_PATH,
        {
            "schema": "carnot.experiment_5064_audited_skillgraph_self_learning.v1",
            "honest_verdict": "complete_guarded_no_promote_minus_0p500",
            "heldout_delta": -0.5,
            "nonforgetting_delta": -0.5,
            "promoted": False,
        },
    )
    for row in _checkpoint_rows():
        _write_json(root / exp.MUSR_CHECKPOINT_RELATIVE_DIR / f"{row['row_id']}.json", row)
    return root


def test_req_learn_5077_spec_declares_group_sc_memory_contract() -> None:
    """REQ-LEARN-5077: OpenSpec anchors guarded group-SC memory evolution."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / exp.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-5077",
        "SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE",
        "experiment_5077_fr11_group_sc_memory.py",
        "results/experiment_5077_fr11_group_sc_memory_v466.json",
        "group_self_consistency_summary",
        "rollback_guard_passed",
        "complete_fr11_group_sc_memory_guarded_no_promote_delta_",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for hf_id in exp.MANDATED_MODEL_SPECS.values():
        assert hf_id in spec
        assert hf_id in module_text


def test_scenario_learn_5077_split_and_contamination_guard(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE: split IDs stay isolated."""

    root = _fixture_root(tmp_path)
    exp5051, memory, exp5059, _exp5064 = exp.load_inputs(root)
    paired = exp.paired_correct(exp5059)
    split = exp.build_train_dev_heldout_split(
        exp5051,
        memory,
        paired,
        train_count=3,
        dev_count=2,
        heldout_count=3,
    )
    proposals = exp.generate_policy_proposals(split, memory)
    summary, candidates = exp.group_self_consistency(proposals, consensus_threshold=2)
    guard = exp.contamination_guard(
        split=split,
        proposals=proposals,
        consensus_candidates=candidates,
        promoted_memory_entries=[],
    )

    assert split["train_ids"] == ["q0000", "q0001", "q0002"]
    assert split["dev_ids"] == ["q0003", "q0004"]
    assert split["heldout_ids"] == ["q0009", "q0010", "q0011"]
    assert set(split["train_ids"]).isdisjoint(split["dev_ids"])
    assert set(split["train_ids"]).isdisjoint(split["heldout_ids"])
    assert guard["passed"] is True
    assert summary["tested_consensus_candidate_ids"] == [
        "candidate_5077_disagreement_fallback"
    ]

    leaked = exp.contamination_guard(
        split=split,
        proposals=[proposals[0] | {"source_row_ids": ["q0010"]}],
        consensus_candidates=candidates,
        promoted_memory_entries=[{"row_id": "q0011"}],
    )

    assert leaked["passed"] is False
    assert "proposal_heldout_id_leak:q0010" in leaked["violations"]
    assert "promoted_memory_heldout_id_leak:q0011" in leaked["violations"]


def test_scenario_learn_5077_group_sc_tests_only_consensus_candidates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE: non-consensus proposals quarantine."""

    root = _fixture_root(tmp_path)
    exp5051, memory, exp5059, _exp5064 = exp.load_inputs(root)
    split = exp.build_train_dev_heldout_split(
        exp5051,
        memory,
        exp.paired_correct(exp5059),
        train_count=4,
        dev_count=2,
        heldout_count=4,
    )
    proposals = exp.generate_policy_proposals(split, memory)
    summary, candidates = exp.group_self_consistency(proposals, consensus_threshold=2)

    assert len(proposals) == 5
    assert len(candidates) == 1
    assert candidates[0]["policy_signature"] == "fallback_to_tuned_on_verifier_sc_disagreement"
    assert summary["consensus_threshold"] == 2
    assert summary["total_proposals"] == 5
    assert summary["tested_consensus_candidate_ids"] == [
        "candidate_5077_disagreement_fallback"
    ]
    assert sorted(summary["quarantined_nonconsensus_proposal_ids"]) == [
        "proposal_5077_abstention_trace_memory",
        "proposal_5077_promote_all_verified_trace_memory",
    ]


def test_scenario_learn_5077_run_no_promotes_and_rolls_back(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE: regression triggers rollback."""

    root = _fixture_root(tmp_path)
    artifact_path = tmp_path / "artifact.json"

    artifact = exp.run(
        root=root,
        artifact_path=artifact_path,
        now=lambda: 100.0,
        write=True,
        train_count=4,
        dev_count=2,
        heldout_count=4,
    )

    assert artifact["honest_verdict"] == (
        "complete_fr11_group_sc_memory_guarded_no_promote_delta_minus_0p250"
    )
    assert artifact["fr11_attempt_completed"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["promoted_count"] == 0
    assert artifact["quarantined_count"] == 3
    assert artifact["contamination_guard_passed"] is True
    assert artifact["rollback_guard_passed"] is True
    assert artifact["heldout_delta"] == pytest.approx(-0.25)
    assert artifact["nonforgetting_delta"] == pytest.approx(-2 / 3)
    assert artifact["ablations"]["baseline"]["accuracy"] == pytest.approx(0.75)
    assert artifact["ablations"]["memory_enabled"]["accuracy"] == pytest.approx(0.5)
    assert artifact["ablations"]["rollback_no_promote"]["accuracy"] == pytest.approx(0.75)
    assert artifact["promotion_decision"]["promoted"] is False
    assert "heldout_delta_negative" in artifact["promotion_decision"]["no_promote_reason"]
    assert "nonforgetting_regressed" in artifact["promotion_decision"]["no_promote_reason"]
    assert artifact["memory_policy"]["policy_signature"] == (
        "fallback_to_tuned_on_verifier_sc_disagreement"
    )
    assert artifact["flagged_adversarial"] is False
    assert exp.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_learn_5077_schema_and_helper_edges(tmp_path: Path) -> None:
    """REQ-LEARN-5077: malformed evidence fails closed before promotion."""

    assert exp.read_json_object(tmp_path / "missing.json") == {}
    assert exp.sha256_file(tmp_path / "missing.bin") is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert exp.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(list_json) == {}
    assert exp.as_binary_list("bad") == []
    assert exp.as_binary_list([1, 0, "1"]) == [1, 0, 1]
    assert exp.as_binary_list([1, 2]) == []
    assert exp.number(True) is None
    assert exp.number("nan") is None
    assert exp.accuracy([]) == 0.0
    assert exp.row_index("q0012") == 12
    assert exp.paired_correct({}) == {
        "verifier": [],
        "tuned_self_consistency": [],
        "oracle_at_k": [],
    }
    assert exp.verified_trace_row_ids({"verified_traces": ["bad"]}, {"q0000"}) == []
    assert exp.candidate_id_for_signature("other_policy").startswith("candidate_5077_")

    with pytest.raises(ValueError, match="split_ids missing"):
        exp.build_train_dev_heldout_split({}, {}, {})
    with pytest.raises(ValueError, match="train and heldout"):
        exp.build_train_dev_heldout_split({"split_ids": {"train_ids": [], "heldout_ids": []}}, {}, {})

    fallback_split = exp.build_train_dev_heldout_split(
        {"split_ids": {"train_ids": ["q0000", "q0001"], "heldout_ids": ["q0002"]}},
        {"verified_traces": []},
        {"verifier": [], "tuned_self_consistency": [], "oracle_at_k": []},
        train_count=1,
        dev_count=1,
        heldout_count=3,
    )
    assert fallback_split["train_ids"] == ["q0000"]
    assert fallback_split["dev_ids"] == ["q0001"]

    bad_ck_root = tmp_path / "bad_ck"
    (bad_ck_root / exp.MUSR_CHECKPOINT_RELATIVE_DIR).mkdir(parents=True)
    (bad_ck_root / exp.MUSR_CHECKPOINT_RELATIVE_DIR / "q0000.json").write_text(
        "{bad",
        encoding="utf-8",
    )
    assert exp.load_checkpoint_rows(bad_ck_root) == {}

    split = {"train_ids": ["q0000"], "dev_ids": ["q0001"], "heldout_ids": ["q0002"]}
    paired = {"verifier": [1, 0, 1], "tuned_self_consistency": [1, 1, 0]}
    checkpoints = {
        "q0002": {"row_id": "q0002", "verifier_sc_disagreement": True},
    }
    evaluation = exp.evaluate_policy(
        paired,
        checkpoints,
        ["q0002"],
        {"policy_signature": "unknown_policy"},
    )
    assert evaluation["memory_correct"] == [1]
    out_of_range = exp.evaluate_policy(paired, checkpoints, ["q9999"], {})
    assert out_of_range["n_rows"] == 0

    overlap_guard = exp.contamination_guard(
        split={"train_ids": ["q0000", "q0001"], "dev_ids": ["q0001", "q0002"], "heldout_ids": ["q0002", "q0000"]},
        proposals=[],
        consensus_candidates=[],
        promoted_memory_entries=[],
    )
    assert set(overlap_guard["violations"]) == {
        "split_overlap_train_dev:q0001",
        "split_overlap_train_heldout:q0000",
        "split_overlap_dev_heldout:q0002",
    }

    decision = exp.promotion_decision(
        heldout_delta=0.0,
        nonforgetting_delta=0.0,
        contamination_guard_passed=True,
        rollback_guard_passed=True,
        consensus_candidate_count=1,
    )
    assert decision == {"promoted": True, "no_promote_reason": ""}
    blocked_decision = exp.promotion_decision(
        heldout_delta=0.0,
        nonforgetting_delta=0.0,
        contamination_guard_passed=False,
        rollback_guard_passed=False,
        consensus_candidate_count=0,
    )
    assert blocked_decision["no_promote_reason"] == (
        "no_consensus_candidate;contamination_guard_failed;rollback_guard_failed"
    )
    assert exp.upstream_flags({"flagged_adversarial": True}, {"flagged_adversarial": True}) == [
        exp.EXP5059_RESULT_RELATIVE_PATH,
        exp.EXP5064_RESULT_RELATIVE_PATH,
    ]

    errors = exp.artifact_schema_errors(
        {
            "schema": "bad",
            "spec_refs": [],
            "honest_verdict": "bad",
            "duration_s": "bad",
            "inference_substrate": "",
            "model_specs": {},
            "fr11_attempt_completed": "yes",
            "heldout_delta": "bad",
            "nonforgetting_delta": 0.0,
            "contamination_guard_passed": True,
            "rollback_guard_passed": True,
            "promoted_count": -1,
            "quarantined_count": -1,
            "memory_policy": {},
            "group_self_consistency_summary": {},
            "flagged_adversarial": False,
            "field_principles": "bad",
        }
    )

    for field in (
        "schema",
        "spec_refs",
        "honest_verdict",
        "duration_s",
        "inference_substrate",
        "model_specs",
        "fr11_attempt_completed",
        "heldout_delta",
        "promoted_count",
        "quarantined_count",
        "memory_policy",
        "group_self_consistency_summary",
        "field_principles",
    ):
        assert field in errors

    promoted_errors = exp.artifact_schema_errors(
        {
            "schema": exp.SCHEMA,
            "spec_refs": list(exp.SPEC_REFS),
            "honest_verdict": "success_fr11_group_sc_memory_promoted_plus_0p000",
            "duration_s": 0.0,
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "model_specs": {"mandated_sota": dict(exp.MANDATED_MODEL_SPECS)},
            "fr11_attempt_completed": True,
            "heldout_delta": -0.1,
            "nonforgetting_delta": -0.1,
            "contamination_guard_passed": False,
            "rollback_guard_passed": False,
            "promoted_count": 1,
            "quarantined_count": 0,
            "memory_policy": {"policy_signature": "fallback_to_tuned_on_verifier_sc_disagreement"},
            "group_self_consistency_summary": {"tested_consensus_candidate_ids": []},
            "flagged_adversarial": False,
            "field_principles": dict(exp.FIELD_PRINCIPLES),
        }
    )
    for field in (
        "promoted_with_negative_heldout_delta",
        "promoted_with_negative_nonforgetting_delta",
        "promoted_with_contamination",
        "promoted_without_rollback_guard",
    ):
        assert field in promoted_errors

    bad_principle_errors = exp.artifact_schema_errors(
        {
            "schema": exp.SCHEMA,
            "spec_refs": list(exp.SPEC_REFS),
            "honest_verdict": "complete_fr11_group_sc_memory_guarded_no_promote_delta_plus_0p000",
            "duration_s": 0.0,
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "model_specs": {"mandated_sota": dict(exp.MANDATED_MODEL_SPECS)},
            "fr11_attempt_completed": True,
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "contamination_guard_passed": True,
            "rollback_guard_passed": True,
            "promoted_count": 0,
            "quarantined_count": 1,
            "memory_policy": {"policy_signature": "fallback_to_tuned_on_verifier_sc_disagreement"},
            "group_self_consistency_summary": {"tested_consensus_candidate_ids": []},
            "flagged_adversarial": False,
            "field_principles": {field: {} for field in exp.REQUIRED_ARTIFACT_FIELDS},
        }
    )
    assert "field_principles" in bad_principle_errors
