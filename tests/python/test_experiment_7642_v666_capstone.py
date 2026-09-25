"""V666 capstone custody and cold-reduction tests.

Spec refs: REQ-REPORT-7642 and SCENARIO-REPORT-7642-CUSTODY,
SCENARIO-REPORT-7642-BLOCK, SCENARIO-REPORT-7642-REDUCE,
SCENARIO-REPORT-7642-MUTATIONS, SCENARIO-REPORT-7642-TERMINAL.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_7642_v666_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def test_consumed_authority_and_literal_custody() -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-CUSTODY."""

    authority = capstone.load_authority(ROOT)
    rows = capstone.collect_dispositions(ROOT, authority["tasks"])
    assert authority["selected_roadmap_path"] == "research-roadmap.yaml"
    assert authority["comparison_passed"] is True
    assert len(rows) == 14
    assert [row["task_id"] for row in rows] == list(capstone.EXPECTED_TASK_IDS)
    assert rows[-1]["custody_kind"] == "current_self"
    assert rows[-1]["sha256"] is None
    assert rows[3]["custody_kind"] == "conductor_pre_gate"
    assert rows[6]["custody_kind"] == "missing_work"
    assert rows[11]["custody_kind"] == "conductor_pre_gate"


def test_external_missing_is_complete_blocked_with_distinct_gates() -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-BLOCK."""

    value = capstone.build_artifact_for_test(ROOT)
    assert value["honest_verdict"] == "complete_blocked_required_v666_external_evidence"
    assert value["verdict_class"] == "blocked"
    assert value["capstone_complete_score"] == 1
    assert value["MODEL_SPECS"] == []
    assert value["model_invoked"] is False
    assert value["inference_substrate_class"] == "aggregation"
    assert set(value["acceptance_gate_results"]) == {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }
    assert all("principle" in gate for gate in value["acceptance_gate_results"].values())
    assert all(
        {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= check.keys()
        for check in value["gate_check_summary"]["failed_checks"]
    )
    assert not any(
        row.get("actual_path") == capstone.RESULT_PATH.as_posix()
        for row in value["source_artifact_hashes"]
    )


def test_reduction_keeps_scientific_and_shipping_limits() -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-REDUCE."""

    value = capstone.build_artifact_for_test(ROOT)
    summary = value["evidence_summary"]
    assert summary["static_evidence"]["probability_benefit"] is None
    assert summary["causal_learning"]["retention_benefit"] is None
    assert summary["arc_wrapper"]["scored_comparison_available"] is False
    assert summary["arc_method"]["eligible"] is False
    assert summary["native_packaging"]["independent_units"] == 12
    assert summary["native_packaging"]["historical_direct_native_ratio"] > 7
    assert summary["native_packaging"]["new_10x_claim"] is False
    assert value["sample_size_budget"]["milestone_tasks"]["observed"] == 14
    assert all("raw_provenance" in row for row in value["rows"])
    assert len(value["remaining_prd_gaps"]) == 3
    assert len(value["next_decisions"]) >= 4
    assert value["publication_gates"]["claim_boundary"] == "historical_fover_only"


def test_cold_mutations_rejected() -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-MUTATIONS."""

    value = capstone.build_artifact_for_test(ROOT)
    assert capstone.independent_reduce(value, root=ROOT) == []
    for mutation in ("deleted", "reordered", "wrong_field", "self_hash"):
        changed = capstone.mutate_for_test(deepcopy(value), mutation)
        assert capstone.independent_reduce(changed, root=ROOT), mutation


def test_terminal_metadata_is_complete() -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-TERMINAL."""

    value = capstone.build_artifact_for_test(ROOT)
    assert value["capstone_note_path"] == "docs/research-notes/v666-capstone.md"
    assert value["verifier_is_oracle"] is False
    assert value["flagged_adversarial"] is False
    assert value["reproducibility_checksum"]
    assert set(value).issubset(value["field_principles"])
    assert value["publication_performed"] is False
    assert value["roadmap_activation_performed"] is False


def test_authority_and_classification_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-BLOCK."""

    authority = capstone.load_authority(ROOT)
    tasks = deepcopy(authority["tasks"])
    tasks.pop()
    with pytest.raises(ValueError, match="roster"):
        capstone.collect_dispositions(ROOT, tasks)
    monkeypatch.setattr(
        capstone.contract,
        "compare_contract_authorities",
        lambda *_: {"passed": False, "errors": ["wrong_field"]},
    )
    with pytest.raises(ValueError, match="authority"):
        capstone.load_authority(ROOT)
    assert capstone.classify_terminal(False, True, False)[1] == "partial"
    assert capstone.classify_terminal(True, False, True)[1] == "positive"
    assert capstone.classify_terminal(True, False, False)[1] == "null"


def test_cold_reader_rejects_false_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-MUTATIONS."""

    baseline = capstone.build_artifact_for_test(ROOT)
    with pytest.raises(ValueError, match="unknown mutation"):
        capstone.mutate_for_test(deepcopy(baseline), "unknown")
    assert capstone.independent_reduce(None, root=ROOT) == ["artifact_object_required"]
    changed = deepcopy(baseline)
    changed["reproducibility_checksum"] = "wrong"
    assert "checksum_mismatch" in capstone.independent_reduce(changed, root=ROOT)
    changed = deepcopy(baseline)
    changed["honest_verdict"] = "complete_positive_unearned"
    changed["MODEL_SPECS"] = [{"model": "unearned"}]
    changed["inference_substrate_class"] = "model_bounded_generation"
    changed["field_principles"] = {}
    changed["publication_performed"] = True
    errors = capstone.independent_reduce(changed, root=ROOT)
    assert {
        "terminal_classification_mismatch",
        "current_model_provenance_mismatch",
        "inference_substrate_class_mismatch",
        "field_principles_incomplete",
        "unauthorized_action_claim",
    } <= set(errors)
    monkeypatch.setattr(
        capstone,
        "build_artifact_for_test",
        lambda *_args, **_kw: (_ for _ in ()).throw(ValueError("bad source")),
    )
    assert "cold_reduction_failed:ValueError" in capstone.independent_reduce(baseline, root=ROOT)


def test_e2e_control_detects_broken_mutation_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7642 / SCENARIO-REPORT-7642-TERMINAL."""

    baseline = capstone.build_artifact_for_test(ROOT)
    assert capstone.task_specific_e2e(baseline, ROOT) == []
    monkeypatch.setattr(capstone, "independent_reduce", lambda *_args, **_kw: [])
    assert capstone.task_specific_e2e(baseline, ROOT) == [
        f"mutation_not_rejected:{name}"
        for name in ("deleted", "reordered", "wrong_field", "self_hash")
    ]
