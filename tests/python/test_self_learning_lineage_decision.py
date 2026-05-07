"""Tests for Exp 1459 self-learning lineage decision.

Spec: REQ-LEARN-1459, SCENARIO-LEARN-1459, SCENARIO-LEARN-1460.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from carnot.reporting import self_learning_lineage_decision as mod


def _exp1433() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline",
        "headline_result_allowed": False,
        "self_learning_delta_overall": 0,
        "nonforgetting_rate": 1.0,
        "session_memory_updated": False,
    }


def _exp1447(*, delta: int = 156, nonforgetting_rate: float = 1.0) -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "fr11_v7_positive_verified_growth_persisted_without_forgetting",
        "headline_result_allowed": delta > 0,
        "baseline_fresh_verified_sample_count": 1508,
        "fresh_verified_sample_count": 1508 + delta,
        "self_learning_delta_overall": delta,
        "nonforgetting_rate": nonforgetting_rate,
        "session_memory_updated": delta > 0,
        "memory_entries_added": delta,
        "new_promoted_count": delta,
    }


def _exp1449() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "ltlzinc_temporal_adapter_ready_verified_cases_only_no_training",
        "ltlzinc_adapter_ready": True,
        "temporal_cases_generated": 24,
        "accepted_case_count": 12,
        "rejected_case_count": 12,
    }


def _classification_rows() -> list[dict[str, str]]:
    return [
        {
            "experiment_id": "1303",
            "path": "results/experiment_1303_querybandits_ngc_online_memory_policy.json",
            "honest_verdict": "online_memory_policy_improved_non_headline",
            "headline_fields": "headline_result_allowed=false",
            "classification": "SIGNAL",
        },
        {
            "experiment_id": "1315",
            "path": "results/experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json",
            "honest_verdict": "cerce_nonforgetting_preserved_improved_non_headline",
            "headline_fields": "headline_result_allowed=false",
            "classification": "SIGNAL",
        },
        {
            "experiment_id": "1447",
            "path": "results/experiment_1447_fr11_v7_memory_policy_growth.json",
            "honest_verdict": "fr11_v7_positive_verified_growth_persisted_without_forgetting",
            "headline_fields": "headline_result_allowed=true",
            "classification": "SIGNAL",
        },
    ]


def test_req_learn_1459_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1459-1/2: bootstrap artifact exposes the required fields first."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["self_learning_artifacts_reviewed"] == []
    assert artifact["decision_note_path"] is None
    assert artifact["self_learning_headline_pivot_selected"] is False
    assert artifact["self_learning_lineage_retired"] is False
    assert artifact["exp1447_delta_overall"] is None
    assert artifact["nonforgetting_rate"] is None
    assert artifact["ltlzinc_benchmark_role"] is None
    assert artifact["next_allowed_experiment_shape"] is None
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_learn_1459_selects_narrow_headline_pivot(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1459: Exp 1447 growth allows one narrow headline pivot."""

    note_path = tmp_path / "docs" / "self_learning_lineage_decision.md"

    artifact = mod.build_decision_artifact(
        exp1433_artifact=_exp1433(),
        exp1447_artifact=_exp1447(),
        exp1449_artifact=_exp1449(),
        classification_rows=_classification_rows(),
        decision_note_path=note_path,
        project_root=tmp_path,
    )
    mod.validate_artifact(artifact)
    written_note = mod.write_decision_note(artifact, note_path)

    assert artifact["self_learning_headline_pivot_selected"] is True
    assert artifact["self_learning_lineage_retired"] is False
    assert artifact["exp1447_delta_overall"] == 156
    assert artifact["nonforgetting_rate"] == 1.0
    assert artifact["next_allowed_experiment_shape"]["allowed_count"] == 1
    assert artifact["next_allowed_experiment_shape"]["nonforgetting_threshold"] == 0.99
    assert "fresh_verified_sample_count" in artifact["next_allowed_experiment_shape"][
        "required_metrics"
    ]
    assert artifact["ltlzinc_benchmark_role"].startswith("supporting benchmark feed")
    assert "Exp 1447" in written_note
    assert "not a standalone headline claim" in written_note


def test_scenario_learn_1460_retires_without_persisted_growth(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1460: missing persisted growth retires headline scope."""

    artifact = mod.build_decision_artifact(
        exp1433_artifact=_exp1433(),
        exp1447_artifact=_exp1447(delta=0, nonforgetting_rate=1.0),
        exp1449_artifact=_exp1449(),
        classification_rows=_classification_rows(),
        decision_note_path=tmp_path / "decision.md",
        project_root=tmp_path,
    )

    mod.validate_artifact(artifact)
    assert artifact["self_learning_headline_pivot_selected"] is False
    assert artifact["self_learning_lineage_retired"] is True
    assert artifact["next_allowed_experiment_shape"]["scope"] == "retired_from_headline_scope"
    assert "internal_memory_policy_only" in artifact["honest_verdict"]


def test_req_learn_1459_run_loads_sources_and_writes_outputs(tmp_path: Path) -> None:
    """REQ-LEARN-1459-2/5: run writes the final artifact and decision note."""

    results_dir = tmp_path / "results"
    exp1433_path = results_dir / "experiment_1433.json"
    exp1447_path = results_dir / "experiment_1447.json"
    exp1449_path = results_dir / "experiment_1449.json"
    classification_path = tmp_path / "classification.csv"
    out_path = results_dir / mod.OUTPUT_FILE
    note_path = tmp_path / "docs" / "research-notes" / "self_learning_lineage_decision.md"

    for path, payload in (
        (exp1433_path, _exp1433()),
        (exp1447_path, _exp1447()),
        (exp1449_path, _exp1449()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    with classification_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "path",
                "honest_verdict",
                "headline_fields",
                "classification",
            ],
        )
        writer.writeheader()
        writer.writerows(_classification_rows())

    artifact = mod.run(
        exp1433_path=exp1433_path,
        exp1447_path=exp1447_path,
        exp1449_path=exp1449_path,
        classification_path=classification_path,
        out_path=out_path,
        decision_note_path=note_path,
        project_root=tmp_path,
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert note_path.exists()
    assert artifact["status"] == "complete"
    assert artifact["decision_note_path"] == str(note_path)
    assert "results/experiment_1447_fr11_v7_memory_policy_growth.json" in artifact[
        "self_learning_artifacts_reviewed"
    ]
