"""Tests for Exp 1471 FR-11 v8 verified memory-growth pivot.

Spec: REQ-LEARN-1471, SCENARIO-LEARN-1471, SCENARIO-LEARN-1472.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.pipeline.session_memory import SessionMemory
from carnot.reporting import fr11_v8_verified_memory_growth_pivot as mod


def _write_checkpoint(path: Path, *, secl_threshold: float = 0.500001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([10.0], dtype=np.float32),
            secl_bin_values=np.full(10, 0.5, dtype=np.float32),
            secl_global_value=np.asarray([0.5], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            dvi_incorrect_threshold=np.asarray([0.72], dtype=np.float32),
            secl_confidence_threshold=np.asarray([secl_threshold], dtype=np.float32),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp1395() -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": 1508,
        "memory_updates": {"promoted": ["dvi_v2:fover:baseline"], "demoted": []},
    }


def _exp1432(checkpoint_path: Path, *, nonforgetting_rate: float = 1.0) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_v3_deployed": True,
        "dvi_v3_checkpoint_path": str(checkpoint_path),
        "nonforgetting_rate": nonforgetting_rate,
    }


def _exp1446() -> dict[str, Any]:
    return {
        "status": "complete",
        "recommended_v7_policy": {
            "policy_name": "fr11_v7_asymmetric_fresh_threshold",
            "changes_exp1433_policy": True,
            "fresh_secl_confidence_threshold": 0.5,
            "replay_nonforgetting_secl_confidence_threshold": 0.500001,
            "dvi_incorrect_threshold": 0.72,
            "expected_promotions_under_v7_policy": 1,
        },
    }


def _exp1447(*, delta: int = 156, nonforgetting_rate: float = 1.0) -> dict[str, Any]:
    return {
        "status": "complete",
        "headline_result_allowed": delta > 0,
        "baseline_fresh_verified_sample_count": 1508,
        "fresh_verified_sample_count": 1508 + delta,
        "self_learning_delta_overall": delta,
        "new_promoted_count": delta,
        "memory_entries_added": delta,
        "session_memory_updated": delta > 0,
        "nonforgetting_rate": nonforgetting_rate,
        "memory_updates": {"promoted": ["dvi_v7:fover:already_promoted"]},
    }


def _exp1459(*, selected: bool = True) -> dict[str, Any]:
    return {
        "status": "complete",
        "self_learning_headline_pivot_selected": selected,
        "self_learning_lineage_retired": not selected,
        "next_allowed_experiment_shape": {
            "allowed_count": 1 if selected else 0,
            "scope": "exp1447_verified_memory_policy_growth_pivot",
            "minimum_new_promotions": 1,
            "nonforgetting_threshold": 0.99,
        },
    }


def _temporal_case(case_id: str, *, satisfied: bool) -> dict[str, Any]:
    return mod.temporal.make_case(
        case_id,
        "always",
        "safe",
        [{"safe": True}] if satisfied else [{"safe": False}],
        satisfied,
    )


def test_req_learn_1471_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1471-1/7: bootstrap artifact exposes all required fields."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["self_learning_artifact_ready"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req_learn_1471_temporal_rows_require_local_verification() -> None:
    """REQ-LEARN-1471-3: Exp 1449 rows enter only after verifier confirmation."""

    verified = _temporal_case("temporal-repair", satisfied=False)
    mismatched = dict(_temporal_case("temporal-bad", satisfied=False))
    mismatched["expected_satisfied"] = True
    mismatched["certificate_state"] = "SAT"
    mismatched["dvi_label"] = 0

    loaded = mod.temporal_cases_to_dvi_candidates([verified, mismatched])

    assert [case.case_id for case in loaded.cases] == ["temporal-repair"]
    assert loaded.counts["temporal_supporting_feed_count"] == 2
    assert loaded.counts["temporal_verifier_mismatch"] == 1
    assert loaded.counts["verified_temporal_candidates"] == 1


def test_scenario_learn_1471_reuses_policy_persists_growth_and_preserves_pivot(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1471: v8 positive growth requires persisted SessionMemory rows."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(),
        exp1447_artifact=_exp1447(delta=156),
        exp1459_artifact=_exp1459(selected=True),
        fover_rows=[],
        temporal_cases=[_temporal_case("temporal-repair", satisfied=False)],
        session_memory_dir=tmp_path / "session",
        project_root="/repo",
        commands_run=["pytest targeted"],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["baseline_fresh_verified_sample_count"] == 1664
    assert artifact["fresh_verified_sample_count"] == 1665
    assert artifact["self_learning_delta_overall"] == 1
    assert artifact["new_promoted_count"] == 1
    assert artifact["memory_entries_added"] == 1
    assert artifact["session_memory_updated"] is True
    assert artifact["nonforgetting_rate"] == 1.0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["headline_result_allowed"] is True
    assert artifact["pivot_preserved"] is True
    assert artifact["pivot_retired"] is False
    assert artifact["ltlzinc_benchmark_role"].startswith("supporting benchmark feed")

    loaded = SessionMemory(str(tmp_path / "session"), mod.SESSION_MEMORY_MODEL_ID).load()
    assert loaded is not None
    case_memory, _, _ = loaded
    assert len(case_memory.entries()) == 1
    assert case_memory.entries()[0].provenance[0].source_experiment == 1471


def test_scenario_learn_1472_failed_gate_retires_pivot(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1472: no persisted growth writes an explicit future-block rule."""

    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    _write_checkpoint(checkpoint_path)

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(),
        exp1432_artifact=_exp1432(checkpoint_path),
        exp1446_artifact=_exp1446(),
        exp1447_artifact=_exp1447(delta=156),
        exp1459_artifact=_exp1459(selected=True),
        fover_rows=[],
        temporal_cases=[],
        session_memory_dir=tmp_path / "session",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["headline_result_allowed"] is False
    assert artifact["self_learning_delta_overall"] == 0
    assert artifact["pivot_preserved"] is False
    assert artifact["pivot_retired"] is True
    assert artifact["future_block_rule"].startswith("Do not rerun")


def test_req_learn_1471_validation_enforces_headline_gate() -> None:
    """REQ-LEARN-1471-5/6/7: validation protects gate and retirement invariants."""

    artifact = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1471.json"))
    artifact.update(
        {
            "status": "complete",
            "self_learning_artifact_ready": True,
            "baseline_fresh_verified_sample_count": 10,
            "fresh_verified_sample_count": 11,
            "self_learning_delta_overall": 1,
            "new_promoted_count": 1,
            "memory_entries_added": 1,
            "session_memory_updated": True,
            "nonforgetting_rate": 0.5,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "headline_result_allowed": True,
            "pivot_preserved": True,
            "pivot_retired": False,
            "honest_verdict": "bad",
        }
    )

    with pytest.raises(AssertionError, match="headline gate"):
        mod.validate_artifact(artifact)

    artifact["nonforgetting_rate"] = 1.0
    artifact["memory_entries_added"] = 2
    with pytest.raises(AssertionError, match="memory_entries_added"):
        mod.validate_artifact(artifact)

    artifact["memory_entries_added"] = 1
    artifact["fresh_verified_sample_count"] = 10
    artifact["self_learning_delta_overall"] = 0
    artifact["new_promoted_count"] = 0
    artifact["memory_entries_added"] = 0
    artifact["session_memory_updated"] = False
    artifact["headline_result_allowed"] = False
    artifact["pivot_preserved"] = False
    artifact["pivot_retired"] = False
    with pytest.raises(AssertionError, match="failed pivot"):
        mod.validate_artifact(artifact)


def test_req_learn_1471_validation_and_helper_edge_branches() -> None:
    """REQ-LEARN-1471-3/5/6: edge helpers keep audit branches deterministic."""

    assert mod._exp1447_promoted_case_ids({"memory_updates": {"promoted": "bad"}}) == set()
    assert mod._policy_mistakes(
        [
            {"semantic_state": mod.fr11.STATE_SAT, "certificate_state": mod.fr11.STATE_REPAIR_HINT},
            {"semantic_state": mod.fr11.STATE_REPAIR_HINT, "certificate_state": mod.fr11.STATE_SAT},
        ]
    ) == (1, 1)
    assert (
        mod._honest_verdict(status="blocked", headline_allowed=False, nonforgetting_rate=None)
        == "fr11_v8_blocked_pivot_prerequisite_missing"
    )
    assert (
        mod._honest_verdict(status="complete", headline_allowed=False, nonforgetting_rate=0.5)
        == "fr11_v8_pivot_retired_nonforgetting_gate_failed"
    )

    base = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1471_edges.json"))
    base.update(
        {
            "status": "complete",
            "self_learning_artifact_ready": False,
            "baseline_fresh_verified_sample_count": 10,
            "fresh_verified_sample_count": 10,
            "self_learning_delta_overall": 0,
            "new_promoted_count": 0,
            "memory_entries_added": 0,
            "session_memory_updated": False,
            "nonforgetting_rate": 1.0,
            "headline_result_allowed": False,
            "pivot_preserved": False,
            "pivot_retired": True,
            "future_block_rule": "blocked",
            "honest_verdict": "retired",
        }
    )

    bad_fresh_count = dict(base, fresh_verified_sample_count=11)
    with pytest.raises(AssertionError, match="fresh_verified_sample_count"):
        mod.validate_artifact(bad_fresh_count)

    bad_delta = dict(base, self_learning_delta_overall=1, fresh_verified_sample_count=11)
    with pytest.raises(AssertionError, match="self_learning_delta_overall"):
        mod.validate_artifact(bad_delta)

    stale_session_flag = dict(base, session_memory_updated=True)
    with pytest.raises(AssertionError, match="session_memory_updated"):
        mod.validate_artifact(stale_session_flag)


def test_req_learn_1471_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1471-1/7: run writes bootstrap then terminal artifact."""

    results = tmp_path / "results"
    checkpoint_path = tmp_path / "verify" / "dvi_v3.pt"
    fover_path = tmp_path / "fover.jsonl"
    temporal_path = tmp_path / "temporal.jsonl"
    out_path = results / mod.OUTPUT_FILE
    _write_checkpoint(checkpoint_path)
    _write_json(results / "experiment_1395.json", _exp1395())
    _write_json(results / "experiment_1432.json", _exp1432(checkpoint_path))
    _write_json(results / "experiment_1446.json", _exp1446())
    _write_json(results / "experiment_1447.json", _exp1447(delta=156))
    _write_json(results / "experiment_1459.json", _exp1459(selected=True))
    _write_jsonl(fover_path, [])
    _write_jsonl(temporal_path, [_temporal_case("temporal-repair", satisfied=False)])

    artifact = mod.run(
        exp1395_path=results / "experiment_1395.json",
        exp1432_path=results / "experiment_1432.json",
        exp1446_path=results / "experiment_1446.json",
        exp1447_path=results / "experiment_1447.json",
        exp1459_path=results / "experiment_1459.json",
        fover_path=fover_path,
        temporal_dataset_path=temporal_path,
        out_path=out_path,
        session_memory_dir=tmp_path / "session",
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["commands_run"] == ["pytest targeted"]
