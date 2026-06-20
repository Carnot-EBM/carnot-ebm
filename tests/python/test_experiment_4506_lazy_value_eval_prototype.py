"""Tests for Exp 4506 lazy ARC value-head evaluation.

Spec refs: REQ-REPORT-4506, SCENARIO-REPORT-4506-LAZY-TOPK,
SCENARIO-REPORT-4506-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4506_lazy_value_eval_prototype as exp4506


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _candidate(
    trial_id: int,
    slot: int,
    *,
    cheap_priority: float,
    value_score: float,
    frame_hash: str | None = None,
) -> exp4506.FrontierCandidate:
    return exp4506.FrontierCandidate(
        trial_id=trial_id,
        candidate_id=f"t{trial_id}-c{slot}",
        frame_hash=frame_hash or f"h{trial_id}-{slot}",
        cheap_priority=cheap_priority,
        value_score=value_score,
    )


def _two_frontiers_with_repeated_topk_hashes() -> list[list[exp4506.FrontierCandidate]]:
    first = [
        _candidate(0, 0, cheap_priority=0.0, value_score=0.8, frame_hash="shared-a"),
        _candidate(0, 1, cheap_priority=0.1, value_score=-0.2, frame_hash="shared-b"),
        _candidate(0, 2, cheap_priority=0.2, value_score=0.4, frame_hash="shared-c"),
        _candidate(0, 3, cheap_priority=1.5, value_score=0.0),
        _candidate(0, 4, cheap_priority=2.0, value_score=-0.1),
    ]
    second = [
        _candidate(1, 0, cheap_priority=0.0, value_score=0.8, frame_hash="shared-a"),
        _candidate(1, 1, cheap_priority=0.1, value_score=-0.2, frame_hash="shared-b"),
        _candidate(1, 2, cheap_priority=0.2, value_score=0.4, frame_hash="shared-c"),
        _candidate(1, 3, cheap_priority=1.5, value_score=0.0),
        _candidate(1, 4, cheap_priority=2.0, value_score=-0.1),
    ]
    return [first, second]


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "fixture-torch",
        "cached_candidate_frontiers_built": True,
    }


def test_req_report_4506_spec_declares_lazy_value_eval_contract() -> None:
    """REQ-REPORT-4506: OpenSpec names the lazy value-head measurement contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4506" in spec
    assert "SCENARIO-REPORT-4506-LAZY-TOPK" in spec
    assert "SCENARIO-REPORT-4506-SCHEMA" in spec
    assert exp4506.RESULT_RELATIVE_PATH in spec
    assert "top-K" in spec
    assert "frame hash" in spec
    for field, principle in exp4506.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4506_lazy_topk_scores_only_topk_and_reuses_cache() -> None:
    """SCENARIO-REPORT-4506-LAZY-TOPK: lazy mode avoids per-node value calls."""

    benchmark = exp4506.run_lazy_value_eval_benchmark(
        frontiers=_two_frontiers_with_repeated_topk_hashes(),
        lazy_top_k=3,
        value_weight=1.0,
        work_units=0,
    )

    assert benchmark["eager"]["value_head_evals"] == 10
    assert benchmark["lazy"]["value_head_evals"] == 3
    assert benchmark["lazy"]["cache_hits"] == 3
    assert benchmark["value_head_call_reduction_factor"] == pytest.approx(10 / 3)
    assert benchmark["routing_quality_match_rate"] == pytest.approx(1.0)
    assert benchmark["routing_quality_preserved"] is True
    assert [row["lazy_selected_candidate_id"] for row in benchmark["per_trial"]] == [
        "t0-c1",
        "t1-c1",
    ]
    assert all(row["selection_matches_eager"] is True for row in benchmark["per_trial"])

    default_benchmark = exp4506.run_lazy_value_eval_benchmark(
        lazy_top_k=2,
        value_weight=1.0,
        work_units=1,
    )
    assert default_benchmark["frontier_width"] == exp4506.DEFAULT_FRONTIER_WIDTH
    assert default_benchmark["trial_count"] == exp4506.DEFAULT_TRIAL_COUNT
    assert default_benchmark["routing_quality_preserved"] is True

    with pytest.raises(ValueError, match="frontiers must include"):
        exp4506.run_lazy_value_eval_benchmark(frontiers=[], work_units=0)


def test_scenario_report_4506_schema_builds_principle_annotated_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4506-SCHEMA: artifacts carry required fields and principles."""

    benchmark = exp4506.run_lazy_value_eval_benchmark(
        frontiers=_two_frontiers_with_repeated_topk_hashes(),
        lazy_top_k=3,
        value_weight=1.0,
        work_units=0,
    )
    artifact = exp4506.build_artifact(
        benchmark=benchmark,
        preconditions_checked=_preconditions(),
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == exp4506.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"] == _preconditions()
    assert artifact["lazy_top_k"] == 3
    assert artifact["frontier_width"] == 5
    assert artifact["trial_count"] == 2
    assert artifact["cache_by_frame_hash"] is True
    assert artifact["speedup_factor"] >= 0.0
    assert artifact["value_head_call_reduction_factor"] == pytest.approx(10 / 3)
    assert artifact["routing_quality_match_rate"] == pytest.approx(1.0)
    assert artifact["routing_quality_preserved"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["field_principles"] == exp4506.FIELD_PRINCIPLES
    assert "REQ-REPORT-4506" in artifact["spec_refs"]
    assert exp4506.artifact_schema_errors(artifact) == []

    written = exp4506.write_artifact(tmp_path, artifact)
    assert json.loads(written.read_text(encoding="utf-8")) == artifact


def test_req_report_4506_schema_rejects_unprincipled_or_unmeasured_artifact() -> None:
    """REQ-REPORT-4506: schema rejects missing substrate, measurements, and quality."""

    benchmark = exp4506.run_lazy_value_eval_benchmark(
        frontiers=_two_frontiers_with_repeated_topk_hashes(),
        lazy_top_k=3,
        value_weight=1.0,
        work_units=0,
    )
    artifact = exp4506.build_artifact(
        benchmark=benchmark,
        preconditions_checked=_preconditions(),
    )
    bad = {
        **artifact,
        "honest_verdict": "done",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "preconditions_checked": [],
        "field_principles": {"honest_verdict": {"principle": "wrong"}},
        "lazy_top_k": "3",
        "frontier_width": 0,
        "trial_count": 0,
        "value_weight": 0,
        "cache_by_frame_hash": "yes",
        "speedup_factor": {"value": 2.0},
        "value_head_call_reduction_factor": None,
        "routing_quality_match_rate": 0.5,
        "routing_quality_preserved": False,
        "eager": {"value_head_evals": "10", "wall_seconds": "1.0"},
        "lazy": {"value_head_evals": 0, "wall_seconds": "0.1"},
        "per_trial": [{"selection_matches_eager": False}],
    }

    errors = exp4506.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must equal verifier_ensemble_against_cached_candidates" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "field_principles must match required principles" in errors
    assert "lazy_top_k must be bare positive int" in errors
    assert "frontier_width must be bare positive int" in errors
    assert "trial_count must be bare positive int" in errors
    assert "value_weight must be bare positive float" in errors
    assert "cache_by_frame_hash must be bare bool" in errors
    assert "speedup_factor must be bare float" in errors
    assert "value_head_call_reduction_factor must be bare float" in errors
    assert "routing_quality_preserved must be true for the complete prototype" in errors
    assert "eager.value_head_evals must be bare int" in errors
    assert "lazy.wall_seconds must be bare float" in errors
    assert "per_trial[0] must include selected candidate ids and score gap" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        exp4506.write_artifact(Path("/tmp"), bad)


def test_req_report_4506_schema_rejects_nested_edge_cases() -> None:
    """REQ-REPORT-4506: schema checks nested summaries and trial rows."""

    benchmark = exp4506.run_lazy_value_eval_benchmark(
        frontiers=_two_frontiers_with_repeated_topk_hashes(),
        lazy_top_k=3,
        value_weight=1.0,
        work_units=0,
    )
    artifact = exp4506.build_artifact(
        benchmark=benchmark,
        preconditions_checked=_preconditions(),
    )

    missing = dict(artifact)
    missing.pop("lazy_top_k")
    assert "missing required field lazy_top_k" in exp4506.artifact_schema_errors(missing)

    bad_checks = {
        **artifact,
        "preconditions_checked": {
            **_preconditions(),
            "offline_arcade_import_smoke": False,
            "torch_import": False,
        },
    }
    errors = exp4506.artifact_schema_errors(bad_checks)
    assert "preconditions_checked must record offline_arcade_import_smoke=true" in errors
    assert "preconditions_checked must record torch_import=true" in errors

    assert exp4506.artifact_schema_errors({**artifact, "eager": []}) == [
        "eager must be a mapping"
    ]
    assert "routing_quality_match_rate must be bare float in [0,1]" in exp4506.artifact_schema_errors(
        {**artifact, "routing_quality_match_rate": 1.5}
    )
    assert "leaderboard_submission must be false" in exp4506.artifact_schema_errors(
        {**artifact, "leaderboard_submission": True}
    )
    assert "per_trial must be a non-empty list" in exp4506.artifact_schema_errors(
        {**artifact, "per_trial": []}
    )
    assert "per_trial[0] must be a mapping" in exp4506.artifact_schema_errors(
        {**artifact, "per_trial": [None]}
    )

    bad_trial = {
        **artifact["per_trial"][0],
        "selection_matches_eager": "yes",
        "score_gap": "0",
        "eager_value_head_evals": "5",
        "lazy_value_head_evals": "3",
        "lazy_cache_hits": "0",
    }
    errors = exp4506.artifact_schema_errors({**artifact, "per_trial": [bad_trial]})
    assert "per_trial[0].selection_matches_eager must be bare bool" in errors
    assert "per_trial[0].score_gap must be bare float" in errors
    assert "per_trial[0].eager_value_head_evals must be bare int" in errors
    assert "per_trial[0].lazy_value_head_evals must be bare int" in errors
    assert "per_trial[0].lazy_cache_hits must be bare int" in errors


def test_req_report_4506_preconditions_record_verified_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-4506: preconditions list the import smoke and Torch resource."""

    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")

    class FakeKit:
        @staticmethod
        def offline_arcade() -> object:
            return object()

    monkeypatch.setattr(exp4506, "_import_arc_solver_kit", lambda: FakeKit)
    monkeypatch.setattr(exp4506, "_import_torch_version", lambda: "fixture-torch")

    checks = exp4506.check_preconditions(tmp_path)

    assert checks["agents_md_read"] is True
    assert checks["codex_md_read"] is True
    assert checks["offline_arcade_import_smoke"] is True
    assert checks["torch_import"] is True
    assert checks["torch_version"] == "fixture-torch"
    assert checks["cached_candidate_frontiers_built"] is True
