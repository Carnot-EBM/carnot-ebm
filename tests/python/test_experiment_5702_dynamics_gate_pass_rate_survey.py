"""Tests for Exp 5702: real-world pass-rate survey of the live pipeline's
min_heldout_accuracy=1.0 dynamics gate (task 11 completion).
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5702_dynamics_gate_pass_rate_survey as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5593_4_spec_declares_pass_rate_survey() -> None:
    """REQ-ARC-WMTE-5593-4: OpenSpec declares the corpus-wide pass-rate survey and its
    real result, including the honest per-row-vs-bounded-retry-loop limitation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593-4") :]
    section = section[: section.index("### REQ-ARC-WMTE-5593:")]

    for marker in (
        "SCENARIO-ARC-WMTE-5593-4-CORPUS-PASS-RATE",
        "pass_rate_at_live_threshold=0.1263",
        "Honest conclusion and limitation",
    ):
        assert marker in section


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def test_walk_heldout_accuracy_rows_finds_nested_values() -> None:
    out: list[float] = []
    mod._walk_heldout_accuracy_rows(
        {
            "rounds": [
                {"round": 1, "heldout_accuracy": 0.5},
                {"round": 2, "heldout_accuracy": 1.0},
            ],
            "nested": {"heldout_accuracy": 0.25},
            "not_a_number": {"heldout_accuracy": "n/a"},
            "boolean_should_not_count": {"heldout_accuracy": True},
        },
        out,
    )
    assert sorted(out) == [0.25, 0.5, 1.0]


def test_collect_rows_excludes_configured_substrings(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1_x.json").write_text(
        json.dumps(
            {
                "inference_substrate": "live_llm_inference",
                "rounds": [{"heldout_accuracy": 0.4}],
            }
        )
    )
    (results / "experiment_5700_excluded.json").write_text(
        json.dumps(
            {
                "inference_substrate": "live_llm_inference",
                "rounds": [{"heldout_accuracy": 1.0}],
            }
        )
    )
    (results / "experiment_2_wrong_substrate.json").write_text(
        json.dumps(
            {
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "rounds": [{"heldout_accuracy": 1.0}],
            }
        )
    )
    values, cited, excluded = mod.collect_rows(root=tmp_path)
    assert values == [0.4]
    assert cited == ["results/experiment_1_x.json"]
    assert excluded == ["results/experiment_5700_excluded.json"]


def test_build_artifact_blocked_when_results_dir_missing(tmp_path) -> None:
    artifact = mod.build_artifact(root=tmp_path)
    assert artifact["honest_verdict"] == "complete: blocked_results_dir_present"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert len(artifact["reproducibility_checksum"]) == 64


def test_build_artifact_no_rows_found_is_reported_honestly(tmp_path) -> None:
    (tmp_path / "results").mkdir()
    artifact = mod.build_artifact(root=tmp_path)
    assert artifact["honest_verdict"] == "complete: no_real_induction_rows_found"
    assert artifact["n_rows"] == 0


def test_build_artifact_computes_pass_rate_and_sweep(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1_x.json").write_text(
        json.dumps(
            {
                "inference_substrate": "live_llm_inference",
                "rounds": [
                    {"heldout_accuracy": 1.0},
                    {"heldout_accuracy": 0.0},
                    {"heldout_accuracy": 0.0},
                    {"heldout_accuracy": 0.5},
                ],
            }
        )
    )
    artifact = mod.build_artifact(root=tmp_path)
    assert artifact["n_rows"] == 4
    assert artifact["pass_rate_at_live_threshold"] == 0.25
    assert artifact["exact_zero_rate"] == 0.5
    assert artifact["threshold_sweep"]["1.0"] == 0.25
    assert artifact["threshold_sweep"]["0.5"] == 0.5
    assert "dynamics_gate_moderately_strict_pass_rate" in artifact["honest_verdict"]


def test_req_arc_wmte_5593_3_repository_artifact_is_a_real_measured_survey() -> None:
    """The checked-in real survey aggregated the actual corpus of live_llm_inference
    artifacts and found a low real-world pass rate at the live threshold -- a
    genuine, citable finding about gate calibration, not a synthetic fixture."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert result["n_rows"] > 50
    assert result["n_source_files"] > 5
    assert result["pass_rate_at_live_threshold"] is not None
    assert result["pass_rate_at_live_threshold"] < 0.3
    assert not any("5700" in f for f in result["cited_upstream_artifacts"])
    assert any("5700" in f for f in result["excluded_artifacts"])
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64
