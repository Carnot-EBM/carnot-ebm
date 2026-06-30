"""Tests for Exp 5047 KAN/PURM energy calibration.

Spec refs: REQ-VERIFY-5047, SCENARIO-VERIFY-5047.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5047_kan_purm_energy_calibration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _raw_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "q0",
            "gold": "A",
            "candidates": [
                {"candidate_id": "q0/c0", "answer": "A", "cache_index": 0},
                {"candidate_id": "q0/c1", "answer": "B", "cache_index": 1},
            ],
        },
        {
            "row_id": "q1",
            "gold": "A",
            "candidates": [
                {"candidate_id": "q1/c0", "answer": "B", "cache_index": 0},
                {"candidate_id": "q1/c1", "answer": "A", "cache_index": 1},
            ],
        },
        {
            "row_id": "q2",
            "gold": "A",
            "candidates": [
                {"candidate_id": "q2/c0", "answer": "B", "cache_index": 0},
                {"candidate_id": "q2/c1", "answer": "A", "cache_index": 1},
            ],
        },
        {
            "row_id": "q3",
            "gold": "B",
            "candidates": [
                {"candidate_id": "q3/c0", "answer": "A", "cache_index": 0},
                {"candidate_id": "q3/c1", "answer": "B", "cache_index": 1},
            ],
        },
        {
            "row_id": "q4",
            "gold": "A",
            "candidates": [
                {"candidate_id": "q4/c0", "answer": "A", "cache_index": 0},
                {"candidate_id": "q4/c1", "answer": "B", "cache_index": 1},
            ],
        },
        {
            "row_id": "q5",
            "gold": "B",
            "candidates": [
                {"candidate_id": "q5/c0", "answer": "A", "cache_index": 0},
                {"candidate_id": "q5/c1", "answer": "B", "cache_index": 1},
            ],
        },
    ]


def _energy_by_id() -> dict[str, float]:
    return {
        "q0/c0": 0.0,
        "q0/c1": 2.0,
        "q1/c0": 0.0,
        "q1/c1": 0.02,
        "q2/c0": 0.0,
        "q2/c1": 0.01,
        "q3/c0": 0.0,
        "q3/c1": 0.03,
        "q4/c0": 0.0,
        "q4/c1": 2.0,
        "q5/c0": 0.0,
        "q5/c1": 0.02,
    }


def _feature_rows() -> list[dict[str, Any]]:
    return mod.build_readout_rows(
        _raw_rows(),
        _energy_by_id(),
        tuned_sc_predictions=["A", "A", "A", "B", "A", "B"],
    )


def test_req_verify_5047_spec_declares_contract() -> None:
    """REQ-VERIFY-5047: OpenSpec anchors the KAN/PURM calibration contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5047",
        "SCENARIO-VERIFY-5047",
        "experiment_5047_kan_purm_energy_calibration.py",
        "results/experiment_5047_kan_purm_energy_calibration.json",
        "delta_vs_powered_d1",
        "degeneracy_guard_fired",
        "held-out-safe question splits",
        "additive fuzzy/KAN/PURM readout",
    ):
        assert marker in spec


def test_scenario_verify_5047_builds_energy_margin_feature_rows() -> None:
    """SCENARIO-VERIFY-5047: candidate energies become row margins and labels."""

    rows = _feature_rows()
    assert rows[0]["powered_d1_answer"] == "A"
    assert rows[0]["powered_margin"] == 2.0
    assert rows[1]["powered_margin"] == 0.02
    assert rows[1]["powered_d1_correct"] == 0
    assert rows[1]["tuned_sc_correct"] == 1
    assert rows[1]["candidates"][0]["label"] == 0
    assert rows[1]["candidates"][1]["label"] == 1
    assert rows[1]["candidates"][0]["energy_delta"] == 0.0
    assert rows[1]["candidates"][1]["energy_delta"] == 0.02


def test_req_verify_5047_split_integrity_has_no_question_overlap() -> None:
    """REQ-VERIFY-5047: cross-validation splits never leak question ids."""

    splits = mod.make_cv_splits(11, n_folds=4, seed=7)
    all_test_indices: list[int] = []
    for split in splits:
        train = set(split["train_indices"])
        test = set(split["test_indices"])
        assert train
        assert test
        assert train.isdisjoint(test)
        all_test_indices.extend(split["test_indices"])
    assert sorted(all_test_indices) == list(range(11))
    assert mod.split_integrity_errors(splits, n_rows=11) == []
    bad = [{"train_indices": [0, 1], "test_indices": [1, 2]}]
    assert "train_test_overlap" in mod.split_integrity_errors(bad, n_rows=3)


def test_req_verify_5047_collapse_and_abstention_detection() -> None:
    """REQ-VERIFY-5047: collapse to D1 and >50% abstention both fire the guard."""

    collapsed = mod.degeneracy_guard(
        ["A", "B", "C"],
        ["A", "B", "C"],
        abstention_rate=0.0,
    )
    abstaining = mod.degeneracy_guard(
        ["A", "B", "A", "B"],
        ["A", "A", "B", "B"],
        abstention_rate=0.75,
    )
    clean = mod.degeneracy_guard(
        ["A", "B", "A", "B"],
        ["A", "A", "B", "B"],
        abstention_rate=0.25,
    )
    assert collapsed["degeneracy_guard_fired"] is True
    assert "collapsed_to_powered_d1" in collapsed["reasons"]
    assert abstaining["degeneracy_guard_fired"] is True
    assert "abstention_gt_0p50" in abstaining["reasons"]
    assert clean["degeneracy_guard_fired"] is False


def test_scenario_verify_5047_run_with_injected_panel_writes_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5047: injected held-out run reports powered-D1 deltas."""

    upstream = {
        "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
        "model_specs": {"lora_ebm": {"base_model": "Qwen/Qwen3.5-2B"}},
        "powered_lora_ebm_accuracy": 0.5,
        "genuine_tuned_sc_accuracy": 1.0,
        "headroom_present": True,
    }
    upstream_path = tmp_path / mod.EXP5045_RELATIVE_PATH
    _write_json(upstream_path, upstream)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        upstream_loader=lambda _root: upstream,
        panel_loader=lambda _root, _upstream: _feature_rows(),
        bootstrap_samples=32,
        n_folds=3,
        logistic_epochs=60,
        now=lambda: 42.0,
        write=True,
    )

    assert artifact["calibration_available"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is True
    assert artifact["powered_d1_accuracy"] == 0.333333
    assert artifact["calibrated_accuracy"] >= artifact["powered_d1_accuracy"]
    assert artifact["delta_vs_powered_d1"] == round(
        artifact["calibrated_accuracy"] - artifact["powered_d1_accuracy"], 6
    )
    assert isinstance(artifact["paired_ci95"], list)
    assert "logistic_baseline_accuracy" in artifact["baselines"]
    assert artifact["readout"]["kind"] == "additive_fuzzy_kan_purm"
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_5047_defensive_helpers_and_blocked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5047: malformed panels block honestly and schema checks fire."""

    valid = tmp_path / "valid.json"
    _write_json(valid, {"ok": True})
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod._read_json(valid) == {"ok": True}
    assert mod._read_json(bad) is None
    assert mod._read_json(tmp_path / "missing.json") is None
    assert mod._number(True, 9.0) == 9.0
    assert mod._number("bad", 8.0) == 8.0
    assert mod._normalized_entropy(mod.Counter()) == 0.0
    assert mod._std([1.0]) == 0.0
    assert mod._standardize([]) == ([], [])
    assert mod._select_answer({"candidates": []}, {}, threshold=0.0, kind="kan") == (None, True)

    assert mod.build_readout_rows(
        [{"row_id": "empty", "gold": "A", "candidates": [{"candidate_id": "missing"}]}],
        {},
        tuned_sc_predictions=[],
    ) == []
    assert mod.make_cv_splits(1) == [{"train_indices": [0], "test_indices": [0]}]
    split_errors = mod.split_integrity_errors(
        [{"train_indices": [], "test_indices": []}, {"train_indices": [4], "test_indices": [5]}],
        n_rows=2,
    )
    assert "empty_train_split" in split_errors
    assert "empty_test_split" in split_errors
    assert "split_index_out_of_range" in split_errors
    assert "test_coverage_not_exactly_once" in split_errors
    assert mod._tuned_predictions_from_upstream(
        {"evaluation": {"tuned_self_consistency": {"predictions": ["A"]}}}
    ) == ["A"]
    assert mod._tuned_predictions_from_upstream({}) == []

    no_upstream = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "no_upstream.json",
        upstream_loader=lambda _root: {},
        now=lambda: 1.0,
        write=True,
    )
    assert no_upstream["honest_verdict"] == "blocked_exp5045_artifact_unavailable"
    assert no_upstream["calibration_available"] is False

    upstream = {"model_specs": {}, "headroom_present": True}
    blocked_panel = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_panel.json",
        upstream_loader=lambda _root: upstream,
        panel_loader=lambda _root, _upstream: [],
        now=lambda: 2.0,
        write=True,
    )
    assert blocked_panel["honest_verdict"] == "blocked_candidate_energy_panel_unavailable"
    assert "only 0 calibrated rows" in blocked_panel["blocked_error"]

    monkeypatch.setattr(mod, "split_integrity_errors", lambda _splits, n_rows: ["bad_split"])
    with pytest.raises(ValueError, match="split integrity"):
        mod.evaluate_cross_validated_readout(_feature_rows(), n_folds=3, bootstrap_samples=8)


def test_req_verify_5047_verdict_and_schema_branches() -> None:
    """REQ-VERIFY-5047: success/null verdicts and schema errors are explicit."""

    rows = _feature_rows()
    upstream = {"model_specs": {"lora_ebm": {"base_model": "Qwen/Qwen3.5-2B"}}, "headroom_present": True}
    evaluation = mod.evaluate_cross_validated_readout(
        rows,
        n_folds=3,
        bootstrap_samples=16,
        logistic_epochs=20,
    )

    success_eval = dict(evaluation)
    success_eval.update(
        {
            "delta_vs_powered_d1": 0.2,
            "paired_ci95": [0.1, 0.3],
            "mcnemar_p": 0.01,
            "degeneracy_guard": {
                "degeneracy_guard_fired": False,
                "abstention_rate": 0.0,
                "collapse_rate": 0.5,
                "reasons": [],
            },
        }
    )
    degenerate_eval = dict(success_eval)
    degenerate_eval["degeneracy_guard"] = {
        "degeneracy_guard_fired": True,
        "abstention_rate": 0.0,
        "collapse_rate": 1.0,
        "reasons": ["collapsed_to_powered_d1"],
    }
    ci_eval = dict(success_eval)
    ci_eval.update({"paired_ci95": [-0.1, 0.2], "degeneracy_guard": success_eval["degeneracy_guard"]})

    success = mod._complete_artifact(
        upstream=upstream,
        rows=rows,
        evaluation=success_eval,
        duration_s=1.0,
    )
    degenerate = mod._complete_artifact(
        upstream=upstream,
        rows=rows,
        evaluation=degenerate_eval,
        duration_s=1.0,
    )
    ci_null = mod._complete_artifact(
        upstream=upstream,
        rows=rows,
        evaluation=ci_eval,
        duration_s=1.0,
    )

    assert success["honest_verdict"].startswith("success_kan_purm_beats_powered_d1")
    assert degenerate["honest_verdict"].endswith("_degenerate")
    assert ci_null["honest_verdict"].endswith("_ci_incl_0")
    assert mod._ci_includes_zero([-0.1, 0.0]) is True
    assert mod._ci_includes_zero([0.1]) is False
    assert mod._format_delta(0.125) == "plus_0p125"
    assert mod.reproducibility_checksum({"x": 1}).startswith("sha256:")

    assert mod.artifact_schema_errors(success) == []
    missing = dict(success)
    missing.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing)
    assert "spec_refs" in mod.artifact_schema_errors({**success, "spec_refs": []})
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**success, "verifier_is_oracle": True}
    )
    assert "calibration_available" in mod.artifact_schema_errors(
        {**success, "calibration_available": "yes"}
    )
    assert "calibrated_accuracy" in mod.artifact_schema_errors(
        {**success, "calibrated_accuracy": 2.0}
    )
    assert "delta_vs_powered_d1" in mod.artifact_schema_errors(
        {**success, "delta_vs_powered_d1": "bad"}
    )
    assert "paired_ci95" in mod.artifact_schema_errors({**success, "paired_ci95": [0.0]})
    assert "model_specs" in mod.artifact_schema_errors({**success, "model_specs": []})
    assert "split_diagnostics" in mod.artifact_schema_errors(
        {**success, "split_diagnostics": []}
    )
    assert "baselines" in mod.artifact_schema_errors({**success, "baselines": []})
    assert "readout" in mod.artifact_schema_errors({**success, "readout": []})
    assert "honest_verdict" in mod.artifact_schema_errors({**success, "honest_verdict": "bad"})
