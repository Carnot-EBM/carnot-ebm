"""Tests for Exp 2867 residual-drift MUS prioritizer diagnostic.

Spec: REQ-VERIFY-2867,
      SCENARIO-VERIFY-2867,
      SCENARIO-VERIFY-2867-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import drift_mus_prioritizer_v2_2867 as mod


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fover_source_payload() -> dict[str, object]:
    return {
        "honest_verdict": "complete: FoVer clean source",
        "condition_a_production_auroc_mean": 0.91,
        "condition_b_architecture_only_auroc_mean": 0.89,
        "n_examples": 1000,
        "n_seeds": 2,
        "per_verifier_condition_a_auroc": {
            "fr11_session_memory": [0.87, 0.88],
            "tier0r_curry_howard": [0.90, 0.91],
            "tier0s_arithmetic_gap": [0.29, 0.31],
            "tier0u_logical_consistency": [0.51, 0.52],
        },
    }


def _halueval_fever_source_payload() -> dict[str, object]:
    return {
        "honest_verdict": "complete: HaluEval/FEVER local calibration ready",
        "halueval_fever_ready": True,
        "full_benchmark_ready": True,
        "halueval_auroc": 0.553,
        "fever_auroc": 0.331,
        "halueval_n_examples": 500,
        "fever_n_examples": 500,
    }


def _matrix_payload() -> dict[str, object]:
    return {
        "artifact": "experiment_2865_cross_corpus_matrix_v5",
        "honest_verdict": "complete: cross-corpus matrix built from 2 clean corpus rows",
        "cross_corpus_matrix_built": True,
        "verifier_corpus_dual_matrix": {
            "FoVer": {
                "corpus": "FoVer",
                "row_status": "clean",
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
                "production_auroc": 0.91,
                "architecture_only_auroc": 0.89,
                "learning_contribution": 0.02,
                "n_examples": 1000,
                "n_seeds": 2,
            },
            "HaluEval/FEVER": {
                "corpus": "HaluEval/FEVER",
                "row_status": "clean",
                "source_artifact": (
                    "results/experiment_2864_halueval_fever_full_calibration_v3.json"
                ),
                "measured_auroc_by_dataset": {
                    "halueval": 0.553,
                    "fever": 0.331,
                },
                "n_examples_by_dataset": {"halueval": 500, "fever": 500},
                "n_examples": 1000,
            },
        },
        "row_status_by_corpus": {
            "FoVer": "clean",
            "HaluEval/FEVER": "clean",
            "MBPP": "missing",
            "HumanEval": "missing",
            "TruthfulQA": "missing",
        },
    }


def _write_clean_fixture(root: Path) -> None:
    _write_json(root, "results/experiment_2865_cross_corpus_matrix_v5.json", _matrix_payload())
    _write_json(
        root,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _fover_source_payload(),
    )
    _write_json(
        root,
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        _halueval_fever_source_payload(),
    )


def test_scenario_verify_2867_clean_matrix_builds_prioritizer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2867: clean rows drive diagnostic, hypergraph, and baselines."""

    _write_clean_fixture(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    required = {
        "honest_verdict",
        "drift_mus_diagnostic_ready",
        "n_failure_rows",
        "failure_class_counts",
        "hypergraph_nodes",
        "hypergraph_hyperedges",
        "hgnn_inspired_heuristic_name",
        "baseline_random_checks_to_conflict",
        "baseline_degree_checks_to_conflict",
        "heuristic_checks_to_conflict",
        "heuristic_improvement_vs_best_baseline",
        "recommended_repairs",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["drift_mus_diagnostic_ready"] is True
    assert artifact["n_failure_rows"] == 4
    assert artifact["failure_class_counts"] == {
        "below_random_auroc": 2,
        "near_random_auroc": 2,
    }
    assert artifact["hypergraph_nodes"] > 0
    assert artifact["hypergraph_hyperedges"] >= artifact["n_failure_rows"]
    assert artifact["hgnn_inspired_heuristic_name"].endswith("_not_trained_hgnn")
    assert artifact["baseline_random_checks_to_conflict"] >= artifact[
        "heuristic_checks_to_conflict"
    ]
    assert artifact["heuristic_improvement_vs_best_baseline"] == pytest.approx(
        min(
            artifact["baseline_random_checks_to_conflict"],
            artifact["baseline_degree_checks_to_conflict"],
        )
        - artifact["heuristic_checks_to_conflict"]
    )
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["random_seed"] == 2867
    assert len(artifact["reproducibility_checksum"]) == 64
    assert all(row["corpus"] != "MBPP" for row in artifact["failure_rows"])
    assert any("not a trained HGNN" in repair for repair in artifact["recommended_repairs"])
    assert artifact["field_principles"]["hgnn_inspired_heuristic_name"].startswith(
        "Descriptive"
    )

    out = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert out == tmp_path / "results/experiment_2867_drift_mus_prioritizer_v2.json"
    assert saved["drift_mus_diagnostic_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.25)


def test_scenario_verify_2867_blocked_matrix_writes_zero_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2867-BLOCKED: unbuilt matrix blocks without inferred failures."""

    blocked_matrix = {
        **_matrix_payload(),
        "cross_corpus_matrix_built": False,
        "verifier_corpus_dual_matrix": {},
    }
    _write_json(tmp_path, "results/experiment_2865_cross_corpus_matrix_v5.json", blocked_matrix)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=4.0)

    assert artifact["honest_verdict"] == "blocked_cross_corpus_matrix_not_built"
    assert artifact["drift_mus_diagnostic_ready"] is False
    assert artifact["n_failure_rows"] == 0
    assert artifact["failure_class_counts"] == {}
    assert artifact["hypergraph_nodes"] == 0
    assert artifact["hypergraph_hyperedges"] == 0
    assert artifact["baseline_random_checks_to_conflict"] == 0.0
    assert artifact["baseline_degree_checks_to_conflict"] == 0.0
    assert artifact["heuristic_checks_to_conflict"] == 0.0
    assert artifact["heuristic_improvement_vs_best_baseline"] == 0.0
    checks = {row["name"]: row for row in artifact["preconditions_checked"]}
    assert checks["cross_corpus_matrix_built"]["ok"] is False
    assert artifact["duration_s"] == pytest.approx(1.0)


def test_req_verify_2867_helpers_are_deterministic_and_guard_bad_inputs(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2867: helpers reject bad sources and rank conflicts deterministically."""

    missing = tmp_path / "missing.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")

    assert mod.read_json(missing) == {}
    assert mod.read_json(bad) == {}
    assert mod.read_json(array) == {}
    assert mod.finite_float(True) is None
    assert mod.finite_float(float("nan")) is None
    assert mod.classify_failure(0.49) == ("below_random_auroc", pytest.approx(0.01))
    assert mod.classify_failure(0.55) == ("near_random_auroc", pytest.approx(0.01))
    assert mod.classify_failure(0.65) is None
    assert mod._mean([]) is None
    assert mod._clean_matrix_rows({"verifier_corpus_dual_matrix": []}) == {}
    assert mod._constraint_family("Other", "unmapped_metric") == "verifier_calibration"
    assert mod._fover_verifier_means({}) == {}
    assert mod._fover_verifier_means(
        {"per_verifier_condition_a_auroc": {"bad": "not-list", "empty": []}}
    ) == {}
    assert mod._source_payloads(
        tmp_path,
        {"NoSource": {"row_status": "clean", "source_artifact": ""}},
    ) == {}
    assert mod._halueval_fever_metrics(
        {},
        {"halueval_auroc": 0.7, "fever_auroc": 0.49},
    ) == {"halueval": 0.7, "fever": 0.49}

    no_failure_matrix = {
        "verifier_corpus_dual_matrix": {
            "HaluEval/FEVER": {
                "row_status": "clean",
                "source_artifact": "",
                "measured_auroc_by_dataset": {"halueval": 0.8},
            }
        }
    }
    assert mod.extract_clean_failure_evidence(tmp_path, no_failure_matrix, {"": {}}) == []

    rows = [
        mod.FailureEvidence(
            row_id="a",
            corpus="FoVer",
            source_metric="tier0s_arithmetic_gap",
            auroc=0.30,
            failure_class="below_random_auroc",
            severity=0.20,
            residual_drift=0.60,
            constraint_family="arithmetic_consistency",
            verifier_failure="tier0s_arithmetic_gap",
            source_artifact="results/a.json",
        ),
        mod.FailureEvidence(
            row_id="b",
            corpus="HaluEval/FEVER",
            source_metric="fever",
            auroc=0.33,
            failure_class="below_random_auroc",
            severity=0.17,
            residual_drift=0.58,
            constraint_family="factual_support",
            verifier_failure="fever_local_calibration",
            source_artifact="results/b.json",
        ),
        mod.FailureEvidence(
            row_id="c",
            corpus="HaluEval/FEVER",
            source_metric="halueval",
            auroc=0.553,
            failure_class="near_random_auroc",
            severity=0.007,
            residual_drift=0.36,
            constraint_family="factual_support",
            verifier_failure="halueval_local_calibration",
            source_artifact="results/b.json",
        ),
    ]

    hypergraph = mod.build_hypergraph(rows)
    degree_ranking = mod.rank_nodes_by_degree(hypergraph)
    heuristic_ranking = mod.rank_nodes_by_residual_message_passing(hypergraph)
    empty_graph = mod.Hypergraph(nodes=(), hyperedges=())

    assert mod.checks_to_first_conflict(heuristic_ranking, hypergraph) <= mod.checks_to_first_conflict(
        degree_ranking,
        hypergraph,
    )
    assert mod.rank_nodes_by_residual_message_passing(empty_graph) == []
    assert mod.checks_to_first_conflict([], hypergraph) == 0.0
    assert mod.random_baseline_checks(empty_graph) == 0.0
    assert mod.random_baseline_checks(hypergraph, seed=2867, trials=32) == mod.random_baseline_checks(
        hypergraph,
        seed=2867,
        trials=32,
    )
    checksum_a = mod.reproducibility_checksum(
        matrix_payload={"cross_corpus_matrix_built": True},
        source_payloads={"results/a.json": {"x": 1}},
        failure_rows=rows,
        hypergraph=hypergraph,
        metrics={"heuristic": 1.0},
    )
    checksum_b = mod.reproducibility_checksum(
        matrix_payload={"cross_corpus_matrix_built": True},
        source_payloads={"results/a.json": {"x": 2}},
        failure_rows=rows,
        hypergraph=hypergraph,
        metrics={"heuristic": 1.0},
    )
    assert checksum_a != checksum_b
