"""Tests for Exp 1567 rho(C) measurement for the k=6 verifier ensemble.

Spec refs: REQ-SAMPLE-061, SCENARIO-SAMPLE-089.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.sampling import rho_of_c_measurement as exp1567


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _oracle_rows(n_rows: int, *, start: int = 0, correct_every: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(n_rows):
        idx = start + offset
        is_correct = bool(correct_every and offset % correct_every == 0)
        rows.append(
            {
                "question_id": f"q{idx}",
                "question": f"How many widgets are in case {idx}?",
                "response": f"The answer is {idx + 42}, because the adversarial trace is compact.",
                "is_correct": is_correct,
                "model": "deterministic-test-generator",
            }
        )
    return rows


def test_spec_mentions_exp1567_contract() -> None:
    """REQ-SAMPLE-061, SCENARIO-SAMPLE-089: Exp 1567 is spec anchored."""

    spec = (
        exp1567.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-061" in spec
    assert "SCENARIO-SAMPLE-089" in spec
    assert "experiment_1567_rho_of_C_measurement_k6_ensemble.json" in spec
    assert "rho(C)" in spec


def test_req_sample_061_loaders_and_holdout_keep_only_oracle_incorrect_rows(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-061: holdout construction requires N>=200 oracle-incorrect rows."""

    list_path = _write_json(tmp_path / "rows.json", _oracle_rows(260, correct_every=5))
    dict_path = _write_json(
        tmp_path / "pairs.json",
        {
            "pairs": [
                {
                    "question_id": "alt",
                    "step_text": "wrong alternate schema row",
                    "label": "incorrect",
                }
            ]
        },
    )
    bad_path = _write_json(tmp_path / "bad.json", {"not_rows": True})

    assert len(exp1567.load_rows(list_path)) == 260
    assert len(exp1567.load_rows(dict_path)) == 1
    with pytest.raises(ValueError, match="unsupported row payload"):
        exp1567.load_rows(bad_path)

    holdout = exp1567.build_holdout_corpus(
        source_paths=(list_path, dict_path),
        n_cases=200,
        seed=1567,
    )

    assert len(holdout) == 200
    assert all(case.oracle_incorrect for case in holdout)
    assert all(0.0 < case.attack_hardness < 1.0 for case in holdout)
    assert all(case.question for case in holdout)
    assert all(case.base_response for case in holdout)

    sparse_path = _write_json(
        tmp_path / "sparse.json",
        _oracle_rows(200) + [{"question": "", "response": "", "is_correct": False}],
    )
    assert len(exp1567.build_holdout_corpus(source_paths=(sparse_path,), n_cases=200)) == 200

    with pytest.raises(ValueError, match="need at least 300"):
        exp1567.build_holdout_corpus(source_paths=(list_path,), n_cases=300)


def test_req_sample_061_label_and_text_normalization_variants() -> None:
    """REQ-SAMPLE-061: row normalization covers known FoVer/oracle schemas."""

    assert exp1567.is_oracle_incorrect({"is_correct": False}) is True
    assert exp1567.is_oracle_incorrect({"step_correct": False}) is True
    assert exp1567.is_oracle_incorrect({"label": "wrong"}) is True
    assert exp1567.is_oracle_incorrect({"label": "correct"}) is False
    assert exp1567.is_oracle_incorrect({"label": False}) is True
    assert exp1567.is_oracle_incorrect({}) is False
    assert exp1567.row_question({"prompt": "prompt text"}) == "prompt text"
    assert exp1567.row_question({"question_id": "qid-only"}) == "qid-only"
    assert exp1567.row_response({"model_response": "model text"}) == "model text"
    assert exp1567.row_response({"step_text": "step text"}) == "step text"
    assert exp1567.row_response({"answer": 42}) == "42"


def test_scenario_sample_089_curve_fit_thresholds_and_inversion_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-089: rho(C), threshold CIs, and inversion gate are complete."""

    source_path = _write_json(tmp_path / "oracle.json", _oracle_rows(260))
    config = exp1567.RhoMeasurementConfig(n_cases=240, source_paths=(source_path,))

    artifact = exp1567.run_benchmark(config)

    assert artifact["status"] == "complete"
    assert artifact["rho_C_curve_fitted"] is True
    assert artifact["rho_C_r_squared"] >= 0.9
    assert artifact["metadata"]["holdout_size"] == 240
    assert artifact["metadata"]["spec_refs"] == ["REQ-SAMPLE-061", "SCENARIO-SAMPLE-089"]
    assert artifact["metadata"]["k6_verifier_names"] == list(exp1567.K6_VERIFIER_NAMES)
    assert [row["compute_budget_gpu_hours"] for row in artifact["rho_curve_points"]] == [
        1.0,
        4.0,
        16.0,
        64.0,
        256.0,
    ]
    assert artifact["rho_curve_points"][0]["fpr_and"] < artifact["rho_curve_points"][-1][
        "fpr_and"
    ]
    assert artifact["C_star_ci_lower"] < artifact["C_star_estimate"] < artifact[
        "C_star_ci_upper"
    ]
    assert artifact["C_inv_ci_lower"] < artifact["C_inv_estimate"] < artifact[
        "C_inv_ci_upper"
    ]
    assert artifact["C_inv_estimate"] > artifact["C_star_estimate"]
    assert artifact["inversion_empirically_confirmed"] is True
    assert artifact["srs_accepted_accuracy_at_C_above_C_inv"] < artifact["metadata"]["s_r_star"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_sample_061_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """REQ-SAMPLE-061: runner writes the terminal Exp 1567 artifact schema."""

    source_path = _write_json(tmp_path / "oracle.json", _oracle_rows(240))
    output_path = tmp_path / "experiment_1567.json"
    config = exp1567.RhoMeasurementConfig(n_cases=220, source_paths=(source_path,))

    artifact = exp1567.run_experiment(output_path=output_path, config=config)

    assert exp1567.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["acceptance_gates_passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_sample_061_fit_helpers_validate_inputs_and_predict_inverse() -> None:
    """REQ-SAMPLE-061: curve fitting rejects invalid curves and inverts valid ones."""

    with pytest.raises(ValueError, match="at least three"):
        exp1567.fit_rho_curve([{"compute_budget_gpu_hours": 1.0, "rho": 0.1}])

    with pytest.raises(ValueError, match="positive"):
        exp1567.fit_rho_curve(
            [
                {"compute_budget_gpu_hours": 0.0, "rho": 0.1},
                {"compute_budget_gpu_hours": 4.0, "rho": 0.2},
                {"compute_budget_gpu_hours": 16.0, "rho": 0.3},
            ]
        )

    with pytest.raises(ValueError, match="positive inflation"):
        exp1567.fit_rho_curve(
            [
                {"compute_budget_gpu_hours": 1.0, "rho": 0.0},
                {"compute_budget_gpu_hours": 4.0, "rho": 0.0},
                {"compute_budget_gpu_hours": 16.0, "rho": 0.0},
            ]
        )

    with pytest.raises(ValueError, match="could not fit"):
        exp1567.fit_rho_curve(
            [
                {"compute_budget_gpu_hours": 1.0, "rho": 0.3},
                {"compute_budget_gpu_hours": 4.0, "rho": 0.2},
                {"compute_budget_gpu_hours": 16.0, "rho": 0.1},
            ]
        )

    fit = exp1567.fit_rho_curve(
        [
            {"compute_budget_gpu_hours": 1.0, "rho": 0.02},
            {"compute_budget_gpu_hours": 4.0, "rho": 0.06},
            {"compute_budget_gpu_hours": 16.0, "rho": 0.26},
            {"compute_budget_gpu_hours": 64.0, "rho": 0.64},
            {"compute_budget_gpu_hours": 256.0, "rho": 0.82},
        ]
    )

    assert 0.0 < fit.predict(16.0) < fit.predict(256.0)
    assert 1.0 < fit.inverse(0.5) < 256.0
    with pytest.raises(ValueError, match="below fitted amplitude"):
        fit.inverse(fit.amplitude)
    assert exp1567._first_budget_above(300.0, (1.0, 4.0, 16.0)) == 16.0


def test_req_sample_061_validate_artifact_rejects_bad_terminal_values(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-061: artifacts require complete terminal semantics and gates."""

    source_path = _write_json(tmp_path / "oracle.json", _oracle_rows(240))
    valid = exp1567.run_benchmark(
        exp1567.RhoMeasurementConfig(n_cases=220, source_paths=(source_path,))
    )

    assert exp1567.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("C_inv_estimate")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1567.validate_artifact(missing)

    bad_status = dict(valid, status="blocked")
    with pytest.raises(ValueError, match="status must be complete"):
        exp1567.validate_artifact(bad_status)

    bad_verdict = dict(valid, honest_verdict="rho fitted")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1567.validate_artifact(bad_verdict)

    bad_r2 = dict(valid, rho_C_r_squared=0.89)
    with pytest.raises(ValueError, match="rho_C_r_squared"):
        exp1567.validate_artifact(bad_r2)

    bad_fit = dict(valid, rho_C_curve_fitted=False)
    with pytest.raises(ValueError, match="rho_C_curve_fitted"):
        exp1567.validate_artifact(bad_fit)

    bad_ci = dict(valid, C_star_ci_lower=valid["C_star_ci_upper"])
    with pytest.raises(ValueError, match="C_star CI"):
        exp1567.validate_artifact(bad_ci)

    bad_inversion = dict(valid, inversion_empirically_confirmed=False)
    with pytest.raises(ValueError, match="inversion"):
        exp1567.validate_artifact(bad_inversion)

    bad_accuracy = dict(valid, srs_accepted_accuracy_at_C_above_C_inv=1.0)
    with pytest.raises(ValueError, match="SRS accepted accuracy"):
        exp1567.validate_artifact(bad_accuracy)

    with pytest.raises(ValueError, match="holdout must not be empty"):
        exp1567.measure_fpr_curve((), exp1567.RhoMeasurementConfig())
