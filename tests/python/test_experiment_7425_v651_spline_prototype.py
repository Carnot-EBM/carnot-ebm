"""Tests for REQ-KAN-7425 additive cubic-spline energy prototype."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7425_v651_spline_prototype as exp


ROOT = Path(__file__).resolve().parents[2]


def _training_fixture() -> np.ndarray:
    """Return finite analytic training inputs with repeated quantiles."""

    return np.asarray(
        [
            [0.0, 2.0, -1.0, 0.0, 1.0, 4.0],
            [0.0, 2.0, -0.5, 0.2, 1.0, 3.0],
            [0.0, 2.0, 0.0, 0.4, 1.0, 2.0],
            [0.5, 2.0, 0.5, 0.6, 1.0, 1.0],
            [0.5, 2.0, 1.0, 0.8, 1.0, 0.0],
            [1.0, 2.0, 1.5, 1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )


def _event(
    event_id: str = "feedback-1",
    *,
    label: int = 1,
    reveal_time: int = 3,
    features: np.ndarray | None = None,
) -> dict[str, object]:
    """Build one immutable analytic feedback event."""

    values = features if features is not None else np.asarray([0.2, 2.0, 0.1, 0.7, 1.0, 2.5])
    return {
        "event_id": event_id,
        "source_version": "analytic-fixture-v1",
        "reveal_time": reveal_time,
        "features": np.asarray(values, dtype=np.float64).tolist(),
        "label": label,
    }


def _passing_receipts() -> list[dict[str, object]]:
    """Build the exact receipt-name set used by cold fixture validation."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_kan_7425_spec_precedes_implementation() -> None:
    """REQ-KAN-7425: the implementation has a complete driving contract."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-KAN-7425:" in text
    for number in range(1, 9):
        assert f"SCENARIO-KAN-7425-{number:02d}" in text


def test_scenario_kan_7425_01_partition_endpoints_duplicates_and_clamps() -> None:
    """SCENARIO-KAN-7425-01: cubic bases are deterministic, local, and normalized."""

    training = _training_fixture()
    knots = exp.fit_quantile_knots(training)
    assert knots.shape == (exp.INPUT_COUNT, exp.KNOT_VECTOR_SIZE)
    assert np.array_equal(knots, exp.fit_quantile_knots(training))
    for feature in range(exp.INPUT_COUNT):
        for value in (
            knots[feature, 0],
            np.mean(knots[feature, [exp.DEGREE, -exp.DEGREE - 1]]),
            knots[feature, -1],
        ):
            evaluated = exp.cubic_basis(float(value), knots[feature])
            assert np.sum(evaluated.values) == pytest.approx(1.0, abs=1e-14)
            assert len(evaluated.active_indices) <= exp.DEGREE + 1
        below = exp.cubic_basis(float(knots[feature, 0] - 10.0), knots[feature])
        above = exp.cubic_basis(float(knots[feature, -1] + 10.0), knots[feature])
        assert below.clamped is True
        assert above.clamped is True
        assert below.active_indices == (0,)
        assert above.active_indices == (exp.COEFFICIENTS_PER_INPUT - 1,)

    prediction = exp.SplineEnergyHead.from_training(training, seed=651).predict(
        [-10.0, 2.0, 0.0, 0.5, 1.0, 20.0]
    )
    assert prediction["clamped_feature_count"] == 2
    assert prediction["active_coefficient_count"] <= exp.INPUT_COUNT * (exp.DEGREE + 1)


def test_scenario_kan_7425_02_energy_normalizer_equals_fixed_basis_logistic() -> None:
    """SCENARIO-KAN-7425-02: two-state energy is fixed-basis logistic regression."""

    head = exp.SplineEnergyHead.from_training(_training_fixture(), seed=13)
    x = np.asarray([0.25, 2.0, 0.3, 0.55, 1.0, 2.25], dtype=np.float64)
    prediction = head.predict(x)
    dense_basis, _ = exp.dense_design_vector(x, head.knots)
    fixed_logit = float(dense_basis @ head.coefficients.reshape(-1) + head.bias)
    normalizer = math.exp(-prediction["energy_y0"]) + math.exp(-prediction["energy_y1"])
    normalized_p1 = math.exp(-prediction["energy_y1"]) / normalizer
    assert prediction["energy_y0"] == 0.0
    assert prediction["energy_y1"] == pytest.approx(-fixed_logit, abs=1e-14)
    assert prediction["probability"] == pytest.approx(exp.sigmoid(fixed_logit), abs=1e-14)
    assert normalized_p1 == pytest.approx(prediction["probability"], abs=1e-14)
    assert head.parameter_count == 49


def test_scenario_kan_7425_03_sparse_dense_update_parity_and_footprint() -> None:
    """SCENARIO-KAN-7425-03: sparse and dense float64 updates agree within 1e-10."""

    sparse = exp.SplineEnergyHead.from_training(_training_fixture(), seed=23)
    dense = exp.DenseSplineReference.from_head(sparse)
    event = _event()
    before = sparse.predict(event["features"])
    sparse_receipt = sparse.commit_feedback(event, visible_at=3)
    dense_receipt = dense.commit_feedback(event, visible_at=3)
    after_sparse = sparse.predict(event["features"])
    after_dense = dense.predict(event["features"])
    assert sparse_receipt["status"] == dense_receipt["status"] == "committed"
    assert sparse_receipt["loss"] == pytest.approx(dense_receipt["loss"], abs=1e-14)
    assert sparse_receipt["gradient_norm_after_clip"] <= exp.GRADIENT_NORM_CAP
    assert sparse_receipt["active_coefficient_count"] <= 24
    assert sparse_receipt["touched_parameter_count"] <= 25
    assert (
        sparse_receipt["coefficient_write_bytes"] == 8 * sparse_receipt["touched_parameter_count"]
    )
    assert np.max(np.abs(sparse.state_vector - dense.state_vector)) <= exp.PARITY_TOLERANCE
    assert abs(after_sparse["probability"] - after_dense["probability"]) <= exp.PARITY_TOLERANCE
    assert before["probability"] != after_sparse["probability"]


def test_scenario_kan_7425_04_finite_difference_gradient_and_nonfinite_rejection() -> None:
    """SCENARIO-KAN-7425-04: analytic loss gradients match centered differences."""

    head = exp.SplineEnergyHead.from_training(_training_fixture(), seed=31)
    x = np.asarray([0.3, 2.0, 0.2, 0.65, 1.0, 1.8], dtype=np.float64)
    loss, gradient, _ = head.loss_and_gradient(x, 1)
    assert math.isfinite(loss)
    numeric = exp.finite_difference_gradient(head, x, 1, epsilon=1e-6)
    assert np.max(np.abs(gradient - numeric)) < 2e-8
    for bad in (math.nan, math.inf, -math.inf):
        changed = x.copy()
        changed[2] = bad
        with pytest.raises(ValueError, match="finite"):
            head.predict(changed)
    with pytest.raises(ValueError, match="finite"):
        exp.fit_quantile_knots(np.asarray([[0.0] * 6, [math.nan] * 6]))
    with pytest.raises(ValueError, match="binary"):
        head.loss_and_gradient(x, 2)


def test_scenario_kan_7425_05_feedback_identity_timing_duplicates_and_revocation() -> None:
    """SCENARIO-KAN-7425-05: inadmissible and revoked events cannot alter weights."""

    head = exp.SplineEnergyHead.from_training(_training_fixture(), seed=41)
    event = _event()
    initial = head.state_vector.copy()
    assert head.commit_feedback(event, visible_at=2)["status"] == "not_revealed"
    assert np.array_equal(initial, head.state_vector)
    assert head.commit_feedback(event, visible_at=3)["status"] == "committed"
    committed = head.state_vector.copy()
    assert head.commit_feedback(event, visible_at=4)["status"] == "duplicate"
    assert np.array_equal(committed, head.state_vector)
    changed_identity = {**event, "label": 0}
    assert head.commit_feedback(changed_identity, visible_at=4)["status"] == "identity_conflict"
    assert np.array_equal(committed, head.state_vector)
    revoked = head.revoke_feedback("feedback-1")
    assert revoked["status"] == "revoked"
    assert np.array_equal(initial, head.state_vector)
    assert head.commit_feedback(event, visible_at=5)["status"] == "revoked"
    assert head.revoke_feedback("unknown")["status"] == "unknown_event"
    assert np.array_equal(initial, head.state_vector)

    for malformed in (
        {**event, "source_version": ""},
        {**event, "event_id": ""},
        {**event, "label": None},
        {**event, "reveal_time": -1},
    ):
        with pytest.raises(ValueError):
            exp.SplineEnergyHead.from_training(_training_fixture()).commit_feedback(
                malformed, visible_at=3
            )


def test_scenario_kan_7425_06_checkpoint_cold_restart_and_interrupted_replace(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7425-06: atomic checkpoints survive cold and interrupted writes."""

    head = exp.SplineEnergyHead.from_training(_training_fixture(), seed=53)
    head.commit_feedback(_event(), visible_at=3)
    path = tmp_path / "numeric-checkpoint.json"
    manifest = head.save_checkpoint(path)
    assert manifest["byte_size"] <= exp.MAX_CHECKPOINT_BYTES
    restored = exp.SplineEnergyHead.load_checkpoint(path)
    assert restored.checkpoint_hash == head.checkpoint_hash
    assert restored.predict(_event()["features"])["probability"] == pytest.approx(
        head.predict(_event()["features"])["probability"], abs=1e-14
    )

    restored.commit_feedback(_event("feedback-2", label=0, reveal_time=4), visible_at=4)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        restored.save_checkpoint(path, interrupt_before_replace=True)
    still_committed = exp.SplineEnergyHead.load_checkpoint(path)
    assert still_committed.checkpoint_hash == head.checkpoint_hash
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["coefficients"][0][0] = math.nan
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        exp.SplineEnergyHead.load_checkpoint(path)


def test_scenario_kan_7425_07_artifact_reduction_and_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7425-07: readiness is recomputed from raw fixture evidence."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    assert exp.validate_artifact(artifact) == []
    reduced = exp.independent_reduce(artifact)
    assert reduced == {"spline_prototype_ready_score": 1, "promotion_score": 0}
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert all(row["data_role"] == "analytic_fixture" for row in artifact["rows"])

    for field, replacement, expected in (
        ("spline_prototype_ready_score", 0, "spline_prototype_ready_score_mismatch"),
        ("model_invoked", True, "model_declaration_mismatch"),
        ("reproducibility_checksum", "sha256:changed", "reproducibility_checksum_mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert expected in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["parity_rows"][0]["parameter_linf_gap"] = 1e-5
    assert "parity_tolerance_failed:0" in exp.validate_artifact(changed)

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path) == []
    path.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(path) == ["artifact_unreadable_or_not_object"]


def test_scenario_kan_7425_08_preconditions_manifest_and_blocked_contract() -> None:
    """SCENARIO-KAN-7425-08: sources and scoped validation are frozen before execution."""

    checks, hashes = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert exp.SPEC_PATH.as_posix() in hashes
    assert exp.V651_MANIFEST.test_paths == (exp.TEST_PATH.as_posix(),)
    assert exp.V651_MANIFEST.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert "tests/python" not in exp.V651_MANIFEST.test_paths

    failed = deepcopy(checks[0])
    failed["passed"] = False
    failed["observed"] = None
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["spline_prototype_ready_score"] == 0
    assert exp.validate_artifact(blocked) == []


def test_cli_reader_modes_and_defensive_shapes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-KAN-7425: thin CLI readers reject bad dates, shapes, and changed artifacts."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path)]) == 0
    assert '"errors": []' in capsys.readouterr().out
    assert exp.main(["--date", exp.RUN_DATE, "--independent-reduce", str(path)]) == 0
    assert '"spline_prototype_ready_score": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--date"):
        exp.run_experiment(ROOT, "20260101", output_path=tmp_path / "bad.json")
    with pytest.raises(ValueError, match="six columns"):
        exp.fit_quantile_knots(np.zeros((3, 5), dtype=np.float64))
    head = exp.SplineEnergyHead.from_training(_training_fixture())
    with pytest.raises(ValueError, match="six features"):
        head.predict([0.0] * 5)


def test_req_kan_7425_defensive_numeric_and_checkpoint_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KAN-7425: malformed numeric state and checkpoint bytes fail closed."""

    repaired = exp._repair_breakpoints(  # noqa: SLF001 - required defensive control.
        np.asarray([1e10, 1e10, 1e10, 1e10, 1e10, 1e10 + 2e-6])
    )
    assert np.all(np.diff(repaired) > 0.0)
    knots = exp.fit_quantile_knots(_training_fixture())
    invalid = knots[0].copy()
    invalid[exp.DEGREE] = invalid[-exp.DEGREE - 1]
    with pytest.raises(ValueError, match="nondegenerate"):
        exp.cubic_basis(0.0, invalid)
    with pytest.raises(ValueError, match="finite"):
        exp.cubic_basis(math.nan, knots[0])
    coefficients = np.zeros((exp.INPUT_COUNT, exp.COEFFICIENTS_PER_INPUT))
    with pytest.raises(ValueError, match="bias"):
        exp.SplineEnergyHead(knots, coefficients, math.nan)
    with pytest.raises(ValueError, match="initial bias"):
        exp.SplineEnergyHead(knots, coefficients, 0.0, initial_bias=math.nan)
    head = exp.SplineEnergyHead(knots, coefficients, 0.0)
    head.commit_feedback(_event("first"), visible_at=3)
    head.commit_feedback(_event("second", label=0, reveal_time=4), visible_at=4)
    assert head.revoke_feedback("first")["status"] == "revoked"
    assert head.update_count == 1
    with pytest.raises(ValueError, match="epsilon"):
        exp.finite_difference_gradient(head, _event()["features"], 1, epsilon=0.0)

    checkpoint = tmp_path / "checkpoint.json"
    head.save_checkpoint(checkpoint)
    original_limit = exp.MAX_CHECKPOINT_BYTES
    monkeypatch.setattr(exp, "MAX_CHECKPOINT_BYTES", 1)
    with pytest.raises(ValueError, match="64 KiB"):
        head.save_checkpoint(tmp_path / "too-large.json")
    monkeypatch.setattr(exp, "MAX_CHECKPOINT_BYTES", original_limit)
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="readable JSON"):
        exp.SplineEnergyHead.load_checkpoint(missing)
    bad_schema = tmp_path / "bad-schema.json"
    bad_schema.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        exp.SplineEnergyHead.load_checkpoint(bad_schema)
    changed = json.loads(checkpoint.read_text(encoding="utf-8"))
    changed["bias"] += 1.0
    checkpoint.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        exp.SplineEnergyHead.load_checkpoint(checkpoint)

    head.save_checkpoint(checkpoint)
    changed = json.loads(checkpoint.read_text(encoding="utf-8"))
    changed["events"].append(_event("uncommitted"))
    payload = {key: value for key, value in changed.items() if key != "checkpoint_sha256"}
    changed["checkpoint_sha256"] = exp.canonical_hash(payload)
    checkpoint.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="state reconstruction"):
        exp.SplineEnergyHead.load_checkpoint(checkpoint)


def test_scenario_kan_7425_03_dense_reference_rejects_repeat_and_early_events() -> None:
    """SCENARIO-KAN-7425-03: the dense control uses the same admission boundary."""

    head = exp.SplineEnergyHead.from_training(_training_fixture())
    dense = exp.DenseSplineReference.from_head(head)
    event = _event()
    assert dense.commit_feedback(event, visible_at=2)["status"] == "not_revealed"
    assert dense.commit_feedback(event, visible_at=3)["status"] == "committed"
    assert dense.commit_feedback(event, visible_at=4)["status"] == "duplicate"
    assert dense.commit_feedback({**event, "label": 0}, visible_at=4)["status"] == (
        "identity_conflict"
    )


def test_scenario_kan_7425_07_artifact_identity_and_reader_mutations(tmp_path: Path) -> None:
    """SCENARIO-KAN-7425-07: every terminal identity declaration fails closed."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    mutations = (
        ("schema", "changed", "schema_mismatch"),
        ("experiment_id", "changed", "experiment_id_mismatch"),
        ("milestone", "changed", "run_identity_mismatch"),
        ("invocation_counts", {}, "invocation_counts_mismatch"),
        ("inference_substrate_class", "gpu", "inference_substrate_class_mismatch"),
        ("execution_venue", "external", "execution_venue_mismatch"),
        ("promotion_score", 1, "promotion_score_mismatch"),
        ("verdict_class", "invented", "verdict_class_invalid"),
        ("verifier_is_oracle", False, "verifier_oracle_declaration_mismatch"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert expected in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "blocked"
    assert "blocked_verdict_prefix_invalid" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "unfinished"
    assert "complete_verdict_prefix_invalid" in exp.validate_artifact(changed)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}  # noqa: SLF001 - strict reader control.


def test_req_kan_7425_main_dispatches_public_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-7425: the public CLI dispatches the reusable runner once."""

    called: list[tuple[Path, str, Path]] = []

    def fake_run(root: Path, run_date: str, *, output_path: Path) -> dict[str, object]:
        called.append((root, run_date, output_path))
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--date", exp.RUN_DATE, "--output", "result.json"]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, Path("result.json"))]
