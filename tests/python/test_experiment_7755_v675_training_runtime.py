"""Fixture qualification for REQ-VERIFY-7755 and REQ-REPORT-7755."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carnot.verify import source_set_energy
from carnot.verify import source_alignment
from carnot.verify import training_runtime as runtime
from carnot import experiment_7755_v675_training_runtime as experiment


def examples() -> list[dict]:
    """Keep labels in the private fixture, outside public feature extraction."""
    raw = [
        (b"Alpha is 12.", b"Alpha is 12.", 0, [1]),
        (b"Alpha is 12.", b"Alpha is 13.", 1, [0]),
        (b"Beta is 30.", b"Beta is 30.", 0, [1]),
        (b"Beta is 30.", b"Beta is 31.", 1, [0]),
    ]
    return [
        {
            "view_a": source_set_energy.prepare(source, answer),
            "view_b": source_set_energy.prepare(source + b" " + source, answer),
            "label": label,
            "known": known,
        }
        for source, answer, label, known in raw
    ]


def test_normalization_prior_masks_and_controls() -> None:
    """SCENARIO-VERIFY-7755-LOSS: finite states and masked padding."""
    rows = examples()
    rows[0]["view_a"] = source_set_energy.prepare(
        b"Alpha is 12. Alpha is 12.", b"Alpha is 12. Beta is 30."
    )
    rows[0]["view_b"] = source_set_energy.prepare(b"Alpha is 12.", b"Alpha is 12. Beta is 30.")
    rows[0]["known"] = [1, -1]
    batch = runtime.prepare_batch(rows)
    assert batch["a"]["x"].shape[-1] == source_set_energy.FEATURE_DIM
    assert np.allclose(np.asarray(batch["a"]["prior"])[0, :3], [0.25, 0.25, 0.5])
    assert np.asarray(batch["a"]["unit_mask"])[1, 1] == 0
    assert np.asarray(batch["a"]["place_mask"])[1, 2] == 0
    params = runtime.init_params("energy_local", 67501)
    p = runtime.predict(params, batch["a"], "energy_local")
    assert math.isclose(
        float(p[0]),
        1
        - source_set_energy.enumerate_response(
            rows[0]["view_a"], runtime.energy_parameters(params)
        ),
        abs_tol=1e-6,
    )
    for arm in ("logistic_local", "mlp_local"):
        ctl = runtime.init_params(arm, 67501)
        support = runtime.sentence_support(ctl, batch["a"], arm)
        assert support.shape == (4, 2)
        assert math.isclose(
            float(runtime.predict(ctl, batch["a"], arm)[0]),
            1 - float(np.prod(np.asarray(support[0]))),
            abs_tol=1e-6,
        )


def test_loss_gradient_augmentation_and_duals() -> None:
    """SCENARIO-VERIFY-7755-LOSS: known labels, paired rule and duals."""
    batch = runtime.prepare_batch(examples())
    params = runtime.init_params("energy_local", 67501)
    base = runtime.loss(params, batch, "energy_local", "canonical", (0.0, 0.0))
    response = runtime.loss(params, batch, "response_set", "canonical", (0.0, 0.0))
    assert float(base) > float(response)
    grad = jax.grad(lambda p: runtime.loss(p, batch, "energy_local", "canonical", (0.0, 0.0)))(
        params
    )
    eps = 1e-4
    plus = jax.tree_util.tree_map(lambda x: x.copy(), params)
    minus = jax.tree_util.tree_map(lambda x: x.copy(), params)
    plus["b"] = plus["b"].at[0].add(eps)
    minus["b"] = minus["b"].at[0].add(-eps)
    finite = (
        float(runtime.loss(plus, batch, "energy_local", "canonical", (0.0, 0.0)))
        - float(runtime.loss(minus, batch, "energy_local", "canonical", (0.0, 0.0)))
    ) / (2 * eps)
    assert abs(finite - float(grad["b"][0])) <= max(1e-6, abs(finite) * 1e-4)
    assert runtime.gradient_error(params, batch, "energy_local") <= max(1e-6, abs(finite) * 1e-4)
    a = np.asarray(runtime.predict(params, batch["a"], "energy_local"))
    b = np.asarray(runtime.predict(params, batch["b"], "energy_local"))
    assert np.allclose(
        np.asarray(runtime.deployed_risk(params, batch, "energy_local", True)), (a + b) / 2
    )
    assert float(runtime.loss(params, batch, "energy_local", "ordinary", (0.0, 0.0))) > 0
    j, ce = runtime.constraints(params, batch, "energy_local")
    assert float(j) >= 0 and float(ce) >= 0
    assert runtime.dual_step((0.0, 0.0), (1.0, 2.0)) == (0.0099, 0.013000000000000001)
    assert runtime.dual_step((0.0, 0.0), (0.0, 0.0)) == (0.0, 0.0)
    same = runtime.prepare_batch([{**row, "view_b": row["view_a"]} for row in examples()])
    assert abs(float(runtime.constraints(params, same, "energy_local")[0])) < 1e-12


def test_fit_calibrate_reload_decide(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7755-DEPLOYMENT: real fit and cold typed decision."""
    batch = runtime.prepare_batch(examples())
    result = runtime.fit("energy_local", batch, batch, 67501, 0.05, 40, "ordinary")
    assert result["curve"][-1]["loss"] < result["curve"][0]["loss"]
    assert result["initial_hash"] != result["final_hash"]
    result["temperature"] = runtime.calibrate(result["params"], batch, "energy_local", True)
    path = tmp_path / "head.json"
    runtime.save(path, result)
    loaded = runtime.load(path)
    before = runtime.decide(result, batch, 0, True)
    after = runtime.decide(loaded, batch, 0, True)
    assert before == after
    assert after["action"] in ("accept", "reject", "escalate")
    assert runtime.action(0.05) == "escalate"
    assert runtime.action(0.75) == "escalate"
    assert result["tune_nll"] == pytest.approx(
        runtime.response_nll(result["params"], batch, "energy_local", True)
    )


def test_extreme_and_invalid_inputs() -> None:
    """REQ-VERIFY-7755: stable logits and explicit divergence errors."""
    batch = runtime.prepare_batch(examples())
    params = runtime.init_params("energy_local", 67501)
    params["b"] = params["b"].at[0].set(1000.0)
    assert np.isfinite(np.asarray(runtime.predict(params, batch["a"], "energy_local"))).all()
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.fit(
            "energy_local",
            batch,
            batch,
            67501,
            0.05,
            1,
            "canonical",
            initial={"w": params["w"] * np.nan, "b": params["b"]},
        )
    with pytest.raises(ValueError, match="budget"):
        runtime.fit("energy_local", batch, batch, 67501, 0.05, 41, "canonical")


def test_child_basetemp_and_two_view_control(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7755-TERMINAL: real child setup and shared head."""
    nested = tmp_path / "nested" / "private" / "pytest"
    nested.parent.mkdir(parents=True)
    test_file = tmp_path / "test_child.py"
    test_file.write_text("def test_child():\n    assert True\n")
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={nested}",
            str(test_file),
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    batch = runtime.prepare_batch(examples())
    control = runtime.init_params("logistic_local", 67501)
    assert control["w"].shape == (132, 2)
    pooled = np.asarray(batch["a"]["x"])[0, 0].T @ np.asarray(batch["a"]["prior"])[0]
    expected = (
        np.asarray(source_alignment.pair_features(b"Alpha is 12.", [b"Alpha is 12."]))
        + np.asarray(source_alignment.pair_features(b"", [b"Alpha is 12."]))
    ) / 2
    assert np.allclose(pooled, expected)
    mlp = runtime.init_params("mlp_local", 67501)
    assert mlp["w1"].shape == (132, 16)
    assert runtime.gradient_error(mlp, batch, "mlp_local") < 1e-6
    assert runtime.parameter_count(mlp, 2) <= 4096
    assert float(runtime.loss(control, batch, "logistic_local", "constrained", (0.5, 0.5))) > 0


def test_temperature_is_applied_after_aggregation() -> None:
    """SCENARIO-REPORT-7755-DEPLOYMENT: one response-level scale."""
    assert runtime.temperature_risk(0.1, 2.0) == pytest.approx(0.25)
    assert runtime.temperature_risk(0.5, 1.0) == 0.5
    assert runtime.temperature_risk(0.0, 0.25) == 0.0
    assert runtime.temperature_risk(1.0, 4.0) == 1.0
    assert json.loads(json.dumps(runtime.temperature_grid())) == runtime.temperature_grid()


def test_private_fixture_runner_and_cold_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7755-TERMINAL: raw heads survive fresh reduction."""
    raw = tmp_path / "raw"
    result = experiment.run_fixtures(
        raw, seeds=(67501,), rates=(0.05,), epochs=4, arms=("energy_local", "logistic_local")
    )
    assert result["trial_count"] == 2
    assert all(row["initial_hash"] != row["final_hash"] for row in result["trial_rows"])
    assert all(row["final_loss"] < row["initial_loss"] for row in result["trial_rows"])
    reduced = experiment.cold_reduce(raw)
    assert reduced["trial_count"] == result["trial_count"]
    assert reduced["decision_count"] == 8
    rows = (raw / "decisions.jsonl").read_text().splitlines()
    changed = json.loads(rows[0])
    changed["action"] = "tampered"
    rows[0] = json.dumps(changed)
    (raw / "decisions.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="decision mismatch"):
        experiment.cold_reduce(raw)


def test_budget_and_view_errors_are_explicit() -> None:
    """REQ-VERIFY-7755: invalid inputs cannot silently enter a fit."""
    rows = examples()
    with pytest.raises(ValueError, match="empty fixture"):
        runtime.prepare_batch([])
    bad = [{**rows[0], "view_a": source_set_energy.prepare(b"", b"")}]
    with pytest.raises(ValueError, match="abstained fixture"):
        runtime.prepare_batch(bad)
    mismatch = [
        {
            **rows[0],
            "view_b": source_set_energy.prepare(b"Alpha is 12.", b"Alpha is 12. Beta is 30."),
        }
    ]
    with pytest.raises(ValueError, match="view sentence mismatch"):
        runtime.prepare_batch(mismatch)
    with pytest.raises(ValueError, match="unregistered"):
        runtime.init_params("unknown", 67501)
    batch = runtime.prepare_batch(rows)
    params = runtime.init_params("energy_local", 67501)
    with pytest.raises(ValueError, match="unknown arm"):
        runtime.sentence_support(params, batch["a"], "unknown")
    with pytest.raises(ValueError, match="unknown training mode"):
        runtime.loss(params, batch, "energy_local", "unknown", (0.0, 0.0))
    with pytest.raises(ValueError, match="parameter budget"):
        runtime.fit(
            "energy_local",
            batch,
            batch,
            67501,
            0.05,
            1,
            "canonical",
            initial={**params, "extra": jnp.zeros(4096)},
        )
    constrained = runtime.fit("energy_local", batch, batch, 67501, 0.05, 2, "constrained")
    assert len(constrained["curve"]) == 2
    assert all(0 <= value <= 10 for value in constrained["duals"])
