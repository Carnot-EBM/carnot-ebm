"""REQ-VERIFY-8085 and REQ-REPORT-8085: supplied labels grant no science credit."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from scipy.spatial.distance import cdist, pdist

from carnot.verify import radial_memory_8085 as k


def fit(seed=0):
    """Public fixture vectors contain a constant column and stable source IDs."""
    x = np.random.default_rng(seed).normal(size=(32, 9))
    x[:, -1] = 2
    return x, [f"fit-{i:03}" for i in range(len(x))]


def feedback(slot=64, role="update", count=64):
    """Original release slots constrain all access to supplied labels."""
    return [
        dict(
            source_id=f"{role}-{slot}-{i:03}",
            x=[i / 10] * 9,
            y=i % 2,
            role=role,
            eligible=True,
            release_slot=slot - count + i,
            observed_slot=slot,
            issued_action="accept" if i % 2 else "reject",
        )
        for i in range(count)
    ]


def test_64_reference_systems():
    """SCENARIO-VERIFY-8085: independently calculate geometry and gradients."""
    for seed in range(64):
        x, ids = fit(seed)
        state = k.initialize(x, ids)
        g = state["geometry"]
        assert g["mean"] == pytest.approx(x.mean(0))
        assert g["std"][-1] == 1
        z = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
        distances = pdist(z)
        assert g["sigma"] == pytest.approx(np.median(distances[distances > 0]))
        centers = np.array([r["x"] for r in state["centers"]])
        expected = np.column_stack(
            (np.ones(len(x)), np.exp(-cdist(z, centers, "sqeuclidean") / (2 * g["sigma"] ** 2)))
        )
        actual = k.design(state, x)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        theta = np.random.default_rng(seed).normal(size=17)
        state["coefficients"] = theta.tolist()
        p = k.predict(state, x)
        assert np.max(np.abs(p - expit(expected @ theta))) <= 1e-10
        assert [k.action(float(v)) for v in p] == [
            k.action(float(v)) for v in expit(expected @ theta)
        ]
        y = np.arange(32) % 2
        value, grad = k.objective(theta, actual, y, 0.01)
        assert value == pytest.approx(
            np.mean(np.logaddexp(0, actual @ theta) - y * (actual @ theta))
            + 0.005 * (theta @ theta)
        )
        for i in range(17):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            numeric = (
                k.objective(plus, actual, y, 0.01)[0] - k.objective(minus, actual, y, 0.01)[0]
            ) / 2e-5
            assert numeric == pytest.approx(grad[i], abs=1e-8)
    zero = k.initialize(np.ones((32, 9)), ids)
    assert zero["geometry"]["sigma"] == 1
    assert np.all(k.design(zero, np.ones((1, 9))) == 1)


def test_validation_and_optimizer_failures(monkeypatch):
    """REQ-VERIFY-8085: invalid numerical operands cannot enter the head."""
    x, ids = fit()
    for bad_x, bad_ids in [
        (x * np.nan, ids),
        (x[:, :8], ids),
        (x[:16], ids[:16]),
        (x, ids[:-1]),
        (x, ["dup"] * 32),
    ]:
        with pytest.raises(ValueError):
            k.initialize(bad_x, bad_ids)
    state = k.initialize(x, ids)
    for bad in [np.full((1, 9), np.inf), np.ones((1, 8))]:
        with pytest.raises(ValueError):
            k.design(state, bad)
    state["coefficients"].pop()
    with pytest.raises(ValueError, match="coefficient"):
        k.predict(state, x)
    with pytest.raises(ValueError):
        k.solve(np.ones((2, 2)), np.array([0, 2]), np.zeros(2))
    with pytest.raises(ValueError):
        k.solve(np.ones((2, 2)), np.array([0, 1]), np.array([np.nan, 0]))
    monkeypatch.setattr(
        k,
        "minimize",
        lambda *a, **kw: type("Result", (), dict(x=np.zeros(2), nit=256, success=False))(),
    )
    with pytest.raises(ValueError, match="optimizer"):
        k.solve(np.ones((2, 2)), np.array([0, 1]), np.zeros(2))


def test_lifecycle_and_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8085: commit and replay bind centers with coefficients."""
    x, ids = fit()
    state = k.initialize(x, ids)
    path = tmp_path / "memory.json"
    k.save(path, state)
    for slot in [64, 128, 192]:
        rows = feedback(slot)
        grown = k.candidate(state, rows, slot, "feedback_grown")
        fixed = k.candidate(state, rows, slot, "fixed_center")
        assert len(grown["coefficients"]) == len(fixed["coefficients"])
        assert grown["centers"][:16] == state["centers"][:16]
        assert len(grown["update_rows"]) == 64
        assert grown["solve"]["gradient_inf"] <= 1e-7
        admission = feedback(slot + 80, "admission", 12)
        committed = k.commit(path, state, grown, admission)
        assert k.load(path) == committed
        assert len(committed["centers"]) <= 28
        state = committed
    assert len(state["opportunities"]) == 3
    with pytest.raises(ValueError, match="opportunity"):
        k.candidate(state, feedback(192), 192, "feedback_grown")
    fresh = k.initialize(x, ids)
    for changes in [
        dict(y=None),
        dict(role="retention"),
        dict(release_slot=1000),
        dict(eligible=False),
    ]:
        bad = feedback()
        bad[0].update(changes)
        with pytest.raises(ValueError, match="feedback"):
            k.candidate(fresh, bad, 64, "feedback_grown")
    with pytest.raises(ValueError, match="duplicate"):
        k.candidate(fresh, [*feedback(), feedback()[0]], 64, "feedback_grown")
    with pytest.raises(ValueError, match="feedback"):
        k.candidate(fresh, [], 64, "feedback_grown")
    with pytest.raises(ValueError, match="arm"):
        k.candidate(fresh, feedback(), 64, "bad")
    overflow = deepcopy(fresh)
    overflow["centers"] *= 2
    with pytest.raises(ValueError, match="dictionary"):
        k.candidate(overflow, feedback(), 64, "feedback_grown")
    proposal = k.candidate(fresh, feedback(), 64, "feedback_grown")
    k.save(path, fresh)
    admissions = feedback(144, "admission", 12)
    with pytest.raises(ValueError, match="admission"):
        k.commit(path, fresh, proposal, admissions[:11])
    with pytest.raises(ValueError, match="admission"):
        k.commit(path, fresh, proposal, feedback(64, "admission", 12))
    stale = deepcopy(fresh)
    stale["version"] += 1
    with pytest.raises(ValueError, match="stale"):
        k.commit(path, stale, proposal, admissions)
    original_save = k.save
    monkeypatch.setattr(k, "save", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        k.commit(path, fresh, proposal, admissions)
    assert k.load(path) == fresh
    monkeypatch.setattr(k, "save", original_save)
    monkeypatch.setattr(k.fresh, "guard", lambda *args: ([], None))
    rejected = k.commit(path, fresh, proposal, admissions)
    assert rejected["centers"] == fresh["centers"]
    assert len(rejected["consumed"]) == 12
    tampered = json.loads(path.read_text())
    tampered["state"]["centers"][0]["x"][0] += 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash"):
        k.load(path)


def test_real_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8085: real children run outside the checkout."""
    import os
    import subprocess
    import sys
    from carnot import experiment_8085_v700_radial_memory_kernel as producer

    cli = producer.ROOT / producer.CLI
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8085_COVERAGE_CONFIG"):
        config = Path(env["CARNOT_8085_COVERAGE_CONFIG"])
        prefix += [
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "--data-file=" + str(config.parent / ".coverage"),
        ]
    for condition in ["success", "blocked", "mutation"]:
        output = tmp_path / condition / (producer.NAME + ".json")
        argv = [*prefix, str(cli), "--date", "20261004", "--fixture-output", str(output)]
        if condition == "blocked":
            argv += ["--root", str(tmp_path / "missing")]
        if condition == "mutation":
            argv += ["--mutate"]
        result = subprocess.run(
            argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_text())
        assert (
            value["verdict_class"]
            == {"success": "circular_positive", "blocked": "blocked", "mutation": "disqualified"}[
                condition
            ]
        )
        assert (
            value["kernel_ready_score"] == 0
        )  # This private route deliberately has no owned validation credit.
        replay = subprocess.run(
            [*prefix, str(cli), "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert replay.returncode == 0, replay.stdout + replay.stderr
        if condition == "success":
            raw = Path(value["terminal_validation_sidecar_path"]).parent
            evidence = json.loads((raw / "evidence.json").read_text())
            evidence["systems"][0]["state"]["centers"][0]["x"][0] += 1
            (raw / "evidence.json").write_text(json.dumps(evidence))
            replay = subprocess.run(
                [*prefix, str(cli), "--cold-replay", str(output)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert replay.returncode == 1
    assert producer.main(["--fixture-output", str(output)]) == 1
    assert producer.replay(tmp_path / "absent.json") is False


def test_additional_lifecycle_safety(tmp_path, monkeypatch):
    """REQ-VERIFY-8085: insufficient errors, consumed labels and changed proposals fail."""
    x, ids = fit()
    state = k.initialize(x, ids)
    path = tmp_path / "state.json"
    k.save(path, state)
    correct = feedback()
    for row in correct:
        row["issued_action"] = "reject" if row["y"] else "accept"
    with pytest.raises(ValueError, match="feedback_error_support"):
        k.candidate(state, correct, 64, "feedback_grown")
    proposal = k.candidate(state, feedback(), 64, "feedback_grown")
    bad = deepcopy(proposal)
    bad["coefficients"][0] += 1
    with pytest.raises(ValueError, match="proposal_hash"):
        k.commit(path, state, bad, feedback(144, "admission", 12))
    bad_state = deepcopy(state)
    bad_state["consumed"] = [feedback(144, "admission", 12)[0]["source_id"]]
    k.save(path, bad_state)
    consumed_proposal = k.candidate(bad_state, feedback(), 64, "feedback_grown")
    with pytest.raises(ValueError, match="admission"):
        k.commit(path, bad_state, consumed_proposal, feedback(144, "admission", 12))
    times = iter([0.0, 601.0])
    monkeypatch.setattr(k.time, "monotonic", lambda: next(times))

    def timeout_solver(*args, **kw):
        kw["callback"](np.zeros(2))

    monkeypatch.setattr(k, "minimize", timeout_solver)
    with pytest.raises(TimeoutError):
        k.solve(np.ones((2, 2)), np.array([0.0, 1.0]), np.zeros(2))


def test_parent_failure_and_owned_validation_routes(tmp_path, monkeypatch):
    """REQ-REPORT-8085: owned failures cannot claim readiness or hide normal exit."""
    from carnot import experiment_8085_v700_radial_memory_kernel as p

    original_manifest, original_run = p.manifest, p.run_check

    def short_manifest(private):
        specs = original_manifest(private)
        # This supplied failed report exercises disqualification inside private tests.
        p.atomic_json(
            private / "coverage.json",
            {
                "files": {
                    str(p.ROOT / p.OWNED[0]): {"summary": {"num_statements": 1, "covered_lines": 0}}
                }
            },
        )
        return [next(s for s in specs if s["name"] == "ruff_check")]

    monkeypatch.setattr(p, "manifest", short_manifest)

    def failed_child(root, spec, private, durable):
        if spec["name"] == "measurement_normal_exit":
            log = durable / "injected-failure.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("supplied private child failure\n")
            return dict(
                spec, passed=False, actual_exit=1, log_path=str(log), log_sha256=p.sha256_file(log)
            )
        return original_run(root, spec, private, durable)

    monkeypatch.setattr(p, "run_check", failed_child)
    output = tmp_path / "failed" / (p.NAME + ".json")
    assert p.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert (
        p.main(
            [
                "--root",
                str(tmp_path / "missing"),
                "--worker-output",
                str(tmp_path / "worker" / "work.json"),
            ]
        )
        == 0
    )
    work = json.loads((tmp_path / "worker" / "work.json").read_text())
    malformed = tmp_path / "malformed" / "results"
    malformed.mkdir(parents=True)
    (malformed / "experiment_8075_v699_constraint_projection_kernel.json").write_text("{}")
    checked = p.prerequisites(malformed.parent, tmp_path / "snapshots")
    assert any(r["check"] == "terminal_binding" for r in checked["failures"])
    assert p.build(work, tmp_path / "worker", [], {}, False)["verdict_class"] == "blocked"
    assert k.action(0.1) == k.action(0.5) == "escalate"


def test_primitive_controls_are_retained(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8085: edge outcomes have reconstructable primitive rows."""
    from carnot import experiment_8085_v700_radial_memory_kernel as p

    x, ids = fit()
    state = k.initialize(x, ids)
    controls = p.controls(state, tmp_path)
    required = {
        "zero_distances",
        "constant_columns",
        "nonfinite_input",
        "duplicate_source_ids",
        "dictionary_overflow",
        "missing_feedback",
        "stale_coefficients",
        "interrupted_commit",
        "dictionary_tamper",
        "action_boundary",
    }
    assert {r["condition"] for r in controls} == required
    assert all(r["passed"] for r in controls)
    grown = k.candidate(state, feedback(), 64, "feedback_grown")
    assert all(
        r["identity"] == p.canonical_hash([r["source_id"], r["x"]]) for r in grown["centers"]
    )
    fixed = k.candidate(state, feedback(), 64, "fixed_center")
    assert all(r["feedback_origin"] == "fit_public" for r in fixed["centers"])
    monkeypatch.setattr(k, "matrix", lambda x: np.asarray(x, dtype=float))
    assert any(not r["passed"] for r in p.controls(state, tmp_path / "broken-kernel"))


def test_historical_basis_remains_separate():
    """REQ-VERIFY-8085: historical projection uses the qualified additive basis."""
    from carnot import experiment_8085_v700_radial_memory_kernel as p

    x, _ = fit()
    result = p.historical(x, np.arange(len(x)) % 2)
    assert result["basis_name"] == "historical_conditioned_energy"
    assert len(result["initial"]) == 111
    assert len(result["constraints"]) == 32
    assert result["projection"]["feasible"]
    assert result["gradient_steps"] == 4
