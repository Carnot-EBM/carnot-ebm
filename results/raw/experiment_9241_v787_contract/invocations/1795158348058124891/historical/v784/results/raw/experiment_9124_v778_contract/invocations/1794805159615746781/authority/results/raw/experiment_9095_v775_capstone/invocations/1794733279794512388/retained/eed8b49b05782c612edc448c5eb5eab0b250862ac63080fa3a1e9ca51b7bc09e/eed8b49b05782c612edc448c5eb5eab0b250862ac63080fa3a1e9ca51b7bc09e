"""REQ-REPORT-9094-V775; SCENARIO-REPORT-9094-V775-NUMERICS/PROVENANCE."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from carnot.reporting import susceptibility_9094 as d
from carnot.reporting import susceptibility_runner_9094 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_primary_equations_and_derivatives() -> None:
    """REQ-REPORT-9094-V775-NUMERICS: signs and SHI factors need independent checks."""
    w, fields, direction, force = d.instance(775094)
    assert w.shape == (4, 2)
    np.testing.assert_array_equal(fields, 1 + np.r_[abs(w).sum(1), abs(w).sum(0)])
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert np.linalg.norm(force[4:]) == pytest.approx(1)
    z = direction * 0.1
    for alpha in d.ALPHAS:
        for strength in d.STRENGTHS:
            gradient, hessian = d.derivatives(z, w, fields, alpha, strength)
            fd_gradient, fd_hessian, fd_mixed = d.finite_differences(z, w, fields, alpha, strength)
            np.testing.assert_allclose(gradient, fd_gradient, atol=1e-6, rtol=0)
            np.testing.assert_allclose(hessian, fd_hessian, atol=1e-6, rtol=0)
            np.testing.assert_allclose(hessian[:4, 4:], fd_mixed, atol=1e-6, rtol=0)
            u, v = z[:4, None], z[None, 4:]
            expected = -w * (np.sin(u) * np.sin(v) - (2 * alpha - 1) * np.cos(u) * np.cos(v))
            np.testing.assert_allclose(hessian[:4, 4:], expected, atol=1e-15)
    expected_energy = -np.sum(w) - fields.sum() - 3 * 0.1
    assert d.energy(np.zeros(6), w, fields, 0.5, 0.1) == pytest.approx(expected_energy)


def test_binary_identity_susceptibility_and_loss_force() -> None:
    """SCENARIO-REPORT-9094-V775-NUMERICS: zero nudge is separate from response."""
    w, fields, _, q = d.instance(775094)
    for bits in d.configurations():
        z = np.array(bits) * np.pi
        energies = [d.energy(z, w, fields, a, 1) for a in d.ALPHAS]
        assert max(energies) - min(energies) <= 1e-10
        assert np.linalg.norm(d.loss_gradient(z[4:], np.array([-1.0, 1.0]))) <= 1e-14
    responses = []
    for alpha in [0, 0.5, 1]:
        _, h = d.derivatives(np.zeros(6), w, fields, alpha, 1)
        response = d.response(h, q)
        assert response["status"] == "completed"
        dz = np.array(response["displacement"])
        np.testing.assert_allclose(h @ dz, -q, atol=1e-14)
        np.testing.assert_allclose(dz[:4], response["schur_hidden_response"], atol=1e-14)
        responses.append(np.linalg.norm(dz[:4]))
        for beta in d.BETAS:
            equilibrium = d.equilibrium(w, fields, alpha, 1, q, beta)
            assert equilibrium["status"] == "completed"
            assert equilibrium["residual_norm"] <= 1e-12
            assert np.linalg.norm(np.array(equilibrium["phases"]) - beta * dz) / beta < 1e-5
    assert responses[1] <= 1e-10 < min(responses[0], responses[2])
    bad = d.response(np.zeros((6, 6)), q)
    assert bad["status"] == "ill_conditioned" and bad["displacement"] is None
    assert d.equilibrium(w, fields, 0, 1, q, 0.01, max_steps=0)["status"] == "failed"


def test_retained_rows_and_null() -> None:
    """REQ-REPORT-9094-V775: primitive rows determine all headlines and denominators."""
    rows = d.measure_case(775094)
    assert len(rows["binary_energy_rows"]) == 64 * 3
    assert len(rows["stability_rows"]) == 64 * 3
    assert len(rows["derivative_rows"]) == 5 * 3 * 4
    assert len(rows["susceptibility_rows"]) == 5 * 3
    assert len(rows["finite_perturbation_rows"]) == 5 * 3 * 3
    assert all(r["mixed_curvature_norms"][2] <= 1e-10 for r in rows["stability_rows"])
    assert d.summarize([rows])["mechanism_reproduced"]
    broken = deepcopy(rows)
    broken["derivative_rows"][0]["absolute_metric"] = 1
    broken["derivative_rows"][0]["status"] = "failed"
    assert not d.summarize([broken])["mechanism_reproduced"]
    assert d.summarize([])["counts"]["completed"] == 0
    assert len(d.summarize([])["rows"]) == 32


def test_operation_map_and_missing_costs(tmp_path: Path) -> None:
    """REQ-REPORT-9094-V775-PROVENANCE: absent costs cannot erase the operation map."""
    from carnot.reporting import susceptibility_evidence_9094 as e

    evidence = e.authenticate(tmp_path, tmp_path / "raw", tmp_path)
    assert evidence["gate_check_summary"]
    assert any(c["observed"] == "missing_file" for c in evidence["gate_check_summary"])
    mapping = e.operation_map(evidence)
    assert {r["operation"] for r in mapping} == {
        "exact_graph_bfs",
        "small_head_spline_linear",
        "durable_checkpointing",
        "ising_sampling",
        "physical_learning",
        "fixed_point_local_basis_updates",
    }
    assert all(r["cost_s"] is None for r in mapping)
    assert all("GPU1 reserved" in r["hardware_boundary"] for r in mapping)


def test_source_and_producer_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9094-V775-PROVENANCE: actual zero and absent fields differ."""
    from carnot.reporting import susceptibility_evidence_9094 as e

    producer = tmp_path / "producer.py"
    producer.write_text("original")
    atomic_json(
        tmp_path / e.COST,
        dict(
            experiment_id=9093,
            task_id="exp9093-service-cost",
            complete_service_cost_ready_score=0,
            code_config_hashes=[e.ref(producer)],
        ),
    )
    good = e.authenticate(tmp_path, tmp_path / "raw", e.SOURCE_DIR)
    assert all(c["passed"] for c in good["gate_check_summary"] if c["required"])
    assert any(c["observed"] == 0 for c in good["gate_check_summary"])
    assert good["unavailable_cost_inputs"]
    assert good["cost"]["complete_service_qualified"] is False
    producer.write_text("tampered")
    bad = e.authenticate(tmp_path, tmp_path / "changed", e.SOURCE_DIR)
    assert any(c["check"] == "producer_hash" and not c["passed"] for c in bad["gate_check_summary"])
    atomic_json(tmp_path / e.COST, dict(experiment_id=9093))
    missing = e.authenticate(tmp_path, tmp_path / "missing", e.SOURCE_DIR)
    assert any(c["observed"] == "missing_field" for c in missing["gate_check_summary"])
    (tmp_path / e.COST).write_text("invalid json")
    assert e.authenticate(tmp_path, tmp_path / "invalid", e.SOURCE_DIR)["unavailable_cost_inputs"]
    altered = tmp_path / "altered_sources"
    altered.mkdir()
    for name in e.SOURCE_FILES:
        (altered / name).write_text("changed")
    assert not all(
        c["passed"]
        for c in e.authenticate(tmp_path, tmp_path / "bad", altered)["gate_check_summary"]
    )


def test_cli_success_block_tamper_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9094-V775: real subprocesses must use installed path bootstrapping."""
    from carnot.reporting import susceptibility_evidence_9094 as e

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / d.CLI)]

    def call(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv + list(args), cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120
        )

    output = tmp_path / (d.NAME + ".json")
    ran = call("--date", "20261115", "--sources", str(tmp_path), "--private-output", str(output))
    assert ran.returncode == 0, ran.stdout + ran.stderr
    value = json.loads(output.read_bytes())
    assert value["summary"]["counts"]["completed"] == 32
    assert value["summary"]["mechanism_reproduced"]
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["numerical_diagnostic_ready_score"] == 0
    assert call("--cold-replay", str(output)).returncode == 0
    changed = deepcopy(value)
    changed["susceptibility_rows"][0]["absolute_metric"] = 123
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1
    atomic_json(output, value)
    reference = value["raw_shard_hashes"][0]
    path = Path(reference["path"])
    path.chmod(0o600)
    original = path.read_bytes()
    path.write_text("{}")
    assert call("--cold-replay", str(output)).returncode == 1
    path.write_bytes(original)
    blocked = tmp_path / "blocked" / output.name
    assert (
        call(
            "--date",
            "20261115",
            "--source-dir",
            str(tmp_path / "absent"),
            "--private-output",
            str(blocked),
        ).returncode
        == 0
    )
    b = json.loads(blocked.read_bytes())
    assert b["verdict_class"] == "blocked" and b["summary"]["counts"]["completed"] == 0
    assert b["operation_mapping_rows"] and b["gate_check_summary"]
    assert call("--cold-replay", str(blocked)).returncode == 0
    assert call("--date", "wrong").returncode == 2
    assert call("--date", "20261115", "--private-output", str(d.ROOT / output.name)).returncode == 2
    assert call("--cold-replay", str(tmp_path / "missing.json")).returncode == 1
    assert e.SOURCE_DIR.is_dir()


def test_validation_qualification_is_not_a_boolean(tmp_path: Path) -> None:
    """REQ-REPORT-9094-V775-PROVENANCE: exact commands and byte logs are required."""
    from carnot.reporting import susceptibility_evidence_9094 as e

    scope = [dict(name="owned", argv=["actual", "command"])]
    work = dict(validation_scope_manifest=scope)
    assert not runner.qualify(work, {})
    log = tmp_path / "command.log"
    log.write_text("pass")
    receipt = dict(
        name="owned",
        command_argv=scope[0]["argv"],
        exit_code=0,
        passed=True,
        timed_out=False,
        log_path=str(log),
        log_sha256=e.ref(log)["sha256"],
    )
    validation = dict(receipts=[receipt], coverage={})
    with patch.object(runner, "verify_measured", return_value=True):
        assert runner.qualify(work, validation)
        receipt["command_argv"] = ["different"]
        assert not runner.qualify(work, validation)
        receipt["command_argv"] = scope[0]["argv"]
        log.write_text("changed")
        assert not runner.qualify(work, validation)


@pytest.mark.parametrize("valid", [True, False])
def test_production_runner_paths_use_real_publication(tmp_path: Path, valid: bool) -> None:
    """REQ-REPORT-9094-V775-PROVENANCE: owned failure must stop public readiness."""
    from carnot.reporting import susceptibility_evidence_9094 as e
    from carnot.reporting.primary_publication import publish_primary

    private = tmp_path / "private"
    private.mkdir()
    output = tmp_path / "published" / (d.NAME + ".json")

    def commands(root: Path, specs: list, *, log_dir: Path, **kwargs: object) -> list:
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for spec in specs:
            log = log_dir / (spec.name + ".log")
            log.write_text("unit-test command transport; not production qualification")
            receipts.append(
                dict(
                    name=spec.name,
                    command_argv=list(spec.argv),
                    exit_code=0 if valid else 1,
                    passed=valid,
                    timed_out=False,
                    log_path=str(log),
                    log_sha256=e.ref(log)["sha256"],
                )
            )
        if valid and specs[0].name == "owned_unit_child":
            (private / "checks/.coverage").write_text("mock coverage transport")
        return receipts

    def publication(path: Path, value: dict, validator: object) -> dict:
        return publish_primary(output, value, validator)

    with (
        patch.object(runner.tempfile, "mkdtemp", return_value=str(private)),
        patch.object(d, "SEEDS", [775094]),
        patch.object(runner, "run_commands", side_effect=commands),
        patch.object(runner, "measured", return_value={"unit_test": True}),
        patch.object(runner, "verify_measured", return_value=valid),
        patch.object(runner, "publish_primary", side_effect=publication),
    ):
        result = runner.main(["--date", "20261115", "--sources", str(tmp_path)])
        assert result == (0 if valid else 1)
        if valid:
            value = json.loads(output.read_bytes())
            assert value["required_checks_passed"]
            assert value["verdict_class"] == "circular_positive"
            assert runner.replay(output)["passed"]
            value["reproducibility_checksum"] = "wrong"
            atomic_json(output, value)
            assert not runner.replay(output)["passed"]
        else:
            candidate = json.loads((private / "invocation/candidate.json").read_bytes())
            assert candidate["verdict_class"] == "disqualified"
            assert candidate["numerical_diagnostic_ready_score"] == 0
            assert not output.exists()


def test_failed_curvature_and_forced_measurements_remain_visible() -> None:
    """SCENARIO-REPORT-9094-V775-NUMERICS: unstable or nonconverged rows are not erased."""
    w, fields, _, q = d.instance(775094)
    assert d.equilibrium(w, -fields * 100, 0, 0.1, q, 1e-3)["status"] == "failed"
    result = d.response(np.zeros((6, 6)), q)
    with patch.object(d, "response", return_value=result):
        case = d.measure_case(775094)
    assert all(r["status"] == "ill_conditioned" for r in case["susceptibility_rows"])
    assert all(r["status"] == "censored" for r in case["finite_perturbation_rows"])
    assert not d.summarize([case])["mechanism_reproduced"]


def test_available_cost_metadata_does_not_invent_device_cost(tmp_path: Path) -> None:
    """REQ-REPORT-9094-V775-PROVENANCE: even a service receipt supplies no device crossover."""
    from carnot.reporting import susceptibility_evidence_9094 as e

    producer = tmp_path / "producer.py"
    producer.write_text("original")
    atomic_json(
        tmp_path / e.COST,
        dict(
            experiment_id=9093,
            task_id="exp9093-service-cost",
            complete_service_cost_ready_score=1,
            code_config_hashes=[e.ref(producer)],
            observed_latency_summary=[dict(scope="science", charged_s=1)],
        ),
    )
    evidence = e.authenticate(tmp_path, tmp_path / "raw", e.SOURCE_DIR)
    assert evidence["cost"]["observed_latency_summary"][0]["charged_s"] == 1
    assert all(r["cost_s"] is None for r in e.operation_map(evidence))
