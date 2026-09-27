"""V676 fixture checks for REQ-VERIFY-7769 and its three scenarios."""

from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.verify import training_runtime as runtime
from carnot.verify import training_qualification as qualification
from carnot import experiment_7769_v676_training_qualification as experiment


@pytest.fixture(scope="module")
def qualified_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the private V676 fixture once for both online and cold-replay tests."""
    private = tmp_path_factory.mktemp("exp7769-qualified")
    return experiment.run_fixture(private / "raw", "20260927")


@pytest.mark.parametrize("arm", qualification.ARMS)
def test_all_nine_arms_fit_and_reload(tmp_path: Path, arm: str) -> None:
    """SCENARIO-VERIFY-7769-NUMERICAL: every arm updates a bounded head."""
    records = qualification.fixture_records()
    names = [f"predicate_{i}" for i in range(16)]
    assert len(qualification.ARMS) == 9
    batch, excluded = qualification.make_batch(records, arm, names)
    assert excluded == []
    head = qualification.fit_arm(arm, batch, epochs=2)
    assert head["initial_hash"] != head["final_hash"]
    assert head["parameter_count"] <= 4096
    assert math.isfinite(head["gradient_error"])
    assert head["gradient_error"] < 1e-4
    assert all(math.isfinite(point["loss"]) for point in head["curve"])
    assert len(head["static_predicates"]) == (16 if arm == "complete_static_constrained_set" else 0)
    assert len(head["static_coefficients"]) == len(head["static_predicates"])
    assert np.all(
        np.isfinite(
            np.asarray(
                runtime.deployed_risk(head["params"], batch, head["head_arm"], head["paired"])
            )
        )
    )
    target = tmp_path / f"{arm}.json"
    runtime.save(target, head)
    loaded = runtime.load(target)
    assert qualification.NaturalHeadAdapter(loaded).decide(batch, 0) == (
        qualification.NaturalHeadAdapter(head).decide(batch, 0)
    )


def test_masking_budget_and_aggregate_order() -> None:
    """SCENARIO-VERIFY-7769-NUMERICAL: abstentions and aggregation order stay visible."""
    rows = qualification.fixture_records()
    rows.append({"id": "empty", "source": "", "answer": "", "label": None, "known": []})
    rows.append(
        {
            "id": "over_budget",
            "source": "A.",
            "answer": "A. " * 17,
            "label": None,
            "known": [],
        }
    )
    batch, excluded = qualification.make_batch(rows, "augmented_set", [])
    assert [item["id"] for item in excluded] == ["empty", "over_budget"]
    assert batch["label"].shape[0] == len(rows) - 2
    assert float(batch["a"]["unit_mask"][0, -1]) == 1
    assert qualification.aggregate_temperature(0.1, 0.6, 0.5) != pytest.approx(
        (runtime.temperature_risk(0.1, 0.5) + runtime.temperature_risk(0.6, 0.5)) / 2
    )
    assert qualification.aggregate_temperature(0.1, 0.6, 0.5) == pytest.approx(
        runtime.temperature_risk(0.35, 0.5)
    )


def test_constraints_and_adapter_share_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7769-NUMERICAL: constraints and natural decisions share math."""
    batch, _ = qualification.make_batch(qualification.fixture_records(), "constrained_set", [])
    params = runtime.init_params("energy_local", 67501)
    ordinary = runtime.loss(params, batch, "energy_local", "ordinary", (0.0, 0.0))
    zero = runtime.loss(params, batch, "energy_local", "constrained", (0.0, 0.0))
    active = runtime.loss(params, batch, "energy_local", "constrained", (2.0, 3.0))
    assert float(ordinary) == pytest.approx(float(zero))
    assert float(active) != pytest.approx(float(zero))
    assert runtime.dual_step((10.0, 0.0), (100.0, 0.0))[0] == 10.0
    assert runtime.dual_step((0.0, 0.0), (0.0, 0.0)) == (0.0, 0.0)
    unknown = {**batch, "known": jnp.full_like(batch["known"], -1)}
    assert float(
        runtime.loss(params, unknown, "energy_local", "canonical", (0.0, 0.0))
    ) == pytest.approx(runtime.response_nll(params, unknown, "energy_local", False))
    extreme = {**params, "b": params["b"].at[0].set(1000)}
    assert np.isfinite(
        np.asarray(runtime.deployed_risk(extreme, batch, "energy_local", True))
    ).all()
    assert -math.log(0.35) != pytest.approx((-math.log(0.1) - math.log(0.6)) / 2)
    head = {"params": params, "arm": "energy_local", "temperature": 0.5, "paired": True}
    expected = runtime.decide(head, batch, 0, True)
    called: list[str] = []
    original = runtime.deployed_risk
    original_temperature = runtime.temperature_risk
    original_action = runtime.action

    def tracked(*args: object) -> jnp.ndarray:
        called.append("aggregate")
        return original(*args)

    monkeypatch.setattr(runtime, "deployed_risk", tracked)

    def tracked_temperature(probability: float, temperature: float) -> float:
        called.append("temperature")
        return original_temperature(probability, temperature)

    def tracked_action(probability: float) -> str:
        called.append("action")
        return original_action(probability)

    monkeypatch.setattr(runtime, "temperature_risk", tracked_temperature)
    monkeypatch.setattr(runtime, "action", tracked_action)
    assert qualification.NaturalHeadAdapter(head).decide(batch, 0) == expected
    assert called == ["aggregate", "temperature", "action"]


def test_logistic_feature_average_and_invalid_arm() -> None:
    """SCENARIO-VERIFY-7769-NUMERICAL: paired logistic pools before one head."""
    records = qualification.fixture_records()
    with pytest.raises(ValueError, match="unknown qualification arm"):
        qualification.make_batch(records, "missing", [])
    with pytest.raises(ValueError, match="sixteen predicates"):
        qualification.make_batch(records, "complete_static_constrained_set", [])
    batch, _ = qualification.make_batch(records, "local_logistic", [])
    averaged = qualification.averaged_logistic_batch(batch)
    params = runtime.init_params("logistic_local", 67501)
    head = {"params": params, "arm": "logistic_local", "temperature": 0.75, "paired": True}
    expected = runtime.decide(head, averaged, 0, True)
    assert qualification.NaturalHeadAdapter(head).decide(batch, 0) == expected
    a = np.asarray(batch["a"]["x"])
    b = np.asarray(batch["b"]["x"])
    assert a.shape[-1] == b.shape[-1]


def test_failed_online_preflight_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7769-TERMINAL: a bad bank producer stops before fitting."""
    monkeypatch.setattr(experiment.online, "preflight", lambda root: ([{"passed": False}], {}, []))
    with pytest.raises(ValueError, match="required online source failed"):
        experiment.run_fixture(tmp_path / "blocked", "20260927")
    with pytest.raises(ValueError, match="online bank preflight failed"):
        experiment.qualify_online(tmp_path / "blocked_online")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)

    def terminal(root: Path, commands: list, *, log_dir: Path, **kwargs: object) -> list[dict]:
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for command in commands:
            log = log_dir / f"{command.name}.log"
            log.write_text(json.dumps({"flagged_count": 0}))
            failed = command.name == "strict_row_consistency"
            receipts.append(
                {
                    "name": command.name,
                    "passed": not failed,
                    "exit_code": int(failed),
                    "log_path": str(log),
                    "log_sha256": "sha256:fixture",
                }
            )
        return receipts

    monkeypatch.setattr(experiment, "run_commands", terminal)
    blocked = experiment.launch("20260927")
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]
    assert any(
        row["field"] == "strict_row_consistency.exit_code" for row in blocked["gate_check_summary"]
    )
    assert len(blocked["rows"]) == 54
    assert blocked["training_runtime_ready_score"] == 0


@pytest.mark.parametrize("bad", [False, True])
def test_validation_orchestration_keeps_required_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: bool
) -> None:
    """SCENARIO-VERIFY-7769-TERMINAL: failed affected checks close both readiness gates."""
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "direct_tests": [
                    "tests/python/test_experiment_7769_v676_training_qualification.py"
                ],
                "transitive_tests": [],
                "changed_modules": ["python/carnot/verify/training_qualification.py"],
                "cli": ["scripts/experiments/experiment_7769_v676_training_qualification.py"],
            }
        )
    )
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "SCOPE", scope)
    monkeypatch.setattr(experiment.online, "preflight", lambda root: ([{"passed": True}], {}, []))
    monkeypatch.setattr(
        experiment,
        "run_fixture",
        lambda folder, date: {
            "rows": [{"family": "fixture-0"}],
            "validation_receipts": {},
            "acceptance_gate_results": {"readiness": 1, "validity": True},
            "honest_verdict": "complete_circular_positive_training_fixture",
            "verdict_class": "circular_positive",
            "training_runtime_ready_score": 1,
            "online_runtime_ready_score": 1,
            "phase_spans": [],
            "flagged_adversarial": False,
        },
    )
    monkeypatch.setattr(
        experiment,
        "build_scoped_commands",
        lambda *args, **kwargs: [experiment.CommandSpec("focused_pytest", ("true",), "test")],
    )
    monkeypatch.setattr(experiment, "reduce_required_checks", lambda receipts: {"passed": True})

    def fake_commands(root: Path, commands: list, *, log_dir: Path, **kwargs: object) -> list[dict]:
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for command in commands:
            path = log_dir / f"{command.name}.log"
            path.write_text(
                json.dumps({"flagged_count": int(bad)})
                if command.name == "adversarial_verify"
                else "ok"
            )
            failed = bad and command.name in {"focused_pytest", "strict_row_consistency"}
            receipts.append(
                {
                    "name": command.name,
                    "passed": not failed,
                    "exit_code": int(failed),
                    "log_path": str(path),
                    "log_sha256": "sha256:fixture",
                    "command_argv": list(command.argv),
                }
            )
        return receipts

    monkeypatch.setattr(experiment, "run_commands", fake_commands)
    artifact = experiment.launch("20260927")
    assert artifact["training_runtime_ready_score"] == int(not bad)
    assert artifact["online_runtime_ready_score"] == int(not bad)
    assert bool(artifact["gate_check_summary"]) == bad
    assert artifact["flagged_adversarial"] == bad


def test_thin_entrypoint_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7769-TERMINAL: both CLI paths use the same experiment module."""
    script = experiment.ROOT / "scripts/experiments/experiment_7769_v676_training_qualification.py"
    spec = importlib.util.spec_from_file_location("exp7769_cli_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.experiment,
        "launch",
        lambda date: {
            "honest_verdict": "complete_circular_positive_training_fixture",
            "training_runtime_ready_score": 1,
            "online_runtime_ready_score": 1,
        },
    )
    monkeypatch.setattr(sys, "argv", [str(script), "--date", "20260927"])
    assert module.main() == 0
    monkeypatch.setattr(module.experiment, "run_fixture", lambda folder, date: {"rows": []})
    monkeypatch.setattr(module.experiment, "cold_reduce", lambda candidate: {"valid": True})
    monkeypatch.setattr(sys, "argv", [str(script), "--private-e2e", "--date", "20260927"])
    assert module.main() == 0


def test_online_requalification_and_private_child(tmp_path: Path, qualified_artifact: dict) -> None:
    """SCENARIO-VERIFY-7769-ONLINE: current fixture has causal and durable receipts."""
    outcome = qualified_artifact["online_fixture"]
    assert outcome["valid"]
    assert outcome["prediction_before_feedback"]
    assert outcome["rejected_admissions"]
    assert outcome["applied_updates"]
    assert outcome["shuffled_delayed_labels"]
    assert outcome["complete_static_initial_coefficients"] == 16
    assert outcome["hard_exit"] and outcome["cold_restart_parity"]
    nested = tmp_path / "child" / "basetemp"
    nested.parent.mkdir(parents=True)
    child_test = tmp_path / "test_private_child.py"
    child_test.write_text("def test_private_child():\n    assert True\n")
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
            str(child_test),
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_artifact_contract_and_cold_reduce(tmp_path: Path, qualified_artifact: dict) -> None:
    """SCENARIO-VERIFY-7769-TERMINAL: raw fixture rows replay in a fresh process."""
    artifact = qualified_artifact
    assert artifact["verdict_class"] == "circular_positive"
    assert len(artifact["fixture_training_rows"]) == 9
    protocol = json.loads(Path(artifact["training_protocol_path"]).read_text())
    assert len(protocol["complete_static_initial_coefficients"]) == 16
    assert set(protocol["complete_static_initial_coefficients"].values()) == {0.0}
    assert artifact["training_runtime_ready_score"] == 0
    assert artifact["online_runtime_ready_score"] == 0
    assert artifact["acceptance_gate_results"]["validity"]
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invocation_counts"]["calls"] == 0
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_7769_v676_training_qualification",
            "--cold-reduce",
            str(candidate),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert experiment.cold_reduce(candidate)["decision_count"] == 36
    assert experiment.main(["--cold-reduce", str(candidate)]) == 0
    changed = json.loads(candidate.read_text())
    changed["rows"][0]["action"] = (
        "reject" if changed["rows"][0]["action"] != "reject" else "accept"
    )
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="candidate_raw_rows_invalid"):
        experiment.cold_reduce(candidate)
    candidate.write_text(json.dumps(artifact))
    changed = json.loads(candidate.read_text())
    changed["fixture_training_rows"][0]["head_hash"] = "sha256:changed"
    raw_training = Path(changed["raw_paths"]["training"])
    original_training = raw_training.read_text()
    raw_training.write_text(json.dumps(changed["fixture_training_rows"]))
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="head_hash_invalid"):
        experiment.cold_reduce(candidate)
    raw_training.write_text(original_training)
    candidate.write_text(json.dumps(artifact))
    changed = json.loads(candidate.read_text())
    raw_rows = Path(changed["raw_paths"]["rows"])
    original_rows = raw_rows.read_text()
    shorter = changed["rows"][1:]
    changed["rows"] = shorter
    raw_rows.write_text(json.dumps(shorter))
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="unit_count_invalid"):
        experiment.cold_reduce(candidate)
    changed["rows"] = json.loads(original_rows)
    changed["rows"][0]["action"] = (
        "reject" if changed["rows"][0]["action"] != "reject" else "accept"
    )
    raw_rows.write_text(json.dumps(changed["rows"]))
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cold_decision_invalid"):
        experiment.cold_reduce(candidate)
    raw_rows.write_text(original_rows)
    changed = json.loads(json.dumps(artifact))
    changed["online_fixture"]["raw_hash"] = "sha256:changed"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="online_raw_invalid"):
        experiment.cold_reduce(candidate)
