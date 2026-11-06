"""REQ-REPORT-8830 and REQ-VERIFY-8830: causal head and producer contracts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from carnot.memory import categorical_memory_8830 as c
from carnot.reporting import categorical_memory_runner_8830 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.mark.parametrize("objective", ["mixed", "categorical", "linear"])
def test_gradient_and_normalization(objective: str) -> None:
    """SCENARIO-VERIFY-8830-CAUSAL: gradients and serving share exact logits."""
    head = c.head(11, objective)
    counts = [5, 4, 1, 2, 0, 1]
    flat, restore = ravel_pytree([jnp.asarray(x) for x in head["weights"]])
    x = jnp.asarray([c.m.features(counts, i) for i in range(3)])
    fn = lambda v: c.d.loss(restore(v), x, jnp.asarray(1), objective)
    analytic = np.asarray(jax.grad(fn)(flat))
    finite = np.array(
        [
            (float(fn(flat.at[i].add(0.01))) - float(fn(flat.at[i].add(-0.01)))) / 0.02
            for i in range(len(flat))
        ]
    )
    assert np.max(np.abs(analytic - finite)) < 0.0001
    new, receipt = c.update(head, counts, "refute")
    assert receipt["optimizer_steps"] == 1
    assert canonical_hash(new) != canonical_hash(head)
    assert sum(c.probabilities(new, counts)) == pytest.approx(1, abs=1e-6)


def test_causal_noops_and_atomicity(tmp_path: Path) -> None:
    """REQ-VERIFY-8830: invalid feedback and failed writes cannot partly commit."""
    s = c.Store(tmp_path / "state.json", c.head(11, "categorical"))
    payload = c.m.world("a", "support")
    sealed = s.predict("q0", payload, 0.5)
    assert s.read()["pending"] == {}
    with pytest.raises(RuntimeError):
        s.predict("q1", payload, 0.5)
    assert s.release("q0", payload, 8, label="support")["status"] == "invalid"
    s.close()
    for event, evidence, ordinal, label in [
        ("missing", payload, 8, "support"),
        ("q0", payload, 7, "support"),
        ("q0", payload, 8, None),
        ("q0", payload, 8, "refute"),
        ("q0", c.m.world("b", "support"), 8, "support"),
    ]:
        before = s.path.read_bytes()
        assert s.release(event, evidence, ordinal, label=label)["optimizer_steps"] == 0
        assert s.path.read_bytes() == before
    state = deepcopy(s.state)
    with patch.object(c, "_atomic_write", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            s.release("q0", payload, 8, label="support")
    assert s.state == state
    assert json.loads(s.path.read_text()) == state
    row = s.release("q0", payload, 8, label="support")
    assert row["template_added"] and row["optimizer_steps"] == 1
    witness = s.state["template_provenance"][row["template"]]["binding_witnesses"][0]
    assert witness["variables"] == {"?x": "aa", "?y": "ab"}
    assert witness["fact_indices"] == list(range(5))
    assert row["prediction_hash"] == canonical_hash(sealed)
    assert s.release("q0", payload, 99, label="support")["optimizer_steps"] == 0
    with pytest.raises(ValueError):
        s.predict("q0", payload, 0.5)
    used = s.predict("q1", c.m.world("later", "support"), 0.5)
    assert used["template_uses"]
    s.close()
    s.state["pending"] = {f"q{i}": sealed for i in range(32)}
    with pytest.raises(ValueError):
        s.predict("q33", payload, 0.5)
    s.state["pending"] = {"q1": dict(used)}
    s.state["templates"] = [str(i) for i in range(32)]
    assert not s.release("q1", c.m.world("later", "support"), 9, label="support")["template_added"]


def cli(tmp_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-8830: run genuine private children outside the checkout."""
    req, out = tmp_path / "request.json", tmp_path / "response.json"
    atomic_json(req, request)
    result = subprocess.run(
        [
            str(r.ROOT / ".venv/bin/python"),
            "-u",
            str(r.ROOT / r.CLI),
            "--request",
            str(req),
            "--response",
            str(out),
        ],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_fresh_process_restart(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8830-CAUSAL: restart preserves sealed work exactly once."""
    request = dict(
        state=str(tmp_path / "state.json"),
        head=c.head(11, "linear"),
        event="q0",
        payload=c.m.world("private", "support"),
        operation="predict",
    )
    predicted = cli(tmp_path, request)
    request.update(operation="release", ordinal=8, label="support")
    released = cli(tmp_path, request)
    assert released["prediction_hash"] == canonical_hash(predicted)
    assert released["optimizer_steps"] == 1
    assert cli(tmp_path, request)["optimizer_steps"] == 0


def test_learning_controls() -> None:
    """REQ-REPORT-8830: shuffled labels must harm a signal-bearing fixture."""
    rows = c.controls()
    assert all(row["passed"] for row in rows)
    assert {row["check"] for row in rows} >= {"shuffled_damage", "finite_difference"}


@pytest.fixture(scope="module")
def measured(tmp_path_factory: Any) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """REQ-REPORT-8830: use actual qualified operands and matched fits once."""
    raw = tmp_path_factory.mktemp("measured8830")
    _, evidence, pre = r.preflight(raw)
    work = r.execute(raw, evidence, pre["configuration"])
    work["restart_rows"] = r.restart(raw / "restart")
    atomic_json(raw / "evidence.json", work)
    return raw, evidence, pre, work


def test_preflight_mutations(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8830: resource, gate, config and operand failures precede measurement."""
    with pytest.raises(ValueError):
        c.head(11, "invented")
    upstream = json.loads(r.UPSTREAM.read_bytes())
    candidate = tmp_path / "source.json"
    for key, value, error in [
        ("objective_candidate_ready_score", 0, PermissionError),
        ("objective_selection", {}, KeyError),
        ("selected_configuration_hash", "bad", RuntimeError),
        ("code_config_hashes", [dict(path=str(r.UPSTREAM), sha256="bad")], ValueError),
    ]:
        changed = dict(upstream, **{key: value})
        atomic_json(candidate, changed)
        with pytest.raises(error):
            r.preflight(tmp_path / "raw", candidate)
    evidence = json.loads(Path(upstream["raw_shard_hashes"][0]["path"]).read_bytes())
    atomic_json(tmp_path / "snapshot.json", dict(private_fixture=True))
    evidence["snapshot"] = dict(path=str(tmp_path / "snapshot.json"), sha256="bad")
    atomic_json(tmp_path / "evidence.json", evidence)
    upstream["raw_shard_hashes"] = [r.reference(tmp_path / "evidence.json")]
    atomic_json(candidate, upstream)
    with pytest.raises(ValueError):
        r.preflight(tmp_path / "raw", candidate)
    evidence["snapshot"]["sha256"] = r.reference(Path(evidence["snapshot"]["path"]))["sha256"]
    atomic_json(tmp_path / "evidence.json", evidence)
    upstream["raw_shard_hashes"] = [r.reference(tmp_path / "evidence.json")]
    atomic_json(candidate, upstream)
    with pytest.raises(PermissionError):
        r.preflight(tmp_path / "raw", candidate)
    with pytest.raises(ValueError):
        r.bound(dict(path=str(candidate), sha256="wrong"))
    with patch.object(r.shutil, "disk_usage", return_value=type("Disk", (), dict(free=0))()):
        with pytest.raises(RuntimeError):
            r.preflight(tmp_path / "raw")


def test_fit_and_restart_failures(tmp_path: Path, measured: Any, monkeypatch: Any) -> None:
    """REQ-VERIFY-8830: owned cohort and real-child control failures fail closed."""
    _, evidence, pre, _ = measured
    with patch.object(c.d, "fit", return_value={}):
        with pytest.raises(RuntimeError, match="historical_fit_drift"):
            r.execute(tmp_path, evidence, pre["configuration"])
    with patch.object(r.validation, "commands", return_value=[dict(passed=False)]):
        with pytest.raises(RuntimeError, match="restart_child"):
            r.restart(tmp_path)
    response = dict(prediction_hash="wrong", optimizer_steps=0)

    def forged(plan: Any, logs: Path) -> list[dict[str, Any]]:
        atomic_json(Path(plan[0][1][-1]), response)
        return [dict(passed=True)]

    with patch.object(r.validation, "commands", side_effect=forged):
        with pytest.raises(RuntimeError, match="restart_atomicity"):
            r.restart(tmp_path)


def test_producer_and_replay(tmp_path: Path, measured: Any, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-8830-CONTRACT: real operands survive publication and reject mutation."""
    raw, evidence, pre, work = measured

    def admitted(raw: Path, upstream: Path = r.UPSTREAM) -> Any:
        if not upstream.exists():
            raise FileNotFoundError(2, "missing", str(upstream))
        if upstream.name == "zero.json":
            raise PermissionError("objective_candidate_ready_score:observed_zero")
        return {}, evidence, pre

    monkeypatch.setattr(r, "preflight", admitted)
    checked = dict(passed=True, receipts=[], coverage=dict(complete=True, percent=100))
    monkeypatch.setattr(r, "validate_owned", lambda raw: deepcopy(checked))
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=True, receipts=[]))
    monkeypatch.setattr(r, "execute", lambda raw, evidence, config: deepcopy(work))
    monkeypatch.setattr(r, "restart", lambda raw: work["restart_rows"])
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--private-output", str(output), "--raw-dir", str(raw)]) == 0
    value = json.loads(output.read_bytes())
    assert value["memory_execution_ready_score"] == 1
    assert len(value["rows"]) == 360
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert r.replay(output)["passed"]
    assert r.main(["--replay", str(output)]) == 0
    mutations = [
        ("code_config_hashes", [dict(path=str(r.UPSTREAM), sha256="bad")], "replay_hash"),
        (
            "validation_receipts",
            [dict(log_path=str(r.UPSTREAM), log_sha256="bad")],
            "validation_log",
        ),
        ("rows", [], "reduction_drift"),
        ("parameter_hashes", [], "parameter_drift"),
        ("completed_count", 1, "readiness_drift"),
    ]
    for key, change, message in mutations:
        atomic_json(output, dict(value, **{key: change}))
        with pytest.raises(ValueError, match=message):
            r.replay(output)
    atomic_json(output, dict(value, verdict_class="blocked", memory_execution_ready_score=0))
    assert r.replay(output)["passed"]
    atomic_json(output, dict(value, verdict_class="blocked"))
    with pytest.raises(ValueError, match="blocked_readiness"):
        r.replay(output)
    assert r.main(["--replay", str(tmp_path / "absent")]) == 1
    wrong = deepcopy(work)
    wrong["restart_rows"][0]["response"] = {}
    atomic_json(tmp_path / "wrong.json", wrong)
    atomic_json(output, dict(value, raw_shard_hashes=[r.reference(tmp_path / "wrong.json")]))
    with pytest.raises(ValueError, match="restart_response"):
        r.replay(output)
    monkeypatch.setattr(r, "execute", lambda raw, evidence, config: dict(work, final_state_refs=[]))
    atomic_json(output, value)
    with pytest.raises(ValueError, match="final_state_drift"):
        r.replay(output)
    monkeypatch.setattr(r, "execute", lambda raw, evidence, config: deepcopy(work))
    work_bad = deepcopy(work)
    work_bad["checkpoint_refs"][0]["parameter_hash"] = "bad"
    atomic_json(tmp_path / "badwork.json", work_bad)
    atomic_json(output, dict(value, raw_shard_hashes=[r.reference(tmp_path / "badwork.json")]))
    with pytest.raises(ValueError, match="parameter_hash"):
        r.replay(output)
    for failed in ("control", "owned", "terminal", "missing", "internal", "zero"):
        checked["passed"] = failed != "owned"
        monkeypatch.setattr(
            r, "terminal", lambda path, raw: dict(passed=failed != "terminal", receipts=[])
        )
        if failed == "control":
            monkeypatch.setattr(
                r,
                "execute",
                lambda raw, evidence, config: dict(work, controls=[dict(passed=False)]),
            )
        else:
            monkeypatch.setattr(r, "execute", lambda raw, evidence, config: deepcopy(work))
        if failed == "internal":
            monkeypatch.setattr(
                r, "execute", lambda *args: (_ for _ in ()).throw(RuntimeError("internal"))
            )
        args = ["--private-output", str(output), "--raw-dir", str(raw)]
        if failed == "missing":
            args += ["--upstream", str(tmp_path / "absent")]
        if failed == "zero":
            atomic_json(tmp_path / "zero.json", dict(objective_candidate_ready_score=0))
            args += ["--upstream", str(tmp_path / "zero.json")]
        assert r.main(args) == 0
        failed_value = json.loads(output.read_bytes())
        assert failed_value["memory_execution_ready_score"] == 0
        assert failed_value["verdict_class"] == (
            "blocked" if failed in {"missing", "zero"} else "disqualified"
        )


def test_shared_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8830: shared validators keep the current producer and owned scope."""
    with patch.object(r.validation, "validate", return_value=dict(passed=True)):
        assert r.validate_owned(tmp_path)["passed"]
    with patch.object(r.validation, "terminal", return_value=dict(passed=True)):
        assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_rejected_producer_and_relative_log(tmp_path: Path) -> None:
    """REQ-REPORT-8830: reject unqualified authority and authenticate relative logs."""
    rejected = json.loads(r.UPSTREAM.read_bytes())
    rejected["required_checks_passed"] = False
    source = tmp_path / "rejected.json"
    atomic_json(source, rejected)
    with pytest.raises(PermissionError, match="rejected_producer"):
        r.preflight(tmp_path / "raw", source)
    candidate = tmp_path / "candidate.json"
    atomic_json(
        candidate,
        dict(
            code_config_hashes=[],
            source_artifact_hashes=[],
            validation_receipts=[
                dict(log_path=str(r.UPSTREAM.relative_to(r.ROOT)), log_sha256="wrong")
            ],
        ),
    )
    with pytest.raises(ValueError, match="validation_log"):
        r.replay(candidate)
