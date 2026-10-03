"""REQ-REPORT-8046: verify frozen methods before evaluator labels are available."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot import experiment_8046_v697_branch_protocols as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8046_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8046_COVERAGE_CONFIG"]]
    return subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_methods_and_original_roles(tmp_path):
    """SCENARIO-REPORT-8046-METHODS: role custody never selects on new errors."""
    plan = e.seal(e.ROOT, tmp_path)
    assert all(c["passed"] for c in plan["checks"])
    assert {k: len(v) for k, v in plan["roles"].items()} == e.ROLE_COUNTS
    assert plan["evaluator_access_log"] == []
    assert all(r["historically_exposed"] for r in plan["rows"])
    assert e.METHODS["source"]["conditions"] == ["full_A", "full_B", "no_source_A", "no_source_B"]
    assert e.METHODS["source"]["context_limit"] == 4096
    assert e.METHODS["statistics"]["draws"] == 10000
    assert e.METHODS["safety"]["later_support"] == [120, 15]
    stream = plan["roles"]["stream"]
    assert {r["family_id"] for r in plan["guard_partition_rows"]} == {
        r["family_id"] for r in stream
    }
    assert all(
        r["bucket"] == int(e.canonical_hash(r["source_cluster_id"]).split(":")[1], 16) % 4
        for r in plan["guard_partition_rows"]
    )
    assert len(plan["head"]["parameters"]) == 110


def test_guard_diagnostics_and_reset():
    """SCENARIO-REPORT-8046-GUARD: gradients cannot use guard feedback."""
    x = np.ones((8, 1))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    theta = np.array([0.0])
    delta = np.array([-10.0])
    constrained = e.accept_candidate(theta, delta, theta, x, y, "feedback_constrained")
    unconstrained = e.accept_candidate(theta, delta, theta, x, y, "unconstrained")
    assert constrained["diagnostics"] == unconstrained["diagnostics"]
    assert constrained["alpha"] == 0 and constrained["rejected"]
    assert unconstrained["alpha"] == 1
    assert constrained["diagnostics"][0]["new_false_accepts"] == 4
    assert (
        e.accept_candidate(theta, delta, theta, x[:2], y[:2], "feedback_constrained")["status"]
        == "waiting_guard"
    )
    reset = e.accept_candidate(np.array([-1000.0]), delta, theta, x, y, "feedback_constrained")
    assert reset["reset"] and reset["alpha"] is None
    assert reset["parameters"] == theta.tolist()
    assert e.accept_candidate(theta, delta, theta, x, y, "frozen_no_write")["status"] == "frozen"
    with pytest.raises(ValueError):
        e.accept_candidate(theta, delta, theta, x, y, "bad")


def test_future_labels_and_durable_issue(tmp_path):
    """SCENARIO-REPORT-8046-CAUSAL: future outcomes cannot affect earlier work."""
    rows = [
        dict(family_id=f"s{i}", source_cluster_id=f"s{i}", slot=i, x=[1.0], y=i % 2)
        for i in range(80)
    ]
    earlier = e.causal_step(rows, 55, np.array([0.0]), tmp_path / "one")
    future = copy.deepcopy(rows)
    for r in future[36:]:
        r["y"] = 1 - r["y"]
    assert earlier == e.causal_step(future, 55, np.array([0.0]), tmp_path / "two")
    assert (tmp_path / "one" / "prediction.json").is_file()
    assert not set(earlier["selected_ids"]) & set(earlier["guard_ids"])
    unknown = copy.deepcopy(rows)
    unknown[0]["y"] = None
    e.causal_step(unknown, 55, np.array([0.0]), tmp_path / "unknown")
    assert e.causal_step(rows, 0, np.array([0.0]), tmp_path / "empty")["selected_ids"] == []


def test_cli_routes_and_blocked_contract(tmp_path):
    """SCENARIO-REPORT-8046-TERMINAL: real private exits and tamper checks."""
    output = tmp_path / "results" / (e.NAME + ".json")
    run = cli(tmp_path, "--output", output, "--seal-only")
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "null" and value["source_protocol_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["learning_protocol_ready_score"] = 1
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--date", "bad").returncode == 2
    assert cli(tmp_path, "--output", output, "--seal-only").returncode == 1
    blocked = tmp_path / "blocked" / "results" / (e.NAME + ".json")
    assert (
        cli(tmp_path, "--root", tmp_path / "missing", "--output", blocked, "--seal-only").returncode
        == 0
    )
    v = json.loads(blocked.read_text())
    assert v["verdict_class"] == "blocked" and v["honest_verdict"].startswith("complete_blocked_")
    assert any(c["observed"] == "missing_resource" for c in v["gate_check_summary"])
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0


def test_owned_validation_and_normal_main(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8046-TERMINAL: reducers require exact owned checks and coverage."""
    specs = e.commands(tmp_path / "venue")
    assert all(e.venue.old.OWNED[0] not in s.argv for s in specs)
    assert any(s.scope == "repository_health" for s in specs)
    assert any("--strict" in s.argv for s in specs)
    monkeypatch.setattr(
        e, "commands", lambda p: [e.CommandSpec("private", (sys.executable, "-c", "pass"), "owned")]
    )

    def child(root, commands, **kwargs):
        raw = kwargs["log_dir"].parent
        raw.mkdir(parents=True, exist_ok=True)
        log = raw / "private.log"
        log.write_text("Synthetic unit receipt; not production acceptance.\n")
        cov = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in [e.MODULE, e.CLI]}
        atomic_json(tmp_path / "coverage.json", dict(files=cov))
        ref = e.reference(log)
        return [
            dict(
                name="private",
                scope="owned",
                exit_code=0,
                passed=True,
                log_path=ref["path"],
                log_sha256=ref["sha256"],
            )
        ]

    monkeypatch.setattr(e, "run_commands", child)
    measured = e.validate(tmp_path, tmp_path / "raw")
    assert measured["coverage"][e.MODULE]["missing_lines"] == 0
    (tmp_path / "coverage.json").unlink()
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [])
    assert e.validate(tmp_path, tmp_path / "empty")["coverage"] == {}
    monkeypatch.setattr(e, "validate", lambda *a: measured)
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    output = tmp_path / "normal" / "results" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    v = json.loads(output.read_text())
    assert v["source_protocol_ready_score"] == v["learning_protocol_ready_score"] == 1
    e.replay(v)
    raw = Path(v["terminal_validation_sidecar_path"]).parent
    plan, work, validation = [
        json.loads((raw / p).read_text()) for p in ["methods.json", "work.json", "validation.json"]
    ]
    validation["receipts"][0]["passed"] = False
    assert e.build(plan, work, raw, validation)["verdict_class"] == "disqualified"
    for field, match in [
        ("methods", "methods_drift"),
        ("roles", "role_drift"),
        ("rows", "primitive_row_drift"),
    ]:
        changed = copy.deepcopy(plan)
        if field == "methods":
            changed[field] = {}
        elif field == "roles":
            changed[field]["fit"][0]["slot"] += 1
        else:
            changed[field][0]["source_cluster_id"] = "tampered"
        atomic_json(raw / "methods.json", changed)
        bad = copy.deepcopy(v)
        bad["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in bad["raw_shard_hashes"]]
        with pytest.raises(ValueError, match=match):
            e.replay(bad)
    atomic_json(raw / "methods.json", plan)
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=False))
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "reject/results" / (e.NAME + ".json")),
            ]
        )
        == 1
    )
    reports = iter([dict(passed=True), dict(passed=False)])
    monkeypatch.setattr(e, "terminal", lambda p: next(reports))
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "post/results" / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_missing_sidecars_and_historical_chunk_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8046-TERMINAL: missing fields and rehashed chunks fail closed."""
    primary = tmp_path / "results" / (e.prior.NAME + ".json")
    atomic_json(primary, dict(experiment_id=8032))
    plan = e.seal(tmp_path, tmp_path / "blocked")
    assert any(c["artifact_field"] == "terminal_exists" and not c["passed"] for c in plan["checks"])
    side = tmp_path / "side.json"
    atomic_json(side, {})
    atomic_json(primary, dict(experiment_id=8032, terminal_validation_sidecar_path=str(side)))
    assert any(
        c["artifact_field"] == "contract"
        for c in e.seal(tmp_path, tmp_path / "missingfield")["checks"]
    )
    part = tmp_path / "part.bin"
    part.write_bytes(b"actual bytes")
    manifest = tmp_path / "x-chunks.json"
    atomic_json(
        manifest,
        dict(chunks=[e.reference(part)], original=dict(sha256="sha256:wrong"), byte_count=11),
    )
    with pytest.raises(ValueError, match="historical_chunk_drift"):
        e.replay(
            dict(
                raw_shard_hashes=[e.reference(manifest)],
                code_config_hashes=[],
                cited_upstream_artifacts=[],
            )
        )
    with pytest.raises(ValueError, match="hash"):
        e.copy_evidence(dict(path=str(tmp_path / "absent"), sha256="missing"), tmp_path)
