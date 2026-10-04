"""REQ-REPORT-8108 and REQ-VERIFY-8108: private custody and precision controls."""

from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from carnot.reporting import radial_hardware_8108 as h
from carnot import experiment_8108_v701_radial_hardware_boundary as cli
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def system(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        seed=seed,
        x=rng.uniform(-2, 2, (4, 9)).tolist(),
        state=dict(
            geometry=dict(mean=[0.0] * 9, std=[1.0] * 9, sigma=2.0),
            centers=[dict(x=rng.uniform(-2, 2, 9).tolist()) for _ in range(16)],
            coefficients=rng.uniform(-1, 1, 17).tolist(),
        ),
    )


def data():
    return dict(
        systems=[system(i) for i in range(64)],
        boards=[
            dict(
                board=name,
                processor_class=contract[0],
                k_max=contract[1],
                blocker=contract[2],
                custody_valid=True,
                source_path=name,
                source_hash="saved",
                scope="original",
                receipt_date="original",
            )
            for name, contract in h.CONTRACTS.items()
        ],
        checks=[],
        references=[],
        costs=[],
    )


def pair():
    components = {k: 1 for k in h.COMPONENTS}
    components["scoring_ns"] = 10
    return dict(
        unit_id="one",
        source_id="family",
        mode="warm",
        centers=16,
        arms=[dict(arm=a, total_ns=100, components=components) for a in ("python", "rust")],
    )


def test_precision_bounds_and_ties():
    """REQ-VERIFY-8108: the observed error fits an analytic outward envelope."""
    for bits in (8, 12, 16):
        row = h.numerical(system(), bits)
        assert row["maximum_probability_error"] <= row["probability_error_bound"]
        assert row["action_disagreements"] == 0
    tied = system()
    tied["state"]["coefficients"] = [0.0] * 17
    row = h.numerical(tied, 8)
    assert all(row["fallback_flags"])
    assert set(row["issued_actions"]) == {"escalate"}
    clipped = system()
    clipped["state"]["centers"][0]["x"][0] = 5
    clipped["x"][0][0] = 5
    assert all(h.numerical(clipped, 12)["fallback_flags"])
    assert h.numerical(clipped, 12)["clipping_count"] == 1
    with pytest.raises(ValueError):
        h.numerical(system(), 4)
    broken = system()
    broken["state"]["geometry"]["sigma"] = 0
    with pytest.raises(ValueError):
        h.numerical(broken, 8)


def test_independent_board_and_bound_failures():
    """SCENARIO-REPORT-8108: board failures do not erase another board."""
    d = data()
    value = h.reduce(d)
    assert value["hardware_boundary_ready_score"] == 1
    assert value["independent_count"] == 0
    d["boards"][0]["custody_valid"] = False
    value = h.reduce(d)
    assert value["verdict_class"] == "blocked"
    assert value["board_rows"][1]["custody_valid"]
    d["boards"][0]["custody_valid"] = True
    d["boards"][0]["k_max"] = 6
    assert h.reduce(d)["board_rows"][0]["status"] == "blocked"
    d["systems"] = []
    assert h.reduce(d)["honest_verdict"] == "complete_blocked_radial_fixtures"
    rows = [h.numerical(system(), 8)]
    rows[0]["probability_error_bound"] = 0
    assert not h.summarize(rows)["passed"]


def test_service_complete_costs():
    """REQ-REPORT-8108: free arithmetic cannot remove durable overhead."""
    value = h.service([pair()])
    assert value["conditional_speedup_bound"] == pytest.approx(100 / 90)
    assert value["arithmetic_fraction"] == 0.1
    assert value["rows"][0]["transport_ns"] is None
    assert h.service([])["status"] == "unavailable"
    for mutate in ("missing", "negative", "duplicate", "total"):
        p = pair()
        if mutate == "missing":
            p["arms"][0]["components"].pop("durable_commit_ns")
        elif mutate == "negative":
            p["arms"][0]["components"]["scoring_ns"] = -1
        elif mutate == "total":
            p["arms"][0]["total_ns"] = 1
        assert h.service([p, p] if mutate == "duplicate" else [p])["status"] == "unavailable"


def authenticated(root, eid, name, extra):
    p = root / "results" / f"experiment_{eid}_{name}.json"
    terminal = root / "results" / "raw" / p.stem / "terminal.json"
    value = dict(
        experiment_id=eid,
        task_id=f"exp{eid}-test",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        flagged_adversarial=False,
        required_checks_passed=True,
        terminal_validation_sidecar_path=str(terminal),
        **extra,
    )
    binding = publish_primary(p, value, lambda _: dict(passed=True))
    atomic_json(terminal, dict(publication=binding))
    return p


def test_authentication_and_missing_inputs(tmp_path, monkeypatch):
    """REQ-REPORT-8108: exact sidecars and immutable snapshots bind each gate."""
    raw = tmp_path / "raw"
    raw.mkdir()
    ledger = dict(checks=[], references=[])
    p = authenticated(tmp_path, 8085, "v700_radial_memory_kernel", dict(kernel_ready_score=1))
    value, good = h.authenticate(p, 8085, "kernel_ready_score", raw, ledger)
    assert good and value["kernel_ready_score"] == 1
    p.write_text("{}")
    assert not h.authenticate(p, 8085, "kernel_ready_score", raw, ledger)[1]
    assert h.authenticate(tmp_path / "absent", 8106, "service_cost_ready_score", raw, ledger) == (
        {},
        False,
    )
    assert h.load(tmp_path, raw)["systems"] == []
    d = data()
    receipts = []
    for board in d["boards"]:
        source = tmp_path / board["source_path"]
        atomic_json(source, dict(workload="original", code_hash="original"))
        board["source_hash"] = sha256_file(source)
    transcript = tmp_path / "transcript.json"
    atomic_json(transcript, dict(workload="original"))
    source = tmp_path / d["boards"][0]["source_path"]
    atomic_json(
        source,
        dict(
            kv260_terminal_transcript_path=str(transcript),
            kv260_terminal_transcript_sha256=sha256_file(transcript),
        ),
    )
    d["boards"][0]["source_hash"] = sha256_file(source)
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, dict(systems=d["systems"]))
    primitive = tmp_path / "primitive_rows.json"
    atomic_json(primitive, dict(paired_service_rows=[pair()]))

    def fake_auth(path, eid, field, raw, ledger):
        if eid == 8081:
            return dict(board_rows=d["boards"]), True
        ref = dict(
            path=str(evidence if eid == 8085 else primitive),
            sha256=sha256_file(evidence if eid == 8085 else primitive),
        )
        return dict(raw_shard_hashes=[ref]), True

    monkeypatch.setattr(h, "authenticate", fake_auth)
    monkeypatch.setattr(h, "RESOURCES", [])
    loaded = h.load(tmp_path, raw)
    assert len(loaded["systems"]) == 64 and len(loaded["costs"]) == 1
    receipts.extend(loaded["boards"])
    (tmp_path / receipts[0]["source_path"]).unlink()
    assert not h.load(tmp_path, raw)["boards"][0]["custody_valid"]
    primitive.write_text("{}")
    assert h.load(tmp_path, raw)["costs"] == []


def test_private_real_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8108: cold and hostile children run outside the checkout."""
    script = cli.ROOT / cli.SCRIPT
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    atomic_json(source, data())
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    python = str(cli.ROOT / ".venv/bin/python")
    prefix = [python, "-u"]
    if env.get("CARNOT_8108_COVERAGE_CONFIG"):
        prefix = [
            python,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--rcfile=" + env["CARNOT_8108_COVERAGE_CONFIG"],
        ]
    receipts = []

    def run(args, expected):
        argv = prefix + [str(script), *args]
        began = time.monotonic()
        result = subprocess.run(
            argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == expected, result.stdout + result.stderr
        log = result.stdout + result.stderr
        receipts.append(
            dict(
                argv=argv,
                exit_code=result.returncode,
                normal_exit=True,
                duration_s=time.monotonic() - began,
                execution_cwd=str(tmp_path),
                log=log,
                log_sha256="sha256:" + hashlib.sha256(log.encode()).hexdigest(),
            )
        )

    run(["--worker-input", str(source), "--output", str(output)], 0)
    assert json.loads(output.read_text())["hardware_boundary_ready_score"] == 1
    run(["--cold-replay", str(output)], 0)
    mutated = json.loads(output.read_text())
    mutated["precision_rows"][0]["probability_error_bound"] = 0
    atomic_json(output, mutated)
    run(["--cold-replay", str(output)], 1)
    d = data()
    d["boards"][0]["custody_valid"] = False
    atomic_json(source, d)
    run(["--worker-input", str(source), "--output", str(output)], 0)
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    run(["--date", "wrong"], 2)
    if os.environ.get("CARNOT_8108_CLI_RECEIPTS"):
        atomic_json(Path(os.environ["CARNOT_8108_CLI_RECEIPTS"]), dict(receipts=receipts))


def test_publisher_and_validation_plan(tmp_path, monkeypatch):
    """REQ-REPORT-8108: owned failures cannot publish ready evidence."""
    monkeypatch.setattr(h, "load", lambda root, raw: data())
    plan = cli.commands(tmp_path / "scratch")
    assert {"coverage100", "strict_mypy", "full_pytest"} <= {p.name for p in plan}
    monkeypatch.setattr(cli, "commands", lambda scratch: [])
    monkeypatch.setattr(cli, "terminal", lambda path: dict(passed=True))
    output = tmp_path / "results" / f"{cli.NAME}.json"
    assert cli.main(["--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    assert cli.main(["--cold-replay", str(output)]) == 0
    broken = json.loads(output.read_text())
    broken["rows"][0]["numerator"] += 1
    atomic_json(output, broken)
    assert cli.main(["--cold-replay", str(output)]) == 1
    actual_runner = cli.run_commands

    def failed_checks(root, specs, **kwargs):
        if specs and specs[0].scope == "measurement":
            return actual_runner(root, specs, **kwargs)
        return [dict(passed=False, scope="owned")]

    monkeypatch.setattr(cli, "run_commands", failed_checks)
    assert cli.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["hardware_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    monkeypatch.setattr(cli, "run_commands", lambda *a, **kw: [dict(passed=False)])
    assert cli.main(["--output", str(output)]) == 1
    monkeypatch.setattr(cli, "run_commands", lambda *a, **kw: [dict(passed=True)])
    assert cli.terminal_commands(output)
    monkeypatch.undo()


def test_terminal_validation_and_bad_byte(tmp_path, monkeypatch):
    """REQ-REPORT-8108: terminal checks use normal exits and exact log hashes."""
    monkeypatch.setattr(cli, "run_commands", lambda *a, **kw: [dict(passed=True)])
    assert cli.terminal(tmp_path / "candidate.json")["passed"]
    candidate = tmp_path / "invalid.json"
    candidate.write_text("{}")
    assert cli.main(["--cold-replay", str(candidate)]) == 1


def test_invalid_precision_and_arms():
    """REQ-VERIFY-8108: invalid scaling and incomplete pairs earn no credit."""
    s = system()
    s["state"]["geometry"]["std"][0] = float("inf")
    s["state"]["geometry"]["mean"][0] = float("inf")
    with pytest.raises(ValueError):
        h.numerical(s, 8)
    p = pair()
    p["arms"].pop()
    assert h.service([p])["status"] == "unavailable"


def test_owned_bound_and_independent_reduction_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8108: corrupt owned arithmetic cannot pass readiness."""
    monkeypatch.setattr(
        h,
        "summarize",
        lambda rows: dict(passed=False, fallback_fraction=0, maximum_probability_error=1),
    )
    assert h.reduce(data())["verdict_class"] == "disqualified"
    monkeypatch.undo()
    monkeypatch.setattr(h, "load", lambda root, raw: data())
    monkeypatch.setattr(cli, "commands", lambda scratch: [])
    actual_runner = cli.run_commands

    def corrupt_worker(root, specs, **kwargs):
        receipts = actual_runner(root, specs, **kwargs)
        output = Path(specs[0].argv[-1])
        value = json.loads(output.read_bytes())
        value["rows"][0]["numerator"] += 1
        atomic_json(output, value)
        return receipts

    monkeypatch.setattr(cli, "run_commands", corrupt_worker)
    assert cli.main(["--output", str(tmp_path / "results" / f"{cli.NAME}.json")]) == 1
