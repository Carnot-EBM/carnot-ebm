"""REQ-REPORT-8220 / REQ-VERIFY-8220: current custody cannot repair history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v711_current_contract as q
from carnot.reporting import v711_current_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(root):
    """Use copied real inputs privately so fixtures never rewrite research outputs."""
    root.mkdir(parents=True, exist_ok=True)
    for name in q.INPUTS:
        source = q.ROOT / name
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def cli(parent, *args):
    """Real script imports and exits must work outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(q.ROOT / q.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_authority(tmp_path):
    """SCENARIO-REPORT-8220-CONTRACT: all independent authority operands matter."""
    root = fixture(tmp_path / "root")
    d, s, a = [root / p for p in [q.DESIGN, q.STAGED, q.ACTIVE]]
    valid = q.assess(d, s, a, tmp_path / "valid")
    assert valid["activated"] and len(valid["contract_rows"]) == 14
    controls = q.mutations(d, a, tmp_path / "controls")
    assert {r["control"] for r in controls} == {
        "count",
        "order",
        "prompt",
        "title",
        "gate",
        "model",
        "digest",
    }
    assert all(r["rejected"] for r in controls)
    s.write_bytes(a.read_bytes())
    assert q.assess(d, s, a, tmp_path / "both")["planning_matched"]
    a.unlink()
    staged = q.assess(d, s, a, tmp_path / "staged")
    assert staged["planning_matched"] and not staged["activated"]
    assert not q.assess(tmp_path / "missing", s, a, tmp_path / "missing")["activated"]


def test_history_and_precedence(tmp_path):
    """REQ-REPORT-8220: historical failures do not count as current owned failures."""
    root = fixture(tmp_path / "root")
    work = q.measure(root, tmp_path / "raw")
    receipt = dict(name="actual_fixture", passed=True, exit_code=0, timed_out=False)
    value = q.reduce(work, [receipt])
    assert value["current_contract_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_count"] == value["generalized_learning_benefit_score"] == 0
    assert len(value["historical_dispositions"]) == 2
    assert (
        value["historical_dispositions"][0]["honest_verdict"]
        == "complete_disqualified_original_code_bytes"
    )
    assert value["historical_dispositions"][0]["gate_check_summary"]
    assert value["historical_dispositions"][1]["utility_protocol_ready_score"] == 1
    assert value["historical_dispositions"][1]["H1"]["measured_here"] is False
    assert value["historical_dispositions"][1]["H2"]["measured_here"] is False
    assert len(value["unexecuted_v710_design_entries"]) == 12
    assert [r["experiment_id"] for r in value["unexecuted_v710_design_entries"]] == list(
        range(8220, 8232)
    )
    work["failures"].append(q.failure(root / "missing", "exists", True, None))
    assert q.reduce(work, [receipt])["verdict_class"] == "blocked"
    assert q.reduce(work, [dict(receipt, passed=False)])["current_contract_ready_score"] == 0
    assert q.reduce(work, [])["verdict_class"] == "disqualified"
    assert all(c["passed"] for c in work["receipt_controls"])


def test_real_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8220-CLI: private valid and blocked publication and cold replay."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (q.NAME + ".json")
    run = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["current_contract_ready_score"] == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key in [
        "rows",
        "current_contract_ready_score",
        "historical_dispositions",
        "unexecuted_v710_design_entries",
    ]:
        changed = deepcopy(value)
        changed[key] = [] if isinstance(changed[key], list) else 9
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    Path(value["work_reference"]["path"]).write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert cli(tmp_path, "--date", "20261006").returncode == 2
    absent = tmp_path / "absent"
    absent.mkdir()
    blocked = tmp_path / "blocked" / (q.NAME + ".json")
    run = cli(tmp_path, "--root", absent, "--output", blocked, "--private-fixture")
    assert run.returncode == 0, run.stdout + run.stderr
    v = json.loads(blocked.read_bytes())
    assert v["verdict_class"] == "blocked" and v["current_contract_ready_score"] == 0
    assert v["gate_check_summary"][0]["observed"] is None
    assert v["completed_count"] == 0 and v["censored_count"] == 14
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0


def test_schema_and_missing_design(tmp_path):
    """REQ-REPORT-8220: malformed external inputs block without replacing their bytes."""
    root = fixture(tmp_path / "root")
    (root / q.UPSTREAM[0]).write_text("{")
    work = q.measure(root, tmp_path / "invalid")
    assert any(r["artifact_field"] == "historical_schema" for r in work["failures"])
    (root / q.HISTORY).write_text("incomplete preserved design")
    work = q.measure(root, tmp_path / "incomplete")
    assert any(r["artifact_field"] == "unexecuted_design_count" for r in work["failures"])
    (root / q.DESIGN).write_text("missing contract")
    assert not q.measure(root, tmp_path / "bad-contract")["contract"]["activated"]


def test_controls_fail_closed(tmp_path, monkeypatch):
    """REQ-VERIFY-8220: a permissive reader control cannot turn into readiness."""
    root = fixture(tmp_path / "root")
    work = q.measure(root, tmp_path / "raw")
    monkeypatch.setattr(q, "read_bound_sidecar", lambda *a: {"primary_sha256": "bad"})
    controls = q.receipt_controls(tmp_path / "broken-reader")
    assert not any(r["passed"] for r in controls)
    work["receipt_controls"] = controls
    value = q.reduce(work, [dict(name="owned", passed=True, exit_code=0, timed_out=False)])
    assert value["verdict_class"] == "disqualified" and value["current_contract_ready_score"] == 0


def test_owned_plan_and_real_main(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8220-REPLAY: real child paths qualify ownership and health separation."""
    plan = runner.commands(tmp_path / "plan")
    assert q.CLI in json.dumps(plan) and "--fail-under=100" in json.dumps(plan)
    root = fixture(tmp_path / "root")
    output = tmp_path / (q.NAME + ".json")

    def bounded(private):
        argv = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}'); raise SystemExit(3)",
            str(private / "coverage.json"),
        ]
        return [
            dict(
                name="health_control", argv=argv, expected=0, deadline=10, scope="repository_health"
            )
        ]

    monkeypatch.setattr(runner, "commands", bounded)
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["current_contract_ready_score"] == 1
    assert not value["repository_health"]["receipts"][0]["passed"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    original = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    changed = deepcopy(original)
    changed["contract"]["activated"] = False
    atomic_json(Path(value["work_reference"]["path"]), changed)
    changed_value = deepcopy(value)
    from carnot.reporting.current_work_receipt import sha256_file

    changed_value["work_reference"]["sha256"] = sha256_file(Path(value["work_reference"]["path"]))
    for ref in changed_value["raw_shard_hashes"]:
        if ref["path"] == value["work_reference"]["path"]:
            ref["sha256"] = changed_value["work_reference"]["sha256"]
    atomic_json(output, changed_value)
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        runner.replay(output)
    changed_value.update(q.reduce(changed, value["validation_receipts"]))
    atomic_json(output, changed_value)
    with pytest.raises(ValueError, match="authority_reduction_drift"):
        runner.replay(output)
    changed = deepcopy(original)
    changed["historical_dispositions"].pop()
    atomic_json(Path(value["work_reference"]["path"]), changed)
    changed_value = deepcopy(value)
    digest = sha256_file(Path(value["work_reference"]["path"]))
    changed_value["work_reference"]["sha256"] = digest
    for ref in changed_value["raw_shard_hashes"]:
        if ref["path"] == value["work_reference"]["path"]:
            ref["sha256"] = digest
    changed_value.update(q.reduce(changed, value["validation_receipts"]))
    atomic_json(output, changed_value)
    with pytest.raises(ValueError, match="historical_reduction_drift"):
        runner.replay(output)
    atomic_json(Path(value["work_reference"]["path"]), original)
    atomic_json(output, value)
    log = Path(value["validation_receipts"][0]["stdout_path"])
    log.write_text("altered command stream")
    value["raw_shard_hashes"] = [
        ref for ref in value["raw_shard_hashes"] if ref["path"] != str(log)
    ]
    atomic_json(output, value)
    with pytest.raises(ValueError):
        runner.replay(output)


def test_design_digest(tmp_path):
    """SCENARIO-REPORT-8220-CONTRACT: embedded prompts must match their stated digest."""
    root = fixture(tmp_path / "root")
    path = root / q.DESIGN
    text = path.read_text()
    start = text.index("```json", text.index("V711_TASK_CONTRACT_START")) + len("```json")
    end = text.index("```", start)
    machine = json.loads(text[start:end])
    machine["tasks"][0]["prompt"] += " changed embedded prompt"
    path.write_text(text[:start] + "\n" + json.dumps(machine) + "\n" + text[end:])
    result = q.assess(path, root / q.STAGED, root / q.ACTIVE, tmp_path / "authority")
    assert not result["activated"]
    assert any(r["artifact_field"] == "design_tasks_sha256" for r in result["gate_check_summary"])


def test_terminal_operand_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8220-CLI: a publisher cannot substitute the frozen candidate."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (q.NAME + ".json")

    def drift(output, value, validator):
        return validator(tmp_path / "wrong_candidate")

    monkeypatch.setattr(runner, "publish_primary", drift)
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1
