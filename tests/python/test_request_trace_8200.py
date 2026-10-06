"""REQ-VERIFY-8200 / REQ-REPORT-8200: demand comes from authenticated requests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.verify import request_trace_8200 as e
from carnot.reporting import request_trace_execution_8200 as runner
from carnot.reporting import request_trace_inventory_8200 as custody


def record(index):
    """Synthetic private requests let chronology and identity fail independently."""
    identity = {k: "private" for k in e.IDENTITY_FIELDS}
    identity.update(seed=index, source_bytes=str(index % 24), answer_bytes="answer")
    identity["request_schema"] = e.SCHEMA
    return dict(
        call_id=f"call-{index:03}",
        issued_at=f"2026-10-06T00:{index // 60:02}:{index % 60:02}Z",
        session_id="private-session",
        source_cluster_id=f"source-{index % 24}",
        operation="generation",
        status="completed",
        attempt=1,
        condition="original",
        request={"seed": index},
        identity=identity,
    )


def test_chronology_support_and_mapping():
    """SCENARIO-VERIFY-8200: seeds stay unchanged and runtime mapping is separate."""
    rows = [record(i) for i in range(100)]
    rows[99]["identity"] = deepcopy(rows[98]["identity"])
    rows[99]["identity"]["runtime"] = "old-runtime"
    result = e.census(rows[::-1], {"runtime": "pinned"})
    assert [r["call_id"] for r in result["requests"]] == [r["call_id"] for r in rows[4:]]
    assert result["ready"] and result["independent_count"] == 24
    assert result["original_exact_repeat_frequency"] == 0
    assert result["replay_exact_repeat_frequency"] == 1 / 96
    assert result["request_shape_duplicate_count"] == 72
    assert result["reuse_distance_rows"][-1]["replay_distance"] == 1
    assert result["requests"][0]["session_id"] == "private-session"
    for field in e.IDENTITY_FIELDS:
        changed = deepcopy(rows[0])
        changed["identity"][field] = "changed"
        assert e.exact_key(changed) != e.exact_key(rows[0])


@pytest.mark.parametrize(
    "field", ["issued_at", "request", "identity", "session_id", "source_cluster_id"]
)
def test_unavailable_records(field):
    """REQ-VERIFY-8200: incomplete records cannot manufacture demand."""
    row = record(0)
    del row[field]
    result = e.census([row], {})
    assert not result["ready"] and not result["requests"]
    assert result["exclusion_rows"][0]["exclusion_reason"]


@pytest.mark.parametrize(
    "change",
    [
        dict(operation="model_load"),
        dict(status="failed"),
        dict(attempt=2),
        dict(historical=True),
        dict(condition="forced_refresh"),
        dict(condition="changed_source"),
        dict(call_id="warmup-1"),
        dict(arm="exact_repeat"),
    ],
)
def test_exclusions(change):
    """SCENARIO-VERIFY-8200: constructed traffic never enters the trace."""
    row = record(0)
    row.update(change)
    assert not e.census([row], {})["requests"]


def test_empty_invalid_time_schema_and_ties():
    """REQ-VERIFY-8200: unsupported schemas and chronology block honestly."""
    assert e.census([], {})["original_exact_repeat_frequency"] is None
    row = record(0)
    row["issued_at"] = "not-a-time"
    assert not e.census([row], {})["requests"]
    row["issued_at"] = "2026-10-06T00:00:00"
    assert not e.census([row], {})["requests"]
    rows = [record(i) for i in range(96)]
    for r in rows:
        r["identity"]["request_schema"] = e.SCHEMA
    rows[1]["issued_at"] = rows[0]["issued_at"]
    assert e.census(rows[::-1], {})["requests"][0]["call_id"] == "call-000"
    rows[0]["identity"]["request_schema"] = "unsupported"
    assert not e.census(rows, {})["ready"]


def test_direct_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8200: E2E-015/019 external CLI, missing input and tamper."""
    source = tmp_path / "input.json"
    rows = [record(i) for i in range(96)]
    for row in rows:
        row["identity"]["request_schema"] = e.SCHEMA
    source.write_text(json.dumps(dict(records=rows, replay_identity={"runtime": "pinned"})))
    output = tmp_path / (e.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def run(args):
        return subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )

    result = run(["--private-input", str(source), "--output", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["request_trace_ready_score"] == 1 and value["MODEL_SPECS"] == []
    assert run(["--cold-replay", str(output)]).returncode == 0
    value["original_exact_repeat_frequency"] = 0.5
    output.write_text(json.dumps(value))
    assert run(["--cold-replay", str(output)]).returncode == 1
    value["original_exact_repeat_frequency"] = 0
    output.write_text(json.dumps(value))
    Path(value["trace_path"]).chmod(0o600)
    Path(value["trace_path"]).write_text("{}")
    assert run(["--cold-replay", str(output)]).returncode == 1
    missing = run(["--private-input", str(tmp_path / "missing.json"), "--output", str(output)])
    assert missing.returncode == 0, missing.stdout + missing.stderr
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_custody_and_owned_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8200: authority bytes, failed operands and owned exits stay visible."""
    import runpy
    import yaml

    root = tmp_path / "repo"
    root.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    assert not custody.authorities(root, raw)["checks"][0]["passed"]
    authority = root / "research-complete.yaml"
    authority.write_text(
        yaml.safe_dump(
            dict(
                milestones=[
                    dict(id="2026.10.700", tasks=[]),
                    dict(
                        id="2026.10.707",
                        tasks=[
                            dict(
                                id="exp8188-exact-request-service", deliverable="results/panel.json"
                            )
                        ],
                    ),
                ]
            )
        )
    )
    docs = root / "openspec/change-proposals"
    docs.mkdir(parents=True)
    (docs / "research-roadmap-v707-preserved-private.md").write_text("immutable old methods")
    raw2 = tmp_path / "raw2"
    raw2.mkdir()
    a = custody.authorities(root, raw2)
    assert len(a["tasks"]) == 1
    empty = custody.inventory(root, raw2, a)
    assert not empty["checks"][-1]["passed"]
    input_data = root / "input_data.json"
    binary = root / "runtime.bin"
    binary.write_bytes(b"private-runtime")
    e.atomic_json(
        input_data,
        dict(
            runtime_freeze=dict(tokenizer_sha256="tokenizer", runtime=e.reference(binary)),
            protocol=dict(
                gguf_sha256="gguf",
                model_revision="revision",
                runtime_sha256=e.sha256_file(binary),
                chat_template_sha256="template",
            ),
        ),
    )
    panel = root / "results/panel.json"
    methods = root / "measurement_code/private.py"
    methods.parent.mkdir()
    methods.write_text("# frozen private service methods\n")
    e.atomic_json(
        panel,
        dict(
            experiment_id=8188,
            call_ledger=[record(0)],
            raw_shard_hashes=[e.reference(input_data)],
            source_artifact_hashes=[e.reference(methods)],
        ),
    )
    raw3 = tmp_path / "raw3"
    raw3.mkdir()
    found = custody.inventory(root, raw3, a)
    assert found["replay_identity"]["runtime"] == e.sha256_file(binary)
    assert any(r["sha256"] == e.sha256_file(methods) for r in found["refs"])
    assert found["inventory_rows"][0]["issued_at"] == record(0)["issued_at"]
    copied = custody.copy_bytes(panel, raw3)
    assert custody.copy_bytes(panel, raw3) == copied
    assert not Path(copied["frozen_path"]).stat().st_mode & 0o222
    with pytest.raises(FileExistsError):
        e.seal(raw3 / "inventory.json", {})
    binary.unlink()
    raw4 = tmp_path / "raw4"
    raw4.mkdir()
    assert not custody.inventory(root, raw4, a)["checks"][-1]["passed"]
    input_data.write_text("{}")
    raw5 = tmp_path / "raw5"
    raw5.mkdir()
    assert not custody.inventory(root, raw5, a)["checks"][-1]["passed"]
    log = tmp_path / "validation.log"
    log.write_text("normal private subprocess exit\n")
    receipt = dict(
        passed=True, normal_exit=True, exit_code=0, log_path=str(log), log_sha256=e.sha256_file(log)
    )
    monkeypatch.setattr(runner, "execute", lambda plan, raw: [dict(receipt)])
    assert runner.run_checks(runner.validators(panel), tmp_path)[0]["actual_exit"] == 0
    monkeypatch.setattr(runner, "run_checks", lambda plan, raw: [dict(receipt)])
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    health = tmp_path / "health.json"
    e.atomic_json(health, dict(receipts=[receipt]))
    (root / "ops").mkdir()
    (root / "ops/exclusion_manifest.yaml").write_text("retired: []")
    assert (
        runner.main(
            [
                "--root",
                str(root),
                "--output",
                str(output),
                "--repository-health-receipt",
                str(health),
            ]
        )
        == 0
    )
    health.unlink()
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: e.atomic_json(health, dict(receipts=[receipt]))
    )
    assert (
        runner.main(
            [
                "--root",
                str(root),
                "--output",
                str(output),
                "--repository-health-receipt",
                str(health),
            ]
        )
        == 0
    )
    failed = dict(receipt, passed=False, exit_code=1)
    monkeypatch.setattr(
        runner,
        "run_checks",
        lambda plan, raw: [dict(failed)] if raw.name != "publication" else [dict(receipt)],
    )
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert json.loads(output.read_text())["request_trace_ready_score"] == 0
    monkeypatch.setattr(runner, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert exit_info.value.code == 0


def test_replay_mutations_and_inprocess_private(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8200: rehashed drift and missing sealed bytes fail cold replay."""
    source = tmp_path / "fixture.json"
    monkeypatch.setenv(
        "CARNOT_8200_REPOSITORY_HEALTH_RECEIPT", str(tmp_path / "unused-private-health")
    )
    source.write_text(json.dumps(dict(records=[record(i) for i in range(96)], replay_identity={})))
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--private-input", str(source), "--output", str(output)]) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    assert e.replay(output)
    value["original_exact_repeat_frequency"] = 0.5
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    value["original_exact_repeat_frequency"] = 0
    value["raw_shard_hashes"][0]["sha256"] = "forged"
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    value["raw_shard_hashes"][0]["sha256"] = e.sha256_file(
        Path(value["raw_shard_hashes"][0]["path"])
    )
    value["request_trace_ready_score"] = 0
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    value["request_trace_ready_score"] = 1
    value["trace_sha256"] = "forged"
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    value["trace_sha256"] = e.sha256_file(Path(value["trace_path"]))
    value["source_artifact_hashes"][0]["sha256"] = "forged"
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    output.write_text("{}")
    assert runner.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--private-input", str(source)])
    missing = tmp_path / "missing" / (e.NAME + ".json")
    assert (
        runner.main(["--private-input", str(tmp_path / "absent.json"), "--output", str(missing)])
        == 0
    )
