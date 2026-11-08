"""REQ-REPORT-8934; SCENARIO-REPORT-8934-CONTRACT: pair graphs, never answers."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from carnot.reporting import depth_control_8934 as d
from carnot.inference import depth_runtime_8934 as runtime
from carnot.reporting import graph_source_8931 as s
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def response(answer):
    """Private emitted-token evidence tests capture without claiming model work."""
    return dict(
        id="fixture",
        model=s.SETTINGS["model"],
        choices=[
            dict(
                message=dict(content=answer),
                logprobs=dict(
                    content=[
                        dict(
                            token=answer,
                            logprob=-0.2,
                            top_logprobs=[
                                dict(token=answer, logprob=-0.2),
                                dict(token="??", logprob=-2),
                            ],
                        )
                    ]
                ),
            )
        ],
        usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101),
    )


@pytest.fixture(scope="module")
def units():
    """REQ-REPORT-8934: use the sealed cohort instead of generating extra units."""
    evidence = d.authenticate(d.ROOT)
    assert evidence["failures"] == []
    return d.ordered_units(evidence["protocol"])


def test_authentication_and_order(units, tmp_path):
    """REQ-REPORT-8934: unavailable inputs block; each graph has two stateless queries."""
    assert len(units) == 100
    assert len({u["seed"] for u in units}) == 50
    assert sum(units[i]["k"] == 2 for i in range(0, 100, 2)) == 25
    assert all(units[i]["cluster_id"] == units[i + 1]["cluster_id"] for i in range(0, 100, 2))
    assert all(u["shard"] == "control" for u in units)
    assert d.authenticate(tmp_path)["failures"]
    for name, pin, field, task in d.INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    failures = d.authenticate(tmp_path)["failures"]
    assert any(f["observed"] == "missing_field" for f in failures)


def test_reduction_positive_null_and_censoring(units):
    """SCENARIO-REPORT-8934-CONTRACT: exact discordance and ceiling diagnostics."""
    rows = [
        s.label_response(u, response(u["expected"][0] if u["k"] == 2 else u["start"]))
        for u in units
    ]
    result = d.reduce(units, rows)
    assert result["depth_control"]["accepted"]
    assert result["discordant_pairs"] == dict(
        k2_correct_k3_wrong=50, k2_wrong_k3_correct=0, both_correct=0, both_wrong=0
    )
    assert result["paired_interval"]["mean_difference"] == 1
    assert len(result["paired_depth_rows"]) == 50
    ceiling = [s.label_response(u, response(u["start"])) for u in units]
    null = d.reduce(units, ceiling)
    assert not null["depth_control"]["accepted"]
    assert null["null_inspection"]["ceiling_k2"]
    assert null["error_rate_k2"] == null["error_rate_k3"] == 1
    ceiling[-1] = s.censored(units[-1], "timeout")
    censored = d.reduce(units, ceiling)
    assert censored["paired_depth_rows"][-1]["difference"] is None
    assert censored["completed_count"] == 99
    with pytest.raises(ValueError, match="row_roster"):
        d.reduce(units, rows[:-1])
    bad = deepcopy(rows)
    bad[0]["value"] = 99
    assert not d.reduce(units, bad)["depth_control"]["labels_verified"]


def test_runtime_owned_transport(units, tmp_path, monkeypatch):
    """REQ-REPORT-8934: preserve live usage, bind template and always release the lease."""
    model = dict(
        model_path=str(tmp_path / "model.gguf"),
        hf_id=s.SETTINGS["model"],
        sha256=canonical_hash("model"),
    )
    Path(model["model_path"]).write_bytes(b"fixture")
    gpu = dict(index=0, uuid="GPU-0", used_mb=0, free_mb=24000)
    monkeypatch.setattr(runtime.p, "preflight", lambda _: dict(model=model, gpu=gpu))
    monkeypatch.setattr(runtime.p, "gpu_state", lambda: gpu)
    lease = Mock()
    lease.owner_receipt.return_value = dict(token="fixture")
    lease.release.return_value = dict(released=True)
    monkeypatch.setattr(runtime.GpuLease, "acquire", lambda **_: lease)
    loaded = dict(
        authenticated=True,
        gguf_sha256=model["sha256"],
        offload_layers=[65, 65],
        props=dict(chat_template="fixture"),
    )
    server = Mock()
    server.load.return_value = loaded
    server.close.return_value = dict(leak_free=True)
    monkeypatch.setattr(runtime, "QwenRuntime", lambda *_: server)
    monkeypatch.setattr(runtime, "wire_response", lambda *args: response(units[0]["start"]))
    result = runtime.run(units[:2], tmp_path / "run")
    assert result["counts"]["generation_calls_completed"] == 2
    assert result["resource_lease"]["release"]["released"]
    assert result["settings"]["chat_template_sha256"] == canonical_hash("fixture")
    assert result["rows"][0]["actual_token_usage"]["completion_tokens"] == 1
    assert result["live"] and not result["owned_failure"]
    server.load.side_effect = TimeoutError("load_deadline")
    failed = runtime.run(units[:2], tmp_path / "failed")
    assert failed["blocker"] == "load_deadline"
    assert failed["counts"]["model_loads_failed"] == 1
    assert lease.release.call_count == 2


def test_wire_response(tmp_path, monkeypatch):
    """REQ-REPORT-8934: retain the exact HTTP response bytes, not invented confidence."""
    from io import BytesIO

    body = b'{"usage": {"completion_tokens": 1}}\n'
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **kw: BytesIO(body))
    assert runtime.wire_response(8000, {}, 1, tmp_path / "wire.json") == json.loads(body)
    assert (tmp_path / "wire.json").read_bytes() == body


def test_authentication_mutations(tmp_path, monkeypatch):
    """REQ-REPORT-8934: distinguish absent, zero, malformed and rejected producers."""
    monkeypatch.setattr(d.methods, "authenticate", lambda _: dict(failures=[]))
    for name, pin, field, task in d.INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        for content, reason in [
            ("[]", "rejected_producer"),
            ("oops", "rejected_producer"),
            (json.dumps({field: 0}), "observed_zero"),
            (json.dumps({field: True}), "rejected_producer"),
        ]:
            path.write_text(content)
            assert any(
                r["artifact_field"] == field and r["observed"] == reason
                for r in d.authenticate(tmp_path)["failures"]
            )
        value = dict(task_id=task, required_checks_passed=True, **{field: 1})
        atomic_json(path, value)
        monkeypatch.setattr(
            d.methods, "reference", lambda p, pin=pin: dict(path=str(p), sha256=pin)
        )
        monkeypatch.setattr(d.methods, "replay", lambda _: dict(passed=False))
        monkeypatch.setattr(d.audit, "replay", lambda _: dict(passed=False))
        assert any(
            r["observed"] == "rejected_producer" for r in d.authenticate(tmp_path)["failures"]
        )


@pytest.mark.parametrize(
    "fault",
    [
        "preflight",
        "busy",
        "offload",
        "transport",
        "usage",
        "identity",
        "signals",
        "cleanup",
        "lease",
    ],
)
def test_runtime_failures(units, tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8934-CONTRACT: failures retain slots and release owned resources."""
    gpu = dict(index=0, uuid="GPU-0", used_mb=0, free_mb=24000)
    model = dict(
        model_path=str(tmp_path / "model"),
        hf_id=s.SETTINGS["model"],
        sha256=canonical_hash("model"),
    )
    monkeypatch.setattr(
        runtime.p,
        "preflight",
        Mock(side_effect=RuntimeError("cache_missing"))
        if fault == "preflight"
        else lambda _: dict(model=model, gpu=gpu),
    )
    monkeypatch.setattr(
        runtime.p, "gpu_state", lambda: dict(gpu, used_mb=2000) if fault == "busy" else gpu
    )
    lease = Mock()
    lease.owner_receipt.return_value = {}
    lease.release.return_value = dict(released=True)
    if fault == "lease":
        lease.release.side_effect = runtime.LeaseError("private_release_failure")
    monkeypatch.setattr(runtime.GpuLease, "acquire", lambda **_: lease)
    server = Mock()
    server.load.return_value = dict(
        authenticated=True,
        gguf_sha256=model["sha256"],
        offload_layers=[0, 65] if fault == "offload" else [65, 65],
        props=dict(chat_template="fixture"),
    )
    server.close.return_value = dict(leak_free=True)
    if fault == "cleanup":
        server.close.side_effect = OSError("close_failed")
    monkeypatch.setattr(runtime, "QwenRuntime", lambda *_: server)

    def transport(*args):
        if fault == "transport":
            raise TimeoutError("timeout")
        raw = response(units[0]["start"])
        if fault == "usage":
            raw.pop("usage")
        if fault == "identity":
            raw["model"] = "foreign"
        if fault == "signals":
            raw["choices"][0].pop("logprobs")
        return raw

    monkeypatch.setattr(runtime, "wire_response", transport)
    result = runtime.run(units[:2], tmp_path / "runtime")
    assert len(result["rows"]) == 2
    assert result["owned_failure"] == (fault in {"cleanup", "identity", "lease"})
    if fault != "preflight":
        assert lease.release.call_count == 1
    if fault == "signals":
        assert result["blocker"] == "unsupported_logprobs_capability"
        assert result["rows"][0]["signal_status"] == "blocked"
        assert result["counts"]["generation_calls_attempted"] == 1


def test_cli_and_cold_replay(tmp_path):
    """REQ-REPORT-8934: actual child CLI runs outside the checkout without PYTHONPATH."""
    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]

    def child(args):
        return subprocess.run(
            command + args, env=env, cwd="/tmp", capture_output=True, text=True, timeout=60
        )

    result = child(["--date", "20261108", "--fixture-e2e", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["depth_source_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["completed_count"] = 100
    value["reproducibility_checksum"] = d.pilot.checksum(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode == 2


def test_build_controls(units, tmp_path, monkeypatch):
    """REQ-REPORT-8934: a valid null, blocked capability and failed owned check differ."""
    from carnot.reporting import depth_control_runner_8934 as runner

    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    validation = dict(receipts=[], coverage={})
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d, "qualified", lambda _: True)
    assert d.build(work, validation, False)["verdict_class"] == "null"
    work["runtime"]["rows"] = [
        s.label_response(u, response(u["expected"][0] if u["k"] == 2 else u["start"]))
        for u in units
    ]
    work["runtime"]["live"] = True
    assert d.build(work, validation, False)["depth_source_ready_score"] == 1
    work["authentication"]["failures"] = [dict(observed="missing_file")]
    assert d.build(work, validation, False)["verdict_class"] == "blocked"
    assert runner.replay(tmp_path / "missing")["passed"] is False


@pytest.mark.parametrize("rejected", [False, True, "unexpected"])
def test_main_validation_paths(tmp_path, monkeypatch, rejected):
    """SCENARIO-REPORT-8934-CONTRACT: private mocks test publication, never research."""
    from carnot.reporting import depth_control_runner_8934 as runner

    real_root, original = d.ROOT, runner.freeze

    def private_freeze(root, raw, fixture):
        with monkeypatch.context() as context:
            context.setattr(d, "ROOT", real_root)
            return original(real_root, raw, True)

    monkeypatch.setattr(runner, "freeze", private_freeze)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d, "qualified", lambda _: True)
    original_plan = runner.validation_plan

    def plan(private):
        commands = original_plan(private)
        (private / ".coverage").write_bytes(b"private mocked coverage; never research")
        return commands

    monkeypatch.setattr(runner, "validation_plan", plan)
    monkeypatch.setattr(runner, "measured", lambda *a: {})

    def commands(root, specs, **kwargs):
        return [
            dict(
                name=s.name,
                passed=not (rejected is True and s.name == "cold_replay_cli"),
                exit_code=1 if rejected is True and s.name == "cold_replay_cli" else 0,
            )
            for s in specs
        ]

    monkeypatch.setattr(runner, "run_commands", commands)
    if rejected == "unexpected":
        monkeypatch.setattr(runner, "publish_primary", Mock(side_effect=ValueError("unexpected")))
        with pytest.raises(ValueError, match="unexpected"):
            runner.main(["--date", "20261108"])
        return
    assert runner.main(["--date", "20261108"]) == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("disqualified" if rejected else "null")
    assert value["depth_source_ready_score"] == 0


def test_replay_raw_labels_and_tampering(units, tmp_path, monkeypatch):
    """REQ-REPORT-8934: replay recomputes labels and rejects rehashed aggregate drift."""
    from carnot.reporting import depth_control_runner_8934 as runner

    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    raw = Path(work["raw"])
    model = dict(hf_id=s.SETTINGS["model"], sha256=canonical_hash("private"))
    rows = s.collect(
        units, lambda *a: response(units[0]["start"]), raw / "runtime" / "collection", model
    )
    for row in rows:
        row["actual_token_usage"] = dict(prompt_tokens=100, completion_tokens=1, total_tokens=101)
    work["runtime"]["rows"] = rows
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    path = tmp_path / "candidate.json"
    value = d.build(work, validation, True)
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    row = work["runtime"]["rows"][0]
    row["value"] = 1 - row["value"]
    atomic_json(raw / "work.json", work)
    value["raw_shard_hashes"][0] = d.methods.reference(raw / "work.json")
    atomic_json(path, value)
    assert not runner.replay(path)["passed"]
    work["units"][0]["k"] = 99
    atomic_json(raw / "work.json", work)
    value["raw_shard_hashes"][0] = d.methods.reference(raw / "work.json")
    atomic_json(path, value)
    assert not runner.replay(path)["passed"]
    (raw / "work.json").write_text("changed bytes")
    assert not runner.replay(path)["passed"]


def test_freeze_nonfixture(units, tmp_path, monkeypatch):
    """REQ-REPORT-8934: authenticated live route uses the shared bounded runtime."""
    from carnot.reporting import depth_control_runner_8934 as runner

    def fake_runtime(scheduled, raw):
        result = dict(
            rows=[s.censored(u, "private") for u in scheduled], counts={}, checks=[], live=False
        )
        atomic_json(raw / "runtime.json", result)
        return result

    monkeypatch.setattr(runner.runtime, "run", fake_runtime)
    assert runner.freeze(d.ROOT, tmp_path / "raw", False)["runtime"]["live"] is False


def test_qualification_is_owned(monkeypatch):
    """REQ-REPORT-8934: require all receipts and measurement of this producer's statements."""
    receipts = [dict(name=name, passed=True, exit_code=0) for name in d.methods.CHECKS]
    monkeypatch.setattr(d, "verify_measured", lambda coverage, owned: owned == d.OWNED)
    assert d.qualified(dict(receipts=receipts, coverage={}))
    assert not d.qualified(dict(receipts=receipts[:-1], coverage={}))
    receipts[0]["passed"] = False
    assert not d.qualified(dict(receipts=receipts, coverage={}))
