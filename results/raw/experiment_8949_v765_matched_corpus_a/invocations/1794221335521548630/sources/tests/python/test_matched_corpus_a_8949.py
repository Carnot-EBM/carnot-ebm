"""REQ-REPORT-8949; SCENARIO-REPORT-8949-CONTRACT: first-response shard custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from carnot.reporting import matched_corpus_a_8949 as d
from carnot.reporting import matched_corpus_runner_8949 as runner
from carnot.inference import matched_corpus_runtime_8949 as runtime
from carnot.reporting import graph_source_8931 as s
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from test_matched_depth_control_8948 import response


@pytest.fixture(scope="module")
def units():
    """REQ-REPORT-8949: use only the authenticated frozen shard A."""
    evidence = d.authenticate(d.ROOT)
    assert evidence["failures"] == []
    return d.ordered_units(evidence["protocol"])


def test_roster_and_missing_inputs(units, tmp_path):
    """REQ-REPORT-8949: retain exact roles and independently check equal opportunity."""
    assert len(units) == len({u["graph_hash"] for u in units}) == 160
    assert all(u["shard"] == "A" and u["k"] == 2 for u in units)
    assert d.reduce(units, [s.censored(u, "private") for u in units])["split_counts"] == dict(
        train=80, calibration=40, test=40
    )
    assert d.authenticate(tmp_path)["failures"]
    with pytest.raises(ValueError, match="shard_roster"):
        d.ordered_units(dict(graph_manifest=[]))
    changed = deepcopy(units)
    changed[0]["role"] = "control"
    with pytest.raises(ValueError, match="shard_roster"):
        d.reduce(changed, [s.censored(u, "private") for u in changed])


def test_reduction_labels_and_feature_isolation(units):
    """SCENARIO-REPORT-8949-CONTRACT: labels never enter the detector sidecar."""
    rows = [s.label_response(u, response(u["expected"][0])) for u in units]
    result = d.reduce(units, rows)
    assert result["completed_count"] == result["independent_count"] == 160
    assert result["labels_verified"] and result["usable_signals"]
    assert sum(r["numerator"] for r in result["rows"]) == 0
    assert all(set(r) == set(s.FEATURE_SCHEMA["learned_features"]) for r in result["detector_view"])
    changed = deepcopy(rows)
    changed[0]["value"] = 1
    assert not d.reduce(units, changed)["labels_verified"]
    assert d.reduce(units, changed)["detector_view"] == result["detector_view"]
    changed[0]["features"] = None
    assert not d.reduce(units, changed)["usable_signals"]
    rows[-1] = s.censored(units[-1], "timeout")
    assert d.reduce(units, rows)["censored_count"] == 1
    with pytest.raises(ValueError, match="row_roster"):
        d.reduce(units, rows[:-1])


def test_cli_cold_replay_and_rehashed_tamper(tmp_path):
    """REQ-REPORT-8949: real private CLI, cold cwd, no model and no public writes."""
    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]

    def child(args):
        return subprocess.run(
            command + args, env=env, cwd="/tmp", capture_output=True, text=True, timeout=90
        )

    result = child(["--date", "20261109", "--fixture-e2e", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["shard_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["completed_count"] = 160
    value["reproducibility_checksum"] = d.pilot.checksum(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode == 2


def test_build_verdicts_and_raw_gate(units, tmp_path, monkeypatch):
    """REQ-REPORT-8949: all-correct shards qualify; missing owned receipts disqualify."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    validation = dict(receipts=[], coverage={})
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d, "qualified", lambda _: True)
    assert d.build(work, validation, False)["shard_ready_score"] == 0
    work["runtime"]["rows"] = [s.label_response(u, response(u["expected"][0])) for u in units]
    work["runtime"]["live"] = True
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d, "raw_complete", lambda *args: True)
    value = d.build(work, validation, False)
    assert value["shard_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive" and value["verifier_is_oracle"]
    work["authentication"]["failures"] = [dict(observed="missing_file")]
    assert d.build(work, validation, False)["verdict_class"] == "blocked"
    assert not runner.replay(tmp_path / "missing")["passed"]


def test_runtime_owned_transport(units, tmp_path, monkeypatch):
    """REQ-REPORT-8949: preserve live usage, bind template and always release the lease."""
    model = dict(
        model_path=str(tmp_path / "model.gguf"),
        hf_id=s.SETTINGS["model"],
        sha256=canonical_hash("model"),
    )
    Path(model["model_path"]).write_bytes(b"fixture")
    gpu = dict(index=0, uuid="GPU-0", used_mb=0, free_mb=24000)
    monkeypatch.setattr(runtime, "preflight", lambda _: dict(model=model, gpu=gpu))
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
    """SCENARIO-REPORT-8949-CONTRACT: failures retain slots and release owned resources."""
    gpu = dict(index=0, uuid="GPU-0", used_mb=0, free_mb=24000)
    model = dict(
        model_path=str(tmp_path / "model"),
        hf_id=s.SETTINGS["model"],
        sha256=canonical_hash("model"),
    )
    monkeypatch.setattr(
        runtime,
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


@pytest.mark.parametrize("rejected", [False, True, "unexpected"])
def test_main_validation_paths(tmp_path, monkeypatch, rejected):
    """SCENARIO-REPORT-8949-CONTRACT: private mocks test publication, never research."""
    from carnot.reporting import matched_corpus_runner_8949 as runner

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
            runner.main(["--date", "20261109"])
        return
    assert runner.main(["--date", "20261109"]) == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("disqualified" if rejected else "null")
    assert value["shard_ready_score"] == 0


def test_replay_raw_labels_and_tampering(units, tmp_path, monkeypatch):
    """REQ-REPORT-8949: replay recomputes labels and rejects rehashed aggregate drift."""
    from carnot.reporting import matched_corpus_runner_8949 as runner

    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    raw = Path(work["raw"])
    model = dict(hf_id=s.SETTINGS["model"], sha256=canonical_hash("private"))
    rows = s.collect(
        units, lambda *a: response(units[0]["start"]), raw / "runtime" / "collection", model
    )
    for row in rows:
        row["actual_token_usage"] = dict(prompt_tokens=100, completion_tokens=1, total_tokens=101)
    work["runtime"]["rows"] = rows
    rebuilt = d.reduce(units, rows)
    atomic_json(raw / "evaluation_labels.json", dict(rows=rebuilt["label_sidecar"]))
    atomic_json(raw / "detector_view.json", dict(rows=rebuilt["detector_view"]))
    work["raw_references"] = [
        d.methods.reference(Path(ref["path"])) for ref in work["raw_references"]
    ]
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    path = tmp_path / "candidate.json"
    value = d.build(work, validation, True)
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    assert runner.main(["--cold-replay", str(path)]) == 0
    detector = raw / "detector_view.json"
    original_detector = detector.read_bytes()
    atomic_json(detector, dict(rows=[]))
    tampered = deepcopy(value)
    tampered["raw_shard_hashes"] = [
        d.methods.reference(Path(ref["path"])) for ref in tampered["raw_shard_hashes"]
    ]
    tampered["reproducibility_checksum"] = d.pilot.checksum(tampered)
    atomic_json(path, tampered)
    assert not runner.replay(path)["passed"]
    detector.write_bytes(original_detector)
    atomic_json(path, value)
    row = work["runtime"]["rows"][0]
    row["value"] = 1 - row["value"]
    atomic_json(raw / "work.json", work)
    value["raw_shard_hashes"][0] = d.methods.reference(raw / "work.json")
    atomic_json(path, value)
    assert not runner.replay(path)["passed"]
    assert runner.main(["--cold-replay", str(path)]) == 1
    work["units"][0]["k"] = 99
    atomic_json(raw / "work.json", work)
    value["raw_shard_hashes"][0] = d.methods.reference(raw / "work.json")
    atomic_json(path, value)
    assert not runner.replay(path)["passed"]
    (raw / "work.json").write_text("changed bytes")
    assert not runner.replay(path)["passed"]


def test_freeze_nonfixture(units, tmp_path, monkeypatch):
    """REQ-REPORT-8949: authenticated live route uses the shared bounded runtime."""
    from carnot.reporting import matched_corpus_runner_8949 as runner

    def fake_runtime(scheduled, raw):
        result = dict(
            rows=[s.censored(u, "private") for u in scheduled], counts={}, checks=[], live=False
        )
        atomic_json(raw / "runtime.json", result)
        return result

    monkeypatch.setattr(runner.runtime, "run", fake_runtime)
    assert runner.freeze(d.ROOT, tmp_path / "raw", False)["runtime"]["live"] is False


def test_qualification_is_owned(monkeypatch):
    """REQ-REPORT-8949: require all receipts and measurement of this producer's statements."""
    receipts = [dict(name=name, passed=True, exit_code=0) for name in d.methods.CHECKS]
    monkeypatch.setattr(d, "verify_measured", lambda coverage, owned: owned == d.OWNED)
    assert d.qualified(dict(receipts=receipts, coverage={}))
    assert not d.qualified(dict(receipts=receipts[:-1], coverage={}))
    receipts[0]["passed"] = False
    assert not d.qualified(dict(receipts=receipts, coverage={}))


def test_current_control_gate_errors(tmp_path, monkeypatch):
    """REQ-REPORT-8949: name missing, zero and rejected producer fields precisely."""
    monkeypatch.setattr(
        d.depth, "authenticate", lambda _: dict(failures=[], references=[], protocol={})
    )
    name, pin, field, task = d.INPUTS[-1]
    path = tmp_path / name
    assert d.authenticate(tmp_path)["failures"][0]["failure_kind"] == "missing_file"
    for payload, expected in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: True}, "rejected_producer"),
        ([], "rejected_producer"),
    ]:
        atomic_json(path, payload)
        assert d.authenticate(tmp_path)["failures"][0]["failure_kind"] == expected
    value = dict(
        task_id=task, required_checks_passed=True, collection_authorized=True, **{field: 1}
    )
    atomic_json(path, value)
    assert d.authenticate(tmp_path)["failures"][0]["artifact_field"] == "sha256"
    monkeypatch.setattr(d.methods, "reference", lambda p: dict(path=str(p), sha256=pin))
    monkeypatch.setattr(d.depth, "verify_references", Mock())
    assert d.authenticate(tmp_path)["failures"] == []


def test_raw_receipts_are_required(units, tmp_path):
    """REQ-REPORT-8949: exact wire evidence, model, usage and finite signals gate readiness."""
    raw = tmp_path / "runtime" / "collection"
    rows = []
    for unit in units:
        evidence = response(unit["expected"][0])
        folder = raw / unit["unit_id"]
        atomic_json(folder / "response.json", dict(raw_response=evidence))
        atomic_json(folder / "wire_response.json", evidence)
        rows.append(s.label_response(unit, evidence))
    work = dict(raw=str(tmp_path), units=units, runtime=dict(rows=rows))
    assert d.raw_complete(work)
    folder = raw / units[-1]["unit_id"]
    atomic_json(folder / "wire_response.json", {})
    assert not d.raw_complete(work)
    (folder / "wire_response.json").unlink()
    assert not d.raw_complete(work)


def test_checkpoint_first_response_and_binding(units, tmp_path, monkeypatch):
    """REQ-REPORT-8949: persist before labeling, resume exact bytes and refuse drift."""
    model = dict(hf_id=s.SETTINGS["model"], sha256=canonical_hash("private-model"))
    unit = units[0]
    original = s.label_response

    def persisted(scheduled, raw):
        assert (tmp_path / unit["unit_id"] / "response.json").is_file()
        return original(scheduled, raw)

    monkeypatch.setattr(s, "label_response", persisted)
    transport = Mock(return_value=response(unit["expected"][0]))
    first = s.collect([unit], transport, tmp_path, model)
    assert s.collect([unit], transport, tmp_path, model) == first
    assert transport.call_count == 1
    changed = dict(unit, prompt=unit["prompt"] + " changed")
    with pytest.raises(ValueError, match="checkpoint_binding"):
        s.collect([changed], transport, tmp_path, model)
    assert transport.call_count == 1


def test_external_block_preserves_roster_and_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8949: missing external files remain a terminal replayable block."""
    work = runner.freeze(tmp_path / "absent-inputs", tmp_path / "raw", False)
    validation = dict(receipts=[], coverage={})
    monkeypatch.setattr(d, "qualified", lambda _: True)
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 160
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
