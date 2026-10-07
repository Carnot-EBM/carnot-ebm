"""REQ-VERIFY-8249 / REQ-REPORT-8249: private custody and causal mechanics."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import evidence_view_kernel_8249 as k
from carnot.verify import evidence_view_execution_8249 as e


def public(
    source="Café is open. Café is open. It is not closed.", answer="Café is open, but only today."
):
    return dict(source_bytes=source.encode().hex(), answer_bytes=answer.encode().hex())


def test_views_capture_and_labels():
    """SCENARIO-VERIFY-8249-VIEWS: custody is distinct from semantic truth."""
    row = public()
    view = k.construct_view(row, [0])
    assert view["answer_bytes"] == row["answer_bytes"]
    assert view["sentence_map"][0]["view_index"] is None
    assert view["sentence_map"][1]["original_byte_start"] > 0
    request = k.capture_request(view)
    assert request["status"] == "completed"
    response = "0|E|0.05|[0,0]"
    parsed = k.accept_capture(view, response, canonical_hash(request))
    assert parsed["status"] == "completed" and parsed["semantic_gold"] is False
    for bad in ["0|E|0.05|[9]", "", response + "\n", "0|E|0.05|[-1]"]:
        assert k.accept_capture(view, bad, canonical_hash(request))["status"] == "escalated"
    with pytest.raises(ValueError, match="hash_drift"):
        k.accept_capture(view, response, "wrong")
    assert k.accept_capture(view, "0|B|0.50|[]", canonical_hash(request))["status"] == "completed"
    assert k.construct_view(row, [0, 1, 2])["source_bytes"] == ""
    assert (
        k.capture_request(k.construct_view(public(""), []))["exclusion_reason"] == "missing_source"
    )
    assert (
        k.capture_request(k.construct_view(public("x" * 7000 + "."), []))["exclusion_reason"]
        == "input_token_limit"
    )
    with pytest.raises(ValueError, match="sentence_address"):
        k.construct_view(row, [8])
    with pytest.raises(ValueError, match="sentence_address"):
        k.construct_view(public("unfinished"), [0])
    changed = dict(row, human_label=1)
    assert k.construct_view(changed, [0]) == view
    cached = [dict(sentence_index=0, relation="E", p_unsupported=0.1)]
    assert k.protocol_views(row, cached) == k.protocol_views(changed, cached)
    assert k.protocol_views(public("One."), cached)["status"] == "unavailable"
    assert (
        k.group_key(dict(relation="E", selected_delta=0.11, control_delta=-0.11, human_label=1))
        == "111"
    )
    assert k.group_key(dict(relation="E", selected_delta=0.1, control_delta=0.1)) == "100"
    assert k.group_key(None) is None
    with pytest.raises(ValueError, match="features"):
        k.group_key(dict(relation="X", selected_delta=0, control_delta=0))


def test_state_admission_restart_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8249-STATE: issues persist before delayed feedback."""
    path = tmp_path / "ledger.jsonl"
    state = k.load_state(path)
    features = dict(relation="E", selected_delta=0.2, control_delta=0)
    assert k.probability(state, 0.2, "110", "group") == 0.2
    for t in range(24):
        k.issue(
            path,
            dict(slot=t, source_cluster_id=str(t), p_static=0.2, features=features, role="stream"),
        )
        if t >= 8:
            k.release(path, t - 8, t, 1)
        state = k.load_state(path)
        assert k.probability(state, 0.2, "110", "group") >= 0.2
        if t < 15:
            assert k.probability(state, 0.2, "110", "group") == k.probability(
                state, 0.2, "110", "global"
            )
    assert state["groups"]["110"]["n"] == 16
    assert k.probability(state, 0.2, "110", "group") > k.probability(state, 0.2, "110", "global")
    assert k.load_state(path) == state
    assert all(0 <= k.probability(state, p, "110", "group") <= 1 for p in [0, 1])
    assert k.probability(state, None, None, "group") is None
    for args, reason in [
        ((23, 24, 1), "future_feedback"),
        ((0, 8, 1), "duplicate_source"),
        ((99, 107, 1), "unissued"),
    ]:
        with pytest.raises(ValueError, match=reason):
            k.release(path, *args)
    with pytest.raises(ValueError, match="probability"):
        k.probability(state, float("nan"), None, "group")
    with pytest.raises(ValueError, match="issue_order"):
        k.issue(
            path, dict(slot=0, source_cluster_id="new", p_static=0.5, features=None, role="stream")
        )
    before = deepcopy(state["groups"])
    for t in range(24, 33):
        k.issue(
            path,
            dict(slot=t, source_cluster_id=str(t), p_static=None, features=None, role="stream"),
        )
    k.release(path, 24, 32, 0)
    assert k.load_state(path)["groups"] == before
    k.issue(
        path,
        dict(
            slot=33, source_cluster_id="retained", p_static=0.5, features=features, role="retention"
        ),
    )
    for t in range(34, 42):
        k.issue(
            path,
            dict(slot=t, source_cluster_id=str(t), p_static=0.5, features=features, role="stream"),
        )
    with pytest.raises(ValueError, match="retention"):
        k.release(path, 33, 41, 1)
    with pytest.raises(ValueError, match="label"):
        k.release(path, 25, 33, 2)
    path.write_text(path.read_text() + "{")
    with pytest.raises(ValueError):
        k.load_state(path)
    path.write_text('{"kind":"release"}')
    with pytest.raises(ValueError, match="partial_record"):
        k.load_state(path)
    with pytest.raises(ValueError, match="event_kind"):
        k.transition(k.initial(), dict(kind="invented", record={}))


def cli(tmp_path, *args):
    """SCENARIO-REPORT-8249-CLI: exercise actual script and child statements."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8249 subprocess", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("after private8249 subprocess", result.returncode, flush=True)
    return result


def test_fixture_benefit_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8249-STATE: fixture benefit is circular and shuffled separately."""
    evidence = k.qualify(tmp_path)
    assert evidence["admission_ready"] and evidence["view_ready"]
    assert {c["n"] for c in evidence["state"]["shuffled"].values()} == {47}
    costs = evidence["fixture_costs"]
    assert costs["group"]["later_cost"] < costs["global"]["later_cost"]
    assert costs["group"]["later_cost"] < costs["shuffled"]["later_cost"]
    assert costs["group"]["retention_cost"] <= costs["group"]["earlier_cost"]
    assert k.reconstruct(evidence) == evidence
    bad = deepcopy(evidence)
    bad["events"][10]["record"]["predictions"]["group"] = 0.99
    with pytest.raises(ValueError, match="ledger_drift"):
        k.reconstruct(bad)
    protocol = json.loads((e.ROOT / e.PROTOCOL).read_text())
    assert k.bind_roles(protocol) == dict(
        fit=128, calibration=32, selection=32, stream=96, retention=32
    )
    protocol["role_manifest"]["fit"] = []
    with pytest.raises(ValueError):
        k.bind_roles(protocol)


def test_execution_build_replay_and_manifest(tmp_path):
    """REQ-REPORT-8249: component gates and actual authentication stay separate."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    receipts = [dict(name="private", passed=True)]
    value = e.build(work, raw, receipts, fixture=True)
    assert value["verdict_class"] == "circular_positive"
    assert value["view_kernel_ready_score"] == value["admission_kernel_ready_score"] == 1
    assert not any(value["model_invocation_counts"].values())
    assert value["generalized_learning_benefit_score"] == 0
    assert value["intended_count"] == 384 * 3
    assert value["completed_count"] == 376 * 3 and value["censored_count"] == 8 * 3
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    value["view_kernel_ready_score"] = 3
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, raw, [dict(passed=False)], fixture=True)["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["evidence"]["view_ready"] = False
    separate = e.build(changed, raw, receipts, fixture=True)
    assert (
        separate["view_kernel_ready_score"] == 0 and separate["admission_kernel_ready_score"] == 1
    )
    missing = e.measure(tmp_path / "missing", tmp_path / "blocked")
    blocked = e.build(missing, tmp_path / "blocked", receipts, fixture=True)
    assert blocked["verdict_class"] == "blocked"
    assert any(c["observed"] is None for c in blocked["gate_check_summary"])
    work = e.measure(e.ROOT, tmp_path / "actual")
    assert all(c["passed"] for c in work["checks"])
    candidate = tmp_path / "candidate.json"
    specs = e.manifest(tmp_path, candidate)
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    assert specs["repository_health"]["classification"] == "diagnostic"
    assert e.build(work, tmp_path / "actual", receipts)["verdict_class"] == "disqualified"


def test_real_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8249-CLI: normal exit, bad date and rehashed drift."""
    output = tmp_path / (e.NAME + ".json")
    assert cli(tmp_path, "--date", "bad").returncode == 2
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_text())
    value["admission_kernel_ready_score"] = 5
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert (
        cli(tmp_path, "--fixture-output", e.ROOT / "results" / "private8249.json").returncode == 2
    )
    worker = tmp_path / "worker" / "measurement.json"
    assert cli(tmp_path, "--root", tmp_path / "missing", "--worker-output", worker).returncode == 0


def test_replay_failure_paths_and_authenticated_exception(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8249-CLI: negative and rehashed custody failures remain failures."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    log = tmp_path / "stdout"
    log.write_text("original")
    from carnot.reporting.current_work_receipt import sha256_file

    receipt = dict(
        name="actual_log", passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log)
    )
    value = e.build(work, raw, [receipt], fixture=True)
    path = tmp_path / (e.NAME + ".json")
    assert value["intended_count"] == 1152
    changed = deepcopy(value)
    changed["random_seed"] = 99
    atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["code_config_hashes"][0]["sha256"] = "wrong"
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    atomic_json(path, value)
    log.write_text("drift")
    assert not e.replay(path)
    log.write_text("original")
    primitive = deepcopy(work["evidence"])
    primitive["mutations"][0]["removed"] = [0]
    atomic_json(raw / "primitive_evidence.json", primitive)
    changed = deepcopy(value)
    changed["raw_shard_hashes"] = [
        e.reference(Path(r["path"])) for r in changed["raw_shard_hashes"]
    ]
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)

    def fail(*args):
        raise ValueError("authenticated schema failure")

    monkeypatch.setattr(e, "authenticate", fail)
    failed = e.measure(e.ROOT, tmp_path / "schema")
    assert failed["checks"][-1]["artifact_field"] == "authenticated_input_schema"
