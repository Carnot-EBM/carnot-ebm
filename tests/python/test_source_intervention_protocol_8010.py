"""REQ-REPORT-8010: label-free panels and real private transport custody."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8010_v694_source_intervention_protocol as e
from carnot.reporting.current_work_receipt import atomic_json
from test_qwen_development_capture_7995 import views


def cli(args, tmp_path):
    argv = [sys.executable, str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8010_COVERAGE_CONFIG")
    if config:
        argv[1:1] = ["-m", "coverage", "run", "--rcfile=" + config]
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_frozen_complete_panel_and_derangement():
    """SCENARIO-REPORT-8010-ISOLATION: order does not depend on hidden targets."""
    public = views()["stream"]
    panel = e.freeze(public)
    assert len(panel["roster"]) == 64 and len(panel["slots"]) == 192
    assert panel == e.freeze(public)
    assert len({r["family_id"] for r in panel["slots"]}) == 192
    assert all(k != v for k, v in panel["derangement"].items())
    for group in panel["roster"]:
        cells = {r["arm"]: r for r in panel["slots"] if r["group_id"] == group["family_id"]}
        assert cells["original"]["request"] == cells["duplicate"]["request"]
        assert cells["swap"]["source_hash"] != cells["original"]["source_hash"]
        for cell in cells.values():
            assert cell["request"]["max_tokens"] == 32
            assert cell["request"]["seed"] == 69410
            assert cell["request"]["messages"][0]["content"] == e.c.risk.SYSTEM
    polluted = copy.deepcopy(public)
    polluted["request_rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="public_fields"):
        e.freeze(polluted)
    public["features"] = public["features"][:2]
    with pytest.raises(ValueError):
        e.freeze(public)


def test_oversize_and_duplicate_identity_refused(tmp_path):
    """REQ-REPORT-8010: selected overrun remains visible with no replacement."""
    public = views()["stream"]
    for row in public["request_rows"]:
        row["source_bytes"] = (bytes.fromhex(row["source_bytes"]) + b" qualifier" * 1000).hex()
    panel = e.freeze(public)
    assert len(panel["slots"]) == 192
    assert all(not s["public_eligible"] for s in panel["slots"])
    ledger = e.c.Ledger(tmp_path / "ledger.json")
    ledger.start("generation", "same", {})
    with pytest.raises(ValueError, match="duplicate_call"):
        ledger.start("generation", "same", {})


def test_real_cli_fixture_cold_replay_and_drift(tmp_path):
    """SCENARIO-REPORT-8010-PUBLICATION: natural rows never inherit fixture risk."""
    fixture = tmp_path / "public.json"
    atomic_json(fixture, views()["stream"])
    output = tmp_path / "out" / (e.NAME + ".json")
    run = cli(["--fixture", fixture, "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert len(value["fixture_rows"]) == 192
    assert len(value["rows"]) == 192
    assert all(r["status"] == "censored" and not r["started"] for r in value["rows"])
    assert not any(value["model_invocation_counts"].values())
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    for mutate in [
        lambda v: v["fixture_rows"][0].update(parsed={}),
        lambda v: v["model_invocation_counts"].update(model_loads_attempted=1),
        lambda v: v.update(protocol_ready_score=1, verdict_class="disqualified"),
        lambda v: v["sample_size_budget"].update(completed=1),
    ]:
        changed = copy.deepcopy(value)
        mutate(changed)
        with pytest.raises(ValueError):
            e.replay(changed)
    changed = copy.deepcopy(value)
    changed["rows"][0]["numerator"] = 1
    atomic_json(output, changed)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    atomic_json(output, value)
    ref = value["checkpoint_references"][0]
    Path(ref["path"]).write_text("{}")
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    assert cli(["--cold-replay", tmp_path / "absent"], tmp_path).returncode == 1
    assert cli(["--date", "20261001"], tmp_path).returncode == 2
    assert cli(["--fixture", fixture], tmp_path).returncode == 1


def test_cold_replay_rejects_protocol_and_transport_mutations(tmp_path):
    """SCENARIO-REPORT-8010-ISOLATION: aggregate copies cannot override raw custody."""
    fixture = tmp_path / "public.json"
    atomic_json(fixture, views()["stream"])
    output = tmp_path / "out" / (e.NAME + ".json")
    run = cli(["--fixture", fixture, "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())

    def altered_file(key, mutate):
        changed = copy.deepcopy(value)
        path = Path(changed[key]["path"])
        original = path.read_bytes()
        document = json.loads(original)
        mutate(document)
        atomic_json(path, document)
        digest = e.sha256_file(path)
        changed[key]["sha256"] = digest
        for ref in changed["raw_shard_hashes"]:
            if ref["path"] == str(path):
                ref["sha256"] = digest
        if key == "fixture_reference":
            changed["fixture_rows"] = document["rows"]
        try:
            with pytest.raises(ValueError):
                e.replay(changed)
        finally:
            path.write_bytes(original)

    altered_file("panel_reference", lambda d: d["methods"].update(seed=1))
    altered_file("fixture_reference", lambda d: d["ledger"].append(d["ledger"][0]))
    altered_file("fixture_reference", lambda d: d["rows"][0].update(parsed={}))
    altered_file("fixture_reference", lambda d: d["ledger"][0].update(request_sha256="drift"))


def test_peer_predictor_and_request_id_refusals(tmp_path):
    """REQ-REPORT-8010: reject bad payloads before any HTTP request starts."""
    panel = e.freeze(views()["stream"])
    panel["slots"] = panel["slots"][:1]
    bad = copy.deepcopy(panel)
    body = json.loads(bad["slots"][0]["request"]["messages"][1]["content"])
    body["y"] = 1
    bad["slots"][0]["request"]["messages"][1]["content"] = json.dumps(body)
    with pytest.raises(ValueError, match="target_bearing"):
        e.capture_peer(bad, tmp_path / "bad.json", "normal")
    panel["slots"].append(panel["slots"][0])
    with pytest.raises(ValueError, match="duplicate_request"):
        e.capture_peer(panel, tmp_path / "bad.json", "normal")


def test_owned_main_branches_and_failure_gates(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8010-PUBLICATION: check required failures and terminal exits."""
    public = tmp_path / "stream.json"
    atomic_json(public, views()["stream"])
    plan = dict(checks=[], references=[], public_role_manifests=dict(stream=dict(path=str(public))))
    monkeypatch.setattr(e, "authenticate", lambda root: plan)
    original = e.run_commands
    mode = ["ok"]

    def run(root, specs, **kwargs):
        if specs[0].name == "focused":
            scratch = Path(kwargs["extra_env"]["COVERAGE_FILE"]).parent
            atomic_json(
                scratch / "coverage.json",
                dict(totals=dict(num_statements=1, covered_lines=1, percent_covered=100), files={}),
            )
            return [
                dict(scope="owned", passed=mode[0] != "owned_fail", name="controlled_failure_gate")
            ]
        if specs[0].name == "fresh_process_cold_reduction" and mode[0] == "cold_fail":
            return [dict(passed=False)]
        return original(root, specs, **kwargs)

    monkeypatch.setattr(e, "run_commands", run)
    for case in ["ok", "owned_fail", "cold_fail"]:
        mode[0] = case
        output = tmp_path / case / (e.NAME + ".json")
        code = e.main(["--output", str(output)])
        assert code == int(case == "cold_fail")
        if output.exists():
            value = json.loads(output.read_text())
            assert value["verdict_class"] == ("disqualified" if case == "owned_fail" else "null")
            assert value["protocol_ready_score"] == int(case == "ok")
    mode[0] = "ok"
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=False))
    assert e.main(["--output", str(tmp_path / "terminal" / (e.NAME + ".json"))]) == 1
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert e.main(["--output", str(tmp_path / "readers" / (e.NAME + ".json"))]) == 1


def test_authentication_contract_and_roster_floor(tmp_path):
    """REQ-REPORT-8010: exact upstream bytes and missing fields stay explicit."""
    assert all(r["passed"] for r in e.authenticate(e.ROOT)["checks"])
    path = tmp_path / "results/experiment_7995_v693_qwen_development_capture.json"
    atomic_json(path, {})
    assert any(
        r["observed"] == "missing_field_contract_error" for r in e.authenticate(tmp_path)["checks"]
    )
    public = views()["stream"]
    for feature in public["features"]:
        feature["abstention"] = "not_eligible"
    with pytest.raises(ValueError, match="roster_floor"):
        e.freeze(public)


@pytest.mark.parametrize("mode", ["wrong_model", "timeout", "context_overrun"])
def test_private_peer_failure_modes(tmp_path, mode):
    """SCENARIO-REPORT-8010-ISOLATION: genuine HTTP failures stay terminal."""
    panel = e.freeze(views()["stream"])
    panel["slots"] = panel["slots"][:1]
    manifest = tmp_path / "panel.json"
    atomic_json(manifest, panel)
    transcript = tmp_path / "peer.json"
    run = cli(["--capture-peer", manifest, "--peer-mode", mode, "--output", transcript], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    row = json.loads(transcript.read_text())["rows"][0]
    assert not row["parsed"]["completed"]
    assert row["status"] == (
        "excluded" if mode == "context_overrun" else "failed" if mode == "timeout" else "generated"
    )


def test_missing_upstream_and_blocked_replay(tmp_path):
    """REQ-REPORT-8010: absent contracts fail explicitly while preparation ends."""
    output = tmp_path / "blocked" / (e.NAME + ".json")
    run = cli(["--root", tmp_path / "absent", "--output", output], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["protocol_ready_score"] == 0 and value["verdict_class"] == "blocked"
    assert value["gate_check_summary"]
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
