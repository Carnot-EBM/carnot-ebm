"""REQ-REPORT-8182 / REQ-VERIFY-8182: private fixed-source capture checks."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import fit_sentence_capture_8182 as e
from carnot.verify import sentence_transport_8179 as transport


def sources():
    """Private sources exercise original roles without opening research targets."""
    return [
        dict(
            unit_id=f"{role}{i}",
            source_cluster_id=f"cluster-{role}{i}",
            role=role,
            slot=i + 1,
            source_bytes=b"Evidence. More evidence.".hex(),
            answer_bytes=b"First claim. Second claim.".hex(),
            **transport.requests(
                dict(
                    source_bytes=b"Evidence. More evidence.".hex(),
                    answer_bytes=b"First claim. Second claim.".hex(),
                ),
                lambda _: 30,
            ),
        )
        for role, count in e.ROLES.items()
        for i in range(count)
    ]


class Runtime:
    """Scripted responses grant no current model invocation credit."""

    worker = None

    def __init__(self, mode="ok"):
        self.worker = self
        self.mode = mode

    def post_json(self, endpoint, payload, timeout):
        if endpoint == "/apply-template":
            return dict(prompt=payload["messages"][0]["content"])
        return dict(tokens=list(range(30)))

    def generate(self, payload):
        indices = json.loads(payload["messages"][0]["content"])["answer_sentence_indices"]
        text = "\n".join(f"{i}|B|0.20|[]" for i in indices)
        if self.mode == "missing":
            text = text.split("\n")[0]
        if self.mode == "wrong":
            text = text.replace("[]", "[999]")
        return dict(
            choices=[dict(message=dict(content=text), finish_reason="stop")],
            usage=dict(prompt_tokens=30, completion_tokens=20),
        )


def controls(slots):
    return [
        dict(
            unit_id=s["unit_id"],
            source_cluster_id=s["source_cluster_id"],
            role=s["role"],
            slot=s["slot"],
            status="completed",
            x=[0.0] * 12,
            y=i % 2,
        )
        for i, s in enumerate(slots)
    ]


def test_fixed_roster_and_reuse(tmp_path):
    """SCENARIO-VERIFY-8182-BOUNDARIES: exact identities alone allow reuse."""
    slots = e.freeze(sources())
    calls = e.capture(slots, Runtime(), tmp_path / "calls", dict(fixture=True), [])
    assert len(calls) == 208
    assert sum(c["condition"] != "original" for c in calls) == 16
    assert e.reduce(slots, calls, controls(slots))["completed_count"] == 192
    reused = e.capture(slots, Runtime(), tmp_path / "reuse", dict(fixture=True), calls)
    assert sum(c["historical"] for c in reused) == 192
    assert reused[0]["started_monotonic_ns"] == calls[0]["started_monotonic_ns"]
    changed = e.capture(slots[:1], Runtime(), tmp_path / "drift", {}, calls)
    assert not changed[0]["historical"]
    assert e.cache_key(slots[0], slots[0]["requests"][0], {}) != calls[0]["cache_key"]
    bad = deepcopy(slots)
    bad[0]["slot"] = 2
    with pytest.raises(ValueError, match="original_slot"):
        e.freeze(bad)
    with pytest.raises(ValueError, match="role_count"):
        e.freeze([])


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_whole_source_failure(tmp_path, mode):
    """REQ-VERIFY-8182: partial or wrong-source records cannot make features."""
    slots = sources()[:1]
    calls = e.capture(slots, Runtime(mode), tmp_path, {}, [])
    reduced = e.reduce(slots, calls, controls(slots))
    assert reduced["completed_count"] == 0 and not reduced["feature_rows"]
    assert all(r["human_target"] is None for r in reduced["diagnostic_rows"])


def test_deadline_exclusion_and_support(tmp_path):
    """REQ-VERIFY-8182: preserve masks and report insufficient class support."""
    slots = sources()[:2]
    slots[0].update(status="excluded", exclusion_reason="sentence_capacity", requests=[])
    calls = e.capture(slots, Runtime(), tmp_path, {}, [], started=-10000)
    reduced = e.reduce(slots, calls, controls(slots))
    assert reduced["excluded_count"] == 1 and reduced["censored_count"] == 1
    assert reduced["fit_trainable_score"] == 0


def test_build_replay_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8182-CUSTODY: rehashed headlines do not survive replay."""
    slots = sources()
    atomic_json(tmp_path / "fit-fixture.json", dict(rows=slots, controls=controls(slots)))
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    value["completed_count"] -= 1
    atomic_json(path, value)
    assert not e.replay(path)
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    work["checks"] = [dict(check="transport_canary_ready_score", passed=False, observed=0)]
    blocked = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert blocked["honest_verdict"] == "complete_blocked_transport_canary_ready_score"
    assert blocked["fit_capture_ready_score"] == 0


def input_fixture(tmp_path, monkeypatch):
    """Private custody bytes test the real reader without changing research inputs."""
    from carnot.reporting.current_work_receipt import sha256_file

    root = tmp_path / "checkout"
    (root / "results").mkdir(parents=True)
    primitive = tmp_path / "requests.json"
    atomic_json(primitive, dict(rows=sources()))
    tune = tmp_path / "tune.json"
    atomic_json(tune, {})
    atomic_json(
        root / e.canary.UPSTREAM,
        dict(
            raw_shard_hashes=[e.reference(primitive)], source_manifest=dict(tune=e.reference(tune))
        ),
    )
    calls = tmp_path / "calls.json"
    atomic_json(calls, dict(rows=[]))
    sidecar, terminal = tmp_path / "sidecar.json", tmp_path / "terminal.json"
    atomic_json(sidecar, {})
    atomic_json(terminal, dict(publication=dict(sidecar_path=str(sidecar))))
    atomic_json(
        root / e.UPSTREAM,
        dict(
            transport_canary_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal),
            raw_shard_hashes=[e.reference(calls)],
            qualified_transport_configuration=dict(identity={}),
        ),
    )
    monkeypatch.setattr(e, "PIN", sha256_file(root / e.UPSTREAM))
    features = tmp_path / "features.json"
    atomic_json(features, dict(rows=controls(sources())))
    atomic_json(root / e.CONTROL, dict(raw_shard_hashes=[e.reference(features)]))
    monkeypatch.setitem(e.historical.PINS, e.CONTROL, sha256_file(root / e.CONTROL))
    method = root / e.historical.METHOD
    method.parent.mkdir(parents=True)
    method.write_text("Immutable historical method")
    monkeypatch.setattr(e.historical, "METHOD_HASH", sha256_file(method))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=True)))
    monkeypatch.setattr(
        e.canary, "inputs", lambda *_: dict(checks=[], slots=[], refs=[], upstream=[], identity={})
    )
    return root, features


def test_ingestion_missing_gate_and_structure(tmp_path, monkeypatch):
    """E2E-015/019: authenticate exact primitives and preserve failed operands."""
    root, features = input_fixture(tmp_path, monkeypatch)
    plan = e.inputs(root, tmp_path / "raw")
    assert len(plan["slots"]) == 192 and all(c["passed"] for c in plan["checks"])
    value = json.loads((root / e.UPSTREAM).read_text())
    value["transport_canary_ready_score"] = 0
    atomic_json(root / e.UPSTREAM, value)
    monkeypatch.setattr(e, "PIN", e.reference(root / e.UPSTREAM)["sha256"])
    assert any(
        c["check"] == "transport_canary_ready_score" and not c["passed"]
        for c in e.inputs(root, tmp_path / "gate")["checks"]
    )
    value["transport_canary_ready_score"] = 1
    del value["qualified_transport_configuration"]
    atomic_json(root / e.UPSTREAM, value)
    monkeypatch.setattr(e, "PIN", e.reference(root / e.UPSTREAM)["sha256"])
    assert e.inputs(root, tmp_path / "structure")["checks"][-1]["check"] == "input_structure"
    features.unlink()
    with pytest.raises(ValueError, match="primitive_sha256"):
        e.bind(
            dict(checks=[], refs=[]),
            dict(path=str(features), sha256="missing"),
            tmp_path / "missing",
        )


def test_live_adapter_and_readiness(tmp_path, monkeypatch):
    """REQ-REPORT-8182: qualified identity and live support are separate operands."""
    root, _ = input_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(e.canary, "preflight", lambda plan, raw: plan.update(identity={}))

    def live(plan, raw):
        e.canary.progress("owned_runtime_heartbeat")
        rows = e.canary.capture(plan["slots"], Runtime(), raw / "slots", {})
        return dict(
            rows=rows,
            checks=[],
            model_loads_attempted=1,
            model_loads_completed=1,
            runtime_receipts=[{}] * len(rows),
        )

    monkeypatch.setattr(e.canary, "live", live)
    work = e.measure(root, tmp_path / "live")
    work["duration_s"] = 11
    value = e.build(work, tmp_path / "live", [dict(passed=True)])
    assert value["fit_capture_ready_score"] == value["fit_trainable_score"] == 1
    assert value["verdict_class"] == "positive"
    monkeypatch.setattr(
        e.canary, "preflight", lambda plan, raw: plan.update(identity=dict(drift=True))
    )
    blocked = e.measure(root, tmp_path / "identity")
    assert blocked["checks"][-1]["check"] == "qualified_transport_identity"
    assert not blocked["calls"]


def test_source_and_control_identity_rejected(tmp_path):
    """REQ-VERIFY-8182: wrong original sources or controls cannot join features."""
    slots = e.freeze(sources())
    calls = e.capture(slots, Runtime(), tmp_path, {}, [])
    bad = deepcopy(calls)
    bad[0]["source_cluster_id"] = "wrong source"
    with pytest.raises(ValueError, match="request_or_source_identity"):
        e.reduce(slots, bad, controls(slots))
    paired = controls(slots)
    paired[0]["source_cluster_id"] = "wrong control"
    with pytest.raises(ValueError, match="paired_control_identity"):
        e.reduce(slots, calls, paired)


def test_private_cli_and_custody_tamper(tmp_path):
    """SCENARIO-REPORT-8182-CUSTODY: direct CLI success, missing input and cold replay."""
    import os
    import subprocess
    import sys

    atomic_json(tmp_path / "fit-fixture.json", dict(rows=sources(), controls=controls(sources())))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]

    def run(args):
        return subprocess.run(
            prefix + [str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    output = tmp_path / (e.NAME + ".json")
    assert run(["--root", str(tmp_path), "--fixture-output", str(output)]).returncode == 0
    assert run(["--cold-replay", str(output)]).returncode == 0
    value = json.loads(output.read_text())
    assert (
        value["fit_capture_ready_score"] == 0 and value["model_invocation_counts"]["generate"] == 0
    )
    value["completed_count"] = 0
    atomic_json(output, value)
    assert run(["--cold-replay", str(output)]).returncode == 1
    assert (
        run(["--root", str(tmp_path / "missing"), "--fixture-output", str(output)]).returncode == 0
    )
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert run(["--fixture-output", str(e.ROOT / "results" / "private.json")]).returncode == 2
    assert run(["--date", "20000101"]).returncode == 2


def test_replay_custody_and_manifest(tmp_path, monkeypatch):
    """REQ-REPORT-8182: logs, source masks and transcripts enter immutable custody."""
    atomic_json(tmp_path / "fit-fixture.json", dict(rows=sources(), controls=controls(sources())))
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True, mutation="source")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    missing = deepcopy(value)
    missing["measurement_reference"]["path"] = str(tmp_path / "absent")
    atomic_json(path, missing)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "drift"
    atomic_json(path, changed)
    assert not e.replay(path)
    log = tmp_path / "receipt.log"
    log.write_text("sealed log")
    changed = deepcopy(value)
    changed["validation_receipts"][0].update(log_path=str(log), log_sha256="wrong hash")
    atomic_json(path, changed)
    assert not e.replay(path)
    atomic_json(path, value)
    primitive = Path(work["raw_shard_hashes"][0]["path"])
    original = primitive.read_bytes()
    atomic_json(primitive, dict(rows=[]))
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0] = e.reference(primitive)
    changed["measurement_reference"] = e.reference(Path(value["measurement_reference"]["path"]))
    atomic_json(path, changed)
    assert not e.replay(path)
    primitive.write_bytes(original)
    changed_work = deepcopy(work)
    changed_work["slots"][0]["answer_bytes"] = b"Changed original answer.".hex()
    atomic_json(Path(value["measurement_reference"]["path"]), changed_work)
    atomic_json(
        Path(work["raw_shard_hashes"][1]["path"]),
        dict(rows=changed_work["slots"], controls=work["controls"], config=e.CONFIG),
    )
    changed_work["raw_shard_hashes"][1] = e.reference(Path(work["raw_shard_hashes"][1]["path"]))
    atomic_json(Path(value["measurement_reference"]["path"]), changed_work)
    changed = deepcopy(value)
    changed["raw_shard_hashes"] = changed_work["raw_shard_hashes"]
    changed["measurement_reference"] = e.reference(Path(value["measurement_reference"]["path"]))
    atomic_json(path, changed)
    assert not e.replay(path)
    private = tmp_path / "validation"
    private.mkdir()
    specs = e.manifest(private, private / "candidate.json")
    assert e.TEST in specs["commands"][0]["argv"]
    assert all("::" not in arg for s in specs["commands"][5:] for arg in s["argv"])
    monkeypatch.setattr(e.execution, "main", lambda argv: 7)
    assert e.main([]) == 7
