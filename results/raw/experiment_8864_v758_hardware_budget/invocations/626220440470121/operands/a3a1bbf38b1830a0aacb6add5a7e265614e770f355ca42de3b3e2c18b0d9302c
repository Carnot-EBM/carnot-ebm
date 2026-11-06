"""REQ-REPORT-8828 / REQ-VERIFY-8828: preserve the original numerical protocol."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import signal_replay_qualification_8828 as q


def test_scenario_verify_8828_inventory():
    """All original cells remain scheduled, including the stronger linear arm."""
    cells = q.inventory()
    assert len(cells) == 45
    assert {r["updates"] for r in cells} == {400}
    assert {r["objective"] for r in cells} == {"mixed", "categorical", "linear"}
    assert len({(r["seed"], r["objective"], r["lr"]) for r in cells}) == 45


def test_scenario_verify_8828_pure_memo():
    """An observation depends on template membership, with no dependence on order."""
    payload = q.d.m.world("memo", "support")
    template = q.d.m.normalize(payload)
    expected = q.d.m.observe(payload, [template])
    with q.primitives() as calls:
        a = q.d.m.observe(payload, ["unrelated", template])
        a["counts"][0] = 999
        assert q.d.m.observe(payload, [template]) == expected
        assert q.d.m.observe(payload, [])["primitive_checks"] == 5
    assert calls and q.verify_primitives(calls) is None
    calls[-1]["value"] = {"forged": True}
    with pytest.raises(ValueError, match="primitive_drift"):
        q.verify_primitives(calls)


def snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    q.atomic_json(source, dict(operand=1))
    path = tmp_path / "snapshot.json"
    value = dict(
        owner=str(tmp_path),
        code=[
            q.d.reference(q.ROOT / p)
            for p in q.old.CODE + q.old.TESTS + q.OWNED + q.TESTS + [q.DESIGN]
        ],
        inputs=[q.d.reference(source)],
        primitive_calls=[],
        diagnostic_rows=[],
        source_artifact_hashes=[],
        excluded=[],
        development_rows=[],
        split_manifest={},
    )
    value["audit_reduction"] = dict(rows=[], leakage_checks={})
    prior = tmp_path / "prior.json"
    admitted = tmp_path / "admitted.json"
    audit = tmp_path / "audit.json"
    q.atomic_json(admitted, dict(source_artifact_hashes=[]))
    q.atomic_json(prior, dict(raw_shard_hashes=[q.d.reference(admitted)]))
    q.atomic_json(audit, value["audit_reduction"])
    monkeypatch.setattr(q, "HISTORICAL", prior)
    monkeypatch.setattr(q, "HISTORICAL_PIN", q.sha256_file(prior))
    monkeypatch.setattr(q.old, "AUDIT", str(audit))
    monkeypatch.setattr(q.old, "AUDIT_PIN", q.sha256_file(audit))
    q.atomic_json(tmp_path / "preconditions.json", dict(inputs=value["inputs"], code=value["code"]))
    q.atomic_json(path, value)
    return path, value


def test_scenario_verify_8828_snapshot(tmp_path, monkeypatch):
    """Invocation, source and code binding survive repaired outer envelopes."""
    path, value = snapshot(tmp_path, monkeypatch)
    ref = q.d.reference(path)
    assert q.load_snapshot(ref, verify=True) == value
    for key, change, message in [
        ("owner", "/tmp/wrong-owner", "snapshot_owner"),
        ("code", [dict(path=q.__file__, sha256="wrong")], "code_hash"),
        ("inputs", [dict(path=str(tmp_path / "source.json"), sha256="wrong")], "operand_hash"),
    ]:
        changed = dict(value, **{key: change})
        q.atomic_json(path, changed)
        with pytest.raises(ValueError, match=message):
            q.load_snapshot(q.d.reference(path), verify=True)
    q.atomic_json(path, value)
    (tmp_path / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="operand_hash"):
        q.load_snapshot(ref, verify=True)
    with pytest.raises(ValueError, match="snapshot_hash"):
        q.load_snapshot(dict(ref, sha256="wrong"))


def test_req_report_8828_projection():
    """Measured pilots must fit the frozen total budget; deadlines stay 600s."""
    good = q.projection(10, [1, 2, 3])
    assert good["feasible"] and good["budget_s"] == 4200
    assert good["child_deadline_s"] == 600
    assert not q.projection(590, [100, 100, 100])["feasible"]


def test_scenario_verify_8828_cold_cli(tmp_path):
    """The installed entrypoint imports outside the checkout without PYTHONPATH."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    for args, expected in [
        (["--date", "bad"], "date_rejected"),
        (["--replay", str(tmp_path / "absent")], "replay_rejected"),
    ]:
        result = subprocess.run(
            [sys.executable, "-u", str(q.ROOT / q.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 1 and expected in result.stdout


def fake_evidence():
    return dict(
        diagnostic_rows=[],
        development_rows=[],
        split_manifest={},
        source_artifact_hashes=[],
        fits=[dict(c) for c in q.inventory()],
        controls=[],
        snapshot={},
        partition_rows=[],
        replay_projection={},
    )


def test_req_verify_8828_preconditions(tmp_path, monkeypatch):
    """Preflight binds historical files before numerical work can begin."""
    source = tmp_path / "source.json"
    q.atomic_json(source, {})
    audit = tmp_path / "audit.json"
    q.atomic_json(
        audit,
        dict(
            required_checks_passed=True,
            raw_shard_hashes=[q.d.reference(source)],
            code_config_hashes=[],
        ),
    )
    evidence = tmp_path / "evidence.json"
    q.atomic_json(evidence, dict(source_artifact_hashes=[]))
    prior = tmp_path / "prior.json"
    q.atomic_json(prior, dict(code_config_hashes=[], raw_shard_hashes=[q.d.reference(evidence)]))
    monkeypatch.setattr(q, "HISTORICAL", prior)
    monkeypatch.setattr(q, "HISTORICAL_PIN", q.sha256_file(prior))
    monkeypatch.setattr(q.old, "AUDIT", str(audit))
    monkeypatch.setattr(q.old, "AUDIT_PIN", q.sha256_file(audit))
    value = q.preconditions(tmp_path / "private")
    assert value["free_bytes"] >= value["minimum_free_bytes"]
    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    q.bindings(value["inputs"], "input")


def test_req_verify_8828_authentication(tmp_path, monkeypatch):
    """The unchanged authenticator owns snapshot creation; historical drift fails."""
    evidence = fake_evidence()
    prior = tmp_path / "prior.json"
    path = tmp_path / "old-evidence.json"
    q.atomic_json(path, evidence)
    q.atomic_json(prior, dict(raw_shard_hashes=[q.d.reference(path)]))
    monkeypatch.setattr(q, "HISTORICAL", prior)
    monkeypatch.setattr(q, "HISTORICAL_PIN", q.sha256_file(prior))
    q.atomic_json(tmp_path / "preconditions.json", dict(inputs=[], code=[]))
    monkeypatch.setattr(q.old.audit, "reduce", lambda *_: dict(rows=[], leakage_checks={}))

    def authenticated():
        q.old.audit.reduce({}, {})
        return [], [], set()

    monkeypatch.setattr(q.old, "authenticate", authenticated)
    monkeypatch.setattr(q.d, "development", lambda _: ([], {}))
    value = q.authenticate(tmp_path)
    assert value["owner"] == str(tmp_path)
    assert (tmp_path / "snapshot.json").stat().st_mode & 0o777 == 0o400
    with pytest.raises(ValueError, match="snapshot_already_exists"):
        q.authenticate(tmp_path)


def test_req_verify_8828_fit_controls_and_children(tmp_path, monkeypatch):
    """Every partition uses the complete original fit and exact control values."""
    path, value = snapshot(tmp_path, monkeypatch)
    ref = q.d.reference(path)
    fitted = dict(q.inventory()[0])
    evidence = tmp_path / "historical-evidence.json"
    controls = [dict(passed=True)]
    q.atomic_json(evidence, dict(fits=[fitted], controls=controls))
    prior = tmp_path / "prior.json"
    q.atomic_json(prior, dict(raw_shard_hashes=[q.d.reference(evidence)]))
    monkeypatch.setattr(q, "HISTORICAL", prior)
    monkeypatch.setattr(q, "HISTORICAL_PIN", q.sha256_file(prior))
    monkeypatch.setattr(q.d, "fit", lambda *args: dict(fitted))
    monkeypatch.setattr(q.d, "controls", lambda: deepcopy(controls))
    assert q.fit(value, 0) == fitted
    q.child(tmp_path, "controls", ref)
    q.child(tmp_path, "fit-00", ref, 0)
    assert json.loads((tmp_path / "fit-00.json").read_bytes())["value"] == fitted
    monkeypatch.setattr(q, "authenticate", lambda raw: {})
    q.child(tmp_path, "authenticate")
    with pytest.raises(ValueError, match="missing_snapshot"):
        q.child(tmp_path, "controls")
    monkeypatch.setattr(q.d, "fit", lambda *args: {})
    with pytest.raises(ValueError, match="fit_drift"):
        q.fit(value, 0)


def fake_partition(raw, mode, ref=None, index=0):
    if mode == "authenticate":
        q.atomic_json(raw / "snapshot.json", {})
    else:
        value = [] if mode == "controls" else dict(q.inventory()[index])
        q.atomic_json(raw / (mode + ".json"), dict(value=value))
    return dict(passed=True, duration_s=1, scope="owned")


def test_req_report_8828_complete_partitions(tmp_path, monkeypatch):
    """Pilots count in the original inventory and infeasible work stops early."""
    from carnot.reporting import signal_replay_runner_8828 as r

    evidence = fake_evidence()
    snap = dict(evidence, inputs=[], code=[])
    monkeypatch.setattr(r, "partition", fake_partition)
    monkeypatch.setattr(q, "load_snapshot", lambda *args, **kwargs: snap)
    value = r.execute(tmp_path)
    assert len(value["fits"]) == len(value["partition_rows"]) == 45
    assert [row["index"] for row in value["partition_rows"]] == list(range(45))
    monkeypatch.setattr(q, "projection", lambda *args: dict(feasible=False))
    with pytest.raises(ValueError, match="replay_projection_infeasible"):
        r.execute(tmp_path)


def test_scenario_verify_8828_replay_mutations(tmp_path, monkeypatch):
    """A complete recomputation rejects missing cells and repaired reductions."""
    from carnot.reporting import signal_replay_runner_8828 as r

    evidence = fake_evidence()
    snap = dict(evidence, inputs=[], code=[])
    snap_path = tmp_path / "snapshot.json"
    q.atomic_json(snap_path, snap)
    evidence["snapshot"] = q.d.reference(snap_path)
    ep = tmp_path / "evidence.json"
    q.atomic_json(ep, evidence)
    log = tmp_path / "validation.log"
    log.write_text("actual test fixture validation")
    reduced = dict(diagnosis_rows_ready_score=1, objective_candidate_ready_score=1, rows=[])
    value = dict(
        reduced,
        verdict_class="null",
        code_config_hashes=[],
        validation_receipts=[dict(log_path=str(log), log_sha256=q.sha256_file(log))],
        raw_shard_hashes=[q.d.reference(ep)],
    )
    value.update(
        original_inventory=q.inventory(),
        partition_rows=[],
        replay_projection={},
        linear_comparator_rows=[],
        objective_selection=dict(paired_rows=[]),
    )
    candidate = tmp_path / "candidate.json"
    q.atomic_json(candidate, value)
    monkeypatch.setattr(q, "load_snapshot", lambda *args, **kwargs: snap)
    monkeypatch.setattr(q.old, "reduce", lambda ev: dict(reduced))
    monkeypatch.setattr(r, "partition", fake_partition)
    assert r.replay(candidate)["updates"] == 18000
    value.update(
        verdict_class="disqualified",
        diagnosis_rows_ready_score=0,
        objective_candidate_ready_score=0,
    )
    q.atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    value["rows"] = ["forged"]
    q.atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    evidence["fits"].pop()
    q.atomic_json(ep, evidence)
    value["raw_shard_hashes"] = [q.d.reference(ep)]
    q.atomic_json(candidate, value)
    with pytest.raises(ValueError, match="fit_inventory"):
        r.replay(candidate)
    value.update(raw_shard_hashes=[], verdict_class="blocked")
    q.atomic_json(candidate, value)
    assert r.replay(candidate)["scope"] == "blocked_no_measurement"
    value["validation_receipts"][0]["log_path"] = str(log.relative_to(tmp_path))
    monkeypatch.setattr(r, "ROOT", tmp_path)
    q.atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]


def test_req_verify_8828_supervision_and_validation(tmp_path, monkeypatch):
    """The supervisor forwards original deadlines and reuses existing validators."""
    from carnot.reporting import signal_replay_runner_8828 as r

    seen = []

    def commands(plan, logs):
        seen.extend(plan)
        return [dict(passed=True, duration_s=0.1)]

    monkeypatch.setattr(q.old, "commands", commands)
    r.partition(tmp_path, "controls", dict(path="snapshot", sha256="hash"))
    assert seen[0][3] == 600 and seen[0][1][:5] == ["env", "-u", "PYTHONPATH", "-C", "/tmp"]
    monkeypatch.setattr(q.old, "validate", lambda raw: dict(passed=True))
    monkeypatch.setattr(q.old, "terminal", lambda path, raw: dict(passed=True))
    assert r.validate(tmp_path)["passed"] and r.terminal(tmp_path / "candidate", tmp_path)["passed"]
    for error, status in [
        (FileNotFoundError(2, "absent", str(tmp_path / "absent")), "missing_file"),
        (KeyError("absent"), "missing_field"),
        (ValueError("drift"), "rejected_producer"),
    ]:
        assert r.gate(error)["status"] == status


@pytest.mark.parametrize("failure", ["none", "external", "internal", "owned", "terminal"])
def test_req_report_8828_publication(tmp_path, monkeypatch, failure):
    """Complete publication keeps external absence distinct from owned failures."""
    from carnot.reporting import signal_replay_runner_8828 as r

    evidence = fake_evidence()
    reduced = dict(
        rows=[],
        intended_count=0,
        completed_count=0,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        diagnosis_rows_ready_score=1,
        objective_candidate_ready_score=1,
        objective_selection=dict(paired_rows=[]),
        selected_configuration_hash="fixture",
    )

    def prepare(raw):
        raw.mkdir(parents=True, exist_ok=True)
        if failure == "external":
            raise FileNotFoundError(2, "absent", str(raw / "missing"))
        if failure == "internal":
            raise ValueError("internal_cohort_failure")
        return {}

    def execute(raw):
        q.atomic_json(raw / "evidence.json", evidence)
        return evidence

    monkeypatch.setattr(q, "preconditions", prepare)
    monkeypatch.setattr(r, "execute", execute)
    monkeypatch.setattr(q.old, "reduce", lambda ev: deepcopy(reduced))
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=failure != "owned", receipts=[], coverage={})
    )
    monkeypatch.setattr(
        r, "terminal", lambda path, raw: dict(passed=failure != "terminal", receipts=[])
    )
    output = tmp_path / (q.NAME + ".json")
    assert r.main(["--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    expected = (
        "circular_positive"
        if failure == "none"
        else "blocked"
        if failure == "external"
        else "disqualified"
    )
    assert value["verdict_class"] == expected
    assert value["diagnosis_rows_ready_score"] == (failure == "none")
    assert value["objective_candidate_ready_score"] == (failure == "none")


def test_scenario_verify_8828_raw_features(tmp_path, monkeypatch):
    """Snapshot features are reconstructed from admitted inputs, even after rehashing."""
    path, value = snapshot(tmp_path, monkeypatch)
    payload = q.d.m.world("raw", "support")
    public = tmp_path / "public.json"
    source = tmp_path / "source.json"
    frozen = tmp_path / "frozen.json"
    q.atomic_json(public, dict(online=[payload], retention=[payload]))
    q.atomic_json(source, dict(input_refs=[q.d.reference(public)]))
    recipe = dict(unit_id="u", structural_hash="full")
    q.atomic_json(frozen, dict(streams=[dict(stream_id="s", worlds=[recipe], retention=[recipe])]))
    refs = [
        dict(path=str(q.ROOT / q.old.AUDIT), sha256=q.old.AUDIT_PIN),
        q.d.reference(source),
        q.d.reference(frozen),
    ]
    prior = q.d.load(dict(path=str(q.HISTORICAL), sha256=q.HISTORICAL_PIN))
    admitted = Path(prior["raw_shard_hashes"][0]["path"])
    q.atomic_json(admitted, dict(source_artifact_hashes=refs))
    prior["raw_shard_hashes"] = [q.d.reference(admitted)]
    q.atomic_json(q.HISTORICAL, prior)
    monkeypatch.setattr(q, "HISTORICAL_PIN", q.sha256_file(q.HISTORICAL))
    structure = q.d.structure(payload)
    value["inputs"] = [q.d.reference(source), q.d.reference(public)]
    q.atomic_json(tmp_path / "preconditions.json", dict(inputs=value["inputs"], code=value["code"]))
    value.update(
        source_artifact_hashes=refs,
        excluded=[structure],
        diagnostic_rows=[
            dict(
                unit_id="u",
                cluster_id="s",
                structure=structure,
                template_mode=mode,
                counts=q.d.m.observe(payload, [] if mode == "off" else [q.d.m.normalize(payload)])[
                    "counts"
                ],
                truth="support",
                full_structure="full",
            )
            for _ in range(2)
            for mode in ("off", "on")
        ],
    )
    q.atomic_json(path, value)
    monkeypatch.setattr(q.d, "development", lambda _: ([], {}))
    assert (
        q.load_snapshot(q.d.reference(path), verify=True)["diagnostic_rows"]
        == value["diagnostic_rows"]
    )
    value["diagnostic_rows"][0]["truth"] = "refute"
    q.atomic_json(path, value)
    with pytest.raises(ValueError, match="snapshot_evidence_drift"):
        q.load_snapshot(q.d.reference(path), verify=True)


def test_req_verify_8828_heartbeats_and_cli_dispatch(tmp_path, monkeypatch):
    """Elapsed clocks emit progress without changing work or measured deadlines."""
    from carnot.reporting import signal_replay_runner_8828 as r

    clock = iter(range(0, 10000, 31))
    monkeypatch.setattr(q.time, "monotonic", lambda: next(clock))
    path = tmp_path / "operand"
    path.write_text("bound")
    q.bindings([q.d.reference(path)], "heartbeat")
    q.verify_primitives(
        [
            dict(
                name="normalize",
                args=[q.d.m.world("clock", "support")],
                value=q.d.m.normalize(q.d.m.world("clock", "support")),
            )
        ]
    )
    monkeypatch.setattr(q, "child", lambda *args: None)
    assert r.main(["--child", "authenticate", "--raw-dir", str(tmp_path)]) == 0
    assert (
        r.main(
            [
                "--child",
                "controls",
                "--raw-dir",
                str(tmp_path),
                "--snapshot",
                str(path),
                "--snapshot-hash",
                q.sha256_file(path),
            ]
        )
        == 0
    )
    assert r.main(["--child", "controls"]) == 1


def test_scenario_verify_8828_real_fit_child(tmp_path):
    """A fresh interpreter reproduces all 400 updates of a real original fit."""
    prior = q.d.load(dict(path=str(q.HISTORICAL), sha256=q.HISTORICAL_PIN))
    evidence = q.d.load(prior["raw_shard_hashes"][0])
    path = tmp_path / "snapshot.json"
    q.atomic_json(
        path,
        dict(
            owner=str(tmp_path),
            development_rows=evidence["development_rows"],
            code=[
                q.d.reference(q.ROOT / p)
                for p in q.old.CODE + q.old.TESTS + q.OWNED + q.TESTS + [q.DESIGN]
            ],
        ),
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(q.ROOT / q.CLI),
            "--child",
            "fit-00",
            "--raw-dir",
            str(tmp_path),
            "--snapshot",
            str(path),
            "--snapshot-hash",
            q.sha256_file(path),
        ],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "fit-00.json").read_bytes())["value"] == evidence["fits"][0]
