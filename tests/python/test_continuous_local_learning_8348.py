"""REQ-SELF-8348 / REQ-VERIFY-8348 / REQ-REPORT-8348: delayed replay is causal."""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import continuous_local_learning_8348 as k
from carnot.reporting import continuous_local_learning_8348 as e
from carnot.reporting import continuous_local_execution_8348 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture
def bundle(tmp_path):
    """SCENARIO-SELF-8348-CAUSAL: private numeric controls never replace source data."""
    rows = [
        dict(slot=i, unit_id=str(i), source_cluster_id=str(i), x=[0.0, *([0.3] * 4)])
        for i in range(1, 129)
    ]
    rows[12]["x"] = None
    head = dict(arm="spline34", coefficients=[1.0, 0.0, *([0.0] * 32)], temperature=2.0)
    labels = tmp_path / "labels.json"
    atomic_json(
        labels,
        dict(
            rows=[
                dict(slot=i, unit_id=str(i), source_cluster_id=str(i), y=i % 2)
                for i in range(1, 129)
            ]
        ),
    )
    return dict(head=head, slots=rows, labels=dict(path=str(labels)), fit=rows[:10])


def test_temperature_gradient_and_equivalence(bundle):
    """SCENARIO-SELF-8348-CAUSAL: the deployed logit derivative includes temperature."""
    x = bundle["slots"][0]["x"]
    h = deepcopy(bundle["head"])
    sparse = k.learn(h, x, 1, "online_sparse")
    dense = k.learn(bundle["head"], x, 1, "online_dense")
    assert sparse["coefficients"] == dense["coefficients"]
    phi = k.kernel.design(x)
    expected = 0.01 * 0.5 / 2 * phi[4]
    assert sparse["coefficients"][4] == pytest.approx(expected)
    assert sparse["coefficients"][:2] == h["coefficients"][:2]
    assert k.learn(h, x, None, "online_sparse")["changed"] == []
    assert k.learn(h, None, 1, "online_sparse")["reason"] == "missing_features"
    assert k.learn(h, x, 1, "frozen_spline")["changed"] == []
    c = k.learn(h, x, 1, "calibration_only")
    assert c["coefficients"][1] > 0 and c["coefficients"][2:] == h["coefficients"][2:]
    with pytest.raises(ValueError):
        k.learn(h, x, 1, "unknown")


def test_trajectory_causality_retention_and_mutation(bundle, tmp_path):
    """SCENARIO-SELF-8348-DURABLE: all issues precede releases and windows stay sealed."""
    state = k.run(bundle, tmp_path / "normal")
    changed = k.run(bundle, tmp_path / "future", mutate_from=49)
    assert state["issued"][: 56 * 5] == changed["issued"][: 56 * 5]
    assert len(state["issued"]) == 480 and len(state["updates"]) == 440
    assert len(state["pending"]) == 8 and len(state["retention"]) == 640
    assert (
        state["arms"]["online_sparse"]["coefficients"]
        == state["arms"]["online_dense"]["coefficients"]
    )
    assert all(len(u["basis_support"]) <= 16 for u in state["updates"])
    events = k.journal(tmp_path / "normal" / "events.jsonl")
    assert all(events.index(v) > 0 for v in events if v["kind"] == "release")
    assert all(v["release_slot"] == v["label_slot"] + 8 for v in state["releases"])
    assert k.run(bundle, tmp_path / "normal") == state
    assert k.reachability(bundle)[0]["attainable_logit_change_bound"] > 0
    assert k.control(bundle["head"])["updates"] == 88


def test_label_barrier_and_duplicate(bundle, tmp_path):
    """SCENARIO-SELF-8348-CAUSAL: seals authorize only one due original label."""
    vault = k.Labels(Path(bundle["labels"]["path"]), bundle["slots"])
    for slot, clock, sealed in [(1, 8, True), (97, 105, True), (1, 9, False)]:
        with pytest.raises(ValueError):
            vault.release(slot, clock, sealed=sealed)
    assert vault.release(1, 9, sealed=True)["y"] == 1
    state = k.initial(bundle)
    state["pending"] = [1]
    k.release(state, bundle, dict(slot=1, unit_id="1", source_cluster_id="1", y=1), 9)
    with pytest.raises(ValueError):
        k.release(state, bundle, dict(slot=1, unit_id="1", source_cluster_id="1", y=1), 9)
    with pytest.raises(ValueError):
        k.release(state, bundle, dict(slot=2, unit_id="2", source_cluster_id="2", y=1), 11)


def test_global_and_incomplete_index(bundle):
    """SCENARIO-VERIFY-8348-REPLAY: incomplete metadata conservatively invalidates all."""
    state = k.initial(bundle)["arms"]["online_sparse"]
    state["index_version"] = -1
    result = k.kernel.invalidate(state, [2], "indexed")
    assert result["fallback"] and len(result["invalidated"]) == 96
    result = k.kernel.invalidate(state, [1], "indexed")
    assert result["global_change"] and len(result["invalidated"]) == 96
    clipped = deepcopy(bundle["head"])
    clipped["coefficients"][2:] = [4.0] * 32
    assert max(k.learn(clipped, [0, 0, 0, 0, 0], 1, "online_sparse")["coefficients"]) <= 4
    with pytest.raises(ValueError):
        k.kernel.design([0, 2, 0, 0, 0])


def test_missing_external_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8348-CLI: absent external inputs block, owned failures disqualify."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(name="control", passed=True)])
    assert value["verdict_class"] == "blocked" and value["trajectory_ready_score"] == 0
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    bad = dict(value, trajectory_ready_score=1)
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(path, bad)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing")
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"


def test_independent_numeric_proof(bundle, tmp_path):
    """SCENARIO-VERIFY-8348-REPLAY: only recomputed arithmetic resolves exact zeros."""
    work = dict(bundle=bundle, state=k.run(bundle, tmp_path / "numeric"))
    proof = k.numeric_proof(work)
    assert proof["recomputed"] and proof["deliberate_error_rejected"]
    changed = deepcopy(work)
    next(u for u in changed["state"]["updates"] if u["arm"] == "online_dense")["coefficients"][
        2
    ] += 0.1
    assert not k.numeric_proof(changed)["recomputed"]
    assert not k.numeric_proof(dict(state={}))["recomputed"]


def test_real_worker_cli_restart(bundle, tmp_path):
    """SCENARIO-SELF-8348-DURABLE: actual child exits73 resume exactly."""
    path = tmp_path / "bundle.json"
    atomic_json(path, bundle)
    normal = tmp_path / "normal"
    assert r.check(
        dict(
            name="normal",
            argv=r.cli() + ["--worker", str(path), "--worker-dir", str(normal)],
            deadline_s=120,
        ),
        tmp_path / "logs",
    )["passed"]
    expected = json.loads((normal / "final.json").read_bytes())
    for slot in (32, 64):
        dest = tmp_path / f"crash{slot}"
        args = r.cli() + ["--worker", str(path), "--worker-dir", str(dest)]
        assert r.check(
            dict(
                name=f"crash{slot}",
                argv=args + ["--crash", str(slot)],
                expected_exit=73,
                deadline_s=120,
            ),
            tmp_path / "logs",
        )["passed"]
        assert r.check(dict(name=f"resume{slot}", argv=args, deadline_s=120), tmp_path / "logs")[
            "passed"
        ]
        assert json.loads((dest / "final.json").read_bytes()) == expected
    cp = normal / "state.json"
    value = json.loads(cp.read_bytes())
    value["cursor"] = 1
    atomic_json(cp, value)
    with pytest.raises(ValueError, match="checkpoint_journal_drift"):
        k.run(bundle, normal)


def test_cli_block_publication_and_arguments(tmp_path):
    """SCENARIO-REPORT-8348-CLI: external blocks still publish through real validators."""
    output = tmp_path / (e.NAME + ".json")
    receipt = r.check(
        dict(
            name="blocked_cli",
            argv=r.cli()
            + ["--root", str(tmp_path / "missing"), "--output", str(output), "--private"],
            deadline_s=120,
        ),
        tmp_path / "logs",
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert r.main(["--cold-replay", str(output)]) == 0
    for args in (["--date", "bad"], ["--private"], ["--worker", str(output)]):
        with pytest.raises(SystemExit):
            r.main(args)
    receipt = r.check(
        dict(
            name="absent",
            argv=r.cli() + ["--cold-replay", str(tmp_path / "absent")],
            expected_exit=1,
            deadline_s=60,
        ),
        tmp_path / "logs",
    )
    assert receipt["passed"]


@pytest.fixture(scope="module")
def natural(tmp_path_factory):
    """SCENARIO-REPORT-8348-CLI: use natural operands independently of fixture data."""
    raw = tmp_path_factory.mktemp("8348-natural")
    work = e.measure(e.ROOT, raw)
    assert not work["failures"] and all(work["checks"].values())
    return work, raw


def test_natural_accounting_and_cold_replay(natural, tmp_path):
    """SCENARIO-VERIFY-8348-REPLAY: rehashed state or source forgeries fail replay."""
    work, raw = natural
    value = e.build(work, raw, [dict(name="private_unit_control", passed=True)])
    assert value["trajectory_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["independent_count"] == 74 and value["completed_count"] == 370
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    altered = deepcopy(work)
    altered["state"]["issued"][0]["p"] += 0.1
    target_raw = tmp_path / "tamper"
    atomic_json(target_raw / "measurement.json", altered)
    for p in raw.glob("retention-*.json"):
        (target_raw / p.name).write_bytes(p.read_bytes())
    forged = e.build(altered, target_raw, [dict(passed=True)])
    atomic_json(path, forged)
    assert not e.replay(path)
    assert all(r["targets_opened"] is False for r in value["retention_shadow_rows"])


def test_failures_and_manifest(natural, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8348-CLI: owned worker failure never becomes a partial result."""
    work, raw = natural
    changed = deepcopy(work)
    changed["checks"]["exact_restart"] = False
    atomic_json(tmp_path / "measurement.json", changed)
    for p in raw.glob("retention-*.json"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    assert e.build(changed, tmp_path, [dict(passed=True)])["verdict_class"] == "disqualified"
    plan = r.manifest(tmp_path, tmp_path / "candidate.json")
    assert plan["no_full_repository_suite"] and "--files" in plan["commands"][-1]["argv"]
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    monkeypatch.setattr(e, "child", lambda *a, **kw: dict(passed=False))
    failed = e.measure(e.ROOT, tmp_path / "worker-failure")
    assert failed["checks"] == dict(worker_exits=False)
    assert (
        e.build(failed, tmp_path / "worker-failure", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )


def test_missing_due_feedback_does_not_create_shuffle_label(bundle):
    """SCENARIO-SELF-8348-CAUSAL: a missing due label gives no arm a training target."""
    state = k.initial(bundle)
    state["pending"] = [1, 2]
    k.release(state, bundle, dict(slot=1, unit_id="1", source_cluster_id="1", y=1), 9)
    k.release(state, bundle, dict(slot=2, unit_id="2", source_cluster_id="2", y=None), 10)
    assert all(u["changed"] == [] for u in state["updates"][-5:])
    with pytest.raises(ValueError, match="feedback_identity"):
        state["pending"] = [3]
        k.release(state, bundle, dict(slot=3, unit_id="foreign", source_cluster_id="3", y=1), 11)


def test_failure_paths_and_gradient_cap(bundle, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8348-REPLAY: malformed labels, stale cache and deadline fail closed."""
    labels = json.loads(Path(bundle["labels"]["path"]).read_bytes())
    labels["rows"][0]["unit_id"] = "foreign"
    atomic_json(Path(bundle["labels"]["path"]), labels)
    with pytest.raises(ValueError, match="label_identity"):
        k.Labels(Path(bundle["labels"]["path"]), bundle["slots"]).release(1, 9, sealed=True)
    head = deepcopy(bundle["head"])
    head["temperature"] = 0.5
    result = k.learn(head, [0, 0, 0, 0, 0], 1, "online_sparse")
    assert result["gradient_norm"] > 1
    assert sum(v * v for v in result["coefficient_delta"]) == pytest.approx(0.01**2)
    original = k.initial

    def stale(b):
        s = original(b)
        s["arms"]["online_sparse"]["cache"]["1"]["p"] = 0.1
        return s

    monkeypatch.setattr(k, "initial", stale)
    with pytest.raises(ValueError, match="stale_cache"):
        k.run(bundle, tmp_path / "stale")
    monkeypatch.setattr(k, "initial", original)
    ticks = iter([0, 121])
    monkeypatch.setattr(k.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="trajectory_deadline"):
        k.run(bundle, tmp_path / "timeout")


def test_incomplete_dependency_metadata_rebuilds(bundle):
    """SCENARIO-VERIFY-8348-REPLAY: incomplete metadata forces full recomputation."""
    state = k.initial(bundle)
    state["pending"] = [1]
    state["arms"]["online_sparse"]["index"] = {}
    k.release(state, bundle, dict(slot=1, unit_id="1", source_cluster_id="1", y=1), 9)
    update = next(u for u in state["updates"] if u["arm"] == "online_sparse")
    assert update["fallback"] and len(update["invalidated"]) == 96
    assert set(state["arms"]["online_sparse"]["index"]) == {str(i) for i in range(34)}


def test_normal_command_boundary_and_coverage_retention(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8348-CLI: the normal mode executes frozen bounded commands."""
    import coverage

    original_manifest = r.manifest

    def small_plan(private, candidate):
        plan = original_manifest(private, candidate)
        current = coverage.Coverage.current()
        if current is not None:
            current.save()
            current.json_report(outfile=str(private / "coverage.json"))
        plan["commands"] = [
            dict(
                name="bounded_validation_child",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    "-c",
                    "print('real bounded validation child')",
                ],
                deadline_s=20,
            )
        ]
        return plan

    monkeypatch.setattr(r, "manifest", small_plan)
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["validation_receipts"][0]["actual_exit"] == 0


def test_authentication_failures_and_tamper_bindings(natural, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8348-REPLAY: bad byte bindings and source identity cannot pass."""
    work, raw = natural
    original_bind = e.bind

    def changed_bundle(work, ref, raw, **kwargs):
        v = original_bind(work, ref, raw, **kwargs)
        if Path(ref["path"]).name == "reserved.json" and "predictors" in ref["path"]:
            v["rows"][0]["unit_id"] = "foreign"
        return v

    monkeypatch.setattr(e, "bind", changed_bundle)
    bad = e.measure(e.ROOT, tmp_path / "foreign")
    assert bad["failures"][-1]["observed"] == "frozen_source_order"
    monkeypatch.setattr(e, "bind", original_bind)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, raw, [dict(passed=True)])
    changed = deepcopy(value)
    changed["measurement_reference"]["sha256"] = "sha256:wrong"
    atomic_json(output, changed)
    assert not e.replay(output)
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "sha256:wrong"
    atomic_json(output, changed)
    assert not e.replay(output)


def test_typed_finding_policy_uses_recomputation(natural, tmp_path):
    """SCENARIO-VERIFY-8348-REPLAY: an exit1 is resolved only with independent proof."""
    work, raw = natural
    path = tmp_path / "candidate.json"
    atomic_json(path, e.build(work, raw, [dict(passed=True)]))
    positive = r.findings.audit(path, tmp_path / "valid", k.numeric_proof(work))
    assert positive["passed"] and positive["findings"]
    negative = r.findings.audit(
        path, tmp_path / "deliberate", dict(recomputed=False, deliberate_error_rejected=True)
    )
    assert not negative["passed"] and negative["findings"] == positive["findings"]


@pytest.mark.parametrize(
    "field", ["bundle", "authority", "state", "control", "retention", "crash", "journal"]
)
def test_each_replay_boundary_rejects_rehashed_evidence(natural, tmp_path, monkeypatch, field):
    """SCENARIO-VERIFY-8348-REPLAY: each reducer boundary rejects a rehashed forgery."""
    work, source_raw = natural
    changed = deepcopy(work)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ["uninterrupted", "crash32", "crash64", "future"]:
        if field == "crash" and name == "crash32":
            (raw / name).mkdir()
            atomic_json(raw / name / "final.json", dict(cursor=-1))
        else:
            (raw / name).symlink_to(source_raw / name, target_is_directory=True)
    for path in source_raw.glob("retention-*.json"):
        (raw / path.name).write_bytes(path.read_bytes())
    if field == "bundle":
        changed["bundle"]["head"]["temperature"] = 0.5
    elif field == "authority":
        changed["authority"]["activated"] = False
    elif field == "state":
        changed["state"]["issued"][0]["p"] += 0.1
    elif field == "control":
        changed["update_budget_control"]["passed"] = False
    elif field == "retention":
        data = json.loads((raw / "retention-0.json").read_bytes())
        data["rows"][0]["p"] = 0.1
        atomic_json(raw / "retention-0.json", data)

    def copied_reconstruction(bundle, scratch, **kwargs):
        if field != "journal":
            (scratch / "events.jsonl").write_bytes(
                (source_raw / "uninterrupted/events.jsonl").read_bytes()
            )
        return deepcopy(work["state"])

    monkeypatch.setattr(k, "run", copied_reconstruction)
    atomic_json(raw / "measurement.json", changed)
    value = e.build(changed, raw, [dict(passed=True)])
    path = tmp_path / "forged.json"
    atomic_json(path, value)
    assert not e.replay(path)


def test_projection_pin_and_readiness_guards(natural, monkeypatch):
    """SCENARIO-VERIFY-8348-REPLAY: pinned identity and readiness remain explicit."""
    work, _ = natural
    bad = deepcopy(work)
    bad["refs"][0]["sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="upstream_pin"):
        e.authenticated_projection(bad)
    real_loads = e.json.loads

    def missing_ready(payload, *args, **kwargs):
        v = real_loads(payload, *args, **kwargs)
        if isinstance(v, dict) and v.get("experiment_id") == 8346:
            v["frozen_heads_ready_score"] = 0
        return v

    monkeypatch.setattr(e.json, "loads", missing_ready)
    with pytest.raises(ValueError, match="upstream_readiness"):
        e.authenticated_projection(work)
