"""REQ-REPORT-8517, REQ-VERIFY-8517: causal objective learning with original custody."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import brier_online_learning_8517 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import objective_stream_8517 as e


def fixture(tmp_path, arm="brier"):
    """A private evaluator holds labels while the learner receives only features."""
    blocks = [state.questions(b) for b in range(5)]
    labels = [
        dict(unit_id=a["id"], y=(i + j) % 2)
        for block in blocks
        for i, q in enumerate(block)
        for j, a in enumerate(q["answers"])
    ]
    tune = [
        dict(unit_id=a["id"], question_hash=q["id"], x=a["x"], y=j % 2)
        for block in blocks[:3]
        for q in block
        for j, a in enumerate(q["answers"])
    ][:150]
    atomic_json(tmp_path / "labels.json", dict(rows=labels))
    atomic_json(tmp_path / "tune.json", dict(rows=tune))
    cfg = e.configuration(
        state.head(),
        blocks,
        17,
        arm,
        r.reference(tmp_path / "labels.json"),
        r.reference(tmp_path / "tune.json"),
    )
    atomic_json(tmp_path / "configuration.json", cfg)
    return cfg


@pytest.mark.parametrize("arm", e.ARMS)
def test_stream_and_actual_restart(tmp_path, arm):
    """SCENARIO-VERIFY-8517-STREAM, SCENARIO-VERIFY-8517-RESTART: exact effects survive exit73."""
    cfg = fixture(tmp_path, arm)
    expected = e.worker(cfg, tmp_path / "baseline")
    assert expected["cursor"] == 100 and expected["version"] == 10
    assert expected["parameter_updates"] == (0 if arm == "frozen" else 120)
    assert expected["pending_high_water"] == 24 and expected["pending"] == []
    assert (expected["head"] == cfg["head"]) == (arm == "frozen")
    argv = [
        sys.executable,
        "-u",
        str(r.ROOT / r.CLI),
        "--configuration",
        str(tmp_path / "configuration.json"),
        "--worker",
        str(tmp_path / "resumed"),
    ]
    died = subprocess.run([*argv, "--crash"], capture_output=True, timeout=90)
    assert died.returncode == 73, died.stderr.decode()
    resumed = subprocess.run(argv, capture_output=True, timeout=90)
    assert resumed.returncode == 0, resumed.stderr.decode()
    assert json.loads((tmp_path / "resumed/final.json").read_bytes()) == expected
    left, right = e.shipped.steps(tmp_path / "baseline"), e.shipped.steps(tmp_path / "resumed")
    assert [e.shipped.semantic_step(s) for s in left] == [e.shipped.semantic_step(s) for s in right]
    for step in left:
        assert e.check_step(cfg, step) == step["after_state"]
        assert len(step["released_feedback"]) == 24
        assert step["issued_wall_ns"] < step["released_wall_ns"]
    assert len(e.shipped.retention(cfg, left)["rows"]) == 750
    costs = json.loads((tmp_path / "baseline/costs.json").read_bytes())
    assert len(costs["updates"]) == (0 if arm == "frozen" else 120)
    assert len(costs["commits"]) == 11 and costs["peak_rss_bytes"] > 0


def test_label_blind_admission_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8517-STREAM: all arms share IDs; rehashed coefficient effects fail."""
    chosen = []
    for arm in e.ARMS:
        cfg = fixture(tmp_path, arm)
        with e.binding(cfg):
            sealed = e.shipped.state.seal(e.shipped.state.initial(cfg), cfg, 0, cfg["blocks"][0])
            chosen.append(sealed["selected_cluster_ids"])
    assert chosen.count(chosen[0]) == 4
    cfg = fixture(tmp_path)
    e.worker(cfg, tmp_path / "stream")
    step = deepcopy(e.shipped.steps(tmp_path / "stream")[0])
    step["after_state"]["head"]["parameters"][0] += 0.01
    step["checkpoint_hash"] = canonical_hash(step["after_state"])
    with pytest.raises(ValueError, match="drift"):
        e.check_step(cfg, step)
    with pytest.raises(ValueError):
        e.configuration(state.head(), cfg["blocks"], 17, "foreign", {}, {})


def test_missing_qualification_and_cli(tmp_path):
    """SCENARIO-REPORT-8517-CUSTODY, SCENARIO-VERIFY-8517-CLI: absence remains explicit."""
    checked = r.inspect(tmp_path / "missing", tmp_path / "raw", tmp_path)
    assert checked["failures"]
    assert any(f["observed"] is None for f in checked["failures"])
    from carnot.reporting import brier_online_runner_8517 as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(["--worker", str(tmp_path)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261013"])


def test_private_publication_and_manifest(tmp_path):
    """SCENARIO-VERIFY-8517-CLI: missing evidence publishes an honest validated disqualification."""
    from carnot.reporting import brier_online_runner_8517 as runner

    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.main(["--private-control"]) == 2
    assert (
        runner.main(
            ["--private-control", "--root", str(tmp_path / "missing"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["objective_stream_ready_score"] == 0
    assert value["censored_count"] == 6000
    assert r.replay(output)
    changed = deepcopy(value)
    changed["objective_stream_ready_score"] = 1
    changed["reproducibility_checksum"] = r.checksum(changed)
    atomic_json(output, changed)
    assert not r.replay(output)
    plan = runner.manifest(tmp_path / "checks")
    union = [n for p in plan for n in p.get("collected_node_ids", [])]
    ledger = json.loads((tmp_path / "checks/historical_collection.json").read_bytes())
    assert union == ledger["node_ids"] and len(union) == len(set(union))
    assert all("--no-cov" in p["argv"] for p in plan if p.get("collected_node_ids"))


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """REQ-REPORT-8517: original authenticated operands produce a complete causal stream."""
    directory = tmp_path_factory.mktemp("measured8517")
    from unittest.mock import patch
    from carnot.reporting import brier_online_runner_8517 as runner
    from carnot.reporting.v709_execution import child

    def inline_fixture(name, argv, logs, **kwargs):
        """Unit fixtures execute numeric work directly; planned process deaths remain real."""
        if "--crash" in argv or not any(
            x in argv for x in ("--worker", "--release", "--retention")
        ):
            return child(name, argv, logs, **kwargs)
        status = runner.main(argv[3:])
        return child(
            name,
            [sys.executable, "-c", f"print('private numeric fixture'); raise SystemExit({status})"],
            logs,
            **kwargs,
        )

    with patch.object(r, "child", inline_fixture), patch.object(e.shipped, "child", inline_fixture):
        work = r.measure(r.ROOT, directory / "raw", directory / "private")
    assert not work["failures"] and len(work["cells"]) == 20
    assert all(c["status"] == "completed" for c in work["cells"])
    return work, directory


def test_readiness_and_primitive_reconstruction(measured):
    """SCENARIO-REPORT-8517-CUSTODY: readiness is independent of scientific contrast sign."""
    from carnot.reporting.v709_execution import child

    work, directory = measured
    receipts = [
        child(
            name,
            [sys.executable, "-c", "print('private gate fixture')"],
            directory / "gate_fixtures",
        )
        for name in sorted(r.REQUIRED_CHECKS)
    ]
    output = directory / (r.NAME + ".json")
    value = r.build(work, receipts, Path(work["raw"]), output)
    atomic_json(output, value)
    assert value["objective_stream_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["completed_count"] == 6000 and len(value["retention_rows"]) == 15000
    assert value["label_updates"] == 1800 and value["admission_count"] == 800
    assert value["lost_feedback_count"] == 0 and value["unadmitted_count"] == 1200
    assert (
        value["generalized_learning_benefit_score"]
        == value["independent_generalization_score"]
        == 0
    )
    assert r.replay(output)
    primitive = Path(value["primitive_reference"]["path"])
    mutations = [
        "source",
        "roster",
        "hash",
        "configuration",
        "before",
        "after",
        "retention",
        "restart",
        "receipt",
        "rows",
    ]
    for attack in mutations:
        changed = deepcopy(work)
        restore = []
        if attack == "source":
            changed["source"]["heads"][0]["parameters"][0] += 0.01
        elif attack == "roster":
            changed["cells"].reverse()
        elif attack == "hash":
            changed["code_refs"][0]["sha256"] = "wrong"
        elif attack == "configuration":
            ref = changed["cells"][0]["configuration"]
            path = Path(ref["path"])
            restore.append((path, path.read_bytes()))
            cfg = r.read(ref)
            cfg["seed"] = 29
            atomic_json(path, cfg)
            changed["cells"][0]["configuration"] = r.reference(path)
            changed["primitive_files"] = [
                r.reference(Path(x["path"])) for x in changed["primitive_files"]
            ]
        elif attack in ("before", "after"):
            ref = changed["cells"][0]["steps"][0]
            path = Path(ref["path"])
            restore.append((path, path.read_bytes()))
            step = r.read(ref)
            step[attack + "_state"]["head"]["parameters"][0] += 0.01
            step["checkpoint_hash"] = canonical_hash(step["after_state"])
            atomic_json(path, step)
            changed["cells"][0]["steps"][0] = r.reference(path)
            changed["primitive_files"] = [
                r.reference(Path(x["path"])) for x in changed["primitive_files"]
            ]
        elif attack == "retention":
            ref = changed["cells"][0]["retention"]
            path = Path(ref["path"])
            restore.append((path, path.read_bytes()))
            retained = r.read(ref)
            retained["rows"][0]["probability"] += 0.01
            atomic_json(path, retained)
            changed["cells"][0]["retention"] = r.reference(path)
            changed["primitive_files"] = [
                r.reference(Path(x["path"])) for x in changed["primitive_files"]
            ]
        elif attack == "restart":
            changed["restart_rows"][0]["passed"] = False
        elif attack == "receipt":
            changed["process_receipts"][0]["duration_s"] = -1
        else:
            path = Path(work["raw"]) / "stream_rows.json"
            restore.append((path, path.read_bytes()))
            rows = json.loads(path.read_bytes())
            rows["rows"][0]["probability"] += 0.01
            atomic_json(path, rows)
        atomic_json(primitive, changed)
        bad = r.build(changed, receipts, Path(work["raw"]), output)
        atomic_json(output, bad)
        assert not r.replay(output), attack
        for path, content in restore:
            path.write_bytes(content)
    atomic_json(primitive, work)
    atomic_json(output, value)
    assert r.replay(output)


def test_source_and_gate_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8517-CUSTODY: changed registered or original operands cannot authorize science."""
    original = r.source()
    assert len(original["heads"]) == 5 and len(original["blocks"]) == 5
    original_protocol = r.methods.protocol
    monkeypatch.setattr(
        r.methods,
        "protocol",
        lambda: dict(
            original_protocol(),
            objective_updates=dict(original_protocol()["objective_updates"], capacity=16),
        ),
    )
    with pytest.raises(ValueError, match="registered"):
        r.source()
    monkeypatch.setattr(r.methods, "protocol", original_protocol)
    monkeypatch.setattr(r, "sha256_file", lambda path: "wrong")
    with pytest.raises(ValueError, match="source_hash"):
        r.source()
    with pytest.raises(ValueError, match="original_primitives_unavailable"):
        r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "private")
    monkeypatch.undo()
    path = tmp_path / "inputs/results" / (r.upstream.NAME + ".json")
    atomic_json(path, dict(learning_qualification_ready_score=0))
    result = r.inspect(tmp_path / "inputs", tmp_path / "raw2", tmp_path)
    assert any(
        f.get("field") == "learning_qualification_ready_score" and f["observed"] == 0
        for f in result["failures"]
    )


def test_runner_dispatch_and_receipt_archival(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8517-CLI: bounded dispatch retains original dates and exact receipts."""
    from carnot.reporting import brier_online_runner_8517 as runner

    monkeypatch.setattr(
        runner.historical_runner, "historical", lambda args, script=False: 7 if script else 0
    )
    assert runner.main(["--historical-tests", "--date", "20261013"]) == 0
    assert runner.main(["--historical-exec", "old.py", "--date", "20261013"]) == 7
    assert runner.main(["--historical-tests"]) == 0
    assert runner.main(["--date", "20261014", "--historical-tests"]) == 0
    monkeypatch.setattr(runner.historical_runner, "summarize", lambda *args: 0)
    assert (
        runner.main(["--historical-summary", str(tmp_path / "ledger"), "--receipts", str(tmp_path)])
        == 0
    )
    with pytest.raises(SystemExit):
        runner.main(["--foreign"])
    plan = [
        dict(
            name="historical_shard_0",
            argv=[sys.executable, "-c", "print('actual shard')"],
            expected=0,
            deadline=30,
            scope="owned",
        ),
        dict(
            name="historical_shard_1",
            argv=[sys.executable, "-c", "print('second actual shard')"],
            expected=0,
            deadline=30,
            scope="owned",
        ),
        dict(
            name="historical_union",
            argv=[sys.executable, "-c", "print('actual union')", str(tmp_path / "archive")],
            expected=0,
            deadline=30,
            scope="owned",
        ),
    ]
    receipts = runner.qualify(plan, tmp_path / "logs")
    assert all(x["passed"] for x in receipts)
    assert (
        json.loads((tmp_path / "archive/historical_shard_0.receipt.json").read_bytes())
        == receipts[0]
    )
    assert runner.qualify([], tmp_path / "empty") == []
    cfg = fixture(tmp_path)
    bad = tmp_path / "bad_issue.json"
    atomic_json(bad, dict(state=dict(version=1, cursor=0), predictions=[]))
    monkeypatch.setattr(e.shipped, "predictions", lambda *args: [dict(id="expected")])
    assert (
        runner.main(
            [
                "--configuration",
                str(tmp_path / "configuration.json"),
                "--release",
                str(bad),
                "--operand-output",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert cfg["policy"] == "uniform"


def test_exactly_once_release_and_wrong_rule(tmp_path):
    """SCENARIO-VERIFY-8517-RESTART: redelivery cannot duplicate coefficients or change RNG."""
    cfg = fixture(tmp_path)
    with e.binding(cfg):
        store = e.shipped.state.Store(tmp_path / "store", cfg)
        store.initialize()
        store.seal(0, cfg["blocks"][0])
        labels = e.shipped.released(cfg, store.read())
        ack = store.release(0, labels)
        committed = store.read()
        assert store.release(0, labels) == ack
        assert store.read() == committed
        conflict = dict(labels)
        conflict[next(iter(conflict))] ^= 1
        with pytest.raises(ValueError, match="duplicate_conflict"):
            store.release(0, conflict)
        future = dict(labels, future_answer=1)
        with pytest.raises(ValueError):
            e.shipped.state.release(committed, cfg, 1, future)
        changed = dict(cfg, update_rule_hash="wrong")
        with pytest.raises(ValueError):
            e.shipped.state.Store(tmp_path / "store", changed).read()
    from carnot.reporting import brier_online_runner_8517 as runner

    assert (
        runner.main(
            [
                "--configuration",
                str(tmp_path / "configuration.json"),
                "--release",
                str(tmp_path / "not_issued"),
            ]
        )
        == 1
    )


def test_collection_archive_is_durable(tmp_path):
    """SCENARIO-VERIFY-8517-CLI: collection logs survive removal of private validation scratch."""
    from carnot.reporting import brier_online_runner_8517 as runner

    private = tmp_path / "private"
    atomic_json(private / "collection/collect.receipt.json", dict(passed=True))
    atomic_json(private / "historical_collection.json", dict(node_ids=["private::node"]))
    plan = [
        dict(
            name="historical_union",
            argv=[
                sys.executable,
                "-c",
                "print('union fixture')",
                "--historical-summary",
                str(private / "historical_collection.json"),
            ],
            expected=0,
            deadline=30,
            scope="owned",
        )
    ]
    receipts = runner.qualify(plan, tmp_path / "raw/logs")
    assert receipts[0]["passed"]
    refs = json.loads((tmp_path / "raw/collection_archive.json").read_bytes())
    assert len(refs) == 1 and r.read(refs[0])["passed"]


def test_replay_authority_and_bound_consumer_rejections(measured, monkeypatch):
    """SCENARIO-REPORT-8517-CUSTODY: primitive authority and bound consumer failures remain decisive."""
    work, directory = measured
    raw, output = Path(work["raw"]), directory / "consumer_rejection.json"
    primitive = raw / "measurement.json"
    cases = ["authority", "upstream_pin", "qualification", "sidecar", "final"]
    for attack in cases:
        changed = deepcopy(work)
        if attack == "authority":
            changed["authority"]["activated"] = False
        elif attack == "upstream_pin":
            ref = next(
                x
                for x in changed["source_refs"]
                if Path(x["path"]).name == r.upstream.NAME + ".json"
            )
            forged = directory / "foreign" / (r.upstream.NAME + ".json")
            atomic_json(forged, dict(flagged_adversarial=True))
            changed["source_refs"] = [
                r.reference(forged) if x == ref else x for x in changed["source_refs"]
            ]
        elif attack == "qualification":
            monkeypatch.setattr(
                r.old, "qualification_failures", lambda *args: [dict(field="flagged_adversarial")]
            )
        elif attack == "sidecar":
            monkeypatch.setattr(
                r, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False))
            )
        else:
            changed["cells"][0]["final"] = changed["cells"][1]["final"]
        atomic_json(primitive, changed)
        atomic_json(output, r.build(changed, [], raw, output))
        assert not r.replay(output), attack
        monkeypatch.undo()
    atomic_json(raw / "collection_archive.json", [])
    value = r.build(work, [], raw, output)
    assert value["collection_archive_reference"]
    (raw / "collection_archive.json").unlink()
    atomic_json(primitive, work)
