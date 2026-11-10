"""REQ-REPORT-8963 / SCENARIO-REPORT-8963-CONTRACT: custody precedes inference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import corpus_admission_8963 as d
from carnot.reporting import corpus_admission_runner_8963 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.corpus_views_8963 import detector_view, release_final_test


@pytest.fixture(scope="module")
def corpus():
    """REQ-REPORT-8963: use immutable producers rather than manufactured measurements."""
    evidence = d.authenticate(d.ROOT)
    assert not evidence["failures"]
    units, rows = d.join(evidence)
    result = d.reduce(units, rows, evidence["history"])
    return evidence, units, rows, result


def test_complete_contract(corpus):
    """REQ-REPORT-8963: two shards give the exact prospective denominator."""
    evidence, units, rows, result = corpus
    assert len(units) == len(rows) == result["completed_count"] == 1024
    assert result["role_counts"] == dict(calibration=128, test=128, online=768)
    assert result["class_support_passed"]
    assert result["independent_error_count"] >= 200
    assert all(r["count"] >= 5 for r in result["class_support_rows"])
    assert len(result["stream_counts"]) == 24
    assert set(result["stream_counts"].values()) == {32}
    assert result["labels_verified"] and result["usable_signals"]
    assert not evidence["history"]["failures"]
    assert all("label" not in r and "expected" not in r for r in result["corpus_manifest"])


@pytest.mark.parametrize("fault", ["duplicate", "role", "label", "expected", "history"])
def test_reject_drift(corpus, fault):
    """SCENARIO-REPORT-8963-CONTRACT: rehashing cannot repair a semantic change."""
    evidence, units, rows, _ = deepcopy(corpus)
    if fault == "duplicate":
        units[1] = deepcopy(units[0])
    elif fault == "role":
        units[0]["role"] = "other"
    elif fault == "label":
        rows[0]["value"] = 1 - rows[0]["value"]
    elif fault == "expected":
        units[0]["expected"] = []
    else:
        evidence["history"]["old_graphs"].append(units[0])
    with pytest.raises(ValueError):
        d.reduce(units, rows, evidence["history"])


def test_support_shortfall_and_failures(corpus, monkeypatch):
    """REQ-REPORT-8963: null support never changes the fixed role roster."""
    evidence, units, rows, _ = deepcopy(corpus)
    monkeypatch.setattr(
        d,
        "all_pairs",
        lambda nodes, edges: {
            u["start"]: {n: 2 if n in u["expected"] else 0 for n in nodes}
            for u in units
            if u["nodes"] == nodes
        },
    )
    for unit, row in zip(units, rows, strict=True):
        row.update(answer=unit["expected"][0], parser_category="correct", value=0, numerator=0)
    result = d.reduce(units, rows, evidence["history"])
    assert not result["class_support_passed"]
    assert result["independent_error_count"] == 0
    assert result["role_counts"] == dict(calibration=128, test=128, online=768)
    rows[0] = d.source.censored(units[0], "generation_failure")
    result = d.reduce(units, rows, evidence["history"])
    assert result["generation_failure_count"] == 1
    assert result["completed_count"] == 1023 and result["censored_count"] == 1


def test_label_and_oracle_mutation_isolation(corpus, tmp_path):
    """REQ-REPORT-8963: scorer inputs and scores cannot depend on truth sidecars."""
    _, units, rows, _ = corpus
    before = detector_view(units[:2], rows[:2])
    changed_units, changed_rows = deepcopy(units[:2]), deepcopy(rows[:2])
    for unit, row in zip(changed_units, changed_rows, strict=True):
        unit.update(expected=["ZZ"], distances={"ZZ": 1}, label=99)
        row.update(value=99, numerator=99, parser_category="changed", label=99)
    assert before == detector_view(changed_units, changed_rows)
    controls = d.isolation(units[:2], rows[:2], tmp_path)
    assert all(r["passed"] for r in controls)


def test_final_test_seal_boundary(tmp_path):
    """SCENARIO-REPORT-8963-CONTRACT: labels require exact complete saved predictions."""
    labels, seal = tmp_path / "labels.json", tmp_path / "predictions.json"
    atomic_json(labels, dict(rows=[dict(unit_id="a", label=1)]))
    with pytest.raises((ValueError, FileNotFoundError)):
        release_final_test(labels, seal)
    predictions = [dict(unit_id="a", p_error=0.2)]
    atomic_json(seal, dict(rows=predictions, prediction_hash=canonical_hash(predictions)))
    assert release_final_test(labels, seal) == [1]
    predictions[0]["unit_id"] = "other"
    atomic_json(seal, dict(rows=predictions, prediction_hash=canonical_hash(predictions)))
    with pytest.raises(ValueError):
        release_final_test(labels, seal)


def test_private_publication_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8963-CONTRACT: private cold CLI checks real publication bytes."""
    output = tmp_path / "results" / (d.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)

    def child(*args):
        completed = subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(completed.stdout, flush=True)
        return completed

    assert child("--date", "bad").returncode == 2
    assert (
        child("--date", "20261110", "--fixture-e2e", str(d.ROOT / "invalid.json")).returncode == 2
    )
    result = child("--date", "20261110", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert child("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["corpus_ready_score"] == 0
    value["corpus_ready_score"] = 1
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert child("--cold-replay", str(output)).returncode == 1
    assert not runner.replay(tmp_path / "missing")["passed"]


@pytest.mark.parametrize("fault", ["missing", "hash", "json", "field", "zero", "boolean", "raw"])
def test_authentication_failure_kinds(tmp_path, monkeypatch, fault):
    """REQ-REPORT-8963: exact upstream fields distinguish absence from rejected evidence."""
    path = tmp_path / "producer.json"
    value = dict(
        task_id="upstream",
        score=1,
        required_checks_passed=True,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    if fault == "field":
        del value["score"]
    elif fault == "zero":
        value["score"] = 0
    elif fault == "boolean":
        value["score"] = True
    elif fault == "raw":
        value["raw_shard_hashes"] = [dict(path=str(tmp_path / "gone"), sha256="sha256:expected")]
    if fault != "missing":
        path.write_text("bad JSON" if fault == "json" else json.dumps(value))
    pin = "wrong" if fault in ("missing", "hash") else d.calibration.reference(path)["sha256"]
    monkeypatch.setattr(d, "INPUTS", [(path.name, pin, "score", "upstream")])
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["upstream"] == "upstream"
    assert failure["failure_kind"] == {
        "missing": "missing_file",
        "field": "missing_field",
        "zero": "observed_zero",
        "raw": "missing_file",
    }.get(fault, "rejected_producer")


def test_blocked_slots_and_qualification(corpus, tmp_path, monkeypatch):
    """REQ-REPORT-8963: support nulls, external blocks and owned failures differ."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True)
    validation = dict(receipts=[], coverage={})
    value = d.build(work, validation, True)
    assert value["verdict_class"] == "blocked" and len(value["rows"]) == 1024
    monkeypatch.setattr(d.shard, "qualified", lambda *args: True)
    work["prior_terminal"] = dict(
        verdict_class="blocked",
        gate_check_summary=work["evidence"]["failures"],
        honest_verdict="complete_blocked_original_scope",
    )
    assert d.build(work, validation, False)["honest_verdict"] == "complete_blocked_original_scope"
    work["evidence"]["failures"] = []
    work["result"] = deepcopy(corpus[3])
    work["controls"] = [dict(passed=True)]
    assert d.build(work, validation, False)["corpus_ready_score"] == 1
    work["result"]["class_support_passed"] = False
    assert d.build(work, validation, False)["verdict_class"] == "null"
    work["controls"] = [dict(passed=False)]
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    work["controls"] = [dict(passed=True)]
    validation["terminal_rejected"] = True
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"


def test_inference_views_and_delayed_broker(corpus, tmp_path):
    """REQ-REPORT-8963: no held-out labels enter views; online release needs saved seals."""
    evidence, units, rows, result = corpus
    views, brokers = runner.write_views(tmp_path, evidence, units, rows, result)
    assert set(views) == {"fit", "guard", "calibration", "test", "online"}
    for role, count in (
        ("fit", 256),
        ("guard", 64),
        ("calibration", 128),
        ("test", 128),
        ("online", 768),
    ):
        view = json.loads(Path(views[role]["path"]).read_text())
        assert len(view["rows"]) == count
        assert all(set(r) == d.methods.g.KEYS for r in view["rows"])
        if role in ("test", "online"):
            assert "labels" not in view
    stream = brokers["online"]["streams"][0]
    broker = runner.LabelBroker(Path(stream["label_reference"]["path"]))
    seal = tmp_path / "predictions.json"
    assert broker.release(7, seal) == []
    with pytest.raises(ValueError):
        broker.release(8, seal)
    predictions = [dict(p_error=0.2)] * 32
    atomic_json(seal, dict(query=8, rows=predictions, prediction_hash=canonical_hash(predictions)))
    assert len(broker.release(8, seal)) == 4
    with pytest.raises(PermissionError):
        broker.release(8, seal, role="test")


def test_seal_rejects_bad_probabilities(tmp_path):
    """SCENARIO-REPORT-8963-CONTRACT: a rehashed invalid score is not a prediction."""
    path = tmp_path / "seal.json"
    for value in (True, float("nan"), -1, 2):
        rows = [dict(unit_id="a", p_error=value)]
        atomic_json(path, dict(rows=rows, prediction_hash=canonical_hash(rows)))
        with pytest.raises(ValueError):
            release_final_test(tmp_path / "unread_truth", path)


def test_successful_freeze_and_semantic_replay(corpus, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8963-CONTRACT: rebuilding views rejects rehashed scientific drift."""
    evidence, units, rows, result = corpus
    monkeypatch.setattr(d, "authenticate", lambda root: deepcopy(evidence))
    monkeypatch.setattr(d, "reduce", lambda *args: deepcopy(result))
    raw = tmp_path / "raw"
    work = runner.freeze(d.ROOT, raw, False)
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.calibration.reference(p)
        for p in raw.rglob("*.json")
        if "inputs" not in p.parts and "sources" not in p.parts
    ]
    atomic_json(raw / "work.json", work)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, True))
    assert runner.replay(output)["passed"]
    work["result"]["independent_error_count"] += 1
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]
    work["result"] = deepcopy(result)
    work["controls"][0]["passed"] = False
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]
    work["controls"][0]["passed"] = True
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, True)
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    value = deepcopy(d.build(work, validation, True))
    value["code_config_hashes"][0]["source_path"] = str(tmp_path / "unexecuted")
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    value = d.build(work, validation, True)
    atomic_json(output, value)
    monkeypatch.setattr(d, "authenticate", lambda root: dict(failures=["changed"]))
    assert not runner.replay(output)["passed"]


@pytest.mark.parametrize("kind", ["custody", "structure"])
def test_freeze_rejects_custody(corpus, tmp_path, monkeypatch, kind):
    """REQ-REPORT-8963: failed joins preserve a blocked complete schedule."""
    evidence = deepcopy(corpus[0])
    monkeypatch.setattr(d, "authenticate", lambda root: evidence)

    def fail(evidence):
        if kind == "structure":
            raise ValueError("broken")
        raise d.Rejected(
            dict(
                upstream="upstream",
                check="custody",
                artifact_field="response",
                expected="original",
                observed="changed",
                path="private",
                hash=None,
            )
        )

    monkeypatch.setattr(d, "join", fail)
    work = runner.freeze(tmp_path / "absent", tmp_path / "raw", False)
    assert work["evidence"]["failures"]
    assert len(work["result"]["rows"]) == 1024


def test_main_owned_checks_and_terminal_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8963-CONTRACT: owned failures disqualify without retrying science."""
    monkeypatch.delenv("CARNOT_8963_GLOBAL_RECEIPT", raising=False)
    original = runner.freeze(d.ROOT, tmp_path / "initial", True)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", lambda *args: deepcopy(original))
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda *args, **kwargs: [
            runner.CommandSpec("owned_unit_child", ("private",), "test"),
            runner.CommandSpec("other", ("private",), "test"),
        ],
    )
    monkeypatch.setattr(runner, "measured", lambda *args: dict(complete=True))
    private = tmp_path / "private"
    private.mkdir()
    (private / ".coverage").write_text("fake orchestration receipt")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(private))
    monkeypatch.setattr(runner, "run_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "replay", lambda *args: dict(passed=True))
    monkeypatch.setattr(
        d.calibration,
        "terminal_commands",
        lambda path: [runner.CommandSpec("terminal", (d.calibration.CLI,), "test")],
    )
    calls = []

    def publish(output, value, validator):
        calls.append(value)
        assert validator(output)["passed"]
        if len(calls) == 1:
            raise ValueError("candidate_rejected")
        return dict(primary_path=str(output))

    monkeypatch.setattr(runner, "publish_primary", publish)
    assert runner.main(["--date", "20261110"]) == 0
    assert calls[-1]["verdict_class"] == "disqualified"
    assert not calls[-1]["corpus_ready_score"]
    log = tmp_path / "global.log"
    log.write_text("private orchestration receipt")
    receipt = tmp_path / "global.json"
    atomic_json(
        receipt,
        dict(
            command_argv=[str(tmp_path / ".venv/bin/pytest"), "tests/python", "-q"],
            log_path=str(log),
            log_sha256=d.calibration.reference(log)["sha256"],
        ),
    )
    monkeypatch.setenv("CARNOT_8963_GLOBAL_RECEIPT", str(receipt))
    assert runner.main(["--date", "20261110"]) == 0
    log.write_text("changed")
    with pytest.raises(ValueError, match="global_receipt_custody"):
        runner.main(["--date", "20261110"])
    monkeypatch.delenv("CARNOT_8963_GLOBAL_RECEIPT")
    monkeypatch.setattr(
        runner,
        "publish_primary",
        lambda *args: (_ for _ in ()).throw(ValueError("conflicting_primary")),
    )
    with pytest.raises(ValueError, match="conflicting_primary"):
        runner.main(["--date", "20261110"])


def test_construction_failure_is_external(tmp_path, monkeypatch):
    """REQ-REPORT-8963: a pinned but invalid construction is rejected before joining."""
    path = tmp_path / "producer.json"
    atomic_json(
        path,
        dict(
            task_id="upstream",
            score=1,
            required_checks_passed=True,
            source_artifact_hashes=[],
            code_config_hashes=[],
            raw_shard_hashes=[],
            graph_manifest=[],
        ),
    )
    monkeypatch.setattr(
        d, "INPUTS", [(path.name, d.calibration.reference(path)["sha256"], "score", "upstream")]
    )
    monkeypatch.setattr(d.methods, "authenticate", lambda root: dict(failures=[], references=[]))
    monkeypatch.setattr(d.methods, "verify_graphs", lambda *args: False)
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["artifact_field"] == "source_construction_and_roles"


def test_roster_and_empty_reduction(corpus):
    """REQ-REPORT-8963: missing scheduled identities cannot silently shrink the corpus."""
    _, units, rows, _ = corpus
    with pytest.raises(ValueError, match="corpus_roster"):
        d.reduce(units, rows[:-1], {})
    with pytest.raises(ValueError, match="corpus_roster"):
        d.reduce(units[:1], rows[:1], {})
    result = d.reduce([], [], {})
    assert result["censored_count"] == 1024
    assert not result["class_support_passed"]


def test_duplicate_first_response_refusal(corpus, monkeypatch):
    """REQ-REPORT-8963: two references to shard A cannot masquerade as two shards."""
    evidence, units, _, _ = deepcopy(corpus)
    evidence["producers"][2] = evidence["producers"][1]
    monkeypatch.setattr(d.shard, "ordered_units", lambda *args: units[:512])
    with pytest.raises(ValueError, match="first_response_identity_overlap"):
        d.join(evidence)
