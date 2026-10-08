"""REQ-VERIFY-8305 / REQ-REPORT-8305: custody without current inference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import cached_sentence_custody_8305 as e


@pytest.fixture
def work(tmp_path):
    """Read authentic cached receipts; tests write only into private scratch."""
    return e.measure(e.ROOT, tmp_path / "raw")


def test_reconstruction_and_target_separation(work, tmp_path):
    """SCENARIO-VERIFY-8305-CUSTODY: count slots rather than quota filling."""
    r = e.reconstruct(work["bundle"])
    assert r["historical_capture_counts"]["fit_tune_transport_completed"] == 188
    assert r["historical_capture_counts"]["fit_tune_feature_rows"] == 158
    assert r["historical_capture_counts"]["fit_tune_labeled_rows"] == 157
    assert [
        r["class_support_by_role"][k]["usable"]
        for k in (
            "fit",
            "calibration",
            "comparator_selection",
            "reserved",
            "stream",
            "later",
            "retention",
        )
    ] == [104, 25, 28, 97, 74, 67, 23]
    assert r["fit_support_ready_score"] == 1
    assert len(r["rows"]) == 320
    assert all(
        "y" not in row and "historical_paired_control" not in row
        for shard in r["predictors"].values()
        for row in shard
    )
    e.check_predictor(r["predictors"]["reserved"][0])
    bad = dict(r["predictors"]["reserved"][0], nested={"y": 0})
    with pytest.raises(ValueError, match="predictor_fields"):
        e.check_predictor(bad)
    for field, value in [("x", [float("nan")] * 16), ("x", [0.0] * 12 + [2.0] * 4)]:
        with pytest.raises(ValueError, match="finite_features"):
            e.check_predictor(dict(r["predictors"]["reserved"][0], **{field: value}))
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    value["completed_count"] -= 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)


@pytest.mark.parametrize(
    "mutation", ["target", "role", "reply", "source", "identity", "label", "roster"]
)
def test_structural_canaries(work, mutation):
    """REQ-VERIFY-8305: identity and targets cannot authorize predictor input."""
    b = deepcopy(work["bundle"])
    slot = b["slots"][0]
    if mutation == "target":
        slot["y"] = 0
    elif mutation == "role":
        b["slots"][128]["source_cluster_id"] = slot["source_cluster_id"]
    elif mutation == "reply":
        b["calls"][0]["response"]["choices"][0]["message"]["content"] += "!"
    elif mutation == "source":
        slot["source_bytes"] = b"Changed source.".hex()
    elif mutation == "identity":
        b["calls"][0]["identity"] = {}
        b["reply_hashes"] = [canonical_hash(c) for c in b["calls"]]
    elif mutation == "label":
        b["labels"][slot["unit_id"]] = 2
    else:
        b["protocol"]["original_roles"]["fit"].reverse()
    with pytest.raises(ValueError):
        e.reconstruct(b)


def test_support_and_external_block(work, tmp_path):
    """REQ-REPORT-8305: support and custody measure different obligations."""
    b = deepcopy(work["bundle"])
    b["labels"] = dict.fromkeys(b["labels"], None)
    assert e.reconstruct(b)["fit_support_ready_score"] == 0
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked")
    value = e.build(blocked, tmp_path / "blocked", [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert next(c for c in value["gate_check_summary"] if not c["passed"])["observed"] is None
    assert value["cached_cohort_ready_score"] == 0
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    path = tmp_path / "missing.json"
    assert not e.replay(path)
    atomic_json(path, {})
    assert not e.replay(path)


def test_byte_tamper_rejected(work, tmp_path):
    """REQ-VERIFY-8305: a changed shard invalidates the frozen evidence."""
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    shard = Path(value["predictor_shards"]["reserved"]["path"])
    shard.chmod(0o600)
    shard.write_text("{}")
    assert not e.replay(path)


def test_cli_replay_and_date(work, tmp_path):
    """SCENARIO-REPORT-8305-CLI: direct children work outside the checkout."""
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    prefix = [sys.executable]
    if os.environ.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    result = subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        cwd=tmp_path,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    value["completed_count"] -= 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert (
        subprocess.run(
            [*prefix, str(e.ROOT / e.CLI), "--cold-replay", str(path)],
            cwd=tmp_path,
            capture_output=True,
            timeout=180,
        ).returncode
        == 1
    )
    assert (
        subprocess.run(
            [*prefix, str(e.ROOT / e.CLI), "--date", "20000101"],
            cwd=tmp_path,
            capture_output=True,
            timeout=30,
        ).returncode
        == 2
    )


def test_authentication_structure_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8305: malformed external evidence names its failed operand."""
    root = tmp_path / "repo"
    p = root / "results" / "experiment_8304_v717_contract_methods.json"
    atomic_json(p, dict(required_checks_passed=True, flagged_adversarial=False))
    monkeypatch.setattr(e, "PINS", {p.stem: e.reference(p)["sha256"][7:]})
    plan = e.authenticate(root, tmp_path / "inputs")
    assert plan["checks"][-1]["artifact_field"] == "structure"


def test_gate_report_keeps_individual_targets_private(tmp_path):
    """REQ-VERIFY-8305: public custody gates cannot expose individual gold targets."""
    plan = e.authenticate(e.ROOT, tmp_path / "protected")
    gates = [c for c in plan["checks"] if c["artifact_field"] == "original_human_target"]
    assert len(gates) == 128
    assert all(c["expected"] is True and c["observed"] is True for c in gates)


def test_rehashed_shard_and_log_failures(work, tmp_path):
    """REQ-VERIFY-8305: byte and semantic replay detect different failures."""
    p = tmp_path / "candidate.json"
    log = tmp_path / "log"
    log.write_text("actual private validation output")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=e.reference(log)["sha256"])]
    value = e.build(work, tmp_path / "raw", receipts)
    atomic_json(p, value)
    assert e.replay(p)
    log.write_text("changed")
    assert not e.replay(p)
    receipts[0]["log_sha256"] = e.reference(log)["sha256"]
    ref = work["predictor_shards"]["reserved"]
    shard = Path(ref["path"])
    data = json.loads(shard.read_text())
    data["rows"][0]["x"][12] = 0.123
    atomic_json(shard, data)
    work["predictor_shards"]["reserved"] = e.reference(shard)
    atomic_json(tmp_path / "raw" / "measurement.json", work)
    atomic_json(p, e.build(work, tmp_path / "raw", receipts))
    assert not e.replay(p)


def test_request_and_baseline_identity(work, monkeypatch):
    """REQ-VERIFY-8305: public bytes and matched baseline roles must agree."""
    b = deepcopy(work["bundle"])
    b["slots"][0]["answer_bytes"] = b"Different answer.".hex()
    with pytest.raises(ValueError, match="public_answer_requests"):
        e.reconstruct(b)
    baseline = deepcopy(e.public_baseline(json.dumps(work["bundle"]["base_calls"], sort_keys=True)))
    baseline[b["slots"][0]["unit_id"]]["role"] = "evaluation"
    monkeypatch.setattr(e, "public_baseline", lambda _: baseline)
    with pytest.raises(ValueError, match="baseline_source_identity"):
        e.reconstruct(work["bundle"])


def test_rehashed_primitive_anchors(work, tmp_path):
    """REQ-VERIFY-8305: modified measurements cannot replace pinned observations."""
    b = deepcopy(work)
    b["bundle"]["calls"][0]["response"]["choices"][0]["message"]["content"] = "0|B|0.00|[]"
    b["bundle"]["reply_hashes"] = [canonical_hash(c) for c in b["bundle"]["calls"]]
    assert e.bound_bundle(b) != b["bundle"]
    b = deepcopy(work)
    unit = next(k for k, y in b["bundle"]["labels"].items() if y in (0, 1))
    b["bundle"]["labels"][unit] = 1 - b["bundle"]["labels"][unit]
    atomic_json(tmp_path / "raw" / "measurement.json", b)
    path = tmp_path / "rehashed.json"
    atomic_json(path, e.build(b, tmp_path / "raw", [dict(passed=True)]))
    assert not e.replay(path)
    first = next(iter(b["anchors"]))
    b["anchors"][first]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="primary_anchor"):
        e.bound_bundle(b)
    b = deepcopy(work)
    original = next(p for p in b["origins"] if p.endswith("primitive_calls.json"))
    b["origins"][original]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="primitive_anchor"):
        e.bound_bundle(b)


def test_private_main_and_failure_recovery(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8305-CLI: actual children and failed publication recover."""
    from carnot.reporting import cached_sentence_execution_8305 as runner

    output = tmp_path / "private" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--fixture-output", str(output)]) == 0
    assert Path(json.loads(output.read_text())["terminal_validation_sidecar_path"]).is_file()
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--worker-output",
                str(tmp_path / "worker" / "measurement.json"),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit, match="2"):
        e.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    value = json.loads(output.read_text())
    raw = Path(value["measurement_reference"]["path"]).parent
    spec = dict(
        name="actual_failed_validator",
        argv=["/bin/false"],
        deadline_s=10,
        expected_exit=0,
        classification="required",
    )
    runner.publish(value, output, tmp_path / "scratch", raw, [spec], False)
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    original = runner.manifest

    def bounded(private, candidate):
        specs = original(private, candidate)
        specs["commands"] = [
            dict(
                name="actual_child",
                argv=["/bin/true"],
                deadline_s=10,
                expected_exit=0,
                classification="required",
            )
        ]
        (private / "coverage.json").write_text('{"private_path_control":true}')
        return specs

    monkeypatch.setattr(runner, "manifest", bounded)
    target = tmp_path / "normal" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(target)]) == 0
    assert json.loads(target.read_text())["verdict_class"] == "blocked"
