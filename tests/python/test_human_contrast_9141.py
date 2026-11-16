"""REQ-REPORT-9141-V779 / SCENARIO-REPORT-9141-V779: human custody and private CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import human_contrast_9141 as d
from carnot.reporting import human_contrast_runner_9141 as c


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Measure real original annotation evidence once, with no model access."""
    return c.measure(d.ROOT, tmp_path_factory.mktemp("contrast9141"), "20261116")


def test_original_labels_and_roles(work):
    """REQ-REPORT-9141-V779: all original siblings and missing annotations survive."""
    inspected = work["inspection"]
    assert all(r["passed"] for r in inspected["checks"])
    rows = inspected["rows"]
    assert len(rows) == 300 and sum(r["sibling_count"] for r in rows) == 900
    assert {role: sum(r["role"] == role for r in rows) for role in d.ROLES} == d.ROLES
    assert sum(r["a2_missing_count"] for r in rows) > 0
    assert all(r["a1_error_count"] + r["a1_correct_count"] == 3 for r in rows)
    assert all(
        r["agreement_count"] + r["disagreement_count"] == r["a2_observed_count"] for r in rows
    )
    assert len({r["unit_id"] for r in rows}) == 300
    reduced = d.reduce(inspected)
    assert sum(reduced["group_counts"].values()) == 300
    assert reduced["panel_resolution"]["TEST100"]["intended_qids"] == 100
    assert reduced["panel_resolution"]["future40"]["intended_qids"] == 40
    assert len(reduced["agreement_intervals"]) == 5
    mask = json.loads(Path(work["evaluator_mask_reference"]["path"]).read_bytes())
    assert (
        len(mask["rows"]) == 300
        and work["evaluator_mask_reference"]["role"] == "evaluator_only_after_prediction_seal"
    )
    assert Path(work["evaluator_mask_reference"]["path"]).stat().st_mode & 0o077 == 0
    protocol = json.loads(Path(work["diagnostic_protocol_reference"]["path"]).read_bytes())
    assert protocol == d.protocol()
    assert protocol["online"]["probability_change_gte"] == 0.05
    assert protocol["online"]["classification_threshold"] == 0.5
    assert len(protocol["static_bounds"]) == 2 and len(protocol["online_bounds"]) == 7


def test_qid_controls_and_missingness():
    """SCENARIO-REPORT-9141-V779: heterogeneous siblings and absent A2 never inflate N."""
    labels = [
        dict(unit_id=f"q__{i}", qid="q", group_id="g", role="TEST", annotator1=a, annotator2=b)
        for i, (a, b) in enumerate([(0, 0), (1, -1), (1, 0)])
    ]
    row = d.group_rows(labels)[0]
    assert row["slice"] == "mixed" and row["a1_error_count"] == 2
    assert row["a2_observed_count"] == 2 and row["disagreement_count"] == 1
    assert row["siblings"][1]["annotator2"] == -1
    assert d.agreement([row])["estimate"] == 0.5
    assert d.agreement([row])["width"] == 0
    assert d.agreement([])["status"] == "unavailable"
    missing = deepcopy(row)
    missing.update(a2_observed_count=0, agreement_count=0)
    assert d.agreement([missing])["observed_qids"] == 0
    for a, expected in [(0, "all_correct"), (1, "all_wrong")]:
        changed = [dict(r, annotator1=a) for r in labels]
        assert d.group_rows(changed)[0]["slice"] == expected
    for mutation in ["duplicate", "label", "role", "group", "count", "missing_a1"]:
        changed = deepcopy(labels)
        if mutation == "duplicate":
            changed[1]["unit_id"] = changed[0]["unit_id"]
        elif mutation == "label":
            changed[0]["annotator2"] = 2
        elif mutation == "role":
            changed[0]["role"] = "TRAIN"
        elif mutation == "group":
            changed[0]["group_id"] = "other"
        elif mutation == "count":
            changed.pop()
        else:
            changed[0]["annotator1"] = None
        with pytest.raises(ValueError):
            d.group_rows(changed)
    varying = [dict(row, agreement_count=k) for k in [0, 1, 2]]
    interval = d.agreement(varying)
    rng = np.random.default_rng(d.SEED)
    expected = np.quantile(
        np.array([0, 0.5, 1])[rng.integers(0, 3, (10000, 3))].mean(axis=1), [0.025, 0.975]
    )
    assert interval["ci95"] == list(expected)
    assert interval["width"] == expected[1] - expected[0]


def test_authentication_and_failed_operands(tmp_path, monkeypatch):
    """REQ-REPORT-9141-V779: pin before fields, with missing distinct from zero."""
    absent = d.inspect(tmp_path)
    assert absent["checks"][0]["observed"] == "missing_file"
    assert absent["rows"] == []
    original_ref = d.ref
    contract = json.loads((d.ROOT / d.UPSTREAM).read_bytes())
    upstream = tmp_path / d.UPSTREAM
    upstream.parent.mkdir()
    upstream.write_text(json.dumps(contract))
    monkeypatch.setattr(d, "UPSTREAM_PIN", original_ref(upstream)["sha256"])
    assert d.inspect(tmp_path)["rows"]
    for operand in ["missing_field", 0, False]:
        changed = deepcopy(contract)
        if operand == "missing_field":
            changed.pop("source_ready_score")
        else:
            changed["source_ready_score"] = operand
        upstream.write_text(json.dumps(changed))
        monkeypatch.setattr(d, "UPSTREAM_PIN", original_ref(upstream)["sha256"])
        result = d.inspect(tmp_path)
        gate = next(r for r in result["checks"] if r["artifact_field"] == "source_ready_score")
        assert gate["observed"] == operand and not gate["passed"] and not result["rows"]
    upstream.write_text(json.dumps(contract))
    monkeypatch.setattr(d, "UPSTREAM_PIN", original_ref(upstream)["sha256"])
    label_path = contract["evaluator_label_reference"]["path"]

    def tampered(path):
        return (
            dict(original_ref(path), sha256=None) if str(path) == label_path else original_ref(path)
        )

    monkeypatch.setattr(d, "ref", tampered)
    assert not d.inspect(tmp_path)["rows"]
    manifest = json.loads(Path(contract["source_manifest_reference"]["path"]).read_bytes())
    master_path = manifest["release_files"]["data/human_judge_release_master.csv"]["path"]
    monkeypatch.setattr(
        d,
        "ref",
        lambda path: (
            dict(original_ref(path), sha256=None)
            if str(path) == master_path
            else original_ref(path)
        ),
    )
    assert not d.inspect(tmp_path)["rows"]
    changed = deepcopy(contract)
    changed.pop("evaluator_label_reference")
    upstream.write_text(json.dumps(changed))
    monkeypatch.setattr(d, "UPSTREAM_PIN", original_ref(upstream)["sha256"])
    assert d.inspect(tmp_path)["checks"][-1]["observed"] == "missing_field"
    upstream.write_text("{}")
    assert not d.inspect(tmp_path)["rows"]


def child(tmp_path, *args):
    """Use actual standalone entry points in foreign private directories, without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9141] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9141] child_after",
        result.returncode,
        result.stdout[-300:],
        result.stderr[-300:],
        flush=True,
    )
    return result


def test_private_cli_and_tamper(work, tmp_path):
    """SCENARIO-REPORT-9141-V779: real success/refusal/cold replay/rehashed tamper."""
    source, output = tmp_path / "work.json", tmp_path / (d.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["contrast_protocol_ready_score"] == 0
    value["group_counts"]["mixed"] += 1
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )


def test_validation_and_deadlines(work, tmp_path, monkeypatch):
    """REQ-REPORT-9141-V779: affected failures suppress readiness and deadlines terminate."""
    assert c.build(work, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    value = c.build(work, {})
    assert value["verdict_class"] == "null" and value["contrast_protocol_ready_score"] == 1
    assert value["sample_size_budget"]["effective_independent_groups"] == 300
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["MODEL_SPECS"] == [] and value["verifier_is_oracle"] is False
    blocked = c.measure(tmp_path / "absent", tmp_path / "blocked", "20261116")
    value = c.build(blocked, {})
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 300
    assert value["inference_substrate_class"] == "blocked_no_run"
    monkeypatch.setattr(c.lifecycle, "qualify", lambda _: dict(delegation=True))
    assert c.qualify(work) == dict(delegation=True)
    monkeypatch.setattr(c, "measure", lambda *_: deepcopy(work))
    monkeypatch.setattr(c, "qualify", lambda _: {})
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + 1)
    monkeypatch.setattr(c, "receipts_pass", lambda *_: False)
    output = tmp_path / (d.NAME + ".json")
    assert c.main(["--output", str(output)]) == 1
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["contrast_protocol_ready_score"] == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + int(3301e9))
    assert c.main(["--output", str(output)]) == 1
    times = iter([work["start_ns"] + 1, work["start_ns"] + int(4801e9)])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    assert c.main(["--output", str(output)]) == 1


def test_replay_custody_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9141-V779: independent reconstruction rejects forged sidecars."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    for key in [
        "current_work_reference",
        "diagnostic_protocol_reference",
        "evaluator_mask_reference",
    ]:
        ref = changed[key]
        path = tmp_path / Path(ref["path"]).name
        path.write_bytes(Path(ref["path"]).read_bytes())
        changed[key] = dict(ref, path=str(path))
    output = tmp_path / (d.NAME + ".json")

    def save():
        c.atomic_json(tmp_path / "work.json", changed)
        value = c.build(changed, {})
        c.atomic_json(output, value)
        return value

    value = save()
    assert c.replay(output)["passed"]
    value["work_reference"]["sha256"] = "sha256:forged"
    c.atomic_json(output, value)
    with pytest.raises(ValueError, match="work_hash"):
        c.replay(output)
    base = deepcopy(changed)
    changed["current_work_reference"]["sha256"] = "sha256:forged"
    save()
    with pytest.raises(ValueError, match="operand_hash"):
        c.replay(output)
    changed = deepcopy(base)
    changed["evaluator_mask_reference"]["sha256"] = "sha256:forged"
    save()
    with pytest.raises(ValueError, match="mask_hash"):
        c.replay(output)
    changed = deepcopy(base)
    protocol = Path(changed["diagnostic_protocol_reference"]["path"])
    c.atomic_json(protocol, dict(d.protocol(), altered=True))
    changed["diagnostic_protocol_reference"] = d.ref(protocol)
    save()
    with pytest.raises(ValueError, match="protocol_drift"):
        c.replay(output)
    c.atomic_json(protocol, d.protocol())
    changed = deepcopy(base)
    changed["inspection"]["rows"][0]["a1_error_count"] += 1
    save()
    with pytest.raises(ValueError, match="source_drift"):
        c.replay(output)
    changed = deepcopy(base)
    mask = Path(changed["evaluator_mask_reference"]["path"])
    original_mask = json.loads(mask.read_bytes())
    c.atomic_json(mask, dict(rows=[]))
    changed["evaluator_mask_reference"] = dict(
        d.ref(mask), role="evaluator_only_after_prediction_seal"
    )
    save()
    with pytest.raises(ValueError, match="mask_drift"):
        c.replay(output)
    c.atomic_json(mask, original_mask)
    changed = deepcopy(base)
    save()
    monkeypatch.setattr(c, "validate_current_work_receipt", lambda *_: ["forged event"])
    with pytest.raises(ValueError, match="current_work_drift"):
        c.replay(output)


def test_group_identity_rejection():
    """SCENARIO-REPORT-9141-V779: two qids cannot claim one independent group."""
    rows = [
        dict(
            qid=q,
            group_id="shared",
            role="TEST",
            unit_id=f"{q}__{i}",
            annotator1=0,
            annotator2=None,
        )
        for q in ["a", "b"]
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="duplicate_group_identity"):
        d.group_rows(rows)
