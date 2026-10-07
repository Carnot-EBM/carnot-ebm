"""REQ-REPORT-8248 / REQ-VERIFY-8248: freeze methods without benefit credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import evidence_intervention_methods_8248 as q
from carnot.reporting import evidence_intervention_runner_8248 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import evidence_intervention_8248 as n


def fixture(root):
    """Private copies let negative tests alter inputs without changing research history."""
    for name in q.INPUTS:
        source = q.ROOT / name
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target.write_bytes(source.read_bytes())
    for ref in q.PROTOCOL_VALUE["source_artifact_hashes"]:
        source = Path(ref["path"])
        target = root / source.relative_to(q.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def cli(parent, *args):
    """Exercise real imports and exits in a fresh process outside this checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )


def row(source="Café ALPHA. Other beta. Third zeta.", answer="Café alpha. Beta answer."):
    """Whole sentences make both selection ties and complete answer preservation visible."""
    return dict(source_bytes=source.encode().hex(), answer_bytes=answer.encode().hex())


def test_views():
    """SCENARIO-VERIFY-8248-VIEWS: labels and cached citations cannot choose views."""
    cached = [dict(sentence_index=i, p_unsupported=0.9, source_indices=[2]) for i in [1, 0]]
    got = n.view(row(), cached, len)
    assert got["status"] == "completed"
    assert got["focal_sentence_index"] == 0
    assert got["selected_sentence_index"] == 0
    assert got["control_sentence_index"] == 1
    assert got["lexical_overlap_strength"] == 1
    assert got["removal_length_difference"] == pytest.approx((11 - 11) / 34)
    for view in got["views"].values():
        assert view["answer_bytes"] == row()["answer_bytes"]
        assert all(
            json.loads(r["payload"])["answer"] == "Café alpha. Beta answer."
            for r in view["requests"]
        )
    changed = [dict(r, source_indices=[0], human_target=1) for r in cached]
    assert n.view(row(), changed, len) == got
    zero = n.view(row(answer="Unknown words."), cached[:1], len)
    assert zero["unavailable_reason"] == "missing_focal_cached_probability"
    zero = n.view(row(answer="Unknown words."), cached, len)
    assert zero["selected_sentence_index"] == 0 and zero["lexical_overlap_strength"] == 0
    for source, reason in [
        ("Only one.", "no_second_complete_source_sentence"),
        ("", "no_second_complete_source_sentence"),
    ]:
        assert n.view(row(source=source), cached, len)["unavailable_reason"] == reason
    assert n.view(row(), [], len)["unavailable_reason"] == "missing_focal_cached_probability"
    assert (
        n.view(row(answer="Unfinished"), cached, len)["unavailable_reason"]
        == "missing_focal_cached_probability"
    )
    assert n.added_features([0.2, 0.6, 0.3], -0.05) == pytest.approx([0.2, 0.4, 0.1, 0.3, -0.05])
    assert n.added_features([0.2, None, 0.3], 0) is None
    for ps in [[1.1, 0.5, 0.2], [float("nan"), 0.5, 0.2]]:
        with pytest.raises(ValueError, match="probability"):
            n.added_features(ps, 0)


def test_roles():
    """REQ-VERIFY-8248: original source hashes prove all five partitions are disjoint."""
    p = q.PROTOCOL_VALUE
    assert n.validate_roles(p) == dict(
        fit=128, calibration=32, selection=32, stream=96, retention=32
    )
    assert p["H1"]["minimum_complete"] == 72 and p["H1"]["seed"] == 7138248
    assert p["H2"]["retention"]["minimum_complete"] == 20
    assert p["H2"]["sensitivity_blocks"] == [4, 16]
    assert len(p["features"]) == 21 and p["head_config"]["ridges"] == [0.0001, 0.001, 0.01, 0.1, 1]
    assert all(h["coefficients"] == 22 for h in p["trained_head_specs"])
    for key in ["unit_id", "source_cluster_id", "original_source_sha256"]:
        bad = deepcopy(p)
        bad["role_manifest"]["retention"][0][key] = bad["role_manifest"]["fit"][0][key]
        with pytest.raises(ValueError, match="roles"):
            n.validate_roles(bad)
    bad = deepcopy(p)
    bad["role_manifest"]["stream"].reverse()
    with pytest.raises(ValueError, match="partition"):
        n.validate_roles(bad)


def test_measurement(tmp_path):
    """SCENARIO-REPORT-8248-CUSTODY: administrative agreement cannot award science."""
    root = fixture(tmp_path / "root")
    work = q.measure(root, tmp_path / "raw")
    assert not work["failures"], work["failures"]
    assert len(work["public_rows"]) == 320
    assert len(work["contract"]["tasks"]) == 14
    assert len(work["history"][2]["task_dispositions"]) == 14
    assert all("y" not in r and "human_target" not in r for r in work["public_rows"])
    labels = json.loads(Path(work["label_reference"]["path"]).read_bytes())
    assert len(labels) == 192 and {r["role"] for r in labels} == {"fit", "calibration", "selection"}
    assert any(r["frozen_v707"]["p"] is not None for r in work["public_rows"] if r["role"] == "fit")
    value = q.reduce(work, [dict(name="owned", passed=True)])
    assert value["intervention_protocol_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["verdict_class"] == "null" and value["independent_generalization_score"] == 0
    assert q.reduce(work, [])["verdict_class"] == "disqualified"
    assert q.reduce(work, [dict(passed=False)])["intervention_protocol_ready_score"] == 0
    for field in ["stream", "retention"]:
        assert len(value["frozen_roles"][field]) == (96 if field == "stream" else 32)
    missing = root / "results/experiment_8239_v712_margin_decision_audit.json"
    missing.unlink()
    blocked = q.measure(root, tmp_path / "missing")
    assert q.reduce(blocked, [dict(passed=True)])["verdict_class"] == "blocked"
    assert blocked["failures"][0]["observed"] is None


def test_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8248-CLI: real children reject altered primitive evidence."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (q.NAME + ".json")
    result = cli(tmp_path, "--private-fixture", "--root", root, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["intervention_protocol_ready_score"] == 1
    assert value["random_seed"] == 7138248
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    altered = deepcopy(value)
    altered["completed_count"] = 999
    atomic_json(tmp_path / "aggregate.json", altered)
    assert cli(tmp_path, "--cold-replay", tmp_path / "aggregate.json").returncode == 1
    original_work = Path(value["work_reference"]["path"])
    work = json.loads(original_work.read_bytes())
    work["public_rows"][0]["intervention"]["selected_sentence_index"] = 999
    atomic_json(original_work, work)
    altered = deepcopy(value)
    altered["work_reference"]["sha256"] = sha256_file(original_work)
    for ref in altered["raw_shard_hashes"]:
        if ref["path"] == str(original_work):
            ref["sha256"] = sha256_file(original_work)
    altered["reproducibility_checksum"] = canonical_hash(
        [altered["work_reference"], altered["code_config_hashes"], q.PIN]
    )
    atomic_json(tmp_path / "rehashed.json", altered)
    bad = cli(tmp_path, "--cold-replay", tmp_path / "rehashed.json")
    assert bad.returncode == 1 and "source_primitive_drift" in bad.stdout
    assert cli(tmp_path, "--date", "20261008").returncode == 2
    assert cli(tmp_path, "--private-fixture").returncode == 1


def test_schema_and_tool_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8248-CUSTODY: duplicate sources, changed hashes and missing tools fail."""
    read = lambda ref: json.loads(Path(ref["path"]).read_bytes())
    fit_path = q.PROTOCOL_VALUE["fit_slots_reference"]
    original = read(dict(path=fit_path))
    duplicate = deepcopy(original)
    duplicate["slots"][0] = duplicate["slots"][1]
    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(
            lambda ref: duplicate if ref["path"] == fit_path else read(ref), q.PROTOCOL_VALUE, len
        )
    changed = deepcopy(original)
    changed["slots"][0]["source_cluster_id"] = "changed"
    with pytest.raises(ValueError, match="source_identity"):
        q.primitives(
            lambda ref: changed if ref["path"] == fit_path else read(ref), q.PROTOCOL_VALUE, len
        )
    root = fixture(tmp_path / "root")
    original_is_file = Path.is_file
    tool = q.ROOT / ".venv/bin/mypy"
    monkeypatch.setattr(
        Path, "is_file", lambda path: False if path == tool else original_is_file(path)
    )
    work = q.measure(root, tmp_path / "raw")
    assert any(f["path"] == str(tool) for f in work["failures"])


def test_replay_rejects_claims(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8248-CUSTODY: receipt, protocol, checksum and authority drift fail replay."""
    root = fixture(tmp_path / "root")
    work = q.measure(root, tmp_path / "raw")
    work["root"] = str(root)
    stream = tmp_path / "stdout"
    stream.write_text("actual child")
    receipts = [
        dict(
            passed=True,
            stdout_path=str(stream),
            stderr_path=str(stream),
            stdout_sha256=sha256_file(stream),
            stderr_sha256=sha256_file(stream),
        )
    ]
    wp = tmp_path / "work.json"
    atomic_json(wp, work)
    value = q.reduce(work, receipts)
    value.update(work_reference=dict(path=str(wp), sha256=sha256_file(wp)), raw_shard_hashes=[])
    value["reproducibility_checksum"] = canonical_hash(
        [value["work_reference"], value["code_config_hashes"], q.PIN]
    )
    candidate = tmp_path / "candidate.json"
    for key, replacement, error in [
        ("reproducibility_checksum", "wrong", "checksum_drift"),
        ("protocol_sha256", "wrong", "protocol_drift"),
    ]:
        atomic_json(candidate, dict(value, **{key: replacement}))
        with pytest.raises(ValueError, match=error):
            runner.replay(candidate)
    altered = deepcopy(value)
    altered["validation_receipts"][0]["stdout_sha256"] = "wrong"
    atomic_json(candidate, altered)
    with pytest.raises(ValueError, match="receipt_drift"):
        runner.replay(candidate)
    atomic_json(candidate, value)
    label_path = Path(work["label_reference"]["path"])
    original_labels = label_path.read_bytes()
    atomic_json(label_path, [])
    with pytest.raises(ValueError, match="label_seal_drift"):
        runner.replay(candidate)
    label_path.write_bytes(original_labels)
    forged = deepcopy(work)
    forged["failures"].append(dict(artifact_field="exists", path="invented"))
    atomic_json(wp, forged)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(wp)
    changed["reproducibility_checksum"] = canonical_hash(
        [changed["work_reference"], changed["code_config_hashes"], q.PIN]
    )
    atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="source_gate_drift"):
        runner.replay(candidate)
    atomic_json(wp, work)
    atomic_json(candidate, value)
    original_assess = runner.authority.assess_authorities
    monkeypatch.setattr(
        runner.authority,
        "assess_authorities",
        lambda *args, **kwargs: dict(original_assess(*args, **kwargs), activated=False),
    )
    with pytest.raises(ValueError, match="authority_drift"):
        runner.replay(candidate)
    monkeypatch.setattr(runner.authority, "assess_authorities", original_assess)
    (root / "CODEX.md").unlink()
    (root / "results/experiment_8239_v712_margin_decision_audit.json").unlink()
    blocked = q.measure(root, tmp_path / "blocked")
    blocked["root"] = str(root)
    atomic_json(wp, blocked)
    value = q.reduce(blocked, receipts)
    value.update(work_reference=dict(path=str(wp), sha256=sha256_file(wp)), raw_shard_hashes=[])
    value["reproducibility_checksum"] = canonical_hash(
        [value["work_reference"], value["code_config_hashes"], q.PIN]
    )
    atomic_json(candidate, value)
    assert runner.replay(candidate)["passed"]


def test_owned_plan_and_external_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8248-CUSTODY: required validators and each failure stay visible."""
    plan = runner.commands(tmp_path / "validation")
    assert {"coverage_report", "consumers_E2E018", "private_E2E021", "scoped_spec_coverage"} <= {
        p["name"] for p in plan
    }
    assert next(p for p in plan if p["name"] == "full_python_suite")["scope"] == "repository_health"
    root = fixture(tmp_path / "root")
    (root / "CODEX.md").unlink()
    (root / q.DESIGN).write_text("missing task design")
    blocked = q.measure(root, tmp_path / "broken")
    assert blocked["failures"] and blocked["contract"]["tasks"] == []
    assert q.reduce(blocked, [dict(passed=True)])["censored_count"] == 14
    root = fixture(tmp_path / "valid-root")
    monkeypatch.setattr(q, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    failed = q.measure(root, tmp_path / "stale")
    assert any("qualified_terminal" in str(f) for f in failed["failures"])
