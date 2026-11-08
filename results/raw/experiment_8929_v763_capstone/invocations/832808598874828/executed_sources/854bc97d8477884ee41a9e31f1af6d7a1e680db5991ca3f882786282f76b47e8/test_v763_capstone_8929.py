"""REQ-REPORT-8929: administrative completeness cannot promote failed science."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import v763_capstone as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_pilot_boundaries():
    """SCENARIO-REPORT-8929-CONTRACT: exactly fifty calls and ten errors are required."""
    for completed, errors, passed in [(50, 10, True), (50, 9, False), (49, 43, False)]:
        result = m.science.collection_plan(
            dict(
                completed_count=completed,
                n_errors=errors,
                error_rate=errors / 50,
                error_interval=dict(low=0.1, high=0.9),
            )
        )
        assert result["pilot_gate_passed"] is passed
        assert result["execute_now"] is False
        if passed:
            assert result["recommended_questions"] == 1300
            assert result["split_unit"] == "graph_seed"
        else:
            assert result["recommended_questions"] is None
    actual = m.science.collection_plan(
        dict(
            completed_count=50,
            n_errors=43,
            error_rate=0.86,
            error_interval=dict(low=0.7381380628941446, high=0.9304916657298371),
        )
    )
    assert actual["recommended_questions"] == 303
    assert actual["uncertainty_question_range"] == [280, 353]


def test_admission():
    """SCENARIO-REPORT-8929-CONTRACT: Boolean gates, foreign identities and oracle claims fail."""
    v = dict(
        experiment_id=8918,
        task_id="exp8918-bfs-error-pilot",
        milestone=m.MILESTONE,
        run_date="20261108",
        honest_verdict="complete_null",
        verdict_class="null",
        required_checks_passed=True,
        pilot_error_gate_score=0,
        owned_statement_coverage=dict(complete=True, percent=100),
        validation_receipts=[dict(passed=True, scope="owned")],
    )
    assert m.admission(v, 8918, v["task_id"]) is None
    assert (
        m.admission(
            dict(v, verdict_class="disqualified", required_checks_passed=False), 8918, v["task_id"]
        )
        == "disqualified_primary"
    )
    for change in [
        dict(experiment_id=0),
        dict(task_id="wrong"),
        dict(pilot_error_gate_score=True),
        dict(verifier_is_oracle=True, verdict_class="positive"),
        dict(required_checks_passed=False),
        dict(validation_receipts=[]),
        dict(owned_statement_coverage={}),
    ]:
        assert m.admission(dict(v, **change), 8918, v["task_id"]) == "rejected_producer"
    v.pop("pilot_error_gate_score")
    assert m.admission(v, 8918, v["task_id"]) == "missing_field"


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8929: real producer bytes are reduced without executing new experiments."""
    return m.collect(tmp_path_factory.mktemp("capstone8929"))


def test_real_reductions(work):
    """REQ-REPORT-8929: all thirteen outcomes retain independent scientific scopes."""
    assert len(work["slots"]) == 12
    assert all(s["accounted"] for s in work["slots"])
    reduced = work["reductions"]
    assert reduced["pilot"]["n_errors"] == 43
    assert reduced["pilot"]["collection_plan"]["recommended_questions"] == 303
    assert len(reduced["learning"]["acceptance_gate_rows"]) == 9
    assert reduced["learning"]["interpolated_benefit_score"] == 0
    assert reduced["labels"]["qualified"] is False
    assert reduced["natural"]["annotation_dependence"]["independent_blind_truth"] is False
    assert reduced["observer"]["inline_clock_ready_score"] == 0
    assert len(reduced["durable"]["rows"]) == 48
    assert all(r["compatible_fraction"] is None for r in reduced["hardware"]["amdahl_rows"])
    assert [s["disposition"] for s in work["slots"]].count("authenticated_scheduler_skip") == 2
    assert work["slots"][3]["disposition"] == "disqualified_primary"


def validation(passed=True):
    """REQ-REPORT-8929: fixture qualification tests terminal routing only."""
    return dict(
        passed=passed,
        coverage=dict(complete=passed, percent=100 if passed else 0),
        receipts=[dict(name="fixture", passed=passed, scope="owned")],
    )


def test_build_and_replay(work, tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: real accounting remains distinct from scientific gates."""
    value = m.build(work, validation())
    assert value["verdict_class"] == "blocked"
    assert value["reconciliation_ready_score"] == 1
    assert value["completed_count"] == 13 and value["independent_count"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert m.build(work, validation(False))["verdict_class"] == "disqualified"
    changed = dict(work, slots=[dict(s) for s in work["slots"]])
    changed["slots"][0]["accounted"] = False
    assert m.build(changed, validation())["reconciliation_ready_score"] == 0


def test_rehashed_claims(tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: small blocked cases isolate aggregate rejection controls."""
    work = m.collect(tmp_path / "empty", sources=tmp_path / "missing")
    value = m.build(work, validation())
    path = tmp_path / (m.NAME + ".json")
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key, changed in [
        ("reconciliation_ready_score", 1),
        ("independent_count", 50),
        ("verdict_class", "positive"),
        ("continuation_rows", []),
    ]:
        altered = dict(value, **{key: changed})
        altered["reproducibility_checksum"] = m.checksum(altered)
        atomic_json(path, altered)
        with pytest.raises(ValueError):
            m.replay(path)
    atomic_json(path, dict(value, reproducibility_checksum="wrong"))
    with pytest.raises(ValueError, match="checksum"):
        m.replay(path)


def test_missing_and_tampered_inputs(tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: external absence is terminal blocked, never partial."""
    work = m.collect(tmp_path / "raw", sources=tmp_path / "missing")
    assert not any(s["accounted"] for s in work["slots"])
    assert all(s["gate_check_summary"] for s in work["slots"])
    assert m.build(work, validation())["verdict_class"] == "blocked"
    assert m.build(work, validation())["reconciliation_ready_score"] == 0
    path = tmp_path / "operand"
    path.write_text("original")
    ref = m.reference(path)
    assert m.resolve(ref, {ref["sha256"]: ref})["sha256"] == ref["sha256"]
    path.write_text("changed")
    with pytest.raises(ValueError, match="operand"):
        m.resolve(ref, {})


def test_actual_cli(tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: cold CLI works outside checkout without PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    py = str(m.ROOT / ".venv/bin/python")
    cli = str(m.ROOT / m.CLI)
    output = tmp_path / (m.NAME + ".json")
    command = [py, "-u", cli, "--private-output", str(output), "--date", "20261108"]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    replay = subprocess.run(
        [py, "-u", cli, "--replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    altered = json.loads(output.read_bytes())
    altered["independent_count"] = 50
    altered["reproducibility_checksum"] = m.checksum(altered)
    atomic_json(output, altered)
    reject = subprocess.run(
        [py, "-u", cli, "--replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert reject.returncode == 1
    with pytest.raises(SystemExit):
        m.main(["--date", "20261107"])


def test_scheduler_and_resolved_originals(work, tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: forged skips cannot borrow an authentic zero gate."""
    evidence = {s["experiment_id"]: s["producer"] for s in work["slots"]}
    for slot in work["slots"]:
        if slot["experiment_id"] not in (8923, 8927):
            continue
        n, value = slot["experiment_id"], slot["producer"]
        assert m.scheduler(value, n, slot["task"], evidence)
        for key, forged in [
            ("failed_observed", True),
            ("failed_expected", True),
            ("failed_field", "invented"),
            ("failed_evidence_sha256", "wrong"),
        ]:
            assert not m.scheduler(dict(value, **{key: forged}), n, slot["task"], evidence)
    original, retained = tmp_path / "original", tmp_path / "retained"
    original.write_text("exact")
    retained.write_text("exact")
    ref, snapshot = m.reference(original), m.reference(retained)
    original.write_text("new")
    assert m.resolve(ref, {snapshot["sha256"]: snapshot})["path"] == str(retained)
    with pytest.raises(ValueError):
        m.science.compare(dict(metric=0), dict(metric=1))


def test_production_routing_without_external_writes(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: a failed owned terminal check suppresses readiness."""
    from types import SimpleNamespace

    base = m.collect(tmp_path / "base", sources=tmp_path / "missing")

    def collect(raw, sources):
        raw.mkdir(parents=True)
        result = dict(base, raw=str(raw))
        atomic_json(raw / "work.json", result)
        return result

    def publish(output, value, validator):
        candidate = tmp_path / "candidate.json"
        atomic_json(candidate, value)
        assert validator(candidate)["passed"]
        assert value["verdict_class"] == "disqualified"
        assert value["reconciliation_ready_score"] == 0

    def validate(raw):
        atomic_json(raw / "fixture.json", dict(scope="test_only"))
        return validation()

    def terminal(path, raw):
        atomic_json(raw / "terminal" / "fixture.json", dict(scope="test_only"))
        return dict(passed=False, receipts=[])

    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m, "publish_primary", publish)
    monkeypatch.setattr(
        m,
        "checker",
        SimpleNamespace(validate=validate, terminal=terminal),
    )
    assert m.main(["--date", "20261108", "--sources", str(tmp_path / "missing")]) == 0


def test_rejected_custody_is_retained(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8929-CONTRACT: lost operands reject producers, not their diagnostics."""
    original = m.resolve

    def reject(ref, catalog):
        if "execution_snapshot" not in ref and not ref.get("current_execution_authority"):
            raise ValueError("deliberate_operand_control")
        return original(ref, catalog)

    monkeypatch.setattr(m, "resolve", reject)
    monkeypatch.setattr(m.science, "reduce", lambda evidence, raw: {})
    work = m.collect(tmp_path / "rejected")
    assert any(not s["accounted"] for s in work["slots"])
    assert all(s["producer"] for s in work["slots"])
    assert any(s["gate_check_summary"] for s in work["slots"])
