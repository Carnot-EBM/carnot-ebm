"""REQ-REPORT-8943: complete accounting preserves scientific nulls and boundaries."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import v764_capstone as m
from carnot.reporting.current_work_receipt import atomic_json


def validation(passed=True):
    """REQ-REPORT-8943: private receipts test routing, never establish research."""
    return dict(passed=passed, coverage=dict(complete=passed, percent=100), receipts=[])


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8943: use unchanged original producers for the real census."""
    return m.collect(tmp_path_factory.mktemp("capstone8943"))


def test_real_census(work):
    """SCENARIO-REPORT-8943-CONTRACT: skips and valid nulls retain all fourteen slots."""
    v = m.build(work, validation())
    assert [r["task_id"] for r in v["outcome_rows"]] == [t["id"] for t in work["tasks"]]
    assert len(v["outcome_rows"]) == 14 and v["reconciliation_ready_score"] == 1
    assert v["completed_count"] == 14 and v["independent_count"] == 0
    assert v["paper_ready"] is False and v["independent_generalization_score"] == 0
    assert v["numerical_training_calls"] == 0 and not any(v["model_invocation_counts"].values())
    assert v["scientific_reductions"]["depth"]["error_rate_k2"] == 0.84
    assert v["scientific_reductions"]["depth"]["error_rate_k3"] == 0.78
    assert v["scientific_reductions"]["learning"]["compatible_benefit_score"] == 0
    assert len(v["scientific_reductions"]["cost"]["rows"]) == 288
    assert v["scientific_reductions"]["collection"]["observed_errors"] is None
    assert v["scientific_reductions"]["detector"]["benefit"] is None
    assert len(v["retirement_recommendations"]) == sum(
        len(t["prior_failures"]) for t in work["tasks"]
    )
    assert v["outcome_rows"][6]["disposition"] == "shared_title_scheduler_retirement"
    assert all(v["outcome_rows"][i]["verdict_class"] == "blocked" for i in [5, 6, 8, 9])
    assert all(r["cluster_id"] != "administrative_slots" for r in v["rows"])
    assert m.build(work, validation(False))["verdict_class"] == "disqualified"
    altered = deepcopy(work)
    altered["slots"][0]["accounted"] = False
    assert m.build(altered, validation())["reconciliation_ready_score"] == 0


def test_absence_and_byte_drift(tmp_path):
    """SCENARIO-REPORT-8943-CONTRACT: unavailable inputs are terminal named blocks."""
    w = m.collect(tmp_path / "raw", sources=tmp_path / "missing")
    v = m.build(w, validation())
    assert v["verdict_class"] == "blocked" and v["reconciliation_ready_score"] == 0
    assert all(r["gate_check_summary"] for r in w["slots"])
    sources = tmp_path / "missing"
    sources.mkdir()
    name = m.PINS[8930][0]
    (sources / name).write_text("{}")
    drift = m.collect(tmp_path / "drift", sources=sources)
    assert drift["slots"][0]["gate_check_summary"][0]["check"] == "rejected_producer"


def test_admission():
    """REQ-REPORT-8943: absent fields and failed owned checks never qualify benefit."""
    original = json.loads((m.ROOT / "results" / m.PINS[8934][0]).read_bytes())
    assert m.admission(original, 8934) is None
    for key in ["verdict_class", "acceptance_gates", "required_checks_passed"]:
        changed = dict(original)
        changed.pop(key)
        assert m.admission(changed, 8934) == "missing_field"
    assert m.admission(dict(original, experiment_id=0), 8934) == "rejected_producer"
    assert (
        m.admission(dict(original, verdict_class="positive", verifier_is_oracle=True), 8934)
        == "rejected_producer"
    )
    assert (
        m.admission(
            dict(original, verdict_class="disqualified", required_checks_passed=False), 8934
        )
        == "disqualified_primary"
    )


def test_replay_and_tampering(work, tmp_path):
    """SCENARIO-REPORT-8943-CONTRACT: recomputed hashes cannot bless invented claims."""
    path = tmp_path / (m.NAME + ".json")
    value = m.build(work, validation())
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key, replacement in [
        ("independent_count", 74),
        ("outcome_rows", []),
        ("rows", []),
        ("reconciliation_ready_score", 0),
    ]:
        changed = dict(value, **{key: replacement})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        with pytest.raises(ValueError):
            m.replay(path)
    atomic_json(path, dict(value, reproducibility_checksum="wrong"))
    assert m.main(["--replay", str(path)]) == 1
    atomic_json(path, value)
    assert m.main(["--replay", str(path)]) == 0
    swapped = deepcopy(work)
    swapped["slots"].reverse()
    with pytest.raises(ValueError):
        m.build(swapped, validation())


def test_private_cold_cli(tmp_path):
    """SCENARIO-REPORT-8943-CONTRACT: real child publication and cold replay are private."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    output = tmp_path / (m.NAME + ".json")
    cli = [str(m.ROOT / ".venv/bin/python"), "-u", str(m.ROOT / m.CLI)]
    for args in [
        ["--date", "20261109", "--private-output", str(output)],
        ["--cold-replay", str(output)],
    ]:
        result = subprocess.run(
            cli + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_owned_checks_are_separate(monkeypatch, tmp_path):
    """REQ-REPORT-8943: a historical timeout cannot trigger a new global suite."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    calls = []

    def run(root, commands, **kwargs):
        calls.extend(commands)
        return [dict(name=c.name, passed=True) for c in commands]

    monkeypatch.setattr(m, "run_commands", run)
    owned = CommandSpec("unit", ("true",), "owned")
    health = CommandSpec(
        "global_python_health", ("pytest", "tests/python", "-q"), "repository_health"
    )
    assert m.run_checks(tmp_path, [owned, health])[0]["passed"]
    assert calls == [owned]


def test_checked_publication_path(work, tmp_path, monkeypatch):
    """REQ-REPORT-8943: private injection exercises final routing without public writes."""
    real_root = m.ROOT
    monkeypatch.setattr(m, "ROOT", tmp_path)
    paths = [
        *m.OWNED,
        m.TEST,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]
    for p in paths:
        link = tmp_path / p
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(real_root / p)

    def collected(raw, sources):
        raw.mkdir(parents=True, exist_ok=True)
        return dict(work, raw=str(raw), code=[], references=list(work["references"]))

    monkeypatch.setattr(m, "collect", collected)
    monkeypatch.setattr(m.checker, "validate", lambda raw: validation())
    monkeypatch.setattr(
        m.checker, "terminal", lambda candidate, raw: dict(passed=True, receipts=[])
    )
    assert m.main(["--date", "20261109"]) == 0
    output = tmp_path / "results" / (m.NAME + ".json")
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1


def test_missing_primitive_contract(monkeypatch, tmp_path):
    """REQ-REPORT-8943: a lost original operand is a named producer failure."""

    def reject(ref, catalog):
        raise ValueError("lost original operand")

    monkeypatch.setattr(m.sources_module, "resolve", reject)
    w = m.collect(tmp_path / "lost")
    assert w["slots"][0]["accounted"] is False
    assert w["slots"][0]["gate_check_summary"][0]["check"] == "rejected_producer"


def test_rehashed_work_cannot_invent_missing_primary(work, tmp_path):
    """SCENARIO-REPORT-8943-CONTRACT: work ledgers cannot invent administrative success."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path / "forged")
    changed["slots"][0]["reference"] = None
    atomic_json(Path(changed["raw"]) / "work.json", changed)
    value = m.build(changed, validation())
    path = tmp_path / (m.NAME + ".json")
    atomic_json(path, value)
    with pytest.raises(ValueError, match="missing_primary_accounting"):
        m.replay(path)
