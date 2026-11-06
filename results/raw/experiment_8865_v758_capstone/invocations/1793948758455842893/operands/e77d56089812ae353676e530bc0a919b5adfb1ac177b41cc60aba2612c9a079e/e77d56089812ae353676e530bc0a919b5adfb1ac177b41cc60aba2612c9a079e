"""REQ-REPORT-8810 / REQ-VERIFY-8810: independent sealed-row audit controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import causal_learning_audit_8810 as a
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture
def evidence(tmp_path):
    """Keep all mutable fixture operands private while preserving real sealed probabilities."""
    source = json.loads((a.ROOT / a.UPSTREAM).read_bytes())
    frozen = json.loads(Path(source["raw_shard_hashes"][1]["path"]).read_bytes())
    frozen["streams"] = frozen["streams"][:1]
    frozen["seeds"] = [11]
    for key in [
        "per_query_rows",
        "retention_rows",
        "release_order_rows",
        "complete_learning_cost_rows",
        "reset_rows",
        "final_state_refs",
    ]:
        source[key] = [r for r in source[key] if r["cluster_id"] == "stream-00" and r["seed"] == 11]
    source["heads"] = [h for h in source["heads"] if h["seed"] == 11]
    source["input_refs"] = source["input_refs"][:1]
    source = {
        key: source[key]
        for key in [
            "per_query_rows",
            "retention_rows",
            "release_order_rows",
            "complete_learning_cost_rows",
            "reset_rows",
            "final_state_refs",
            "heads",
            "input_refs",
        ]
    }
    for index, row in enumerate(source["per_query_rows"] + source["retention_rows"]):
        old = Path(row["seal_ref"]["path"])
        path = tmp_path / "seals" / str(index)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(old.read_bytes())
        row["seal_ref"] = a.reference(path)
    for index, ref in enumerate(source["input_refs"] + source["final_state_refs"]):
        path = tmp_path / ("operand-" + str(index))
        path.write_bytes(Path(ref["path"]).read_bytes())
        ref.update(a.reference(path))
    return source, frozen


def test_scores_and_invalid_probabilities():
    """SCENARIO-VERIFY-8810-INDEPENDENT-REDUCTION: proper loss and actions use sealed numbers."""
    assert a.score([1.0, 0.0, 0.0], "support", 0.5) == dict(
        metric=0.0, action="accept", false_accept=0, false_reject=0, covered=1
    )
    assert a.score([0.0, 1.0, 0.0], "support", 0.5)["false_reject"] == 1
    assert a.score([1.0, 0.0, 0.0], "refute", 0.5)["false_accept"] == 1
    assert a.score([0.3, 0.3, 0.4], "insufficient", 0.5)["covered"] == 0
    for p in [[1.0, 1.0, 1.0], [float("nan"), 0.0, 0.0], [-0.1, 0.5, 0.6], [1.0, 0.0]]:
        with pytest.raises(ValueError):
            a.score(p, "support", 0.5)


def test_independent_rows_and_familywise(evidence, monkeypatch, capsys):
    """SCENARIO-VERIFY-8810-FAMILYWISE: paired seeds never count as independent streams."""
    from types import SimpleNamespace

    ticks = iter(range(0, 1000000, 31))
    monkeypatch.setattr(a, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    source, frozen = evidence
    reduced = a.reduce(source, frozen)
    assert "completed=340 pending=0" in capsys.readouterr().out
    assert len(reduced["rows"]) == 340
    assert all(r["status"] == "completed" for r in reduced["rows"])
    assert all(reduced["leakage_checks"]["checks"].values())
    assert reduced["leakage_checks"]["complete_state_identities"] > 0
    summary = a.summarize(reduced["rows"])
    intervals = summary["familywise_intervals"]
    assert len(intervals) == 10
    assert all(r["draws"] == 10000 and r["independent_count"] == 1 for r in intervals)
    assert summary == a.summarize(reduced["rows"])
    assert not a.benefit_gates(reduced, summary)["future_brier"]
    assert summary["coverage_rows"] and summary["headroom_checks"]["exact_oracle_brier"] == 0


def test_census_missing_and_duplicates(evidence):
    """SCENARIO-REPORT-8810-SEALED-CENSUS: every missing slot survives in the denominator."""
    source, frozen = evidence
    removed = source["per_query_rows"].pop()
    reduced = a.reduce(source, frozen)
    assert len(reduced["rows"]) == 340
    assert sum(r["status"] == "censored" for r in reduced["rows"]) == 1
    assert not reduced["leakage_checks"]["checks"]["complete_denominators"]
    source["per_query_rows"].append(removed)
    Path(removed["seal_ref"]["path"]).unlink()
    assert sum(r["status"] == "censored" for r in a.reduce(source, frozen)["rows"]) == 1
    source["per_query_rows"].append(source["per_query_rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        a.reduce(source, frozen)


@pytest.mark.parametrize(
    "mutation", ["metric", "sealed_ns", "state_hash", "changed_later_predicates"]
)
def test_row_tamper_rejected(evidence, mutation):
    """SCENARIO-VERIFY-8810-INDEPENDENT-REDUCTION: producer assertions cannot override seals."""
    source, frozen = evidence
    row = source["per_query_rows"][0]
    row[mutation] = {
        "metric": -1,
        "sealed_ns": row["truth_recorded_ns"] + 1,
        "state_hash": "changed",
        "changed_later_predicates": True,
    }[mutation]
    with pytest.raises(ValueError):
        a.reduce(source, frozen)


def test_seal_and_shuffled_mutation(evidence):
    """REQ-VERIFY-8810: outer rehashing cannot hide broken inner identities or shuffle order."""
    source, frozen = evidence
    row = source["per_query_rows"][0]
    path = Path(row["seal_ref"]["path"])
    value = json.loads(path.read_bytes())
    value["pre_state"]["acks"].append("fabricated")
    atomic_json(path, value)
    with pytest.raises(ValueError):
        a.reduce(source, frozen)
    row["seal_ref"] = a.reference(path)
    with pytest.raises(ValueError):
        a.reduce(source, frozen)


def test_all_threshold_gates(evidence):
    """REQ-VERIFY-8810: each prespecified benefit gate can independently falsify readiness."""
    source, frozen = evidence
    reduced = a.reduce(source, frozen)
    summary = a.summarize(reduced["rows"])
    for row in summary["familywise_intervals"]:
        row["upper"] = -0.02 if row["comparison"].startswith("future_") else 0.01
    assert all(a.benefit_gates(reduced, summary).values())
    summary["familywise_intervals"][0]["upper"] = -0.01
    assert not a.benefit_gates(reduced, summary)["future_brier"]
    summary["familywise_intervals"][2]["upper"] = 0.021
    assert not a.benefit_gates(reduced, summary)["retention"]
    summary["familywise_intervals"][-1]["upper"] = 0.021
    assert not a.benefit_gates(reduced, summary)["errors"]


def test_gate_and_blocked_artifact(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8810-TERMINAL: external absence is blocked and never a zero measurement."""
    from carnot.reporting import causal_learning_runner_8810 as r

    path = tmp_path / "missing.json"
    assert r.upstream_gate(path)["observed"] == "missing_file"
    atomic_json(path, {"experiment_id": 8804})
    assert r.upstream_gate(path)["observed"] == "missing_field"
    atomic_json(path, {"experiment_id": 8804, "memory_rows_ready_score": 0})
    assert r.upstream_gate(path)["observed"] == 0
    atomic_json(path, {"experiment_id": 8804, "verdict_class": "disqualified"})
    assert r.upstream_gate(path)["observed"] == "rejected_producer"
    path.write_text("broken")
    assert r.upstream_gate(path)["observed"] == "rejected_producer"
    monkeypatch.setattr(a, "UPSTREAM", str(tmp_path / "absent.json"))
    value = r.run(tmp_path / "raw", "20261104")
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == value["intended_count"] == 30600
    assert not value["familywise_intervals"] and value["independently_recomputed_rows"] == 0
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    value["learning_benefit_ready_score"] = 1
    atomic_json(candidate, value)
    assert not r.replay(candidate)["passed"]
    candidate.write_text("broken")
    assert not r.replay(candidate)["passed"]


def test_private_measured_cold_replay(evidence, tmp_path):
    """SCENARIO-REPORT-8810-REPLAY: cold replay reconstructs aggregates and rejects rehashing."""
    from carnot.reporting import causal_learning_runner_8810 as r

    value = r.run(tmp_path / "audit", "20261104", evidence=evidence)
    assert value["completed_count"] == 340 and value["verdict_class"] == "disqualified"
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    directory = Path(env.get("CARNOT_8810_COVERAGE_DIRECTORY", str(tmp_path)))
    argv = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        "--parallel-mode",
        "--data-file=" + str(directory / ".coverage.child"),
        str(a.ROOT / r.CLI),
        "--replay",
        str(candidate),
    ]
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0 and '"passed": true' in child.stdout, child.stdout + child.stderr
    for field, altered in [
        ("completed_count", -1),
        ("learning_benefit_ready_score", 1),
        ("verdict_class", "positive"),
    ]:
        mutated = deepcopy(value)
        mutated[field] = altered
        atomic_json(candidate, mutated)
        assert not r.replay(candidate)["passed"]
    changed = deepcopy(value)
    changed["familywise_intervals"][0]["estimate"] += 0.1
    atomic_json(candidate, changed)
    assert not r.replay(candidate)["passed"]
    assert r.main(["--replay", str(candidate)]) == 1
    atomic_json(candidate, value)
    assert r.main(["--replay", str(candidate)]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "20261103"])


def test_preflight_production_and_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8810: custody is charged once and historical authorities use preserved files."""
    from carnot.reporting import causal_learning_runner_8810 as r

    pre, source, frozen = r.preflight(tmp_path / "pre")
    assert pre["passed"], pre["checked"]
    assert len(a.census(frozen, "test")) == 30600
    assert source["experiment_id"] == 8804 and pre["validation_reserve_s"] == 900
    assert pre["measured_cheap_unit_s"] > 0 and pre["sources"]
    assert pre["checked"]["historical_authority"]
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=False, references=[]))
    pre, _, _ = r.preflight(tmp_path / "bad-authority")
    assert not pre["passed"] and not pre["checked"]["current_authority"]


def test_failure_and_incomplete_runner(evidence, tmp_path):
    """REQ-REPORT-8810: invalid owned reductions disqualify; absent input slots block."""
    from carnot.reporting import causal_learning_runner_8810 as r

    source, frozen = evidence
    source["per_query_rows"][-1]["metric"] = -1
    value = r.run(tmp_path / "failure", "20261104", evidence=(source, frozen))
    assert value["verdict_class"] == "disqualified" and value["owned_failure"]
    assert not value["learning_benefit_ready_score"]
    source["per_query_rows"].pop()
    value = r.run(tmp_path / "incomplete", "20261104", evidence=(source, frozen))
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 1


def test_qualification_paths(evidence, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8810-TERMINAL: null readiness differs from benefit and owned failure."""
    from carnot.reporting import causal_learning_runner_8810 as r

    value = r.run(tmp_path / "raw", "20261104", evidence=evidence)
    checked = dict(
        passed=True,
        coverage=dict(complete=True, combined_reference=dict(path="private")),
        receipts=[dict(passed=True)],
    )
    r.qualify(value, checked)
    assert value["verdict_class"] == "null" and value["required_checks_passed"]
    assert value["learning_benefit_ready_score"] == 0
    monkeypatch.setattr(r, "coverage_summary", lambda *args: checked["coverage"])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    changed = deepcopy(value)
    changed["validation_receipts"] = [dict(passed=False)]
    atomic_json(candidate, changed)
    assert not r.replay(candidate)["passed"]
    checked["coverage"]["complete"] = False
    atomic_json(candidate, value)
    assert not r.replay(candidate)["passed"]
    for key in value["acceptance_gates"]:
        value["acceptance_gates"][key] = True
    r.qualify(value, dict(checked, passed=True))
    assert (
        value["verdict_class"] == "circular_positive" and value["learning_benefit_ready_score"] == 1
    )
    r.qualify(value, dict(checked, passed=False))
    assert value["verdict_class"] == "disqualified" and not value["learning_benefit_ready_score"]
    value["acceptance_gates"]["execution"] = False
    r.qualify(value, dict(checked, passed=True))
    assert value["verdict_class"] == "blocked"


def test_validation_and_terminal_commands(tmp_path, monkeypatch):
    """REQ-VERIFY-8810: retain actual command plans, SQLite union and unchanged validators."""
    from types import SimpleNamespace
    from carnot.reporting import causal_learning_runner_8810 as r

    calls = []

    def child(name, argv, logs, **kw):
        calls.append((name, argv))
        if name == "owned_pytest":
            directory = Path(argv[1].split("=", 1)[1])
            (directory / ".coverage.child.fixture").write_bytes(b"private SQLite fixture")
        return dict(name=name, passed=True)

    monkeypatch.setattr(r, "child", child)
    monkeypatch.setattr(
        r, "Coverage", lambda **kw: SimpleNamespace(combine=lambda **kw: None, save=lambda: None)
    )
    monkeypatch.setattr(r, "coverage_summary", lambda *args: dict(complete=True, percent=100))
    assert r.validate(tmp_path / "validation")["passed"]
    assert r.terminal_checks(tmp_path / "candidate.json", tmp_path)["passed"]
    names = {name for name, _ in calls}
    assert {
        "owned_pytest",
        "private_E2E_018_021",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
        "adversarial",
        "row_consistency",
        "cold_replay",
    } <= names
    cold = next(argv for name, argv in calls if name == "cold_replay")
    assert cold[:6] == ["env", "-u", "PYTHONPATH", "-C", "/tmp", str(a.ROOT / ".venv/bin/python")]


def test_main_publication_and_retained_rejection(evidence, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8810-TERMINAL: publish checked bytes and retain rejected candidates."""
    from carnot.reporting import causal_learning_runner_8810 as r

    measured = r.run(tmp_path / "measure", "20261104", evidence=evidence)
    monkeypatch.setattr(r, "run", lambda *args: deepcopy(measured))
    monkeypatch.setattr(
        r,
        "validate",
        lambda raw: dict(passed=True, coverage=dict(complete=True), receipts=[dict(passed=True)]),
    )
    monkeypatch.setattr(r, "terminal_checks", lambda *args: dict(passed=True))
    output = tmp_path / "results" / (a.NAME + ".json")
    assert r.main(["--raw", str(tmp_path / "cli"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "null"

    def rejected(*args, **kwargs):
        raise ValueError("candidate_rejected")

    monkeypatch.setattr(r, "publish_primary", rejected)
    with pytest.raises(ValueError, match="candidate_rejected"):
        r.main(["--raw", str(tmp_path / "reject"), "--output", str(output)])
    assert (tmp_path / "reject/rejected_candidate.json").is_file()


def test_pinned_production_projection_replay(evidence, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8810-REPLAY: production replay requires pins and the exact projection."""
    from carnot.reporting import causal_learning_runner_8810 as r

    source, frozen = evidence
    value = r.run(tmp_path / "raw", "20261104", evidence=evidence)
    upstream = tmp_path / "private-primary.json"
    atomic_json(upstream, source)
    monkeypatch.setattr(a, "UPSTREAM", str(upstream))
    monkeypatch.setattr(a, "PIN", a.sha256_file(upstream))
    monkeypatch.setattr(r, "frozen_manifest", lambda: frozen)
    value["protocol_scope"] = "production"
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    source["per_query_rows"][0]["metric"] = -1
    atomic_json(upstream, source)
    monkeypatch.setattr(a, "PIN", a.sha256_file(upstream))
    assert not r.replay(candidate)["passed"]


def test_shuffled_control_rejects_unreleased_label(evidence):
    """REQ-VERIFY-8810: shuffled feedback can use only its already released prefix."""
    source, frozen = evidence
    row = next(r for r in source["release_order_rows"] if r["arm"] == "shuffled_feedback")
    row["feedback_label"] = "outside-released-prefix"
    with pytest.raises(ValueError, match="release_boundary"):
        a.reduce(source, frozen)
