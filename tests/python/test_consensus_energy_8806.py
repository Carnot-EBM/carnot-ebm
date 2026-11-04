"""REQ-REPORT-8806 / REQ-VERIFY-8806: constructed evidence cannot imply live inference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import consensus_energy_8806 as m
from carnot.reporting import consensus_energy_runner_8806 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v746_retained_evidence import reference


def worlds():
    """Private worlds keep fixtures away from published evidence."""
    return [
        dict(
            world_id=f"private/{i}",
            split="consensus_dev" if i < 6 else "consensus_test",
            edges=[[0, 1], [0, 2], [1, 2], [2, 3]],
        )
        for i in range(9)
    ]


def test_cohort_and_signal_boundary():
    """SCENARIO-REPORT-8806-COHORT: crossing and labels do not enter serving features."""
    cohort = m.construct(worlds())
    assert len(cohort) == 81
    assert all(len(x["candidates"]) == 3 for x in cohort)
    for x in cohort:
        assert len(m.signals(x["input"])) == 5
        assert x["truth"] in m.LABELS
        assert (len(set(x["candidates"])) == 1) == (x["agreement"] != "disagreement")
        assert (x["candidates"][0] == x["truth"]) == (x["agreement"] == "unanimous_correct") or x[
            "agreement"
        ] == "disagreement"
    p = {
        "facts": [["w", "r", "a", True], ["z", "s", "b", True]],
        "query": [["?x", "r", "a"], ["?x", "s", "b"]],
    }
    assert m.signals(p)[1] > 0
    p["facts"].append(["w", "r", "a", False])
    assert m.signals(p)[1] > 0
    with pytest.raises(ValueError):
        m.signals(dict(p, truth="support"))


def test_energy_training_and_world_statistics():
    """SCENARIO-REPORT-8806-ENERGY: real NCE training and calibration precede test scores."""
    cohort = m.construct(worlds())
    fit = m.fit(cohort, [11], steps=12)
    assert len(fit["parameters"]) == 1
    assert fit["numerical_training_calls"] == 9
    assert fit["train_worlds"] and fit["calibration_worlds"]
    assert not set(fit["train_worlds"]) & set(fit["calibration_worlds"])
    rows = m.evaluate(cohort, fit["parameters"])
    summary = m.summarize(rows, draws=100)
    assert len(rows) == 3 * 9 * 3 * 4
    assert len(summary["brier_rows"]) == 4
    assert summary["independent_count"] == 3
    assert all(x["denominator"] > 0 for x in summary["brier_rows"])
    assert m.decide([0.9, 0.05, 0.05], 0.6) == "accept"
    assert m.decide([0.05, 0.9, 0.05], 0.6) == "reject"
    assert m.decide([0.3, 0.3, 0.4], 0.6) == "escalate"
    assert m.target("support", "support") == 0
    assert m.target("refute", "support") == 1
    assert m.target("support", "insufficient") == 2
    assert len(m.interval([1.0, 1.0], 100)) == 3
    return_fit = deepcopy(fit["parameters"])
    return_fit[0]["threshold"] = 1.0
    escalated = m.evaluate(cohort, return_fit)
    assert any(x["action"] == "escalate" for x in escalated)


def test_upstream_gates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8806-CUSTODY: absence, missing, zero and drift are different."""
    source = tmp_path / "producer.json"
    monkeypatch.setattr(r, "SOURCE", source)
    assert r.admit(source)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"consensus_protocol_ready_score": 0}, "observed_zero"),
        ({"consensus_protocol_ready_score": 1}, "rejected_producer"),
        ([], "rejected_producer"),
    ]:
        source.write_text(json.dumps(value))
        assert r.admit(source)["diagnostics"][0]["check"] == check
    payload = {
        "consensus_protocol_ready_score": 1,
        "split_manifest": {"worlds": worlds()},
        "cohort_references": {},
        "required_checks_passed": True,
        "code_config_hashes": [],
    }
    atomic_json(source, payload)
    monkeypatch.setattr(r, "SOURCE_PIN", reference(source)["sha256"])
    assert r.admit(source)["passed"]
    payload["cohort_references"]["missing"] = {
        "path": str(tmp_path / "absent"),
        "sha256": "sha256:0",
    }
    atomic_json(source, payload)
    monkeypatch.setattr(r, "SOURCE_PIN", reference(source)["sha256"])
    assert not r.admit(source)["passed"]


def test_summary_is_not_a_positive_by_oracle_alone():
    """REQ-VERIFY-8806: the exact reference cannot qualify learned benefit."""
    cohort = m.construct(worlds())
    parameters = [
        {
            "seed": 1,
            "models": [
                {
                    "layers": [[[[0.0] * 5] * 4, [0.0] * 4]],
                    "output_weight": [0.0] * 4,
                    "output_bias": 0.0,
                }
                for _ in range(3)
            ],
            "threshold": 0.6,
        }
    ]
    rows = m.evaluate(cohort, parameters)
    summary = m.summarize(rows, draws=50)
    assert not summary["benefit_passed"]
    assert summary["oracle_reference_rows"][0]["brier"] == 0
    assert all(row["status"] == "completed" for row in rows)
    assert np.isfinite([row["metric"] for row in rows]).all()


def sealed_fixture(tmp_path):
    """Private sealed receipts allow replay mutation tests without changing primaries."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    cohort = m.construct(worlds())
    parameters = [
        {
            "seed": 1,
            "models": [
                {
                    "layers": [[[[0.0] * 5] * 4, [0.0] * 4]],
                    "output_weight": [0.0] * 4,
                    "output_bias": 0.0,
                }
                for _ in range(3)
            ],
            "threshold": 0.6,
        }
    ]
    atomic_json(raw / "cohort.json", dict(rows=cohort))
    source = raw / "source.json"
    atomic_json(source, dict(private=True))
    work = dict(
        run_date="20261104",
        preconditions_checked=dict(passed=True, diagnostics=[], worlds=worlds()),
        fit=dict(parameters=parameters, shortcut_controls=[dict(passed=True)], costs=[]),
        per_candidate_rows=m.evaluate(cohort, parameters),
        source_artifact_hashes=[reference(source)],
        code_config_hashes=[],
        raw_shard_hashes=[],
        phase_spans=[],
        duration_s=0.1,
        cost_rows=[],
        cohort_reference=reference(raw / "cohort.json"),
    )
    validation = dict(passed=True, coverage=dict(passed=True), receipts=[], raw_references=[])
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    seal = dict(
        primitive_reference=reference(raw / "work.json"),
        validation_reference=reference(raw / "validation.json"),
    )
    atomic_json(raw / "independent_seal.json", seal)
    work.update(seal)
    value = r.build(work, validation)
    candidate = tmp_path / (r.NAME + ".json")
    atomic_json(candidate, value)
    return candidate, work, validation


def test_build_and_replay_mutations(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8806-CUSTODY: bound rows and aggregates reject local hash repairs."""
    candidate, work, validation = sealed_fixture(tmp_path)
    assert "GibbsModel" in r.build(work, validation)["target_model"]
    assert r.replay(candidate)
    original = json.loads(candidate.read_bytes())
    for field in ("duration_s", "completed_count", "consensus_rows_ready_score"):
        changed = deepcopy(original)
        changed[field] += 1
        atomic_json(candidate, changed)
        assert not r.replay(candidate)
    changed = deepcopy(original)
    changed["primitive_reference"]["sha256"] = "sha256:0"
    atomic_json(candidate, changed)
    assert not r.replay(candidate)
    atomic_json(candidate, original)
    primitive = Path(work["primitive_reference"]["path"])
    saved = primitive.read_bytes()
    mutated = json.loads(saved)
    mutated["per_candidate_rows"][0]["metric"] += 1
    atomic_json(primitive, mutated)
    assert not r.replay(candidate)
    # Repair both ordinary references: the independent seal still refuses the replacement.
    changed = deepcopy(original)
    changed["primitive_reference"] = reference(primitive)
    atomic_json(candidate, changed)
    assert not r.replay(candidate)
    primitive.write_bytes(saved)
    atomic_json(candidate, original)
    assert r.replay(candidate)
    validation["passed"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"
    validation["passed"] = True
    work["fit"]["shortcut_controls"][0]["passed"] = False
    assert r.build(work, validation)["consensus_rows_ready_score"] == 0
    work["preconditions_checked"]["passed"] = False
    blocked = r.build(work, validation)
    assert blocked["verdict_class"] == "blocked" and blocked["censored_count"] == 1080
    work["preconditions_checked"]["passed"] = True
    work["fit"]["shortcut_controls"][0]["passed"] = True
    original_summary = m.summarize
    monkeypatch.setattr(
        m, "summarize", lambda rows: dict(original_summary(rows), benefit_passed=True)
    )
    assert r.build(work, validation)["verdict_class"] == "positive"
    assert not r.replay(tmp_path / "absent.json")


def test_cold_cli_coverage_and_bad_date(tmp_path):
    """SCENARIO-REPORT-8806-REPLAY: actual cold child works without PYTHONPATH."""
    candidate, _, _ = sealed_fixture(tmp_path)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    rcfile = env.get("COVERAGE_RCFILE")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + rcfile]
        if rcfile
        else [sys.executable]
    )
    cli = str(r.ROOT / r.CLI)
    for args, exit_code in [
        (["--cold-replay", str(candidate)], 0),
        (["--cold-replay", str(tmp_path / "missing.json")], 1),
        (["--date", "20261105"], 2),
    ]:
        result = subprocess.run(
            prefix + [cli] + args, cwd=tmp_path, env=env, capture_output=True, timeout=60
        )
        assert result.returncode == exit_code, result.stderr.decode()


def test_preflight_projection(tmp_path, monkeypatch):
    """REQ-VERIFY-8806: setup is charged once and absent cohorts cannot start training."""
    cohort_worlds = [
        dict(
            world_id=f"w/{i}",
            split="consensus_dev" if i < 60 else "consensus_test",
            edges=[[0, 1], [0, 2], [1, 2], [2, 3]],
        )
        for i in range(180)
    ]
    monkeypatch.setattr(
        r,
        "admit",
        lambda path: dict(
            passed=True,
            references=[],
            diagnostics=[],
            producer=dict(split_manifest=dict(worlds=cohort_worlds)),
        ),
    )
    pre = r.preflight(tmp_path / "private.json", tmp_path)
    assert pre["passed"] and pre["validation_reserve_s"] == 900
    assert pre["projected_s"] >= pre["setup_s"] + 900
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=False))
    assert not r.preflight(tmp_path / "private.json", tmp_path)["passed"]
    monkeypatch.setattr(r, "admit", lambda path: dict(passed=False, references=[], diagnostics=[]))
    assert not r.preflight(tmp_path / "private.json", tmp_path)["passed"]


def test_validation_and_sqlite_combination(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8806-VALIDATION: combine real coverage, retain child arguments."""
    from coverage import Coverage

    directory = tmp_path / "sqlite"
    directory.mkdir()
    cov = Coverage(data_file=str(directory / ".coverage"), data_suffix=True, config_file=False)
    cov.start()
    m.target("support", "support")
    cov.stop()
    cov.save()
    actual = r.coverage_summary(directory)
    assert not actual["passed"] and actual["percent"] < 100
    assert Path(actual["database"]["path"]).is_file()
    calls = []

    def child(name, argv, logs, **kwargs):
        calls.append((name, argv))
        return dict(name=name, passed=True)

    monkeypatch.setattr(r, "child", child)

    def summary(directory):
        shutil.copy2(Path(actual["database"]["path"]), directory / ".coverage")
        return dict(passed=True, database=reference(directory / ".coverage"))

    monkeypatch.setattr(r, "coverage_summary", summary)
    assert r.validate(tmp_path)["passed"]
    assert {x[0] for x in calls} == {"unit", "e2e018", "e2e021", "ruff", "format", "mypy", "spec"}
    assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]
    assert {x[0] for x in calls} >= {"adversarial", "row_consistency", "cold_replay"}


def test_main_terminal_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8806: successful measurement and external blocking both terminate."""
    candidate, fixture, validation = sealed_fixture(tmp_path)
    monkeypatch.setattr(r, "validate", lambda raw: validation)
    monkeypatch.setattr(r, "terminal", lambda candidate, raw: dict(passed=True))
    monkeypatch.setattr(
        r,
        "preflight",
        lambda source, private: dict(
            passed=True,
            worlds=worlds(),
            references=fixture["source_artifact_hashes"],
            diagnostics=[],
        ),
    )
    monkeypatch.setattr(m, "fit", lambda cohort, seeds: fixture["fit"])
    assert r.main(["--output", str(candidate)]) == 0
    assert json.loads(candidate.read_text())["completed_count"] == 1080
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(r, "replay", lambda path: True)
    assert r.main(["--cold-replay", str(candidate)]) == 0
    monkeypatch.setattr(
        r,
        "preflight",
        lambda source, private: dict(passed=False, worlds=[], references=[], diagnostics=[]),
    )
    assert r.main(["--output", str(candidate)]) == 0
    assert json.loads(candidate.read_text())["verdict_class"] == "blocked"

    def absent(source, private):
        raise FileNotFoundError("private missing operand")

    monkeypatch.setattr(r, "preflight", absent)
    assert r.main(["--output", str(candidate)]) == 0
    assert json.loads(candidate.read_text())["gate_check_summary"][0]["check"] == "missing_file"


def test_replay_recomputes_despite_repaired_hashes(tmp_path):
    """SCENARIO-REPORT-8806-REPLAY: source and computation both bind the result."""
    candidate, work, _ = sealed_fixture(tmp_path)
    primitive = Path(work["primitive_reference"]["path"])
    seal_path = primitive.parent / "independent_seal.json"
    original_work = primitive.read_bytes()
    original_seal = seal_path.read_bytes()
    original_candidate = candidate.read_bytes()
    for mutation in ("rows", "cohort", "source"):
        primitive.write_bytes(original_work)
        seal_path.write_bytes(original_seal)
        candidate.write_bytes(original_candidate)
        body = json.loads(original_work)
        if mutation == "rows":
            body["per_candidate_rows"][0]["metric"] += 1
        elif mutation == "cohort":
            altered = json.loads(Path(body["cohort_reference"]["path"]).read_bytes())
            altered["rows"][0]["truth"] = "insufficient"
            atomic_json(Path(body["cohort_reference"]["path"]), altered)
            body["cohort_reference"] = reference(Path(body["cohort_reference"]["path"]))
        else:
            Path(body["source_artifact_hashes"][0]["path"]).write_text("{}")
        atomic_json(primitive, body)
        seal = json.loads(original_seal)
        seal["primitive_reference"] = reference(primitive)
        atomic_json(seal_path, seal)
        value = json.loads(original_candidate)
        value["primitive_reference"] = seal["primitive_reference"]
        atomic_json(candidate, value)
        assert not r.replay(candidate)


def test_admitted_operand_drift(tmp_path, monkeypatch):
    """REQ-VERIFY-8806: a matching primary cannot conceal changed bound raw bytes."""
    bound = tmp_path / "bound.json"
    atomic_json(bound, {"original": True})
    source = tmp_path / "source.json"
    atomic_json(
        source,
        dict(
            consensus_protocol_ready_score=1,
            required_checks_passed=True,
            cohort_references=dict(input=reference(bound)),
            code_config_hashes=[],
        ),
    )
    monkeypatch.setattr(r, "SOURCE_PIN", reference(source)["sha256"])
    assert r.admit(source)["passed"]
    bound.write_text("{}")
    assert not r.admit(source)["passed"]


def test_long_loop_progress(capsys):
    """REQ-VERIFY-8806: real completed/pending counts cover long world loops."""
    held = [
        dict(world_id=f"held/{i}", split="consensus_test", edges=[[0, 1], [0, 2], [1, 2], [2, 3]])
        for i in range(21)
    ]
    cohort = m.construct(held)
    parameters = [
        {
            "seed": 1,
            "models": [
                {
                    "layers": [[[[0.0] * 5] * 4, [0.0] * 4]],
                    "output_weight": [0.0] * 4,
                    "output_bias": 0.0,
                }
                for _ in range(3)
            ],
            "threshold": 0.6,
        }
    ]
    assert len(m.evaluate(cohort, parameters)) == 21 * 9 * 3 * 4
    output = capsys.readouterr().out
    assert "construct_worlds completed=20 pending=1" in output
    assert "evaluate_groups completed=180 pending=9" in output


def test_serving_costs_are_measured():
    """REQ-REPORT-8806: oracle checks and shared paid checks have explicit costs."""
    cohort = m.construct(worlds())
    parameters = [
        {
            "seed": 1,
            "models": [
                {
                    "layers": [[[[0.0] * 5] * 4, [0.0] * 4]],
                    "output_weight": [0.0] * 4,
                    "output_bias": 0.0,
                }
                for _ in range(3)
            ],
            "threshold": 0.6,
        }
    ]
    costs = []
    m.evaluate(cohort, parameters, costs)
    assert {x["arm"] for x in costs} == {
        "agreement_only",
        "fixed_evidence",
        "learned_energy",
        "exact_oracle",
    }
    assert all(x["duration_s"] > 0 for x in costs)
    assert next(x for x in costs if x["arm"] == "exact_oracle")["exact_solver_calls"] == 27


def test_rejected_candidate_is_retained_and_disqualified(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8806-CUSTODY: owned terminal failure preserves rejected bytes."""
    candidate, fixture, validation = sealed_fixture(tmp_path)
    monkeypatch.setattr(
        r,
        "preflight",
        lambda source, private: dict(passed=False, worlds=[], references=[], diagnostics=[]),
    )
    monkeypatch.setattr(r, "validate", lambda raw: validation)
    calls = []

    def terminal(candidate, raw):
        calls.append(candidate)
        value = dict(
            passed=len(calls) > 1, receipts=[dict(name="private_rejection", passed=len(calls) > 1)]
        )
        atomic_json(raw / "terminal_validation.json", value)
        return value

    monkeypatch.setattr(r, "terminal", terminal)
    assert r.main(["--output", str(candidate)]) == 0
    value = json.loads(candidate.read_bytes())
    assert value["consensus_rows_ready_score"] == 0
    assert list((tmp_path / "raw" / r.NAME / "invocations").glob("*/rejected_candidate.json"))


def test_publication_identity_failure_propagates(tmp_path, monkeypatch):
    """REQ-VERIFY-8806: publication conflicts cannot replace an existing primary."""
    monkeypatch.setattr(
        r,
        "preflight",
        lambda source, private: dict(passed=False, worlds=[], references=[], diagnostics=[]),
    )
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=True, coverage={}, receipts=[], raw_references=[])
    )

    def conflicting(*args, **kwargs):
        raise ValueError("conflicting_primary")

    monkeypatch.setattr(r, "publish_primary", conflicting)
    with pytest.raises(ValueError, match="conflicting_primary"):
        r.main(["--output", str(tmp_path / (r.NAME + ".json"))])
