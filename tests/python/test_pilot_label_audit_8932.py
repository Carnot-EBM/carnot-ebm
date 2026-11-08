"""REQ-REPORT-8932; SCENARIO-REPORT-8932-CONTRACT: preserve every pilot unit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import pilot_label_audit_8932 as audit
from carnot.reporting import pilot_label_audit_runner_8932 as runner


def operands():
    """Private graphs expose parsing and distance mistakes without loading weights."""
    graphs, responses, labels = [], [], []
    for i in range(50):
        graph = dict(
            unit_id=f"g{i}",
            seed=i,
            nodes=["AA", "BB", "CC", "DD", "EE"],
            edges=[["AA", "BB"], ["BB", "CC"], ["AA", "CC"], ["CC", "DD"]],
            start="AA",
            k=2,
            expected=["DD"],
        )
        answer = ["DD", "BB", "ZZ", "DD BB"][i % 4]
        parsed, reason = pilot.parse_answer(answer, graph["nodes"])
        responses.append(
            dict(
                unit_id=graph["unit_id"],
                status="completed",
                answer=answer,
                raw_response=dict(choices=[dict(message=dict(content=answer))]),
                usage=dict(completion_tokens=2),
            )
        )
        labels.append(
            dict(
                unit_id=graph["unit_id"],
                error=int(answer != "DD"),
                parsed_node=parsed,
                parser_reason=reason,
            )
        )
        graphs.append(graph)
    return dict(graph_manifest=graphs, answer_rows=labels), dict(responses=responses)


def test_independent_distances_and_all_categories():
    """REQ-REPORT-8932: a two-edge walk through a triangle is not distance two."""
    value, runtime = operands()
    graph = value["graph_manifest"][0]
    assert audit.distances(graph) == dict(AA=0, BB=1, CC=1, DD=2, EE=None)
    result = audit.reduce(value, runtime)
    assert result["error_type_counts"] == dict(
        correct=13, valid_wrong_distance=13, out_of_graph=12, malformed=12
    )
    assert result["semantic_error_rate"]["numerator"] == 13
    assert result["semantic_error_rate"]["denominator"] == 26
    assert result["total_error_rate"]["numerator"] == 37
    assert result["total_error_rate"]["denominator"] == 50
    assert result["label_disagreements"] == []
    assert result["rows"][3]["original_response"] == runtime["responses"][3]
    assert result["output_length_distribution"]["characters"] == {"2": 38, "5": 12}
    assert audit.rate(0, 0)["wilson_95"] is None
    assert audit.rate(0, 1)["wilson_95"]["low"] == 0
    assert audit.rate(1, 1)["wilson_95"]["high"] == 1


def test_disagreements_and_scheduled_failures():
    """REQ-REPORT-8932: failures are visible and cannot inflate reasoning errors."""
    value, runtime = operands()
    value["graph_manifest"][0]["expected"] = ["BB"]
    value["answer_rows"][1]["error"] = 0
    runtime["responses"][2]["raw_response"]["choices"][0]["message"]["content"] = "AA"
    runtime["responses"][3]["status"] = "runtime_failed"
    runtime["responses"].pop()
    result = audit.reduce(value, runtime)
    assert result["completed_count"] == 48
    assert result["failed_count"] == result["censored_count"] == 1
    assert result["excluded_count"] == 0
    assert {r["field"] for r in result["label_disagreements"]} >= {
        "expected",
        "error",
        "raw_answer",
    }
    assert result["rows"][3]["value"] is None
    assert result["rows"][-1]["missing_reason"] == "not_started"
    runtime["responses"][0]["status"] = "unknown"
    with pytest.raises(ValueError, match="response_status"):
        audit.reduce(value, runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_response",
        "foreign_response",
        "duplicate_seed",
        "missing_graph",
        "duplicate_label",
    ],
)
def test_roster_rejection(mutation):
    """SCENARIO-REPORT-8932-CONTRACT: duplicate units never become independent data."""
    value, runtime = operands()
    if mutation == "duplicate_response":
        runtime["responses"].append(runtime["responses"][0])
    elif mutation == "foreign_response":
        runtime["responses"][0]["unit_id"] = "foreign"
    elif mutation == "duplicate_seed":
        value["graph_manifest"][0]["seed"] = 1
    elif mutation == "missing_graph":
        value["graph_manifest"].pop()
    else:
        value["answer_rows"][0] = value["answer_rows"][1]
    with pytest.raises(ValueError, match="roster"):
        audit.reduce(value, runtime)


def test_real_pilot_and_authentication(tmp_path):
    """REQ-REPORT-8932: historical answers remain opaque upstream compute."""
    evidence = audit.authenticate(audit.ROOT)
    assert evidence["failures"] == []
    result = audit.reduce(evidence["pilot"], evidence["runtime"])
    assert result["completed_count"] == result["independent_count"] == 50
    assert result["total_error_rate"]["numerator"] == 43
    assert result["label_disagreements"] == []
    assert all(
        r["original_graph"] == g
        for r, g in zip(result["rows"], evidence["pilot"]["graph_manifest"], strict=True)
    )
    absent = audit.authenticate(tmp_path)
    assert absent["failures"]
    assert "missing_file" in {r["observed"] for r in absent["failures"]}


def test_cold_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8932-CONTRACT: actual child works without checkout environment."""
    output = tmp_path / "results" / (audit.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(audit.ROOT / audit.CLI)]

    def child(*args):
        return subprocess.run(
            [*command, *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )

    assert child("--date", "20260101").returncode == 2
    run = child("--date", "20261108", "--fixture-e2e", str(output))
    assert run.returncode == 0, run.stdout + run.stderr
    assert child("--cold-replay", str(output)).returncode == 0
    original = json.loads(output.read_text())
    assert original["pilot_labels_ready_score"] == 0
    for field, changed in [
        ("semantic_error_rate", dict(numerator=0)),
        ("pilot_labels_ready_score", 1),
        ("verdict_class", "positive"),
    ]:
        value = deepcopy(original)
        value[field] = changed
        value["reproducibility_checksum"] = pilot.checksum(value)
        output.write_text(json.dumps(value))
        assert child("--cold-replay", str(output)).returncode == 1
    assert child("--cold-replay", str(tmp_path / "absent.json")).returncode == 1


@pytest.fixture
def frozen(tmp_path):
    """REQ-REPORT-8932: private snapshots leave all historical producer bytes untouched."""
    work = runner.freeze(audit.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    runner.atomic_json(Path(work["raw"]) / "work.json", work)
    runner.atomic_json(Path(work["raw"]) / "validation.json", validation)
    return work, validation


def test_readiness_and_blocked_controls(frozen, monkeypatch):
    """REQ-REPORT-8932: owned failures disqualify; missing external inputs block terminally."""
    work, validation = frozen
    monkeypatch.setattr(runner, "verify_measured", lambda *_: True)
    validation["receipts"] = [
        dict(name=n, passed=True, exit_code=0, timed_out=False) for n in runner.methods.CHECKS
    ]
    result = runner.build(work, validation, False)
    assert result["pilot_labels_ready_score"] == 1
    assert result["verdict_class"] == "circular_positive"
    assert not any(result["model_invocation_counts"].values())
    assert result["calibration_evaluation_exclusion"]["all_pilot_graphs_excluded"]
    assert len(result["calibration_evaluation_exclusion"]["graph_hashes"]) == 50
    validation["receipts"][0]["passed"] = False
    assert runner.build(work, validation, False)["verdict_class"] == "disqualified"
    validation["receipts"][0]["passed"] = True
    work["result"]["label_disagreements"].append(dict(unit_id="g0", field="error"))
    blocked = runner.build(work, validation, False)
    assert blocked["pilot_labels_ready_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]
    work["evidence"] = audit.authenticate(Path(work["raw"]) / "missing")
    work["result"] = {}
    blocked = runner.build(work, validation, False)
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert blocked["censored_count"] == 50
    assert blocked["verdict_class"] == "blocked"


@pytest.mark.parametrize("mode", ["raw_drift", "custody_drift", "missing_raw"])
def test_upstream_custody_rejection(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8932-CONTRACT: no altered raw response can authenticate."""
    authentic = audit.authenticate(audit.ROOT)
    value = deepcopy(authentic["pilot"])
    runtime = deepcopy(authentic["runtime"])
    original_raw = Path(
        next(r["path"] for r in value["raw_shard_hashes"] if Path(r["path"]).name == "runtime.json")
    ).parent
    for directory in ["requests", "responses"]:
        (tmp_path / directory).mkdir()
        for path in (original_raw / directory).glob("*.json"):
            (tmp_path / directory / path.name).write_bytes(path.read_bytes())
    if mode == "raw_drift":
        runtime["responses"][0]["answer"] = "ZZ"
    elif mode == "custody_drift":
        runtime["model_receipt"]["owner"]["owned_by_task"] = False
    else:
        (tmp_path / "responses/graph-00.json").unlink()
    runner.atomic_json(tmp_path / "runtime.json", runtime)
    value["raw_shard_hashes"] = [dict(path=str(tmp_path / "runtime.json"))]
    primary = tmp_path / runner.methods.PILOT
    runner.atomic_json(primary, value)
    monkeypatch.setattr(audit.methods, "authenticate", lambda _: dict(failures=[], references=[]))
    monkeypatch.setattr(audit.pilot, "replay", lambda _: dict(passed=True))
    result = audit.authenticate(tmp_path)
    assert result["failures"][0]["observed"] == (
        "missing_file" if mode == "missing_raw" else "rejected_producer"
    )


def test_replay_operand_and_semantic_drift(frozen, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8932-CONTRACT: recomputation protects both work and aggregate bytes."""
    work, validation = frozen
    output = tmp_path / "artifact.json"
    value = runner.build(work, validation, True)
    runner.atomic_json(output, value)
    assert runner.replay(output)["passed"]
    original_authenticate = audit.authenticate
    monkeypatch.setattr(
        audit,
        "authenticate",
        lambda root: dict(original_authenticate(root), authority_mode="changed"),
    )
    assert not runner.replay(output)["passed"]
    monkeypatch.setattr(audit, "authenticate", original_authenticate)
    original_reduce = audit.reduce
    monkeypatch.setattr(
        audit, "reduce", lambda *args: dict(original_reduce(*args), completed_count=49)
    )
    assert not runner.replay(output)["passed"]
    monkeypatch.setattr(audit, "reduce", original_reduce)
    Path(work["sources"][0]["path"]).write_text("changed bytes")
    assert not runner.replay(output)["passed"]


def test_validation_plan(tmp_path):
    """REQ-REPORT-8932: no historical source statements enter owned coverage."""
    commands = runner.validation_plan(tmp_path)
    assert {c.name for c in commands} == runner.methods.CHECKS
    config = (tmp_path / "coverage.ini").read_text()
    assert all(name in config for name in audit.OWNED)
    assert "bfs_pilot_runtime_8918.py" not in config
    assert "exclude_lines =\n" in config


@pytest.mark.parametrize("mode", ["pass", "no_coverage", "terminal_reject", "publication_error"])
def test_production_control_flow(frozen, tmp_path, monkeypatch, mode):
    """REQ-REPORT-8932: private controls execute every producer path without claiming science."""
    work, _ = frozen
    monkeypatch.setattr(runner.a, "ROOT", tmp_path)

    def private_freeze(root, raw):
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.symlink_to(Path(work["raw"]), target_is_directory=True)
        return work

    monkeypatch.setattr(runner, "freeze", private_freeze)
    private = tmp_path / "validation"
    private.mkdir()
    if mode != "no_coverage":
        (private / ".coverage").write_bytes(b"private fixture; never research evidence")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(runner, "validation_plan", lambda _: [])
    monkeypatch.setattr(runner, "measured", lambda *_: dict(private_control=True))
    monkeypatch.setattr(runner, "verify_measured", lambda value, _: bool(value))
    calls = []

    def commands(root, specs, **kwargs):
        calls.append(specs)
        names = [c.name for c in specs] or list(runner.methods.CHECKS)
        return [
            dict(
                name=n,
                passed=not (mode == "terminal_reject" and n == "strict_rows"),
                exit_code=int(mode == "terminal_reject" and n == "strict_rows"),
                timed_out=False,
            )
            for n in names
        ]

    monkeypatch.setattr(runner, "run_commands", commands)
    published = []

    def publication(output, value, validator):
        if mode == "publication_error":
            raise ValueError("conflicting_identity")
        candidate = tmp_path / "candidate.json"
        runner.atomic_json(candidate, value)
        report = validator(candidate)
        if not report["passed"]:
            raise ValueError("candidate_rejected")
        published.append(value)
        return dict(passed=True)

    monkeypatch.setattr(runner, "publish_primary", publication)
    if mode == "publication_error":
        with pytest.raises(ValueError, match="conflicting_identity"):
            runner.main(["--date", "20261108"])
    else:
        assert runner.main(["--date", "20261108"]) == 0
        assert published
        assert published[-1]["pilot_labels_ready_score"] == int(mode == "pass")
        assert published[-1]["verdict_class"] == (
            "circular_positive" if mode == "pass" else "disqualified"
        )
    assert len(calls) >= 2


def test_changed_and_zero_upstream_inputs(tmp_path):
    """SCENARIO-REPORT-8932-CONTRACT: zero gates and changed bytes remain explicit blockers."""
    for name in [
        *runner.methods.INPUT_PINS,
        runner.methods.branch.DESIGN,
        runner.methods.branch.ACTIVE,
        audit.NOTE,
    ]:
        source = audit.ROOT / name
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    primary = tmp_path / runner.methods.PILOT
    value = json.loads(primary.read_text())
    value["pilot_ready_score"] = 0
    runner.atomic_json(primary, value)
    result = audit.authenticate(tmp_path)
    assert {r["observed"] for r in result["failures"]} >= {"rejected_producer", "observed_zero"}
    primary.write_text("invalid json")
    assert audit.authenticate(tmp_path)["failures"]
