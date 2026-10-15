"""REQ-REPORT-8527, REQ-VERIFY-8527: custody qualifies without new science."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from carnot.reporting import v733_source_consumer_qualification as e
from carnot.reporting import v733_source_consumer_runner as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.reporting.v733_source_diagnostics import diagnostic_rows


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Use actual independently sealed inputs so a favorable fixture cannot replace custody."""
    raw = tmp_path_factory.mktemp("source8527")
    return e.measure(e.ROOT, raw), raw


def test_diagnostic_schema():
    """SCENARIO-VERIFY-8527-CONTROLS: absent and zero remain different observations."""
    rows = diagnostic_rows(
        [
            dict(
                upstream_id="producer",
                artifact_path="missing",
                artifact_hash=None,
                artifact_field="ready",
                op="==",
                expected=1,
                observed=None,
                passed=False,
            ),
            dict(
                check="zero",
                upstream="producer",
                path="present",
                sha256="hash",
                field="ready",
                operator="==",
                expected=1,
                observed=0,
                passed=False,
            ),
        ]
    )
    assert rows[0]["observed"] is None and rows[1]["observed"] == 0
    assert rows[0]["upstream"] == "producer"
    assert all(e.DIAGNOSTIC_FIELDS <= set(r) for r in rows)
    assert diagnostic_rows([dict(check="fallback")])[0]["upstream"] == e.TASK


def test_custody(measured):
    """SCENARIO-REPORT-8527-CUSTODY: retain every original role and exposed loss."""
    work, raw = measured
    assert not work["failures"] and not work["owned_failure"]
    value = e.build(work, [dict(name="cold", passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["experiment_id"] == 8527 and value["run_date"] == "20261015"
    assert value["comparator_inputs_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["intended_count"] == value["completed_count"] == 7500
    assert value["independent_count"] == 100
    manifest = e.read(value["comparator_manifest_reference"])
    assert manifest["arms"] == list(e.COMPARATORS)
    assert len(manifest["slots"]) == 22500
    assert {k: len(v) for k, v in manifest["question_roles"].items()} == dict(
        fit=150, tune=50, test=100
    )
    assert all(
        r["status"] == "future_dependency"
        for r in manifest["slots"]
        if r["arm"].startswith("brier_")
    )
    assert not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert len(value["literature_mapping"]) == 5


def test_absence_owned_failure(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8527-CONTROLS: blocked evidence and owned failure differ."""
    raw = tmp_path / "absent"
    work = e.measure(tmp_path / "missing", raw)
    value = e.build(work, [dict(name="cold", passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 7500
    assert all(e.DIAGNOSTIC_FIELDS <= set(r) for r in value["gate_check_summary"])
    assert any(r["observed"] is None for r in value["gate_check_summary"])
    original, raw = measured
    failed = e.build(
        original, [dict(name="failed", passed=False)], raw, tmp_path / (e.NAME + ".json")
    )
    assert (
        failed["verdict_class"] == "disqualified" and failed["comparator_inputs_ready_score"] == 0
    )
    monkeypatch.setattr(
        e.baseline, "evaluate", lambda *_: (_ for _ in ()).throw(ValueError("owned"))
    )
    assert e.measure(e.ROOT, tmp_path / "owned")["owned_failure"]


@pytest.mark.parametrize(
    "mode", ["valid", "manifest", "diagnostic", "literature", "source", "historical"]
)
def test_rehashed_replay(measured, tmp_path, mode):
    """SCENARIO-VERIFY-8527-CONTROLS: local rehashing cannot replace original operands."""
    original, _ = measured
    work = deepcopy(original)
    if mode == "manifest":
        manifest = e.read(work["comparator_manifest_reference"])
        manifest["slots"][0]["donor_answer_id"] = "tampered"
        atomic_json(tmp_path / "manifest.json", manifest)
        work["comparator_manifest_reference"] = e.reference(tmp_path / "manifest.json")
    if mode == "diagnostic":
        work["diagnostic_schema_rows"][0]["observed"] = "tampered"
    if mode == "literature":
        work["literature_mapping"][0]["adopt"] = "test-selected threshold"
    if mode == "source":
        work["source_refs"][0]["sha256"] = "wrong"
    if mode == "historical":
        work["original_failure_reference"][0]["sha256"] = "wrong"
    receipt = child("real", [sys.executable, "-c", "print('real')"], tmp_path / "logs")
    shutil.copyfile(
        Path(original["prediction_reference"]["path"]).parent / "evaluator_targets.json",
        tmp_path / "evaluator_targets.json",
    )
    atomic_json(tmp_path / "measurement.json", work)
    value = e.build(work, [receipt], tmp_path, tmp_path / (e.NAME + ".json"))
    atomic_json(tmp_path / "candidate.json", value)
    assert e.replay(tmp_path / "candidate.json") == (mode == "valid")


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8527-CONTROLS: children publish final bytes and exercise rejection."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for i, (args, expected) in enumerate(
        [
            (["--private-control", "--date", "20261015", "--output", str(output)], 0),
            (["--cold-replay", str(output)], 0),
            (["--cold-replay", str(tmp_path / "missing")], 1),
            (["--deliberate-error"], 7),
            (["--date", "20261014"], 2),
            (["--private-control"], 2),
            (
                [
                    "--private-control",
                    "--root",
                    str(tmp_path / "missing"),
                    "--output",
                    str(tmp_path / "blocked" / (e.NAME + ".json")),
                ],
                0,
            ),
        ]
    ):
        receipt = child(
            f"cli{i}", [*argv, *args], tmp_path / "cli", expected=expected, deadline=180
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert (
        terminal["consumer"]["passed"]
        and terminal["consumer"]["gate_sha256"] == terminal["primary_sha256"]
    )
    assert value["consumer_receipt"]["scope"] == "prepublication_private_candidate"


def test_validation_manifest(tmp_path, monkeypatch):
    """REQ-VERIFY-8527: freeze the original source node union before bounded shards."""
    monkeypatch.setattr(runner.collected, "manifest", lambda _: [])
    monkeypatch.setattr(
        runner.collected, "collect", lambda _, suite: ([suite + "::test_one"], dict(argv=["true"]))
    )
    plan = runner.manifest(tmp_path)
    assert any(p["name"].startswith("original_source_") for p in plan)
    assert all(p["deadline"] <= 1200 for p in plan)


def test_main_and_source_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8527-CUSTODY: flagged or renamed copies cannot inherit original seals."""
    monkeypatch.setattr(runner.shipped, "run", lambda *a, **kw: 0)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    upstream = e.read(dict(path=str(e.ROOT / e.baseline.UPSTREAM), sha256=e.baseline.UPSTREAM_PIN))
    root = tmp_path / "altered"
    path = root / e.baseline.UPSTREAM
    for field, changed in [
        ("head_controls_ready_score", 0),
        ("task_id", "exp1-wrong"),
        ("flagged_adversarial", True),
    ]:
        value = dict(upstream, **{field: changed})
        atomic_json(path, value)
        with pytest.raises(ValueError, match="source_hash"):
            e.baseline.operands(root)
    with pytest.raises(FileNotFoundError):
        e.baseline.operands(tmp_path / "absent")
