"""REQ-REPORT-8430 / REQ-VERIFY-8430: original bytes, custody and real children."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v726_feature_panel as e
from carnot.reporting import v726_feature_panel_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_original_panel_and_custody(tmp_path):
    """SCENARIO-REPORT-8430-PANEL: every answer inherits its frozen question role."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    panel = work["panel"]
    assert len(panel["rows"]) == 900
    assert panel["support_passed"]
    assert {k: v["usable_questions"] for k, v in panel["class_support"].items()} == {
        "fit": 150,
        "tune": 50,
        "test": 100,
    }
    assert len(panel["trainer_rows"]) == 600
    assert all(row["role"] != "test" for row in panel["trainer_rows"])
    assert len(panel["evaluator_rows"]) == 300
    assert all("y" not in row and "human_primary" not in row for row in panel["rows"])
    assert panel["rows"][0]["x"] == pytest.approx(
        [0.14, *e.extract(work["public_rows"][0])["values"]]
    )
    assert set(panel["partition_hashes"]) == {"fit", "tune", "test"}
    assert all(row["overlap"] == "unknown" for row in panel["exposure_ledger"])
    assert e.independent_sample(work["public_rows"], panel["rows"])
    receipt = child("proof", [sys.executable, "-c", "print('actual child')"], tmp_path / "logs")
    value = e.build(work, [receipt], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["feature_panel_ready_score"] == 1
    failed = e.build(work, [dict(receipt, passed=False)], tmp_path / "raw", tmp_path / "out.json")
    assert failed["verdict_class"] == "disqualified"


@pytest.mark.parametrize("mutation", ["source", "q", "target", "id", "duplicate", "role"])
def test_missing_and_identity_tripwires(tmp_path, mutation):
    """SCENARIO-REPORT-8430-CUSTODY: missing operands never acquire invented values."""
    operands = e.operands()
    master, audit, inherited, protocol = deepcopy(operands)
    if mutation == "source":
        master[0]["ref_str"] = ""
    elif mutation == "q":
        audit[0]["error_p2"] = "unavailable"
    elif mutation == "target":
        master[0]["annotator1_label"] = "-1"
        audit[0]["annotator1_label"] = "-1"
    elif mutation == "id":
        master[0]["item_id"] = "foreign"
    elif mutation == "duplicate":
        master[1] = deepcopy(master[0])
    else:
        protocol["partition"]["roles"]["fit"][0] = "foreign"
    if mutation in {"id", "duplicate", "role"}:
        with pytest.raises(ValueError):
            e.panel(master, audit, inherited, protocol)
    else:
        result, _ = e.panel(master, audit, inherited, protocol)
        assert result["rows"][0]["excluded"]
        assert result["rows"][0]["missing_reason"]
        assert len(result["rows"]) == 900


def test_probability_and_leakage():
    """SCENARIO-REPORT-8430-CUSTODY: only p2 judgment and public text form predictors."""
    assert e.probability(dict(label=1, confidence=0, valid=True)) == 0.0001
    assert e.probability(dict(label=0, confidence=0, valid=True)) == 0.9999
    for value in [
        {},
        dict(label=2, confidence=0.5, valid=True),
        dict(label=1, confidence=float("nan"), valid=True),
    ]:
        assert e.probability(value) is None
    master, audit, inherited, protocol = e.operands()
    base, public = e.panel(master, audit, inherited, protocol)
    master[0]["annotator2_label"] = "1"
    changed, _ = e.panel(master, audit, inherited, protocol)
    assert [row["x"] for row in base["rows"]] == [row["x"] for row in changed["rows"]]
    public[0]["family_id"] = "different-id"
    assert e.extract(public[0])["values"] == base["rows"][0]["x"][1:]
    changed["rows"][0]["x"][0] = 0.8
    assert not e.independent_sample(public, changed["rows"])


def test_blocked_support_and_required_inputs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8430-CUSTODY: external absence blocks without reseeding."""
    master, audit, inherited, protocol = e.operands()
    for row in master:
        row["ref_str"] = ""
    result, _ = e.panel(master, audit, inherited, protocol)
    assert not result["support_passed"]
    blank_public = [dict(family_id="missing", source_bytes="", answer_bytes="61")]
    blank_row = [dict(x=None, factorial_outputs=dict(p2=dict(valid=False)))]
    assert e.independent_sample(blank_public, blank_row)
    monkeypatch.setattr(e, "operands", lambda: (_ for _ in ()).throw(FileNotFoundError("release")))
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    assert work["failures"]
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / "out.json")
    assert value["verdict_class"] == "blocked"
    assert len(value["rows"]) == 900 and value["censored_count"] == 900
    receipt = child(
        "blocked_proof", [sys.executable, "-c", "print('real blocked control')"], tmp_path / "logs"
    )
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], tmp_path / "raw", output))
    assert e.replay(output)


def test_private_cli_replay_and_errors(tmp_path):
    """SCENARIO-VERIFY-8430-REPLAY: fresh children reject self-consistent tampering."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child("cli", [*argv, "--private-control", "--output", str(output)], tmp_path / "logs")
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["feature_panel_ready_score"] == 1
    assert child("cold", [*argv, "--cold-replay", str(output)], tmp_path / "logs")["passed"]
    for key in ["rows", "partition_hashes", "feature_panel_ready_score"]:
        changed = deepcopy(value)
        changed[key] = [] if key == "rows" else 999
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert child(key, [*argv, "--cold-replay", str(output)], tmp_path / "logs", expected=1)[
            "passed"
        ]
    atomic_json(output, value)
    work_path = Path(value["primitive_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    work["panel"]["rows"][0]["x"][0] = 0.7
    atomic_json(work_path, work)
    value["primitive_reference"] = e.reference(work_path)
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert child("primitive", [*argv, "--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    assert child(
        "missing",
        [*argv, "--cold-replay", str(tmp_path / "missing")],
        tmp_path / "logs",
        expected=1,
    )["passed"]
    assert child("error", [*argv, "--deliberate-error"], tmp_path / "logs", expected=7)["passed"]
    assert (
        r.main(
            [
                "--date",
                "20261011",
                "--private-control",
                "--output",
                str(e.ROOT / "results" / output.name),
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit):
        r.main(["--date", "20261010"])
    assert r.manifest(tmp_path / "validation")


def test_precondition_and_sample_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8430: failed independent operands cannot qualify owned outputs."""
    monkeypatch.setattr(
        e.methods.legacy,
        "resources",
        lambda _: dict(
            private_disk_backed_scratch=False, available_memory_bytes=0, minimum_memory_bytes=1
        ),
    )
    monkeypatch.setattr(e, "independent_sample", lambda *_: False)
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert len(work["failures"]) >= 2
    monkeypatch.setattr(e, "read", lambda _: dict(methods_ready_score=0))
    monkeypatch.setattr(e, "operands", lambda: (_ for _ in ()).throw(ValueError("corrupt")))
    # Fallback reads use original pinned reference independently of the injected methods failure.
    original_read = e.methods.read
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: (
            dict(methods_ready_score=0) if ref["sha256"] == e.RECEIPT_PIN else original_read(ref)
        ),
    )
    bad = e.measure(e.ROOT, tmp_path / "bad")
    assert any(g["observed_value"] == "methods_not_ready" for g in bad["failures"])


def test_runner_validation_and_coverage_copy(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8430-REPLAY: real subprocess coverage shards survive publication."""
    import coverage

    private = tmp_path / "private"
    private.mkdir()
    cov = coverage.Coverage(data_file=str(private / ".coverage.actual"), config_file=False)
    source = private / "measured.py"
    source.write_text("print('genuine coverage primitive')\n")
    cov.start()
    exec(compile(source.read_text(), str(source), "exec"))
    cov.stop()
    cov.save()
    atomic_json(private / "coverage.json", dict(scope="private measured copy control"))
    monkeypatch.setattr(
        r,
        "manifest",
        lambda _: [
            dict(
                name="actual_validation_child",
                argv=[sys.executable, "-u", "-c", "print('bounded actual child')"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ],
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.run(e.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["feature_panel_ready_score"] == 1
    assert e.replay(output)
    for mutation in [
        "operand",
        "receipt",
        "primitive",
        "blocked_value",
        "authority_snapshot",
        "authority_reduction",
        "methods_gate",
        "custody",
    ]:
        changed = deepcopy(value)
        work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
        if mutation == "receipt":
            changed["validation_receipts"][0]["stdout_sha256"] = "sha256:foreign"
        else:
            if mutation == "operand":
                work["source_refs"][0]["sha256"] = "sha256:foreign"
            elif mutation == "primitive":
                work["panel"]["rows"][0]["x"][0] = 0.91
            elif mutation == "authority_snapshot":
                work["authority"]["authority_snapshots"]["design"]["sha256"] = "sha256:foreign"
            elif mutation == "authority_reduction":
                work["authority"]["activated"] = False
            elif mutation == "methods_gate":
                work["methods_receipt_reference"]["path"] = str(tmp_path / "missing_methods")
            elif mutation == "custody":
                target = tmp_path / "altered" / "trainer_view.json"
                atomic_json(target, dict(rows=work["panel"]["evaluator_rows"]))
                work["primitive_files"] = [
                    e.reference(target) if Path(ref["path"]).name == "trainer_view.json" else ref
                    for ref in work["primitive_files"]
                ]
            else:
                work["public_rows"] = []
            path = tmp_path / (mutation + ".json")
            atomic_json(path, work)
            changed["primitive_reference"] = e.reference(path)
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / ("candidate_" + mutation + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)
    bad = deepcopy(value)
    bad["inference_substrate_class"] = "unrecognized_class"
    candidate = tmp_path / "flagged.json"
    atomic_json(candidate, bad)
    checks, findings = r.scan(candidate, tmp_path / "flagged_scan")
    assert findings and not checks[0]["passed"]
