"""REQ-REPORT-8375 and REQ-VERIFY-8375: private operands keep truth claims separate."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import external_evidence_8375 as e
from carnot.reporting import external_evidence_runner_8375 as runner
from carnot.reporting.current_work_receipt import atomic_json


def unit(index: int = 0) -> dict:
    """Private structure proves parser behavior without becoming research evidence."""
    return dict(
        id=str(index),
        cluster=str(index),
        lane="grounded_qa",
        context=["source"],
        output="answer",
        label="correct",
        label_authority="independent_human",
        authority_reference="private-human-ledger",
        overlap="disjoint",
        license_usable=True,
        claims=["answer"],
        single_evidence=["source"],
        complete_evidence=["source"],
        oracle_only=False,
    )


def test_structure_controls() -> None:
    """SCENARIO-REPORT-8375-STRUCTURE: field presence cannot imply truth."""
    assert e.inspect(unit())["structurally_complete"]
    for field in ("context", "output", "claims", "cluster", "authority_reference"):
        value = unit()
        value.pop(field)
        assert not e.inspect(value)["eligible"]
    for update in (
        {"overlap": "contaminated"},
        {"label_authority": "model_judge"},
        {"oracle_only": True},
        {"license_usable": False},
    ):
        assert not e.inspect(unit() | update)["eligible"]
    assert e.inspect(unit() | {"label_authority": "model_judge"})["weak_label"]
    assert e.inspect(unit() | {"single_evidence": None})["paired_comparison"] is None


def test_metadata_order_and_support() -> None:
    """SCENARIO-REPORT-8375-SUPPORT: metadata alone owns order and support."""
    rows = [unit(i) | {"label": "correct" if i % 2 else "incorrect"} for i in range(80)]
    a = e.reduce_manifest({"units": rows})
    assert a["external_corpus_ready_score"] == 1
    assert len(a["rows"]) == 96 and a["completed_count"] == 32
    assert a["independent_count"] == 80
    assert e.reduce_manifest({"units": list(reversed(rows))}) == a
    changed = deepcopy(rows)
    changed[0]["output"] = "different content"
    assert [r["id"] for r in e.reduce_manifest({"units": changed})["rows"]] == [
        r["id"] for r in a["rows"]
    ]
    assert e.reduce_manifest({"units": rows[:79]})["external_corpus_ready_score"] == 0
    assert (
        e.reduce_manifest({"units": [unit(i) for i in range(80)]})["external_corpus_ready_score"]
        == 0
    )
    assert e.reduce_manifest({"units": rows + [rows[0]]})["independent_count"] == 80
    assert e.reduce_manifest({"units": []})["censored_count"] == 96
    with pytest.raises(ValueError, match="lane"):
        e.reduce_manifest({"units": [unit() | {"lane": "unknown"}]})


def test_cli_missing_and_deliberate_error(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8375-REPLAY: real children reject bad inputs and date."""
    cli = Path(e.ROOT) / e.CLI
    for args in (
        ["--cold-replay", str(tmp_path / "absent.json")],
        ["--deliberate-error"],
        ["--date", "20261011"],
    ):
        result = subprocess.run(
            [sys.executable, str(cli), *args], capture_output=True, text=True, timeout=20
        )
        assert result.returncode != 0
    missing = tmp_path / "candidate.json"
    atomic_json(missing, {"experiment_id": 8375})
    assert not e.replay(missing)


def test_private_e2e018(authority_paths: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """REQ-VERIFY-8375: unchanged authority lifecycle accepts and rejects controls."""
    from carnot.reporting.v685_authority_lifecycle import assess_authorities

    paths = authority_paths[:3]
    assert assess_authorities(*paths, tmp_path / "valid")["activated"]
    import yaml

    original = paths[2].read_bytes()
    for mutation in ("prompt", "delete", "reorder"):
        value = yaml.safe_load(original)
        if mutation == "prompt":
            value["tasks"][0]["prompt"] += " mutation"
        elif mutation == "delete":
            value["tasks"].pop()
        else:
            value["tasks"].reverse()
        paths[2].write_text(yaml.safe_dump(value))
        assert not assess_authorities(*paths, tmp_path / mutation)["activated"]


@pytest.fixture
def authority_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """REQ-VERIFY-8375: copy frozen lifecycle fixtures into private scratch."""
    import gzip

    fixture = e.ROOT / "tests/fixtures/v685"
    design, staged, active = (tmp_path / n for n in ("design.md", "staged.yaml", "active.yaml"))
    design.write_bytes((fixture / "design.md").read_bytes())
    active.write_bytes(gzip.decompress((fixture / "active.yaml.gz").read_bytes()))
    staged.write_bytes(active.read_bytes())
    return design, staged, active, tmp_path / "snapshots"


def private_work(tmp_path, monkeypatch, units=None, root=None):
    """REQ-VERIFY-8375: private measurement uses the real adapter and sealed operands."""
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, dict(units=[unit()] if units is None else units))
    monkeypatch.setattr(
        runner.authority,
        "authority",
        lambda *a: dict(
            tasks=[dict(id=e.TASK, MODEL_SPECS=[])], gate_check_summary=[], activated=True
        ),
    )
    work = runner.measure(e.ROOT if root is None else root, raw, manifest, tmp_path)
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    return work


def test_build_replay_and_real_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8375-REPLAY: rehashed primitives still fail fresh replay."""
    work = private_work(tmp_path, monkeypatch, [unit() | {"label_authority": "model_judge"}])
    receipts = [dict(name="private", passed=True, scope="owned")]
    value = e.build(work, receipts)
    assert value["verdict_class"] == "blocked"
    assert any(
        g["field"] == "independent_label_context_claims_overlap"
        for g in value["gate_check_summary"]
    )
    assert e.build(work, [])["verdict_class"] == "disqualified"
    assert e.build(work, [dict(passed=True, scope="global")])["verdict_class"] == "disqualified"
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    assert all(r["passed"] for r in runner.controls(path, tmp_path / "controls"))
    assert runner.validate(path, tmp_path / "validators")["passed"]
    changed = deepcopy(value)
    changed["completed_count"] += 1
    atomic_json(path, changed)
    assert not e.replay(path)
    source = Path(work["source_artifact_hashes"][0]["path"])
    source.write_bytes(b"changed")
    atomic_json(path, value)
    assert not e.replay(path)


def test_full_support_and_absence(tmp_path, monkeypatch):
    """REQ-REPORT-8375: independently usable metadata needs80 clusters and8 per class."""
    units = [unit(i) | {"label": "correct" if i % 2 else "incorrect"} for i in range(80)]
    work = private_work(tmp_path, monkeypatch, units)
    # Empty lanes remain blocked despite a metadata support floor in one lane.
    value = e.build(work, [dict(passed=True)])
    assert value["independent_count"] == 80 and value["external_corpus_ready_score"] == 0
    second = tmp_path / "missing"
    second.mkdir()
    monkeypatch.setattr(
        runner.authority,
        "authority",
        lambda *a: dict(
            tasks=[],
            gate_check_summary=[
                dict(
                    upstream_id="private",
                    artifact_path="missing",
                    artifact_hash=None,
                    artifact_field="authority",
                    op="==",
                    expected=True,
                    observed=None,
                )
            ],
        ),
    )
    work = runner.measure(second, second, second / "absent.json", second)
    assert any(g.get("observed_value") is None for g in work["gate_check_summary"])
    assert (
        e.reduce_manifest(
            {"units": [unit() | {"label": "correct"}, unit() | {"label": "incorrect"}]}
        )["independent_count"]
        == 0
    )


def test_release_hydration_and_hash_rejection(tmp_path):
    """REQ-VERIFY-8375: author blob identity and question joins reject byte drift."""
    import hashlib

    source, trace = tmp_path / "source", tmp_path / "trace"
    record = dict(qid="id", paragraphs=[dict(title="title", text="context", is_gold=True)])
    source.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    trace.write_text(
        json.dumps(dict(qid="id", pred="answer", em=1))
        + "\n"
        + json.dumps(dict(qid="id", pred="other", em=0))
        + "\n"
    )
    ref = e.reference(source)
    data = source.read_bytes()
    ref["git_blob"] = hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
    ).hexdigest()
    manifest = dict(
        operands=[ref],
        split_reference=ref,
        trace_references=[e.reference(trace) | {"offset": 0}],
        selection_ordinals=[0, 1],
    )
    assert len(e.hydrate(manifest)["units"]) == 2
    with pytest.raises(ValueError, match="release_blob"):
        e.read_reference(ref | {"git_blob": "wrong"})
    with pytest.raises(ValueError, match="input_hash"):
        e.read_reference(ref | {"sha256": "wrong"})
    trace.write_text(json.dumps(dict(qid="wrong", pred="answer", em=1)) + "\n")
    manifest["trace_references"] = [e.reference(trace) | {"offset": 0}]
    with pytest.raises(ValueError, match="source_identity"):
        e.hydrate(manifest)


def test_main_publication_and_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8375: actual publication exercises CLI orchestration and real children."""
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setenv("TMPDIR", str(private))
    prior = {name: os.environ.get(name) for name in ("COVERAGE_RCFILE", "COVERAGE_FILE")}
    frozen = runner.plan(private)
    assert all(c["deadline_s"] <= 900 for c in frozen["commands"])
    for name, value in prior.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runner, "SCRATCH", private)
    monkeypatch.setattr(
        runner.authority,
        "authority",
        lambda *a: dict(tasks=[dict(id=e.TASK, MODEL_SPECS=[])], gate_check_summary=[]),
    )
    manifest = private / "manifest.json"
    atomic_json(manifest, dict(units=[]))
    assert runner.main(["--manifest", str(manifest), "--inspect-manifest"]) == 0
    monkeypatch.setattr(
        runner,
        "plan",
        lambda *a: dict(
            commands=[
                dict(
                    name="private_child",
                    argv=[sys.executable, "-c", "print('private check')"],
                    deadline_s=20,
                    scope="owned",
                )
            ]
        ),
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    assert e.replay(output)
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--deliberate-error"]) == 1
    monkeypatch.setattr(runner, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned")))
    assert runner.main(["--output", str(output)]) == 1
