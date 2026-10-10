"""REQ-REPORT-8373 / REQ-VERIFY-8373: declared source aliases retain frozen custody."""

import json

from carnot.reporting import v721_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def test_source_alias_reads_and_writes_use_private_custody(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8373-REPLAY: later source changes cannot replace sealed input bytes."""
    root = tmp_path / "repository"
    source = root / "ops/exclusion_manifest.yaml"
    snapshot = tmp_path / "upstream/inputs/manifest.yaml"
    source.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    sealed = b"retired: []\n"
    snapshot.write_bytes(sealed)
    current = b"retired:\n- experiment_id: later\n"
    source.write_bytes(current)
    ref = dict(e.freeze(snapshot, tmp_path / "custody"), source_path=str(source))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(e, "ROOT", root)
    output = root / "results/reconstruction.json"
    with e.frozen_inputs([ref], scratch):
        assert source.read_bytes() == sealed
        source.write_bytes(b"private reconstruction")
        assert source.read_bytes() == sealed
        output.parent.mkdir(parents=True)
        atomic_json(output, dict(mechanical_control=True))
        assert json.loads(output.read_bytes()) == dict(mechanical_control=True)
    assert source.read_bytes() == current
    assert snapshot.read_bytes() == sealed
    assert not output.exists()
    assert json.loads((scratch / "writes/reconstruction.json").read_bytes()) == dict(
        mechanical_control=True
    )


def test_natural_capture_preserves_declared_source_aliases(tmp_path):
    """SCENARIO-VERIFY-8373-REPLAY: authentic utility replay uses its own sealed manifest."""
    task = e.contract.authority(e.ROOT, tmp_path / "authority")["tasks"][1]
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "primary")
    summary = e.outcome(task, primary, tmp_path / "capture")
    alias = str(e.ROOT / "ops/exclusion_manifest.yaml")
    refs = [r for r in summary["closure"] if r.get("source_path") == alias]
    assert refs
    assert all(r["sha256"] == r["expected_sha256"] for r in refs)
    assert summary["branch_replay"]["passed"]
