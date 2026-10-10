"""REQ-VERIFY-8368 / REQ-REPORT-8368: typed paths retain actual byte custody."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import typed_runtime_8368 as t


def test_documented_references_and_prose(tmp_path):
    """SCENARIO-VERIFY-8368-TYPED: annotation words cannot become file paths."""
    p = tmp_path / "operand"
    p.write_text("real bytes")
    ref = dict(path=str(p), sha256=sha256_file(p))
    value = dict(
        field_principles=dict(
            runtime_binding_path="Explain provenance", runtime_binding_sha256="words"
        ),
        arbitrary_path="missing",
        arbitrary_sha256=sha256_file(p),
        nested=dict(primitive_reference=ref),
        source_artifact_hashes=[ref],
    )
    assert t.references(value) == [ref, ref]
    assert t.references([dict(arbitrary=ref), "words", 7]) == []
    assert t.references(
        dict(runtime_binding_path=str(p), runtime_binding_sha256=sha256_file(p))
    ) == [ref]
    assert t.references(dict(primitive_reference=dict(ref, snapshot_path=str(p))))[0][
        "snapshot_path"
    ] == str(p)
    assert t.references(dict(primitive_reference=dict(ref, exists=False))) == [
        dict(ref, exists=False)
    ]


@pytest.mark.parametrize(
    "ref",
    [
        dict(path=8, sha256="sha256:" + "a" * 64),
        dict(path="p", sha256="invalid"),
        dict(path="p"),
        "words",
    ],
)
def test_malformed_required_records(ref):
    """SCENARIO-VERIFY-8368-TYPED: malformed declared operands fail closed."""
    with pytest.raises(ValueError, match="typed_reference"):
        t.references(dict(primitive_reference=ref))


def test_absence_hash_and_terminal(tmp_path):
    """SCENARIO-VERIFY-8368-TYPED: absent bytes differ from rejected terminals."""
    p = tmp_path / "operand"
    with pytest.raises(FileNotFoundError):
        t.authenticate(dict(path=str(p), sha256="sha256:" + "a" * 64))
    p.write_text("present")
    with pytest.raises(ValueError, match="hash"):
        t.authenticate(dict(path=str(p), sha256="sha256:" + "a" * 64))
    assert t.authenticate(dict(path=str(p), sha256=sha256_file(p))) == p
    primary = tmp_path / "results/experiment_8307_control.json"
    atomic_json(primary, dict(terminal_validation_sidecar_path=str(tmp_path / "terminal.json")))
    side = (
        primary.parent / "raw" / primary.stem / "validators" / (sha256_file(primary)[7:] + ".json")
    )
    atomic_json(side, dict(primary_sha256=sha256_file(primary), report=dict(passed=False)))
    with pytest.raises(ValueError, match="terminal_rejected"):
        t.terminal(primary)
    atomic_json(side, dict(primary_sha256=sha256_file(primary), report=dict(passed=True)))
    atomic_json(
        tmp_path / "terminal.json",
        dict(publication=dict(primary_sha256=sha256_file(primary)), normal_process_exit=True),
    )
    assert t.terminal(primary) == [side, tmp_path / "terminal.json"]
    atomic_json(
        tmp_path / "terminal.json",
        dict(publication=dict(primary_sha256="wrong"), normal_process_exit=False),
    )
    with pytest.raises(ValueError, match="terminal_binding"):
        t.terminal(primary)


def test_historical_authority_and_closure(tmp_path):
    """SCENARIO-VERIFY-8368-TYPED: historical sources use their original authority."""
    from carnot.verify import runtime_reader_8353 as old

    auth = t.historical_authority(tmp_path / "authority")
    assert auth["activated"]
    refs = t.closure(old.ROOT, tmp_path / "raw", [old.UPSTREAM])
    assert refs and all(Path(r["snapshot_path"]).is_file() for r in refs)
    p = tmp_path / "absent-root" / old.UPSTREAM
    atomic_json(p, dict(primitive_reference=dict(path=str(tmp_path / "missing"), sha256="invalid")))
    with pytest.raises(FileNotFoundError):
        t.closure(p.parents[1], tmp_path / "bad", [old.UPSTREAM])
    with patch.object(t, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        with pytest.raises(ValueError, match="terminal_rejected"):
            t.closure(old.ROOT, tmp_path / "rejected", [old.UPSTREAM])
    assert t.closure(tmp_path / "none", tmp_path / "empty", [old.UPSTREAM]) == []


def test_metadata_cannot_hide_a_required_operand(tmp_path):
    """SCENARIO-VERIFY-8368-TYPED: non-prose annotations and bad collections are rejected."""
    with pytest.raises(ValueError, match="annotation"):
        t.references(
            dict(
                field_principles=dict(
                    primitive_reference=dict(path="absent", sha256="sha256:" + "a" * 64)
                )
            )
        )
    with pytest.raises(ValueError, match="collection"):
        t.references(dict(source_artifact_hashes="words"))
    with pytest.raises(ValueError, match="path_type"):
        t.authenticate(dict(path=7, sha256="invalid"))
    assert t.references(dict(refs=[dict(path="absent", exists=False, sha256=None)])) == []
    p = tmp_path / "reference"
    p.write_text("valid")
    ref = dict(path=str(p), source_path=str(p), sha256=t.sha256_file(p))
    assert t.references(dict(authority_snapshots=dict(active=ref, staged=dict(exists=False)))) == [
        ref
    ]
    primary = tmp_path / "results/experiment_8307_control.json"
    atomic_json(primary, dict(primitive_reference=dict(path=str(p), sha256=t.sha256_file(p))))
    with (
        patch.object(t, "authenticate", return_value=p),
        patch.object(t, "sha256_file", return_value="sha256:" + "b" * 64),
    ):
        with pytest.raises(ValueError, match="hash"):
            t.closure(tmp_path, tmp_path / "raw", [str(primary.relative_to(tmp_path))])
