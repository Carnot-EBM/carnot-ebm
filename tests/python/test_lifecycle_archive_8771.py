"""REQ-REPORT-8771, REQ-VERIFY-8771: preserve authority while replaying real arithmetic."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import lifecycle_archive_8771 as api
from carnot.reporting import lifecycle_archive_authority_8771 as authority
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def publication(tmp_path_factory):
    """SCENARIO-VERIFY-8771-CUSTODY: use an actual CLI with private output."""
    folder = tmp_path_factory.mktemp("archive8771")
    output = folder / (api.e.NAME + ".json")
    receipt = child(
        "private",
        [sys.executable, api.e.CLI, "--private-control", "--output", str(output)],
        folder / "logs",
        deadline=480,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_archived_authority_and_original_failure(publication):
    """REQ-REPORT-8771: original failure and historical assertion both remain visible."""
    value = json.loads(publication.read_bytes())
    work = api.e.read(value["primitive_reference"])
    rows = value["historical_authority_rows"]
    assert all(row["passed"] for row in rows)
    assert work["historical_failure_reproduction"]["actual_exit"] == 1
    assert rows[0]["canonical_tasks_sha256"] == authority.contract.PIN
    assert (
        authority.contract.PIN == "d0fb6404987d9825a83e7f3b910e1d9a5eaa6a9caabee9aef693faa3050e7c83"
    )
    assert value["experiment_id"] == 8771 and value["milestone"] == "2026.11.751"
    assert value["natural_update_denominator"] == 4096
    assert value["natural_dense_event_count"] == 0
    assert value["ordinary_window_ready_score"] == value["threshold_fixture_ready_score"] == 1
    assert len(value["rows"]) == 24 and len(value["restart_rows"]) == 3
    assert all(row["passed"] for row in value["rows"])
    assert (
        work["native_runtime_reference"]["path"] == authority.compiled.numerical.DEFAULT_EXTENSION
    )


@pytest.mark.parametrize("field", ["prompt", "title", "gated_on"])
def test_immutable_task_mutations(tmp_path, field):
    """REQ-VERIFY-8771: coordinated task edits cannot replace the original fixed digest."""
    root, manifest = authority.fixture(tmp_path / "archive")
    assert (root / "results").is_dir() and not (root / "results").is_symlink()
    value = yaml.safe_load((root / authority.contract.ACTIVE).read_bytes())
    value["tasks"][0][field] = "changed"
    for name in [authority.contract.ACTIVE, authority.contract.STAGED]:
        path = root / name
        path.chmod(0o600)
        path.write_text(yaml.safe_dump(value))
    assert not authority.contract.authority(root)["ready"]
    with pytest.raises(ValueError, match="changed_bytes"):
        authority.verify_fixture(root, manifest["sha256"])


def test_source_artifact_and_manifest_tamper(tmp_path):
    """SCENARIO-VERIFY-8771-CUSTODY: changed source and artifact bytes fail independently."""
    root, manifest = authority.fixture(tmp_path / "archive")
    for name in ["source.py", "artifact.json"]:
        path = tmp_path / name
        path.write_text("original")
        ref = api.e.reference(path)
        path.write_text("changed")
        with pytest.raises(ValueError, match="changed_bytes"):
            api.e.verify(ref)
    path = root / "authority_manifest.json"
    path.chmod(0o600)
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed_bytes"):
        authority.verify_fixture(root, manifest["sha256"])


def test_replay_and_owned_failure(publication, tmp_path):
    """SCENARIO-VERIFY-8771-CUSTODY: rehashed summaries and failed checks cannot release work."""
    api.runtime.replay(publication)
    value = json.loads(publication.read_bytes())
    changed = tmp_path / publication.name
    api.e.atomic_json(changed, dict(value, ordinary_window_ready_score=0))
    with pytest.raises(ValueError, match="report_drift"):
        api.runtime.replay(changed)
    work = api.e.read(value["primitive_reference"])
    bad = api.build(work, [dict(name="owned", passed=False)], publication)
    assert bad["verdict_class"] == "disqualified"
    assert bad["ordinary_window_ready_score"] == bad["threshold_fixture_ready_score"] == 0
    work = deepcopy(work)
    work["historical_authority_rows"][0]["passed"] = False
    assert api.build(work, [], publication)["ordinary_window_ready_score"] == 0


def test_cli_guards_and_historical_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-8771: invalid dates and checkout private destinations reject."""
    assert api.main(["--date", "bad"]) == 1
    assert api.main(["--private-control"]) == 1
    assert api.main(["--worker", str(tmp_path / "missing")]) == 1
    root, manifest = authority.fixture(tmp_path / "archive")
    authority.verify_fixture(root, manifest["sha256"])
    receipt = child(
        "historical_help",
        [sys.executable, str(root / authority.HISTORICAL_CLI), "--help"],
        tmp_path / "help",
        deadline=60,
    )
    assert receipt["passed"]
    with monkeypatch.context() as m:
        m.delenv("CARNOT_8771_ARCHIVE", raising=False)
        authority.pytest_configure(None)
        m.setenv("CARNOT_8771_ARCHIVE", str(root))
        m.setenv("CARNOT_8771_ARCHIVE_SHA256", manifest["sha256"])
        m.setattr(authority.compiled, "ROOT", authority.compiled.ROOT)
        m.setattr(authority.compiled, "CLI", authority.compiled.CLI)
        authority.pytest_configure(None)
        assert authority.compiled.ROOT == root
        assert not authority.compiled.inputs()["failures"]
    with monkeypatch.context() as m:
        m.setattr(authority.runner, "main", lambda: 19)
        m.setattr(authority.compiled, "ROOT", authority.compiled.ROOT)
        m.setattr(authority.compiled, "CLI", authority.compiled.CLI)
        assert authority.historical_main(root, manifest["sha256"]) == 19


def test_fixed_digest_and_manifest_controls(tmp_path, monkeypatch):
    """REQ-VERIFY-8771: a repaired outer hash cannot replace the fixed V747 digest."""
    with monkeypatch.context() as m:
        m.setattr(authority, "canonical_hash", lambda value: "changed")
        with pytest.raises(ValueError, match="original_v747_digest"):
            authority.fixture(tmp_path / "bad_digest")
    root, manifest = authority.fixture(tmp_path / "archive")
    path = Path(manifest["path"])
    value = api.e.read(manifest)
    path.chmod(0o600)
    api.e.atomic_json(path, dict(value, canonical_tasks_sha256="changed"))
    with pytest.raises(ValueError, match="immutable_v747_authority"):
        authority.verify_fixture(root, api.e.reference(path)["sha256"])
    with monkeypatch.context() as m:
        m.setattr(authority, "child", lambda *a, **k: dict(passed=False))
        folder = tmp_path / "reproduction"
        folder.mkdir()
        (folder / "reproduced.xml").write_text("<testsuite/>")
        with pytest.raises(ValueError, match="historical_failure_not_reproduced"):
            authority.reproduce(folder)


def test_validation_adapter_and_complete_union(publication, tmp_path, monkeypatch):
    """REQ-VERIFY-8771: bind only historical children and require their exact node outcomes."""
    value = json.loads(publication.read_bytes())
    work = api.e.read(value["primitive_reference"])
    api.e.atomic_json(tmp_path / "measurement.json", work)

    def validate(raw):
        return [
            api.runtime.checks.child(
                name, [sys.executable, "-c", "pass", argument], raw / "logs", deadline=20
            )
            for name, argument in [
                ("historical", work["original_compiled_nodes"][0]),
                ("current", "current.py"),
            ]
        ]

    with monkeypatch.context() as m:
        m.setattr(api, "base_validate", validate)
        assert all(r["passed"] for r in api.validate(tmp_path))
    work["private_control"] = False
    receipt = dict(
        name="original_disjoint_union",
        passed=True,
        executed_node_ids=work["original_compiled_nodes"],
    )
    assert api.build(work, [receipt], publication)["ordinary_window_ready_score"] == 1
    receipt["executed_node_ids"] = []
    assert api.build(work, [receipt], publication)["ordinary_window_ready_score"] == 0


def test_current_native_identity(monkeypatch):
    """REQ-REPORT-8771: the actual opt-in binary must match the retained numerical witness."""
    admitted = api.inputs()
    assert not admitted["failures"]
    assert admitted["extension"]["path"] == authority.compiled.numerical.DEFAULT_EXTENSION
    with monkeypatch.context() as m:
        m.setattr(api, "base_inputs", lambda: dict(failures=[dict(passed=False)], refs=[]))
        assert api.inputs()["failures"]
    with monkeypatch.context() as m:
        m.setattr(api.e.resident.original, "DEFAULT_EXTENSION", "/missing/native.so")
        assert api.inputs()["failures"][0]["field"] == "current_native_binary"
