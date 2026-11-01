"""REQ-VERIFY-8660: missing or killed children cannot receive passing credit."""

from pathlib import Path

import pytest

from carnot.reporting import installed_validation_8660 as v
from carnot.reporting import installed_report_8660 as r


def test_coverage_requires_every_owned_file():
    """SCENARIO-VERIFY-8660-CUSTODY: absence differs from measured zero misses."""
    assert not v.coverage_result({})["passed"]
    files = {
        str(v.ROOT / p): dict(summary=dict(num_statements=1), missing_lines=[]) for p in r.OWNED
    }
    assert v.coverage_result(files)["passed"]
    files[str(v.ROOT / r.OWNED[0])]["missing_lines"] = [1]
    assert not v.coverage_result(files)["passed"]


def test_empty_collection_is_closed(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8660-CUSTODY: empty collection never expands to all tests."""

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (name + ".stdout")
        path.write_text("")
        return dict(name=name, passed=False, stdout_path=str(path))

    monkeypatch.setattr(v, "child", fake)
    work = dict(original_nodes=[])
    checks = v.validate(work, tmp_path)
    assert not next(x for x in checks if x["name"] == "exact_node_union")["passed"]
    assert not next(x for x in checks if x["name"] == "owned_coverage")["passed"]


@pytest.mark.parametrize("mode", ["pass", "killed", "missing_node"])
def test_disjoint_children_retain_raw_custody(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8660-CUSTODY: authentic raw shards stay separate from combine copies."""
    import json
    from carnot.reporting.current_work_receipt import atomic_json

    historical = json.loads(
        (v.ROOT / "results/experiment_8634_v741_installed_compatibility.json").read_bytes()
    )
    original = historical["original_node_union"]["original_nodes"]
    current = [
        r.TESTS[0] + "::test_current",
        "tests/python/test_primary_publication_7928.py::test_reader",
    ]
    outcomes = {}

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text(
            "\n".join(
                (original[:-1] if mode == "missing_node" else original)
                if name == "collect-historical"
                else current
            )
            if name.startswith("collect-")
            else ""
        )
        if name.startswith(("historical-", "current-")):
            selected = [a for a in argv if a.startswith("tests/python/")]
            xml = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            xml.write_text("<testsuite/>")
            outcomes[str(xml)] = selected
            data = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("COVERAGE_FILE=")))
            data.with_name(".coverage.actual-child").write_bytes(
                b"unit fixture; never claimed as measurements"
            )
        if name == "coverage_json":
            atomic_json(
                logs / "coverage.json",
                dict(
                    files={
                        str(v.ROOT / p): dict(summary=dict(num_statements=1), missing_lines=[])
                        for p in r.OWNED
                    }
                ),
            )
        return dict(
            name=name,
            passed=not (mode == "killed" and name == "historical-0"),
            stdout_path=str(out),
        )

    monkeypatch.setattr(v, "child", fake)
    monkeypatch.setattr(v, "xml_nodes", lambda path: outcomes[str(path)])
    work = dict(original_nodes=original)
    checks = v.validate(work, tmp_path)
    assert all(x["passed"] for x in checks) == (mode == "pass")
    assert {x["historical"] for x in work["child_shard_manifest"]} == {True, False}
    assert work["original_node_union"]["passed"] == (mode == "pass")


def test_sealed_authority_drift_is_rejected(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8660-CUSTODY: changed authority cannot be licensed by its path."""
    manifest = v.c.overlay(tmp_path / "overlay")
    monkeypatch.setattr(v.c, "reference", lambda path: dict(path=str(path), sha256="wrong"))
    import pytest

    with pytest.raises(ValueError, match="historical_authority_hash"):
        v.historical_command(manifest, tmp_path, "tests/python/test_resident_provenance_8593.py")
