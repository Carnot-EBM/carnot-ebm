"""REQ-REPORT-8633 / REQ-VERIFY-8633: checked publication and immutable replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import installed_lean_8633 as e
from carnot.reporting import installed_runner_8633 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.mark.parametrize("candidate", ["valid", "missing", "rehashed"])
def test_standalone_replay_outside_checkout(tmp_path, candidate):
    """SCENARIO-VERIFY-8633-STANDALONE-CLI: resolve readers without PYTHONPATH."""
    authority_path = tmp_path / "authority.json"
    atomic_json(
        authority_path,
        dict(
            raw=str(tmp_path),
            run_date="20261018",
            bundle={},
            evidence={},
            gate_check_summary=[dict(check="private_external_input", observed=None)],
            preconditions_checked={},
            duration_s=0,
            phase_spans=[],
            source_artifact_hashes=[],
            validation_receipts=[],
        ),
    )
    authority = e.reference(authority_path)
    value = runner.artifact(authority, [])
    assert value["verdict_class"] == "blocked"
    path = tmp_path / "candidate.json"
    if candidate == "rehashed":
        value["completed_count"] += 1
        value["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        )
    if candidate != "missing":
        atomic_json(path, value)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(path),
            "--authority",
            authority["path"],
            authority["sha256"],
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if candidate == "valid" else 1), result.stderr
    assert "Traceback" not in result.stderr
    assert "no_model_load" in result.stdout
    if candidate == "missing":
        assert str(path) in result.stderr
    elif candidate == "rehashed":
        assert "immutable_reduction_drift" in result.stderr


def test_artifact_and_anchored_replay(tmp_path):
    """SCENARIO-REPORT-8633-INSTALL: rehashing a changed claim cannot create evidence."""
    bundle = e.inputs(e.ROOT)
    rows = [
        dict(
            unit_id=r["unit_id"],
            arm=arm,
            response=bundle["references"][i],
            baseline_response=bundle["references"][i],
            status="ok",
            parity=True,
        )
        for arm in ["python", "native"]
        for i, r in enumerate(bundle["requests"])
    ]
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"fixture wheel, mechanics only")
    evidence = dict(
        prototype_parity_rows=rows,
        import_trace_rows=[dict(clean=True, clean_shutdown=True)],
        export_contract_rows=[dict(passed=True)],
        wheel_reference=e.reference(wheel),
        installed_entry_command=["python", "-m", "carnot.inference.resident_protocol"],
    )
    work = dict(
        raw=str(tmp_path),
        run_date="20261018",
        bundle=bundle,
        evidence=evidence,
        gate_check_summary=[],
        preconditions_checked={"cpu": True},
        duration_s=1,
        phase_spans=[],
        source_artifact_hashes=bundle["refs"],
    )
    authority = tmp_path / "authority.json"
    atomic_json(authority, work)
    receipts = [dict(name=name, passed=True) for name in runner.REQUIRED]
    value = runner.artifact(e.reference(authority), receipts)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    runner.replay(path, e.reference(authority))
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(path),
            "--authority",
            str(authority),
            e.reference(authority)["sha256"],
        ],
        cwd=e.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    work["raw_shard_hashes"] = [e.reference(wheel)]
    work["validation_receipts"] = receipts
    atomic_json(authority, work)
    value = runner.artifact(e.reference(authority), receipts)
    atomic_json(path, value)
    wheel.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="replay_input_hash"):
        runner.replay(path, e.reference(authority))
    wheel.write_bytes(b"fixture wheel, mechanics only")
    bad_receipts = deepcopy(value)
    bad_receipts["validation_receipts"] = []
    atomic_json(path, bad_receipts)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        runner.replay(path, e.reference(authority))
    for field in [
        "completed_count",
        "installed_prototype_ready_score",
        "rows",
        "required_checks_passed",
    ]:
        bad = deepcopy(value)
        bad[field] = [] if field == "rows" else 0
        bad["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in bad.items() if k != "reproducibility_checksum"}
        )
        atomic_json(path, bad)
        with pytest.raises(ValueError):
            runner.replay(path, e.reference(authority))
    rejected = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(path),
            "--authority",
            str(authority),
            e.reference(authority)["sha256"],
        ],
        cwd=e.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode == 1 and "immutable_reduction_drift" in rejected.stderr
    failed = runner.artifact(e.reference(authority), [dict(name="owned", passed=False)])
    assert failed["verdict_class"] == "disqualified"
    assert failed["installed_prototype_ready_score"] == 0
    work["gate_check_summary"] = [dict(check="input", observed=None)]
    work["evidence"] = {}
    atomic_json(authority, work)
    assert runner.artifact(e.reference(authority), receipts)["verdict_class"] == "blocked"
    with pytest.raises(OSError):
        runner.replay(tmp_path / "missing", e.reference(authority))


def test_main_boundaries(tmp_path, monkeypatch):
    """REQ-VERIFY-8633: real CLI branches reject missing and malformed inputs."""
    assert runner.main(["--date", "bad"]) == 1
    assert (
        runner.main(
            [
                "--cold-replay",
                str(tmp_path / "missing"),
                "--authority",
                str(tmp_path / "missing"),
                "x",
            ]
        )
        == 1
    )
    assert runner.main(["--input-root", str(tmp_path)]) == 1
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    assert runner.main(["--date", "20261018", "--private-output", str(tmp_path / "out.json")]) == 0


def test_validation_missing_reports_fail(tmp_path, monkeypatch):
    """REQ-VERIFY-8633: absent child identities or coverage cannot become readiness."""
    from carnot.reporting import installed_validation_8633 as validation

    def fake(name, argv, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / (name + ".stdout")
        out.write_text("")
        return dict(name=name, passed=False, stdout_path=str(out))

    monkeypatch.setattr(validation, "child", fake)
    receipts = validation.validate(tmp_path)
    assert not next(r for r in receipts if r["name"] == "owned_coverage")["passed"]
    assert not next(r for r in receipts if r["name"] == "exact_node_union")["passed"]


def test_validation_identities_do_not_replace_coverage(tmp_path, monkeypatch):
    """REQ-VERIFY-8633: a complete node union still needs executed statements."""
    import xml.etree.ElementTree as ET
    from carnot.reporting import installed_validation_8633 as validation

    nodes = [paths[0] + "::test_original" for _, paths in validation.GROUPS]

    def fake(name, argv, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / (name + ".stdout")
        out.write_text("\n".join(nodes) if name == "collect" else "")
        for index, (group, _) in enumerate(validation.GROUPS):
            if name == group:
                node = nodes[index]
                suite = ET.Element("testsuite")
                ET.SubElement(
                    suite,
                    "testcase",
                    classname=node.split("::")[0][:-3].replace("/", "."),
                    name="test_original",
                )
                ET.ElementTree(suite).write(folder / (name + ".xml"))
        if name == "coverage_report":
            (folder / "coverage.json").write_text(json.dumps(dict(files={})))
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(validation, "child", fake)
    receipts = validation.validate(tmp_path)
    assert next(r for r in receipts if r["name"] == "exact_node_union")["passed"]
    assert not next(r for r in receipts if r["name"] == "owned_coverage")["passed"]


def test_publication_and_external_block(tmp_path, monkeypatch):
    """REQ-VERIFY-8633: final bytes and real readers agree; missing inputs stay blocked."""
    from carnot.reporting import installed_validation_8633 as validation

    bundle = e.inputs(e.ROOT)
    rows = [
        dict(
            unit_id=r["unit_id"],
            arm=arm,
            response=bundle["references"][i],
            baseline_response=bundle["references"][i],
            status="ok",
            parity=True,
        )
        for arm in ["python", "native"]
        for i, r in enumerate(bundle["requests"])
    ]
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"private fixture")
    evidence = dict(
        prototype_parity_rows=rows,
        import_trace_rows=[dict(clean=True, clean_shutdown=True)],
        export_contract_rows=[dict(passed=True)],
        wheel_reference=e.reference(wheel),
        installed_entry_command=["python", "-m", "carnot.inference.resident_protocol"],
    )
    monkeypatch.setattr(e, "measure", lambda *args: evidence)

    def validated(raw):
        folder = raw / "validation"
        folder.mkdir()
        (folder / ".coverage.fixture").write_bytes(b"private coverage fixture")
        return [dict(name=n, passed=True) for n in runner.REQUIRED]

    monkeypatch.setattr(validation, "validate", validated)

    def fake(name, argv, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / (name + ".stdout")
        out.write_text("{}")
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(runner, "child", fake)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.run(output, "20261018", e.ROOT) == 0
    assert json.loads(output.read_bytes())["installed_prototype_ready_score"] == 1
    value = json.loads(output.read_bytes())
    assert any(Path(r["path"]).name == ".coverage.fixture" for r in value["raw_shard_hashes"])
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert runner.run(blocked, "20261018", tmp_path / "missing") == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"

    def failed_measure(*args):
        raise ValueError("owned execution failed")

    monkeypatch.setattr(e, "measure", failed_measure)
    failed = tmp_path / "failed" / (e.NAME + ".json")
    assert runner.run(failed, "20261018", e.ROOT) == 1
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    from types import SimpleNamespace

    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    resource = tmp_path / "resource" / (e.NAME + ".json")
    assert runner.run(resource, "20261018", e.ROOT) == 0
