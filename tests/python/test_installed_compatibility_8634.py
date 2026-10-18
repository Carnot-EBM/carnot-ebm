"""REQ-REPORT-8634 / REQ-VERIFY-8634: independent wheel qualification."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import installed_compatibility_8634 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """SCENARIO-VERIFY-8634-INSTALLED: run actual installed public and worker children."""
    return e.measure(tmp_path_factory.mktemp("installed8634"), "20261018", e.ROOT)


def test_actual_wheel_and_public_contracts(work):
    """REQ-VERIFY-8634: wheel imports and numerical results retain original contracts."""
    assert not work["gate_check_summary"]
    assert e.reduce(work)["completed_count"] >= 128
    assert {r["surface"] for r in work["public_surface_rows"]} == {"python", "cli", "mcp", "http"}
    assert {r["mode"] for r in work["export_compatibility_rows"]} == {"exports", "absent", "errors"}
    assert all(r["passed"] for r in work["public_surface_rows"] + work["export_compatibility_rows"])
    assert work["installed_module_identity"]["outside_checkout"]
    assert work["installed_module_identity"]["pythonpath_unset"]
    assert Path(work["installed_module_identity"]["coverage_bootstrap"]["path"]).is_file()
    assert all(r["passed"] for r in work["rejection_rows"])
    assert work["historical_8609"]["lean_measurement_ready_score"] == 0
    assert work["historical_8609"]["verdict_class"] == "blocked"


@pytest.mark.parametrize("field", ["parity", "public", "export", "identity", "rejection"])
def test_reducer_rejects_primitive_drift(work, field):
    """REQ-REPORT-8634: success aggregates cannot hide a failed primitive."""
    changed = deepcopy(work)
    if field == "parity":
        changed["installed_parity_rows"][0]["response"] = {}
    elif field == "identity":
        changed["installed_module_identity"]["outside_checkout"] = False
    else:
        key = {
            "public": "public_surface_rows",
            "export": "export_compatibility_rows",
            "rejection": "rejection_rows",
        }[field]
        changed[key][0]["passed"] = False
    with pytest.raises(ValueError):
        e.reduce(changed)


def test_missing_inputs_and_hash_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8634: missing external evidence differs from zero and owned failures."""
    blocked = e.measure(tmp_path / "raw", "20261018", tmp_path)
    assert blocked["gate_check_summary"][0]["observed"] == "missing"
    assert not blocked.get("owned_execution_failed", False)
    with pytest.raises(ValueError, match="input_hash"):
        e.load(dict(path=str(e.ROOT / e.SOURCE), sha256="wrong"))
    original = e.load
    monkeypatch.setattr(
        e, "load", lambda ref: dict(original(ref), installed_prototype_ready_score=0)
    )
    with pytest.raises(ValueError, match="prototype_not_ready"):
        e.inputs(e.ROOT)


def test_artifact_and_cold_replay(work, tmp_path):
    """SCENARIO-REPORT-8634-REPLAY: external authority rejects rehashed claim edits."""
    from carnot.reporting import installed_runner_8634 as r

    authority_path = tmp_path / "authority.json"
    receipts = [dict(name=n, passed=True) for n in r.REQUIRED]
    frozen = dict(
        work, validation_receipts=receipts, original_node_union={}, child_shard_manifest=[]
    )
    atomic_json(authority_path, frozen)
    authority = e.reference(authority_path)
    value = r.artifact(authority)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    r.replay(output, authority)
    assert value["installed_measurement_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    changed = dict(value, completed_count=value["completed_count"] + 1)
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="immutable_reduction_drift"):
        r.replay(output, authority)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    atomic_json(output, value)
    for candidate, code in [(output, 0), (tmp_path / "missing", 1)]:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(candidate),
                "--authority",
                authority["path"],
                authority["sha256"],
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert process.returncode == code, process.stderr
    for receipts, gates, owned, expected in [
        ([], [], False, "disqualified"),
        ([], [dict(check="external")], False, "blocked"),
        ([dict(name="bad", passed=False)], [], False, "disqualified"),
        ([], [], True, "disqualified"),
    ]:
        atomic_json(
            authority_path,
            dict(
                frozen,
                validation_receipts=receipts,
                gate_check_summary=gates,
                owned_execution_failed=owned,
            ),
        )
        assert r.artifact(e.reference(authority_path))["verdict_class"] == expected


def test_cli_boundaries(tmp_path, monkeypatch):
    """REQ-VERIFY-8634: private output and date boundaries fail closed."""
    from carnot.reporting import installed_runner_8634 as r

    for args in [
        ["--date", "bad"],
        ["--date", "2026101"],
        ["--input-root", str(tmp_path)],
        ["--private-output", str(e.ROOT / "forbidden.json")],
        ["--cold-replay", "missing"],
    ]:
        assert r.main(args) == 1
    monkeypatch.setattr(r, "run", lambda *a: 0)
    assert r.main(["--date", "20261018", "--private-output", str(tmp_path / "private.json")]) == 0


def test_private_publication_and_reader(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8634-REPLAY: actual private publication runs cold and hostile readers."""
    from carnot.reporting import installed_runner_8634 as r
    from carnot.reporting import installed_validation_8634 as v

    def measured(raw, *args):
        raw.mkdir(parents=True)
        return dict(deepcopy(work), raw=str(raw))

    def validated(value, raw):
        value.update(original_node_union={}, child_shard_manifest=[])
        return [dict(name=n, passed=True) for n in r.REQUIRED]

    monkeypatch.setattr(e, "measure", measured)
    monkeypatch.setattr(v, "validate", validated)
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, "20261018", e.ROOT) == 0
    value = json.loads(output.read_bytes())
    assert value["installed_measurement_ready_score"] == 1
    report = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert report["report"]["passed"]
    assert report["report"]["actual_findings"]["flagged_count"] == 0
    raw = Path(value["authority_reference"]["path"]).parent
    assert json.loads((raw / "reader_receipt.json").read_bytes())["passed"]
    monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="publication_reader"):
        r.run(tmp_path / "other" / (e.NAME + ".json"), "20261018", e.ROOT)


@pytest.mark.parametrize(
    "kind",
    ["identity_hash", "public_roster", "rejection_roster", "head", "cleanup", "numerical", "count"],
)
def test_all_reduction_rejection_paths(work, tmp_path, kind):
    """REQ-REPORT-8634: immutable identities, arithmetic and complete rosters fail closed."""
    changed = deepcopy(work)
    if kind == "identity_hash":
        changed["installed_module_identity"]["modules"][0]["sha256"] = "wrong"
    elif kind == "public_roster":
        changed["public_surface_rows"] = []
    elif kind == "rejection_roster":
        changed["rejection_rows"].pop()
    elif kind == "cleanup":
        changed["sessions"][0]["installed"]["exit_code"] = 1
    elif kind == "numerical":
        changed["installed_parity_rows"][0]["response"]["probability"] = 0.123
        changed["installed_parity_rows"][0]["baseline_response"] = deepcopy(
            changed["installed_parity_rows"][0]["response"]
        )
    elif kind == "count":
        changed["installed_parity_rows"].append(changed["installed_parity_rows"][-1])
    else:
        config = e.load(changed["sessions"][0]["config_reference"])
        config["head"]["temperature"] = 99
        atomic_json(tmp_path / "bad-head.json", config)
        changed["sessions"][0]["config_reference"] = e.reference(tmp_path / "bad-head.json")
    with pytest.raises(ValueError):
        e.reduce(changed)


def test_input_rejection_branches(tmp_path, monkeypatch):
    """REQ-REPORT-8634: reject wheel substitution and unpublished producer evidence."""
    with monkeypatch.context() as patch:
        patch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
        with pytest.raises(ValueError, match="prototype_publication"):
            e.inputs(e.ROOT)
    original = e.sha256_file
    with monkeypatch.context() as patch:
        patch.setattr(
            e, "sha256_file", lambda path: "bad" if str(path).endswith(".whl") else original(path)
        )
        with pytest.raises(ValueError, match="wheel_hash"):
            e.inputs(e.ROOT)
    archive = e.zipfile.ZipFile

    class WrongWheel:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self, path):
            return b"wrong installed entry"

    monkeypatch.setattr(e.zipfile, "ZipFile", lambda *a: WrongWheel())
    with pytest.raises(ValueError, match="wheel_entry_drift"):
        e.inputs(e.ROOT)
    monkeypatch.setattr(e.zipfile, "ZipFile", archive)
    monkeypatch.setattr(e.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    blocked = e.measure(tmp_path / "missing-disk", "20261018", e.ROOT)
    assert blocked["gate_check_summary"][0]["observed"] == "memory_disk_tools"


def test_owned_failure_deadline_and_replay_hash(work, tmp_path, monkeypatch):
    """REQ-VERIFY-8634: owned execution failures and changed primitive bytes reject readiness."""
    from carnot.reporting import installed_runner_8634 as r

    with monkeypatch.context() as patch:
        patch.setattr(e, "installed", lambda *a: dict(deepcopy(work)))
        patch.setattr(e, "reduce", lambda *a: {})
        ticks = iter([0, 3601_000_000_000, 3602_000_000_000])
        patch.setattr(e.time, "monotonic_ns", lambda: next(ticks))
        failed = e.measure(tmp_path / "deadline", "20261018", e.ROOT)
        assert failed["owned_execution_failed"]
        assert failed["gate_check_summary"][0]["observed"] == "measurement_deadline"
    bad = dict(
        deepcopy(work), validation_receipts=[], original_node_union={}, child_shard_manifest=[]
    )
    bad["raw_shard_hashes"] = [dict(path=str(e.ROOT / e.SOURCE), sha256="bad")]
    authority = tmp_path / "authority.json"
    atomic_json(authority, bad)
    ref = e.reference(authority)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, r.artifact(ref))
    with pytest.raises(ValueError, match="replay_input_hash"):
        r.replay(candidate, ref)


def test_runtime_cap_is_failed_owned_check(tmp_path, monkeypatch):
    """REQ-VERIFY-8634: validation beyond the time reserve fails the owned gate."""
    from carnot.reporting import installed_runner_8634 as r
    from carnot.reporting import installed_validation_8634 as v

    captured = {}
    monkeypatch.setattr(e, "measure", lambda *a: dict(phase_spans=[]))
    monkeypatch.setattr(v, "validate", lambda *a: [])
    ticks = iter([0, 100, 4501_000_000_000])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))

    def reduced(ref):
        captured.update(e.load(ref))
        raise ValueError("stop before publication")

    monkeypatch.setattr(r, "artifact", reduced)
    with pytest.raises(ValueError, match="stop before publication"):
        r.run(tmp_path / (e.NAME + ".json"), "20261018", e.ROOT)
    assert captured["validation_receipts"] == [dict(name="runtime_cap", passed=False)]
