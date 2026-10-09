"""REQ-REPORT-8356 / REQ-VERIFY-8356: evidence controls stay in private scratch."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import kv260_workload_cost_8356 as e
from carnot.reporting import kv260_workload_runner_8356 as r
from carnot.reporting import kv260_table_cost_8356 as t
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def current_root(root: Path) -> Path:
    """Copy current authority without writing the live roadmap."""
    for name in [e.authority_module.DESIGN, e.authority_module.ACTIVE, e.authority_module.PROTOCOL]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    return root


def test_historical_and_current_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-AUTHORITY: versioned bytes retain unchanged assertions."""
    root = e.historical_root(tmp_path / "old")
    assert e.prior.authority_module.authority(root, tmp_path / "old-check")["activated"]
    root = current_root(tmp_path / "current")
    assert e.authority(root, tmp_path / "current-check")["activated"]
    (root / e.authority_module.ACTIVE).write_text("{}")
    with pytest.raises((ValueError, KeyError, TypeError)):
        e.authority(root, tmp_path / "bad")


def test_independent_arithmetic_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-BRANCHES: absent peers cannot suppress ten clocks."""
    work = e.measure(current_root(tmp_path / "root"), tmp_path / "raw")
    assert len(work["timing_rows"]) == 10
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["arithmetic_cost_ready_score"] == 1
    assert value["durable_cost_ready_score"] == value["kv260_execution_ready_score"] == 0
    assert value["compatible_fraction"] is value["amdahl_upper_bound"] is None
    assert value["verdict_class"] == "blocked" and value["independent_count"] == 0
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    value["arithmetic_cost_ready_score"] = 0
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "missing")
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"


def test_owned_numeric_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-8356: an owned arithmetic failure never becomes missingness."""
    with patch.object(e.p, "audit", return_value=dict(passed=False)):
        work = e.measure(current_root(tmp_path / "root"), tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 256


def test_incomplete_arithmetic_pairs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-REPETITIONS: ten rows need all five paired IDs."""
    work = e.measure(current_root(tmp_path / "root"), tmp_path / "raw")
    row = next(r for r in work["timing_rows"] if r["arm"] == "dense" and r["repetition"] == 0)
    row["repetition"] = 4
    assert len(work["timing_rows"]) == 10
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["arithmetic_cost_ready_score"] == 0
    assert value["arithmetic_pair_parity"] is None
    assert not any(r["completed"] for r in value["arithmetic_rows"])


def test_table_operation_semantics() -> None:
    """SCENARIO-VERIFY-8356-TAMPER: exact refresh and lookup must survive replay."""
    head = dict(coefficients=[1.0, 0.0] + [0.2] * 32, temperature=1.0)
    row = t.transaction(head, 65, "float64", "linear", 0)
    t.verify(row)
    assert set(row["operation_ns"]) == {"table_evaluation", "coefficient_writes", "table_refresh"}
    changed = deepcopy(row)
    changed["probabilities"][0] += 0.1
    with pytest.raises(ValueError):
        t.verify(changed)
    changed = deepcopy(row)
    changed["grid_points"] = 66
    with pytest.raises(ValueError, match="configuration"):
        t.verify(changed)
    changed = deepcopy(row)
    changed["operation_ns"]["table_refresh"] = -1
    with pytest.raises(ValueError):
        t.verify(changed)


@pytest.mark.parametrize(
    "changes",
    [dict(repetition=-2), dict(repetition=True), dict(arm="foreign"), dict(branch="arithmetic")],
)
def test_table_clock_identity(changes: dict) -> None:
    """SCENARIO-VERIFY-8356-TAMPER: valid values cannot authorize foreign clock IDs."""
    head = dict(coefficients=[1.0, 0.0] + [0.2] * 32, temperature=1.0)
    row = t.transaction(head, 65, "float64", "linear", 0)
    row.update(changes)
    with pytest.raises(ValueError, match="configuration"):
        t.verify(row)


def test_table_repetition_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-REPETITIONS: rehashing cannot fill missing table repeats."""
    raw = tmp_path / "raw"
    work = e.measure(current_root(tmp_path / "root"), raw)
    head = dict(coefficients=[1.0, 0.0] + [0.2] * 32, temperature=1.0)
    work["table_rows"] = [
        t.transaction(head, size, storage, interpolation, repetition)
        for size, storage, interpolation in t.n.CONFIGS
        for repetition in range(5)
    ]
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, [dict(passed=True)])
    assert value["table_cost_ready_score"] == 1
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for replacement in [
        deepcopy(work["table_rows"][0]),
        dict(work["table_rows"][-1], repetition=-1),
    ]:
        changed = deepcopy(work)
        changed["table_rows"][-1] = replacement
        assert len(changed["table_rows"]) == 60
        atomic_json(raw / "measurement.json", changed)
        value = e.build(changed, raw, [dict(passed=True)])
        assert value["table_cost_ready_score"] == 0
        assert all(r["censored"] for r in value["rows"] if r["branch"] == "table")
        atomic_json(candidate, value)
        assert not e.replay(candidate)


def test_actual_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8356-CLI: run the real child and private publication paths."""
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--root", str(tmp_path / "absent"), "--private-output", str(output)]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert r.main(["--root", str(tmp_path), "--worker-output", str(tmp_path / "worker.json")]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "20000101"])
    child = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0


def test_real_optional_evidence_and_failures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-BRANCHES: real tables qualify without service promotion."""
    work = e.measure(e.ROOT, tmp_path / "real")
    assert len(work["table_rows"]) == 60
    assert len(work["imported_cost_traces"]) == 3
    value = e.build(work, tmp_path / "real", [dict(passed=True)])
    assert value["table_cost_ready_score"] == 1 and value["durable_cost_ready_score"] == 0
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(value)
    changed["table_timing_rows"][0]["probabilities"][0] += 0.1
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    original_probe = e.base.probe

    def foreign(location: Path, raw: Path, data: dict, field: str | None) -> dict | None:
        found = original_probe(location, raw, data, field)
        if found is not None:
            found["milestone"] = "2026.10.719"
        return found

    with (
        patch.object(e.base, "probe", foreign),
        patch.object(
            t, "authenticate", return_value=dict(task_id=t.producer.TASK, milestone="old")
        ),
    ):
        isolated = dict(
            checks=[], refs=[], branches={}, historical_model_provenance=[], owned_failure=False
        )
        e.optional(e.ROOT, tmp_path / "foreign", isolated)
    assert isolated["table_rows"] == [] and not isolated["owned_failure"]
    with patch.object(t, "transaction", side_effect=ValueError("owned_table_benchmark")):
        broken = dict(checks=[], refs=[], owned_failure=False)
        t.optional(e.ROOT, tmp_path / "real", broken)
    assert broken["owned_failure"] and broken["table_rows"] == []
    with patch.object(t.producer, "replay", return_value=False):
        rejected = dict(checks=[], refs=[], owned_failure=False)
        t.optional(e.ROOT, tmp_path / "real", rejected)
    assert not rejected["owned_failure"]
    source = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
    source["table_rows"][0]["probabilities"][0] += 0.1
    atomic_json(Path(value["measurement_reference"]["path"]), source)
    changed = deepcopy(value)
    ref = e.reference(Path(value["measurement_reference"]["path"]))
    changed["measurement_reference"] = ref
    changed["raw_shard_hashes"][0] = ref
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    empty = tmp_path / "empty-configurations.json"
    atomic_json(
        empty,
        dict(
            inputs=dict(head=dict(coefficients=[1.0, 0.0] + [0.2] * 32, temperature=1.0), refs=[]),
            code_refs=[],
            primitive_refs=[],
            configurations=[],
        ),
    )
    with (
        patch.object(
            t,
            "authenticate",
            return_value=dict(
                task_id=t.producer.TASK, milestone=e.MILESTONE, work_reference=e.reference(empty)
            ),
        ),
        patch.object(t.producer, "replay", return_value=True),
    ):
        t.optional(e.ROOT, tmp_path / "empty", dict(checks=[], refs=[], owned_failure=False))
    with patch.object(
        t, "read_bound_sidecar", return_value=dict(primary_path="wrong", report=dict(passed=True))
    ):
        assert (
            t.authenticate(
                e.ROOT / "results" / (t.producer.NAME + ".json"),
                tmp_path / "bad-terminal",
                dict(checks=[], refs=[]),
            )
            is None
        )


def test_historical_authentication_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8356-AUTHORITY: stale primary and terminal failures reject."""
    with patch.object(e, "sha256_file", return_value="wrong"):
        with pytest.raises(ValueError, match="historical_primary_hash"):
            e.historical_root(tmp_path / "wrong")
    with patch.object(e, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        with pytest.raises(ValueError, match="historical_terminal"):
            e.historical_root(tmp_path / "terminal")


def test_manifest_and_production_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8356-CLI: frozen child failure remains disqualified."""
    assert len(r.manifest(tmp_path / "manifest")) == 9
    failure = dict(
        name="deliberate_owned_error",
        argv=[sys.executable, "-c", "raise SystemExit(3)"],
        expected=0,
        deadline=10,
        scope="owned",
    )
    output = tmp_path / "production" / (e.NAME + ".json")
    with patch.object(r, "manifest", return_value=[failure]):
        assert r.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
