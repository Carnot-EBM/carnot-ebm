"""REQ-REPORT-8437, REQ-VERIFY-8437: conditional bounds need authentic operands."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import acceleration_ceiling_8437 as r
from carnot.reporting import acceleration_ceiling_runner_8437 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import acceleration_ceiling_8437 as k


def test_analytic_bounds():
    """SCENARIO-VERIFY-8437-BOUNDS: service work remains in every denominator."""
    for a in (1, 10, 100, None):
        assert k.speed(10, 0, a) == 1
    assert k.speed(10, 3, 1) == 1
    assert k.speed(10, 10, 10) == 10
    assert k.speed(10, 10, None) == "infinite"
    bounds = [k.speed(10, 3, a) for a in (1, 10, 100, None)]
    assert bounds == sorted(bounds)
    assert bounds[-1] == pytest.approx(10 / 7)
    assert k.speed(10, 4, None) != bounds[-1]
    assert k.speed(20, 3, None) != bounds[-1]
    for t, kernel in [(0, 0), (-1, 0), (10, -1), (10, 11), (float("nan"), 0)]:
        with pytest.raises(ValueError):
            k.speed(t, kernel, 10)
    with pytest.raises(ValueError):
        k.speed(10, 3, 0)


def test_paired_pooled_interval():
    """REQ-VERIFY-8437: all three operands use identical seeded repeat indices."""
    rows = [
        dict(cell=k.PRIMARY, complete_wall_s=t, kernel_s=v, conversion_s=c)
        for t, v, c in [(10, 1, 1), (20, 2, 3), (30, 9, 2), (40, 4, 2), (50, 10, 4)]
    ]
    out = k.analyze(rows)
    indices = np.random.default_rng(7268437).integers(0, 5, (10000, 5))
    t = np.array([10, 20, 30, 40, 50])[indices].sum(axis=1)
    kernel = np.array([1, 2, 9, 4, 10])[indices].sum(axis=1)
    plus = np.array([2, 5, 11, 6, 14])[indices].sum(axis=1)
    for key, operand in [("arithmetic", kernel), ("optimistic_conversion", plus)]:
        for a in (1, 10, 100, None):
            label = "infinity" if a is None else str(a)
            ratios = t / (t - operand + (0 if a is None else operand / a))
            assert out["descriptive_interval"][key][label]["lower_95"] == pytest.approx(
                np.quantile(ratios, 0.025)
            )
            assert out["descriptive_interval"][key][label]["upper_95"] == pytest.approx(
                np.quantile(ratios, 0.975)
            )
    assert out["arithmetic_10x_feasible_under_fixed_service"] is False
    assert out["every_primary_repeat_rules_out_arithmetic_10x"]
    assert out["lower_bound_nonkernel_removal"]["range"] == pytest.approx([0.6, 0.8])
    assert k.analyze([])["descriptive_interval"] is None
    all_kernel = [dict(cell=k.PRIMARY, complete_wall_s=10, kernel_s=10, conversion_s=0)] * 5
    assert (
        k.analyze(all_kernel)["descriptive_interval"]["arithmetic"]["infinity"]["upper_95"]
        == "infinite"
    )


@pytest.fixture(scope="module")
def actual_work(tmp_path_factory):
    """SCENARIO-REPORT-8437-CUSTODY: private disk snapshots preserve producer bytes."""
    private = tmp_path_factory.mktemp("actual8437")
    return r.measure(r.ROOT, private / "raw", private)


def receipts():
    return [dict(name=n, passed=True) for n in r.REQUIRED]


def test_actual_reduction(actual_work, tmp_path):
    """REQ-REPORT-8437: qualified primitive rows establish conditional readiness."""
    assert not actual_work["gate_check_summary"]
    assert len(actual_work["rows"]) == 135
    value = r.build(actual_work, receipts())
    assert value["acceleration_analysis_ready_score"] == 1
    assert value["native_parity_ready_score"] == 0
    assert value["actual_native_speedup"] is None
    assert value["every_primary_repeat_rules_out_arithmetic_10x"]
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)
    value["per_repeat_bounds"][0]["fraction_accelerable"] = 0.99
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(candidate, value)
    assert not r.replay(candidate)
    assert not r.replay(tmp_path / "missing")


def test_real_private_cli(tmp_path):
    """SCENARIO-REPORT-8437-CUSTODY: real children test valid and missing dependencies."""
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for root, expected in [(r.ROOT, "disqualified"), (tmp_path / "missing", "blocked")]:
        output = tmp_path / expected / "results" / (r.NAME + ".json")
        child = subprocess.run(
            [
                *argv,
                "--root",
                str(root),
                "--output",
                str(output),
                "--private-control",
                "--date",
                "20261012",
            ],
            capture_output=True,
            timeout=180,
        )
        assert child.returncode == int(expected == "disqualified"), child.stderr.decode()
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == expected
        assert r.replay(output)
        assert all(row["passed"] for row in value["validation_receipts"])
    assert (
        subprocess.run([*argv, "--date", "wrong"], capture_output=True, timeout=30).returncode == 2
    )
    assert (
        subprocess.run([*argv, "--deliberate-error"], capture_output=True, timeout=30).returncode
        == 7
    )
    assert (
        subprocess.run([*argv, "--private-control"], capture_output=True, timeout=30).returncode
        == 2
    )


@pytest.mark.parametrize(
    "mutation", ["checksum", "hash", "authority", "rows", "no_operands", "missing_field"]
)
def test_replay_negative_controls(actual_work, tmp_path, mutation):
    """SCENARIO-REPORT-8437-CUSTODY: coordinated rehashing cannot grant readiness."""
    work = deepcopy(actual_work)
    if mutation == "authority":
        snapshot = tmp_path / "active.yaml"
        snapshot.write_text("tasks: []")
        work["authority"]["authority_snapshots"]["active"]["snapshot_path"] = str(snapshot)
    elif mutation == "rows":
        work["rows"][0]["kernel_s"] *= 2
    elif mutation == "no_operands":
        work["operands"] = []
    work["raw"] = str(tmp_path)
    atomic_json(tmp_path / "measurement.json", work)
    value = r.build(work, receipts())
    if mutation == "checksum":
        value["reproducibility_checksum"] = "wrong"
    elif mutation == "hash":
        value["source_artifact_hashes"][0]["sha256"] = "wrong"
    elif mutation == "missing_field":
        del value["work_reference"]
    if mutation != "checksum":
        value["reproducibility_checksum"] = r.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not r.replay(candidate)


def test_producer_and_reduction_controls(actual_work, monkeypatch):
    """REQ-REPORT-8437: independent producer qualification survives local new hashes."""
    with pytest.raises(ValueError, match="repeat_roster"):
        r.reduce(r.ROOT, [])
    changed = deepcopy(actual_work["operands"])
    changed[0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="producer_row_substitution"):
        r.reduce(r.ROOT, changed)
    real_attribute = r.attribute
    monkeypatch.setattr(r, "attribute", lambda v: dict(real_attribute(v), kernel_s=999))
    with pytest.raises(ValueError, match="producer_attribution_drift"):
        r.reduce(r.ROOT, actual_work["operands"])
    monkeypatch.setattr(r, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="unqualified_8436"):
        r.producer(r.ROOT)


def test_measure_required_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8437: present but unqualified external inputs remain blocked."""
    private = tmp_path / "private"
    private.mkdir(mode=0o755)
    private.chmod(0o755)
    monkeypatch.setattr(
        r, "producer", lambda root: (_ for _ in ()).throw(ValueError("bad upstream"))
    )
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda p: False if str(p).endswith("/.venv/bin/ruff") else real_is_file(p)
    )
    work = r.measure(r.ROOT, tmp_path / "raw", private)
    checks = {g["check"] for g in work["gate_check_summary"]}
    assert {"private_permissions", "tool_available", "qualified_terminal_and_primitives"} <= checks
    assert len(work["rows"]) == 135
    value = r.build(work, receipts())
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 135
    assert r.build(work, [dict(name="failed", passed=False)])["verdict_class"] == "disqualified"
    work["adversarial_findings"] = [dict(severity="warning")]
    assert r.build(work, receipts())["acceleration_analysis_ready_score"] == 0


def test_missing_and_substituted_primitive(actual_work, tmp_path):
    """SCENARIO-REPORT-8437-CUSTODY: missing or rehashed primitive rows fail independently."""
    operands = deepcopy(actual_work["operands"])
    operands[0]["path"] = str(tmp_path / "absent.json")
    with pytest.raises(OSError):
        r.reduce(r.ROOT, operands)
    primitive = r.read(actual_work["operands"][0])
    primitive["end_ns"] += 1000
    path = tmp_path / "rehashed.json"
    atomic_json(path, primitive)
    operands[0] = r.reference(path)
    with pytest.raises(ValueError, match="producer_row_substitution"):
        r.reduce(r.ROOT, operands)
    with pytest.raises(ValueError, match="invalid_conversion"):
        k.bounds(dict(cell=k.PRIMARY, complete_wall_s=10, kernel_s=1, conversion_s=-1))
    with pytest.raises(ValueError, match="invalid_T_K"):
        k.bounds(dict(cell=k.PRIMARY, complete_wall_s=10, kernel_s=9, conversion_s=2))


def test_manifest_and_actual_coverage_child(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8437-CUSTODY: genuine child shards exercise the validation adapter."""
    private = tmp_path / "coverage"
    private.mkdir(mode=0o700)
    plan = runner.manifest(private)
    assert {row["name"] for row in plan} == r.REQUIRED
    assert "-n" in plan[0]["argv"] and "--no-cov" in plan[0]["argv"]
    config = private / "coverage.ini"
    # This small validation invocation is real; production uses the unchanged full manifest.
    cov = str(r.ROOT / ".venv/bin/coverage")
    rc = "--rcfile=" + str(config)
    own = [
        dict(
            name="real_coverage_child",
            argv=[cov, "run", rc, str(r.ROOT / r.CLI), "--deliberate-error"],
            expected=7,
            deadline=60,
            scope="owned",
        ),
        dict(
            name="combine",
            argv=[cov, "combine", rc, "--keep"],
            expected=0,
            deadline=30,
            scope="owned",
        ),
        dict(
            name="json",
            argv=[cov, "json", rc, "-o", str(private / "coverage.json")],
            expected=0,
            deadline=30,
            scope="owned",
        ),
    ]
    monkeypatch.setattr(runner, "manifest", lambda p: own)
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(tmp_path / "missing", output, private, False) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["covered_lines"] > 0
    assert any(Path(ref["path"]).name.startswith(".coverage") for ref in value["raw_shard_hashes"])
    assert (
        next(row for row in value["validation_receipts"] if row["name"] == "real_coverage_child")[
            "exit_code"
        ]
        == 7
    )


def test_main_owned_exception(monkeypatch):
    """SCENARIO-REPORT-8437-CUSTODY: an actual owned exception cannot claim success."""
    monkeypatch.setattr(
        runner, "run", lambda *a: (_ for _ in ()).throw(ValueError("owned failure"))
    )
    assert runner.main([]) == 1
