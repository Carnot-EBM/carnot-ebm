"""REQ-VERIFY-8382 / REQ-REPORT-8382: runtime custody cannot imply benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.verify import runtime_evidence_delta_8382 as q
from carnot.verify import runtime_evidence_execution_8382 as e


def cli(tmp_path, *args):
    """Fresh processes test the actual entry point and its rejection exits."""
    return subprocess.run(
        [e.PY, "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_unchanged_and_missing(tmp_path):
    """SCENARIO-VERIFY-8382-DELTA: missing receipts never trigger a context retry."""
    with patch.object(q, "probe", side_effect=AssertionError("retry")):
        work = q.measure(q.ROOT, tmp_path / "raw")
    assert work["baseline"] and not work["current"]
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = q.build(work, tmp_path / "raw", [dict(passed=True, scope="owned")])
    assert value["honest_verdict"] == "complete_blocked_cuda_environment_unchanged"
    assert value["runtime_reader_ready_score"] == 1
    assert value["runtime_changed_score"] == value["cuda_context_ready_score"] == 0
    assert value["censored_count"] == 7 and value["independent_count"] == 0
    assert value["historical_closure_status"]["driver_errors"]
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    missing = q.measure(tmp_path / "absent", tmp_path / "missing")
    assert missing["failures"] and not missing["baseline"]
    assert q.build(missing, tmp_path / "missing", [])["verdict_class"] == "disqualified"


def test_receipt_controls(tmp_path):
    """SCENARIO-VERIFY-8382-DELTA: only pinned new causal operands admit inspection."""
    base = q.measure(q.ROOT, tmp_path / "base")
    current = deepcopy(base["baseline"])
    operand = tmp_path / "current.json"
    current["driver"] = "fixture-new-driver"
    atomic_json(operand, current)
    receipt = tmp_path / "receipt.json"
    value = dict(
        authority="operator",
        previous_primary_sha256=q.PIN,
        executed_at_ns=q.frontier(base["historical"]) + 1,
        action="driver_repair",
        description="constructed positive control",
        current_reference=reference(operand),
    )
    atomic_json(receipt, value)
    with patch.object(q.old, "inventory", return_value=(current, [])):
        work = q.measure(q.ROOT, tmp_path / "changed", receipt, sha256_file(receipt), fixture=True)
    assert any(r["changed"] for r in q.delta(work))
    result = q.build(work, tmp_path / "changed", [dict(passed=True, scope="owned")])
    assert result["runtime_changed_score"] == result["cuda_context_ready_score"] == 0
    from carnot.reporting.v709_execution import child

    measured = child(
        "fixture_validation", [e.PY, "-c", "print('validated')"], tmp_path / "receipts"
    )
    candidate = tmp_path / "fixture.json"
    atomic_json(candidate, q.build(work, tmp_path / "changed", [measured]))
    assert q.replay(candidate)
    altered = deepcopy(work)
    altered["current"]["driver"] = "forged"
    atomic_json(tmp_path / "changed/measurement.json", altered)
    atomic_json(candidate, q.build(altered, tmp_path / "changed", [measured]))
    assert not q.replay(candidate)
    with patch.object(q.old, "inventory", return_value=(base["baseline"], [])):
        assert q.measure(q.ROOT, tmp_path / "mismatch", receipt, sha256_file(receipt))["failures"]
    for name, change in [
        ("stale", dict(executed_at_ns=0)),
        ("authority", dict(authority="unknown")),
    ]:
        atomic_json(receipt, dict(value, **change))
        assert q.measure(q.ROOT, tmp_path / name, receipt, sha256_file(receipt))["failures"]
    assert q.measure(q.ROOT, tmp_path / "bad_hash", receipt, "invalid")["failures"]
    assert q.measure(q.ROOT, tmp_path / "bad_path", tmp_path / "absent", q.PIN)["failures"]
    atomic_json(receipt, value)
    auth = dict(base["authority"], gate_check_summary=[])
    with (
        patch.object(q.authority, "authority", return_value=auth),
        patch.object(q.old, "inventory", return_value=(current, [])),
        patch.object(q, "probe", return_value=dict(rows=[], checks=[])) as probe,
    ):
        changed = q.measure(q.ROOT, tmp_path / "admitted", receipt, sha256_file(receipt))
    assert probe.call_count == 1 and changed["current"] == current
    current = deepcopy(base["baseline"])
    for library in current["libraries"]:
        if not Path(library["path"]).name.startswith(("libcuda", "libnvidia")):
            library["sha256"] = "sha256:" + "a" * 64
    assert not any(r["changed"] for r in q.delta(dict(baseline=base["baseline"], current=current)))


def test_private_cli_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8382-REPLAY: rehashing cannot authorize forged scores."""
    monkeypatch.setattr(
        e,
        "manifest",
        lambda p: [
            dict(
                name="private_child",
                argv=[e.PY, "-c", "print('real child')"],
                deadline=15,
                expected=0,
                scope="owned",
            )
        ],
    )
    output = tmp_path / (q.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert q.replay(output) and cli(tmp_path, "--cold-replay", output).returncode == 0
    for field in [
        "runtime_reader_ready_score",
        "runtime_changed_score",
        "cuda_context_ready_score",
    ]:
        bad = dict(value, **{field: 1})
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(output, bad)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--date", "wrong").returncode != 0
    assert cli(tmp_path, "--private-run").returncode == 2


def test_manifest(tmp_path):
    """SCENARIO-VERIFY-8382-EXECUTION: freeze coverage and private E2E commands."""
    plan = e.manifest(tmp_path)
    assert any(r["name"] == "private_E2E018" for r in plan)
    assert all(r["deadline"] <= 900 for r in plan)
    assert any(r["scope"] == "global" for r in plan)


@pytest.mark.parametrize("mode", ["success", "child_fail", "lease_fail", "parse_fail"])
def test_direct_probe_child_paths(tmp_path, mode):
    """SCENARIO-VERIFY-8382-EXECUTION: one owned child retains errors and cleanup."""
    from carnot.reporting.v709_execution import child

    released = []
    lease = SimpleNamespace(
        owner_receipt=lambda: dict(fixture=True),
        transition=lambda phase: released.append(phase),
        release=lambda: dict(released=True),
    )
    current = dict(permitted_uuid="GPU-fixture", driver_library="fixture")
    result = dict(context_copy_ready=True, byte_copy_parity=True, cleanup_passed=True)

    def run(name, argv, raw, **kwargs):
        code = (
            "raise SystemExit(1)"
            if mode == "child_fail"
            else (
                "print('invalid json')"
                if mode == "parse_fail"
                else "print(" + repr(json.dumps(result)) + ")"
            )
        )
        return child(name, [e.PY, "-c", code], raw, **kwargs)

    with (
        patch.object(
            q.GpuLease,
            "acquire",
            side_effect=ValueError("busy") if mode == "lease_fail" else None,
            return_value=lease,
        ),
        patch.object(q, "child", run),
    ):
        diagnostic = q.probe(current, tmp_path)
    if mode in {"lease_fail", "parse_fail"}:
        assert diagnostic["checks"]
    else:
        assert len(diagnostic["rows"]) == 1
        assert diagnostic["rows"][0]["primitive"].get("context_copy_ready") == (
            True if mode == "success" else None
        )
    assert bool(diagnostic["cleanup_receipt"]) == (mode != "lease_fail")


def test_private_real_cli(tmp_path):
    """SCENARIO-VERIFY-8382-EXECUTION: actual CLI fixture has no CUDA readiness."""
    output = tmp_path / (q.NAME + ".json")
    result = cli(tmp_path, "--private-run", "--root", tmp_path / "absent", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert not value["cuda_context_ready_score"] and not value["runtime_changed_score"]
    assert q.replay(output)
    probe = cli(tmp_path, "--cuda-probe", "driver", "--library", str(tmp_path / "absent"))
    assert probe.returncode == 0 and "context_copy_ready" in probe.stdout


@pytest.mark.parametrize("failed", [False, True])
def test_owned_failure_and_resources(tmp_path, monkeypatch, failed):
    """SCENARIO-VERIFY-8382-EXECUTION: owned failure disqualifies; global failure stays separate."""

    def plan(private):
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100)))
        return [
            dict(
                name="owned_child",
                argv=[e.PY, "-c", f"raise SystemExit({int(failed)})"],
                deadline=15,
                expected=0,
                scope="owned",
            ),
            dict(
                name="global_child",
                argv=[e.PY, "-c", "raise SystemExit(1)"],
                deadline=15,
                expected=0,
                scope="global",
            ),
        ]

    monkeypatch.setattr(e, "manifest", plan)
    monkeypatch.setattr(
        e,
        "resources",
        lambda p: dict(
            private_disk_backed_scratch=False, available_memory_bytes=0, minimum_memory_bytes=1
        ),
    )
    output = tmp_path / (q.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert q.replay(output)


def test_primitive_substitution_and_independent_scores(tmp_path):
    """SCENARIO-REPORT-8382-REPLAY: sealed producer identity defeats consistent rehashing."""
    from carnot.reporting.v709_execution import child

    work = q.measure(q.ROOT, tmp_path / "raw")
    receipt = child("valid", [e.PY, "-c", "print('completed')"], tmp_path / "logs")
    path = tmp_path / "candidate.json"
    atomic_json(path, q.build(work, tmp_path / "raw", [receipt]))
    assert q.replay(path)
    value = json.loads(path.read_bytes())
    for kind in ("checksum", "identity", "receipt", "baseline", "historical", "primitive"):
        bad = deepcopy(value)
        altered = deepcopy(work)
        if kind == "checksum":
            bad["reproducibility_checksum"] = "invalid"
        elif kind == "identity":
            bad["experiment_id"] = 0
        elif kind == "receipt":
            bad["validation_receipts"][0]["passed"] = False
        else:
            altered["baseline" if kind == "baseline" else "historical"]["tampered"] = True
            if kind == "primitive":
                altered["refs"].append(dict(path=str(tmp_path / "missing"), sha256=q.PIN))
            primitive = tmp_path / (kind + ".json")
            atomic_json(primitive, altered)
            bad["primitive_reference"] = reference(primitive)
            bad["raw_shard_hashes"] = [reference(primitive)]
        if kind != "checksum":
            bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(path, bad)
        assert not q.replay(path)
    changed = deepcopy(work)
    changed["current"] = deepcopy(work["baseline"])
    changed["current"]["driver"] = "new"
    changed["receipt_reference"] = dict(path="fixture", sha256=q.PIN)
    changed["failures"] = []
    changed["diagnostic"] = dict(
        rows=[
            dict(
                receipt=receipt,
                primitive=dict(context_copy_ready=True, byte_copy_parity=True, cleanup_passed=True),
            )
        ],
        checks=[],
    )
    assert q.reduction(changed, True)["cuda_context_ready_score"] == 1
    assert q.reduction(changed, False)["runtime_changed_score"] == 1
    assert q.reduction(changed, False)["cuda_context_ready_score"] == 0
    changed["fixture"] = True
    assert q.reduction(changed, True)["cuda_context_ready_score"] == 0
    work["diagnostic"] = dict(
        rows=[
            dict(
                receipt=child(
                    "primitive", [e.PY, "-c", "print('{}')"], tmp_path / "primitive_logs"
                ),
                primitive={},
            )
        ],
        checks=[],
    )
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(path, q.build(work, tmp_path / "raw", [receipt]))
    assert q.replay(path)
    work["diagnostic"]["rows"][0]["primitive"] = dict(error="forged")
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(path, q.build(work, tmp_path / "raw", [receipt]))
    assert not q.replay(path)
    altered = deepcopy(work)
    altered["refs"] = [
        r for r in altered["refs"] if not r["path"].endswith("v721-deployment-protocol.json")
    ]
    atomic_json(tmp_path / "raw/measurement.json", altered)
    atomic_json(path, q.build(altered, tmp_path / "raw", [receipt]))
    assert not q.replay(path)
    protocol = next(
        r for r in altered["refs"] if r["path"].endswith("v717-local-learning-protocol.json")
    )
    substitute = tmp_path / "substituted-protocol.json"
    atomic_json(substitute, dict(constructed=True))
    protocol.update(sha256=sha256_file(substitute), snapshot_path=str(substitute))
    altered["refs"].append(
        next(r for r in work["refs"] if r["path"].endswith("v721-deployment-protocol.json"))
    )
    atomic_json(tmp_path / "raw/measurement.json", altered)
    atomic_json(path, q.build(altered, tmp_path / "raw", [receipt]))
    assert not q.replay(path)
