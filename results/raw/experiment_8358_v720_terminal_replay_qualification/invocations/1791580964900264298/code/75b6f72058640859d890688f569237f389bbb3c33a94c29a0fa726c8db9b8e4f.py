"""REQ-REPORT-8358 / REQ-VERIFY-8358: exact closure and honest failure controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import runpy

import pytest

from carnot.reporting import v720_terminal_replay as e
from carnot.reporting import v720_replay_closure as c
from carnot.reporting import v720_replay_execution as x
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def test_private_producers_and_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8358-CONTROLS: child results preserve successful and failed work."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        value = e.private_producer(output, failed)
        bundle = c.capture(output, sha256_file(output), tmp_path / str(failed) / "closure", {})
        request = dict(bundle=bundle, authority=e.TASK_PIN, disposition=value["verdict_class"])
        result, receipt = e.invoke(request, tmp_path / str(failed) / "worker", "valid")
        assert receipt["passed"] and result["passed"]
        assert result["verdict_class"] == ("disqualified" if failed else "circular_positive")
        assert result["mutable_authority_access_count"] == 0
        for name in [
            "missing-source",
            "changed-source",
            "wrong-authority",
            "changed-disposition",
            "rehashed-tamper",
        ]:
            changed = deepcopy(request)
            if name == "wrong-authority":
                changed["authority"] = "wrong"
            elif name == "changed-disposition":
                changed["disposition"] = "positive"
            else:
                ref = changed["bundle"]["rows"][-1]["reference"]
                target = tmp_path / str(failed) / (name + ".bin")
                target.write_bytes(Path(ref["path"]).read_bytes())
                ref["path"] = str(target)
                if name == "missing-source":
                    target.unlink()
                else:
                    target.write_bytes(b"{}")
                    if name == "rehashed-tamper":
                        ref["sha256"] = sha256_file(target)
                        changed["bundle"]["rows"][-1]["expected_sha256"] = ref["sha256"]
            actual, negative = e.invoke(
                changed, tmp_path / str(failed) / name / "worker", name, expected=1
            )
            assert negative["passed"] and not actual["passed"]


def test_membership_and_terminal_authentication(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8358-DISPOSITIONS: hashes cannot bless altered membership."""
    output = tmp_path / (e.NAME + ".json")
    e.private_producer(output, False)
    bundle = c.capture(output, sha256_file(output), tmp_path / "closure", {})
    assert c.verify(bundle)["verdict_class"] == "circular_positive"
    bad = deepcopy(bundle)
    bad["rows"] = bad["rows"][:-1]
    with pytest.raises(ValueError):
        c.verify(bad)
    with pytest.raises(ValueError):
        c.capture(output, "sha256:wrong", tmp_path / "bad", {})


def test_real_cli_and_rehashed_aggregate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8358-PUBLICATION: standalone fresh replay checks semantics."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "private",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--private-e2e", "--output", str(output)],
        tmp_path / "logs",
        deadline=240,
    )
    assert receipt["passed"]
    value = json.loads(output.read_bytes())
    assert e.replay(output)
    changed = dict(value, branch_replay_ready_score=1 - value["branch_replay_ready_score"])
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)


def test_frozen_paths_and_cli_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8358-EXECUTION: old path writes and mutable aliases stay private."""
    output = tmp_path / (e.NAME + ".json")
    e.private_producer(output, False)
    bundle = c.capture(output, sha256_file(output), tmp_path / "closure", {})
    value = c.verify(bundle)
    accesses = dict(count=0)
    with c.frozen_reads(bundle, tmp_path / "scratch", accesses):
        assert Path(value["primitive_reference"]["path"]).read_bytes()
        (Path(value["primitive_reference"]["path"]).parent / "new.json").write_text("private")
        atomic_json(
            Path(value["primitive_reference"]["path"]).parent / "nested/new.json",
            dict(private=True),
        )
        (tmp_path / "outside-file").write_text("private")
        (tmp_path / "outside-file").replace(tmp_path / "outside-file-moved")
        assert (tmp_path / "outside").exists() is False
        with pytest.raises(ValueError, match="mutable_authority"):
            (tmp_path / "research-roadmap-vNEXT.md").read_text()
    assert accesses["count"] == 1 and (tmp_path / "scratch/new.json").read_text() == "private"
    assert json.loads((tmp_path / "scratch/nested/new.json").read_bytes())["private"]
    main = runpy.run_path(str(e.ROOT / e.CLI))["main"]
    for args in [["--date", "wrong"], ["--private-e2e"], ["--worker-request", str(output)]]:
        with pytest.raises(SystemExit):
            main(args)
    assert main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert (
        main(
            [
                "--worker-request",
                str(tmp_path / "absent"),
                "--worker-output",
                str(tmp_path / "error.json"),
            ]
        )
        == 1
    )
    with pytest.raises(ValueError):
        e.private_producer(e.ROOT / "results" / output.name, False)
    monkeypatch.setattr(e, "child", lambda *a, **kw: dict(passed=False))
    actual, receipt = e.invoke({}, tmp_path / "no-child", "missing")
    assert not actual["passed"] and not receipt["passed"]


def test_natural_closures_and_named_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8358-DISPOSITIONS: original failures and all current slots survive."""
    work = e.measure(tmp_path / "natural")
    assert len(work["rows"]) == 4 and len(work["historical_dispositions"]) == 2
    assert work["historical_dispositions"][0]["verdict_class"] == "disqualified"
    assert all(not r["science_promoted"] for r in work["historical_dispositions"])
    assert (
        work["historical_dispositions"][1]["failed_stdout"].find("gatemate_change_ledger_8344.py")
        >= 0
    )
    work.update(
        frozen_validation_receipts=[dict(passed=True)],
        authority_refs=[],
        code_refs=[],
        authority={},
        preconditions_checked=[],
    )
    value = e.build(work, [dict(passed=True)], tmp_path / "natural", tmp_path / "natural.json")
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    monkeypatch.setattr(c, "capture", lambda *a, **kw: (_ for _ in ()).throw(ValueError("schema")))
    failed = e.measure(tmp_path / "failed")
    assert all(r["status"] == "disqualified" for r in failed["rows"])
    monkeypatch.setattr(c, "PINS", {9999: "sha256:missing"})
    blocked = e.measure(tmp_path / "missing")
    assert blocked["rows"][0]["status"] == "external_blocked"
    assert blocked["rows"][0]["first_mismatch"]["observed"] is None


def test_frozen_plan_and_memory_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8358-EXECUTION: coverage scope, resources and failures remain real."""
    plan = x.manifest(tmp_path)
    assert all(r["deadline"] <= 600 for r in plan)
    assert "tests/python" not in next(r["argv"] for r in plan if r["name"] == "owned_tests")
    assert len(x.preflight(tmp_path)) == 7
    assert x.freeze([tmp_path / "absent"], tmp_path / "frozen") == []
    output = tmp_path / (e.NAME + ".json")
    e.private_producer(output, False)
    bundle = c.capture(output, sha256_file(output), tmp_path / "closure", {})
    request = tmp_path / "worker.json"
    atomic_json(request, dict(bundle=bundle, authority=e.TASK_PIN, disposition="circular_positive"))
    memories = iter([dict(peak_rss_mb=0), dict(peak_rss_mb=501)])
    monkeypatch.setattr(e, "memory", lambda: next(memories))
    assert e.worker(request, tmp_path / "memory.json") == 1
    assert not json.loads((tmp_path / "memory.json").read_bytes())["memory"]["passed"]


def test_anchor_and_sidecar_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8358-CONTROLS: repaired sidecar hashes cannot alter identity."""
    output = tmp_path / (e.NAME + ".json")
    value = e.private_producer(output, False)
    pin = sha256_file(output)
    bundle = c.capture(output, pin, tmp_path / "closure", {})
    for field, replacement in [
        ("experiment_id", 0),
        ("original_path", "wrong"),
        ("primary_sha256", "wrong"),
    ]:
        changed = deepcopy(bundle)
        changed[field] = replacement
        with pytest.raises(ValueError):
            c.verify(changed)
    with monkeypatch.context() as context:
        context.setattr(c, "PINS", {8358: "wrong"})
        with pytest.raises(ValueError):
            c.capture(output, pin, tmp_path / "wrong-anchor", {})
    with monkeypatch.context() as context:
        context.setattr(c, "TERMINAL_PINS", {8358: ("wrong", "wrong")})
        with pytest.raises(ValueError):
            c.capture(output, pin, tmp_path / "wrong-side", {})
        with pytest.raises(ValueError):
            c.verify(bundle)
    changed = deepcopy(bundle)
    side = tmp_path / "side.json"
    altered = json.loads(Path(changed["rows"][1]["reference"]["path"]).read_bytes())
    atomic_json(side, dict(altered, primary_sha256="wrong"))
    changed["rows"][1]["reference"] = dict(path=str(side), sha256=sha256_file(side))
    changed["rows"][1]["expected_sha256"] = sha256_file(side)
    with pytest.raises(ValueError):
        c.verify(changed)
    live_side = output.parent / "raw" / output.stem / "validators" / (pin[7:] + ".json")
    atomic_json(live_side, dict(altered, report=dict(passed=False)))
    with pytest.raises(ValueError):
        c.capture(output, pin, tmp_path / "failed-side", {})
    assert value["verdict_class"] == "circular_positive"


def test_failed_native_reduction_and_bad_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8358-PUBLICATION: declared failure must agree with primitive work."""
    from carnot.reporting.primary_publication import publish_primary

    output = tmp_path / (e.NAME + ".json")
    value = e.private_producer(output, False)
    value.update(verdict_class="disqualified", honest_verdict="complete_disqualified_control")
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    atomic_json(Path(value["terminal_validation_sidecar_path"]), dict(publication=publication))
    bundle = c.capture(output, sha256_file(output), tmp_path / "closure", {})
    actual, receipt = e.invoke(
        dict(bundle=bundle, authority=e.TASK_PIN, disposition="disqualified"),
        tmp_path / "false-reduction",
        "false",
        expected=1,
    )
    assert receipt["passed"] and actual["error"] == "primitive_reduction_drift"
    original_child = x.child

    def invalid_report(name: str, argv: list[str], logs: Path, **kwargs):
        if name == "adversarial":
            argv = [sys.executable, "-u", "-c", "print('invalid-json')"]
        return original_child(name, argv, logs, **kwargs)

    monkeypatch.setattr(x, "child", invalid_report)
    assert not x.validate(output, tmp_path / "validator")["passed"]


def test_natural_invocation_failure_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8358-EXECUTION: owned and authority failures publish disqualified."""
    work = e.measure(tmp_path / "measurement", private=True)
    probe = tmp_path / "covered_control.py"
    probe.write_text("print('covered private control', flush=True)\n")
    coverage = str(e.ROOT / ".venv/bin/coverage")
    assert child(
        "coverage_run",
        [coverage, "run", "--data-file=" + str(tmp_path / "control.coverage"), str(probe)],
        tmp_path / "coverage_logs",
        deadline=30,
    )["passed"]
    assert child(
        "coverage_report",
        [
            coverage,
            "json",
            "--data-file=" + str(tmp_path / "control.coverage"),
            "-o",
            str(tmp_path / "coverage.json"),
        ],
        tmp_path / "coverage_logs",
        deadline=30,
    )["passed"]
    monkeypatch.setattr(e, "measure", lambda raw, **kw: deepcopy(work))
    monkeypatch.setattr(
        x,
        "manifest",
        lambda private: [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-u", "-c", "raise SystemExit(7)"],
                expected=0,
                deadline=5,
                scope="owned",
            )
        ],
    )
    monkeypatch.setattr(x, "operand_controls", lambda private, raw: [])
    output = tmp_path / "failed" / (e.NAME + ".json")
    assert x.run(output, tmp_path) == 1
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert any(
        r.get("original_path") == str(tmp_path / "coverage.json")
        for r in json.loads(output.read_bytes())["code_config_hashes"]
    )
    assert e.replay(output)
    monkeypatch.setattr(
        x.contract,
        "authority",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("authority-unavailable")),
    )
    original_validate = x.validate
    attempts = []

    def reject_first(path: Path, raw: Path):
        result = original_validate(path, raw)
        attempts.append(result)
        if len(attempts) == 1:
            result["passed"] = False
        return result

    monkeypatch.setattr(x, "validate", reject_first)
    assert x.run(tmp_path / "authority-failed" / output.name, tmp_path) == 1
    assert attempts[0]["passed"] is False and attempts[-1]["passed"]


def test_authority_receipt_and_primitive_rehashes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8358-CONTROLS: repaired primitive and aggregate seals still fail."""
    work = e.measure(tmp_path / "private", private=True)
    actual = x.contract.authority(e.ROOT, tmp_path / "assessment")
    task = next(t for t in actual["tasks"] if t["id"] == e.TASK)
    refs = x.freeze(
        [
            e.ROOT / p
            for p in [x.contract.DESIGN, x.contract.ACTIVE, x.contract.STAGED, x.contract.PROTOCOL]
        ],
        tmp_path / "authority",
    )
    work.update(
        private_control=False,
        authority_refs=refs,
        authority=dict(activated=actual["activated"], task=task),
        frozen_validation_receipts=[dict(passed=True)],
        code_refs=[],
    )
    output = tmp_path / "candidate.json"
    value = e.build(work, work["frozen_validation_receipts"], tmp_path / "raw", output)
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)
    for field in ["authority", "row"]:
        altered = deepcopy(work)
        if field == "authority":
            altered["authority"]["activated"] = not actual["activated"]
        else:
            altered["rows"][0]["status"] = "invented_status"
        candidate = e.build(altered, work["frozen_validation_receipts"], tmp_path / field, output)
        atomic_json(output, candidate)
        assert not e.replay(output)


def test_capstone_seal_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8358-DISPOSITIONS: imported capstone history has a fixed anchor."""
    monkeypatch.setattr(c, "PINS", {})
    monkeypatch.setattr(e, "CAPSTONE_PIN", "wrong")
    with pytest.raises(ValueError, match="capstone_anchor"):
        e.measure(tmp_path)
