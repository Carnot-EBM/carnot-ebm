"""REQ-REPORT-8456 / REQ-VERIFY-8456: honest accounting needs independent operands."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v727_capstone_evidence as e
from carnot.reporting import v727_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(*args):
    """Real processes test publication and failure paths that mocks cannot establish."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v727-capstone")
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_missing_cli(private, tmp_path):
    """SCENARIO-REPORT-8456-ACCOUNTING: thirteen absences cannot become measured nulls."""
    root, output, value = private
    assert value["honest_verdict"] == "complete_blocked_v727_capstone"
    assert value["task_count"] == value["intended_count"] == len(value["rows"]) == 14
    assert value["absent_count"] == 13 and value["producer_count"] == 0
    assert value["milestone_outcomes"][-1]["disposition"] == "current_reconciliation"
    assert not value["MODEL_SPECS"] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert e.replay(output) and cli("--cold-replay", output).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    assert cli("--date", "20260101").returncode == 2
    assert cli("--private-control").returncode == 2
    assert cli("--deliberate-error").returncode == 7
    assert cli("--cold-replay", tmp_path / "absent.json").returncode == 1


def test_rehashed_controls(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: repaired local hashes do not repair evidence."""
    _, output, value = private
    for key, replacement in [
        ("task_count", 13),
        ("scientific_claims", {}),
        ("contract_digest", "wrong"),
    ]:
        changed = deepcopy(value)
        changed[key] = replacement
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert not e.replay(path)
    work = e.load(value["primitive_reference"])
    changed = deepcopy(work)
    changed["slots"][0]["disposition"] = "qualified_producer"
    path = tmp_path / "work.json"
    atomic_json(path, changed)
    tampered = deepcopy(value)
    tampered["primitive_reference"] = e.reference(path)
    tampered["reproducibility_checksum"] = e.checksum(tampered)
    candidate = tmp_path / "rehashed.json"
    atomic_json(candidate, tampered)
    assert not e.replay(candidate)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "wrong"
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    assert e.replay(output)


def test_slot_classification(tmp_path):
    """SCENARIO-REPORT-8456-ACCOUNTING: pre-gates and cascade skips never count as producers."""
    task = {"id": "exp8450-journal-equivalence", "deliverable": "results/no.json"}
    ref = e.freeze(tmp_path / "absent.json", tmp_path / "raw")
    assert e.slot(task, ref, None)["disposition"] == "absent_artifact"
    assert e.slot(task, ref, "GATE_BLOCK upstream retired")["disposition"] == "cascade_skip"
    value = {
        "blocked_at_layer": "conductor_pre_gate",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "missing runtime",
    }
    path = tmp_path / "pre.json"
    atomic_json(path, value)
    ref = e.freeze(path, tmp_path / "raw")
    with patch.dict(e.PINS, {8450: ref["sha256"].split(":")[1]}):
        summary = e.outcome(task, ref, tmp_path)
    assert e.slot(task, ref, None, summary)["disposition"] == "pre_gate_receipt"
    with pytest.raises(ValueError, match="primary_pin"):
        e.outcome(dict(task, id="exp8449-persistence-localization"), ref, tmp_path)
    Path(ref["snapshot_path"]).chmod(0o600)
    Path(ref["snapshot_path"]).write_text("bad JSON")
    with pytest.raises(ValueError):
        e.load(ref)


def test_real_primitive_and_disqualified_reader(tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: independent primitive replay precedes imports."""
    from types import SimpleNamespace

    path = next((e.ROOT / "results").glob("experiment_8449_*.json"))
    task = {"id": "exp8449-persistence-localization"}
    primary = e.freeze(path, tmp_path)
    actual = e.outcome(task, primary, tmp_path)
    assert actual["eligible"] and actual["selected"]["persistence_profile_ready_score"] == 0
    assert actual["selected"]["complete_wall_s"] is None
    with patch.object(
        e.old.importlib, "import_module", return_value=SimpleNamespace(replay=lambda _: False)
    ):
        bad = e.outcome(task, primary, tmp_path)
    assert bad["disposition"] == "disqualified_producer" and bad["checks"]
    with pytest.raises(ValueError, match="task_identity"):
        e.outcome({"id": "exp8449-wrong-task"}, primary, tmp_path)
    big = tmp_path / "large.bin"
    with big.open("wb") as stream:
        stream.truncate(129 * 1024**2)
    with pytest.raises(ValueError, match="operand_size_budget"):
        e.freeze(big, tmp_path / "raw")


def test_pregate_only_cli(private, tmp_path):
    """SCENARIO-REPORT-8456-ACCOUNTING: an actual pre-gate-only root preserves thirteen pending slots."""
    root = tmp_path / "root"
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    source = next((e.ROOT / "results").glob("experiment_8450_*.json"))
    target = root / "results" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    output = root / "results" / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["pregate_count"] == 1 and value["producer_count"] == 0
    assert value["absent_count"] == 12 and e.replay(output)
    request = tmp_path / "request.json"
    atomic_json(
        request,
        dict(
            task={"id": "exp8450-journal-equivalence"},
            task_sha256="wrong",
            primary=e.freeze(source, tmp_path / "source"),
            closure=[],
        ),
    )
    assert (
        cli("--worker-request", request, "--worker-output", tmp_path / "summary.json").returncode
        == 1
    )


def test_all_null_and_failed_accounting(private, tmp_path):
    """SCENARIO-REPORT-8456-ACCOUNTING: all-null accounting still gives no generalization."""
    from carnot.reporting.v727_capstone_report import build

    _, output, value = private
    work = e.load(value["primitive_reference"])
    work["checks"] = []
    for row in work["slots"]:
        row.update(
            status="completed",
            censored=False,
            absolute_metric=1,
            raw_numerator=1,
            disposition="qualified_producer",
            producer_executed=True,
            eligible=True,
            verdict_class="null",
            honest_verdict="complete_null_private_control",
        )
    candidate = build(work, value["validation_receipts"], Path(work["raw"]), output)
    assert candidate["verdict_class"] == "null" and candidate["producer_count"] == 13
    assert (
        candidate["scientific_claims"]["objective_learning"]["decision"]
        == "continue_changed_objective_after_qualification"
    )
    work["slots"][0].update(disposition="disqualified_producer", verdict_class="disqualified")
    work["branch_receipts"] = [dict(passed=False)]
    candidate = build(work, value["validation_receipts"], Path(work["raw"]), output)
    assert candidate["verdict_class"] == "disqualified"
    work["adversarial_findings"] = [dict(kind="unknown", severity="warn")]
    assert build(work, value["validation_receipts"], Path(work["raw"]), output)[
        "flagged_adversarial"
    ]
    with patch.object(
        e,
        "resources",
        return_value=dict(
            disk_backed=False, memory_available_bytes=0, free_disk_bytes=0, tools_available=False
        ),
    ):
        blocked = e.measure(tmp_path, tmp_path / "raw", tmp_path)
    assert any(g.get("field") == "required_resources" for g in blocked["checks"])


def test_receipt_and_authority_tampering(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: hashes cannot excuse changed clocks, tasks or authority."""
    _, output, value = private
    original = e.load(value["primitive_reference"])
    for mutation in ["bytes", "snapshot", "authority", "contract", "task", "argv", "log"]:
        work, changed = deepcopy(original), deepcopy(value)
        if mutation == "bytes":
            work["refs"].append(dict(path=str(e.ROOT / "CODEX.md"), sha256="wrong"))
        elif mutation == "snapshot":
            work["authority"]["authority_snapshots"]["active"]["sha256"] = "wrong"
        elif mutation == "authority":
            work["authority"]["activated"] = False
        elif mutation == "contract":
            work["contract_digest"] = "wrong"
        elif mutation == "task":
            work["slots"][0]["task"] = {"id": "exp8443-foreign"}
        elif mutation == "argv":
            changed["validation_receipts"][0]["argv"] = ["foreign"]
        else:
            changed["validation_receipts"][0]["stdout_sha256"] = "wrong"
        path = tmp_path / (mutation + ".work.json")
        atomic_json(path, work)
        changed["primitive_reference"] = e.reference(path)
        changed["reproducibility_checksum"] = e.checksum(changed)
        candidate = tmp_path / (mutation + ".json")
        atomic_json(candidate, changed)
        assert not e.replay(candidate), mutation


def test_production_path_with_genuine_child_shards(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: normal orchestration preserves real child databases."""
    root, _, _ = private
    scratch = tmp_path / "private"
    scratch.mkdir(mode=0o700)
    plan = runner.manifest(scratch)
    rc = "--rcfile=" + str(scratch / "coverage.ini")
    direct = dict(
        name="genuine_child_coverage",
        argv=[
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            rc,
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(tmp_path / "missing.json"),
        ],
        expected=1,
        deadline=30,
        scope="owned",
    )
    selected = [direct, *[p for p in plan if p["name"] in {"coverage_combine", "coverage_json"}]]
    output = tmp_path / "results" / (e.NAME + ".json")
    with patch.object(runner, "manifest", return_value=selected):
        assert runner.run(root, output, scratch) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and e.replay(output)
    assert any(".coverage" in r["path"] for r in value["raw_shard_hashes"])
    assert any("coverage.json" in r["path"] for r in value["raw_shard_hashes"])


def test_real_partial_measurement_and_rehash(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: child rows and primitive seals resist repaired summaries."""
    root = tmp_path / "root"
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    source = next((e.ROOT / "results").glob("experiment_8449_*.json"))
    target = root / "results" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "Qualify a bounded event journal with unchanged dur GATE_BLOCK upstream retired\n"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    work = e.measure(root, tmp_path / "raw", scratch)
    row = work["slots"][6]
    assert row["eligible"] and row["memory"]["passed"]
    assert row["reader_initialization"]["scope"].endswith("no model weights loaded")
    assert work["slots"][7]["disposition"] == "cascade_skip"
    _, _, original = private
    work.update(
        publication=e.load(original["primitive_reference"])["publication"],
        publication_receipt=e.load(original["primitive_reference"])["publication_receipt"],
    )
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value = runner.build(
        work, original["validation_receipts"], Path(work["raw"]), tmp_path / (e.NAME + ".json")
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    with patch.object(
        e.workers, "invoke", return_value=(dict(memory=dict(passed=False)), dict(passed=False))
    ):
        assert not e.replay(path)
    work["slots"][6]["selected"]["complete_wall_s"] = 99
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value = runner.build(
        work, original["validation_receipts"], Path(work["raw"]), tmp_path / (e.NAME + ".json")
    )
    atomic_json(path, value)
    assert not e.replay(path)


@pytest.mark.parametrize(
    ("args", "expected", "phase"),
    [
        (["--deliberate-error"], 7, "deliberate_error"),
        (["--cold-replay", "absent.json"], 1, "replay_rejected"),
    ],
)
def test_standalone_cli_without_import_path(tmp_path, args, expected, phase):
    """SCENARIO-VERIFY-8456-CONTROLS: script imports follow its checkout from any cwd."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert "phase=" + phase in result.stdout


def test_deadline_and_unconditional_authority(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: measurement deadlines and task gates are real checks."""
    root, _, value = private
    authority = e.load(value["primitive_reference"])["authority"]
    wrong = deepcopy(authority)
    wrong["tasks"][-1]["gated_on"] = [{"upstream": "foreign"}]
    with patch.object(e.methods, "authority", return_value=wrong):
        with pytest.raises(ValueError, match="exact_unconditional"):
            e.measure(root, tmp_path / "invalid", tmp_path)
    with (
        patch.object(e.methods, "authority", return_value=authority),
        patch.object(e.time, "monotonic_ns", side_effect=[0, 3600 * 10**9]),
    ):
        with pytest.raises(TimeoutError, match="measurement_deadline"):
            e.measure(root, tmp_path / "deadline", tmp_path)


def test_original_historical_worker_and_rehashed_evidence(private, tmp_path):
    """SCENARIO-VERIFY-8456-CONTROLS: original authority reconstructs historical scalar science."""
    root = tmp_path / "root"
    for name in [e.methods.DESIGN, e.methods.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    source = next((e.ROOT / "results").glob("experiment_8432_*.json"))
    target = root / "results" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    work = e.measure(root, tmp_path / "raw", tmp_path)
    original = e.load(private[2]["primitive_reference"])
    work.update(
        publication=original["publication"], publication_receipt=original["publication_receipt"]
    )
    assert work["historical_slots"][0]["eligible"]
    output = tmp_path / (e.NAME + ".json")
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value = runner.build(work, private[2]["validation_receipts"], Path(work["raw"]), output)
    atomic_json(output, value)
    assert e.replay(output)
    assert (
        value["scientific_claims"]["V726_preserved"]["8432"]["metrics"]["brier_gain_vs_logistic"]
        > 0
    )
    work["historical_slots"][0]["selected"]["brier_gain_vs_logistic"] = 0
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    atomic_json(output, runner.build(work, value["validation_receipts"], Path(work["raw"]), output))
    assert not e.replay(output)
    work["historical_slots"] = []
    work["predecessors"][0]["sha256"] = "rehashed"
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value["primitive_reference"] = e.reference(Path(work["raw"]) / "measurement.json")
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "missing.json")


def test_blocked_branch_retirement_and_historical_null(private):
    """SCENARIO-REPORT-8456-ACCOUNTING: original nulls and blocked board claims remain separate."""
    _, output, value = private
    work = e.load(value["primitive_reference"])
    for row in work["slots"][-3:]:
        row.update(producer_executed=True, verdict_class="blocked", eligible=True)
    for number in [8435, 8437]:
        path = next((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
        old = json.loads(path.read_bytes())
        work["historical_slots"].append(
            dict(
                experiment_id=number,
                task_id=old["task_id"],
                eligible=True,
                honest_verdict=old["honest_verdict"],
                selected={},
                primary_reference=e.reference(path),
            )
        )
    built = runner.build(work, value["validation_receipts"], Path(work["raw"]), output)
    assert any(r.get("scope") == "board_operations" for r in built["retirement_recommendations"])
    assert built["scientific_claims"]["V726_preserved"]["8435"]["decision"].startswith(
        "keep_learning_null"
    )
    assert built["scientific_claims"]["objective_learning"]["retention_windows"] == [
        dict(window=i, metric=None, missing_reason="Exp8448 objective audit not measured")
        for i in range(5)
    ]
