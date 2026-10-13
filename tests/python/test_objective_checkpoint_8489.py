"""REQ-REPORT-8489 and REQ-VERIFY-8489: preserve original checks and honest gates."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import objective_checkpoint_8489 as e
from carnot.reporting import objective_checkpoint_runner_8489 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_changed_reference_reaches_original_rejection(tmp_path):
    """SCENARIO-REPORT-8489-REFERENCE: envelope rehash cannot repair changed bytes."""
    before = e.reference(e.ORIGINAL)
    result = e.changed_reference_control(tmp_path)
    assert result["valid_original_passed"]
    assert result["changed_reference_rejected"]
    assert result["outer_hashes_consistent"]
    assert result["stored_reference_sha256"] != result["changed_file_sha256"]
    assert result["original_rejection_statement"].endswith(":388")
    assert e.reference(e.ORIGINAL) == before


def test_log_loss_reference_grid_and_rejection_controls():
    """REQ-VERIFY-8489: reference arithmetic retains temperature, bias and full L2."""
    rows = e.log_loss_checks(e.legacy.adapter.legacy.head())
    assert len(rows) == 90 and all(r["passed"] for r in rows)
    assert {r["temperature"] for r in rows} == {0.5, 1, 2}
    assert {r["target"] for r in rows} == {0, 1}
    assert {r["logit"] for r in rows} == {-20, -4, 0, 4, 20}
    assert max(r["absolute_error"] for r in rows) <= 1e-7
    assert max(r["relative_error"] for r in rows) <= 1e-5
    assert all(
        r["omit_temperature_rejected"]
        for r in rows
        if r["temperature"] != 1 and abs(r["logit"]) <= 4
    )
    assert all(r["wrong_sign_rejected"] for r in rows if not r["saturated"])


@pytest.fixture(scope="module")
def measured_current(tmp_path_factory):
    """SCENARIO-REPORT-8489-TERMINAL: real current authority admits sealed historical heads."""
    parent = tmp_path_factory.mktemp("8489-current")
    nodes = [runner.collection.collect(parent, suite)[0] for suite in runner.STORE_TESTS]
    groups = [nodes[0], nodes[1][:12], nodes[1][12:]]
    atomic_json(
        parent / "historical_collection.json",
        dict(
            node_ids=[n for group in groups for n in group],
            shards=groups,
        ),
    )
    work = e.measure(e.ROOT, parent / "raw", parent / "checkpoints")
    receipts = [
        child(
            n,
            [sys.executable, "-c", "print('constructed validation fixture')"],
            parent / "receipts",
            deadline=10,
        )
        for n in sorted(e.REQUIRED_CHECKS)
    ]
    output = parent / (e.NAME + ".json")
    value = e.build(work, receipts, parent / "raw", output)
    atomic_json(output, value)
    return parent, work, value, output


def test_complete_current_authority_qualifies_fixture(measured_current):
    """SCENARIO-REPORT-8489-TERMINAL: complete original checks precede mechanics readiness."""
    parent, work, value, output = measured_current
    assert not work["failures"]
    assert value["objective_kernel_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["completed_count"] == value["intended_count"] == 216
    assert len(value["derivative_rows"]) == 180 and len(value["process_exit_rows"]) == 32
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert e.replay(output)
    changed = deepcopy(work)
    rows = deepcopy(e.read(changed["log_loss_reference"]))
    rows[0]["passed"] = False
    atomic_json(parent / "bad_log_loss.json", rows)
    changed["log_loss_reference"] = e.reference(parent / "bad_log_loss.json")
    assert (
        e.build(changed, value["validation_receipts"], parent / "raw", output)["verdict_class"]
        == "disqualified"
    )


@pytest.mark.parametrize("operand", ["log_loss", "file", "producer_pin", "union"])
def test_self_consistent_additional_tampering(measured_current, tmp_path, operand):
    """SCENARIO-REPORT-8489-REFERENCE: reconstructed operands defeat repaired envelopes."""
    _, _, original, _ = measured_current
    value = deepcopy(original)
    if operand == "log_loss":
        rows = e.read(value["log_loss_reference"])
        rows[0]["absolute_error"] = 999
        atomic_json(tmp_path / "log_loss.json", rows)
        value["log_loss_reference"] = e.reference(tmp_path / "log_loss.json")
    elif operand == "union":
        ledger = e.read(value["historical_collection_reference"])
        ledger["shards"][0].append(ledger["shards"][0][0])
        atomic_json(tmp_path / "collection.json", ledger)
        value["historical_collection_reference"] = e.reference(tmp_path / "collection.json")
    else:
        control = e.read(value["changed_reference_control_reference"])
        if operand == "file":
            control["changed_file_reference"]["sha256"] = "wrong"
        else:
            control["original_terminal_reference"]["sha256"] = "wrong"
        atomic_json(tmp_path / "control.json", control)
        value["changed_reference_control_reference"] = e.reference(tmp_path / "control.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert not e.replay(tmp_path / "candidate.json")


def test_original_authority_byte_hash_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8489-REFERENCE: historical authority cannot authorize changed snapshots."""
    original = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "wrong" if p.name.startswith("design-") else original(p)
    )
    with pytest.raises(ValueError, match="original_authority_pin"):
        with e.original_binding(tmp_path):
            pytest.fail("changed authority admitted")


def test_binding_restores_originals(tmp_path):
    """SCENARIO-REPORT-8489-REFERENCE: current adaptation leaves old identity intact."""
    original = e.legacy.NAME
    with e.binding():
        assert e.legacy.NAME == e.NAME
        assert e.legacy.evidence.TASK == e.TASK
        assert e.legacy.evidence.methods.authority == e.methods.authority
    assert e.legacy.NAME == original
    with e.original_binding(tmp_path):
        assert e.legacy.evidence.methods.authority(e.ROOT, tmp_path / "auth")["activated"]


def test_manifest_exact_bounded_union(tmp_path):
    """SCENARIO-VERIFY-8489-SHARDS: all unchanged nodes occur exactly once."""
    plan = runner.manifest(tmp_path)
    historical = [r for r in plan if r["name"].startswith("historical_store_shard_")]
    collected = json.loads((tmp_path / "historical_collection.json").read_bytes())
    union = [n for row in historical for n in row["collected_node_ids"]]
    assert len(union) == len(set(union)) == 52
    assert union == collected["node_ids"]
    assert len(historical) == 3 and sum(r["deadline"] for r in historical) == 1800
    assert all(r["deadline"] == 600 for r in historical)
    assert collected["assertion_sources"]
    assert e.REQUIRED_CHECKS <= {r["name"] for r in plan}
    assert "patch=subprocess,_exit" in (tmp_path / "coverage.ini").read_text()
    assert all("--no-cov" in r["argv"] for r in historical)


def test_missing_inputs_and_rehashed_terminal(tmp_path):
    """SCENARIO-REPORT-8489-TERMINAL: missing is blocked; a forged summary fails."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "absent", raw, tmp_path / "private")
    receipts = [
        child(n, [sys.executable, "-c", "print('fixture receipt')"], raw / "checks", deadline=10)
        for n in sorted(e.REQUIRED_CHECKS)
    ]
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, receipts, raw, output)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 216
    assert value["objective_kernel_ready_score"] == 0
    assert value["current_llm_invocations"] == 0 and value["MODEL_SPECS"] == []
    assert value["gate_check_summary"]
    assert any(f["observed"] is None for f in value["gate_check_summary"])
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["objective_kernel_ready_score"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [], raw, output)["verdict_class"] == "disqualified"


def test_cli_controls_and_private_publication(tmp_path):
    """SCENARIO-VERIFY-8489-CLI: actual private publication and failures stay bounded."""
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261012"])
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            [
                "--date",
                "20261013",
                "--private-control",
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert json.loads(output.read_bytes())["objective_kernel_ready_score"] == 0


def test_worker_modes_and_wrong_rule(tmp_path):
    """SCENARIO-VERIFY-8489-CLI: inherited workers enforce objective rule seals."""
    cfg = tmp_path / "configuration.json"
    atomic_json(cfg, e.legacy.adapter.config(e.legacy.adapter.legacy.head()))
    assert runner.main(["--worker", str(tmp_path / "store"), "--configuration", str(cfg)]) == 0
    state = tmp_path / "store/final.json"
    assert runner.main(["--state-replay", str(state), "--configuration", str(cfg)]) == 0
    changed = json.loads(cfg.read_bytes())
    changed["update_rule_hash"] = "wrong"
    atomic_json(cfg, changed)
    assert runner.main(["--state-replay", str(state), "--configuration", str(cfg)]) == 1
    assert runner.main(["--worker", str(tmp_path / "missing"), "--configuration", "absent"]) == 1


def test_changed_control_failure_and_missing_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8489-TERMINAL: failed owned control differs from absent input."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "private")
    work["changed_reference_passed"] = False
    value = e.build(work, [], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    monkeypatch.setattr(e, "ORIGINAL", tmp_path / "absent-original")
    work = e.measure(tmp_path / "absent", tmp_path / "raw2", tmp_path / "private2")
    assert work["changed_reference_passed"] is None
    assert any(f["check"] == "original_reference_fixture_input" for f in work["failures"])


@pytest.mark.parametrize("field", ["sidecar", "methods_ready_score", "learning_inputs_ready_score"])
def test_methods_rejection_paths(tmp_path, monkeypatch, field):
    """SCENARIO-REPORT-8489-TERMINAL: a bound producer still needs every independent gate."""
    monkeypatch.setattr(e, "_AUTHENTICATE", lambda root, raw: dict(refs=[], failures=[]))
    if field == "sidecar":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    else:
        original = e.reader_receipt
        monkeypatch.setattr(
            e,
            "reader_receipt",
            lambda task, results, **kw: (
                dict(passed=False) if kw["field"] == field else original(task, results, **kw)
            ),
        )
    result = e.authenticate(e.ROOT, tmp_path)
    assert result["failures"] and not result["refs"]


def test_original_private_worker_branches(tmp_path):
    """SCENARIO-VERIFY-8489-CLI: historical wrappers retain original dates and exits."""
    node = e.ORIGINAL_TESTS[0] + "::test_transition_controls_and_invalid_configuration"
    assert runner.main(["--original-tests", node, "--basetemp=" + str(tmp_path / "tests")]) == 0
    assert (
        runner.main(
            [
                "--original-exec",
                str(e.ROOT / e.ORIGINAL_OWNED[-1]),
                "--cold-replay",
                str(tmp_path / "absent"),
            ]
        )
        == 1
    )
    source = tmp_path / "normal.py"
    source.write_text("print('private normal script')\n")
    assert runner.original_worker([str(source)], tests=False) == 0
    source.write_text("raise SystemExit('private string exit')\n")
    assert runner.original_worker([str(source)], tests=False) == 0


def test_actual_shard_failure_is_retained(tmp_path):
    """SCENARIO-VERIFY-8489-SHARDS: failed children cannot become passing aggregate inputs."""
    plan = [
        dict(
            name="historical_store_shard_control",
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
            deadline=10,
            expected=0,
            receipt_directory=str(tmp_path / "receipts"),
        ),
        dict(
            name="passing_control",
            argv=[sys.executable, "-c", "print('passed')"],
            deadline=10,
            expected=0,
        ),
    ]
    rows = runner.execute(plan, tmp_path / "logs")
    assert rows[0]["exit_code"] == 7 and not rows[0]["passed"]
    assert rows[1]["passed"]
    assert (
        json.loads((tmp_path / "receipts/historical_store_shard_control.receipt.json").read_bytes())
        == rows[0]
    )


def test_independent_shard_summary_requires_exact_passed_nodes(tmp_path):
    """SCENARIO-VERIFY-8489-SHARDS: zero exits cannot conceal omitted or skipped assertions."""
    node = e.TEST + "::test_log_loss_reference_grid_and_rejection_controls"
    receipt = child(
        "historical_store_shard_0",
        [
            sys.executable,
            "-m",
            "pytest",
            *runner.COMMON,
            "-vv",
            node,
            "--basetemp=" + str(tmp_path / "pytest"),
        ],
        tmp_path / "receipts",
        deadline=60,
    )
    assert receipt["passed"]
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, dict(node_ids=[node], shards=[[node]]))
    assert runner.summarize(ledger, tmp_path / "receipts") == 0
    atomic_json(ledger, dict(node_ids=[node, "missing"], shards=[[node, "missing"]]))
    assert runner.summarize(ledger, tmp_path / "receipts") == 1
    atomic_json(ledger, dict(node_ids=[node], shards=[[node]]))
    receipt["timed_out"] = True
    atomic_json(tmp_path / "receipts/historical_store_shard_0.receipt.json", receipt)
    assert runner.summarize(ledger, tmp_path / "receipts") == 1
    assert runner.summarize(tmp_path / "absent", tmp_path / "receipts") == 1
    assert (
        runner.main(
            ["--shard-summary", str(ledger), "--shard-receipts", str(tmp_path / "receipts")]
        )
        == 1
    )
    with pytest.raises(SystemExit):
        runner.main(["--shard-summary", str(ledger)])


def test_original_child_preserves_explicit_coverage_prefix(tmp_path):
    """SCENARIO-VERIFY-8489-COVERAGE-PREFIX: historical routing retains real tracing.

    A private script starts the original CLI through Coverage.py. The wrapper
    must retain both its failure exit and independently configured data shards.
    """
    config = tmp_path / "coverage.ini"
    config.write_text(f"[run]\nparallel=true\ndata_file={tmp_path / '.coverage'}\n")
    script = tmp_path / "coverage_child.py"
    arguments = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        str(e.ROOT / e.ORIGINAL_OWNED[-1]),
        "--deliberate-error",
    ]
    script.write_text(
        "import subprocess\n"
        f"assert subprocess.run({arguments!r}, check=False).returncode == 7\n"
        "from pathlib import Path\n"
        f"assert list(Path({str(tmp_path)!r}).glob('.coverage.*'))\n"
    )
    assert runner.original_worker([str(script)], tests=False) == 0
