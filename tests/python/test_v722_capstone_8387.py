"""REQ-REPORT-8387 / REQ-VERIFY-8387: terminal accounting cannot invent missing science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v722_capstone_evidence as e
from carnot.reporting import v722_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def private_root(root):
    """Actual task bytes qualify private mechanics without creating research outcomes."""
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    return root


def cli(*args):
    """Use the standalone script so child imports and exit paths are measured."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
    )


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8387-REPLAY: missing producers stay blocked after checked publication."""
    root = private_root(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["missing_output_count"] == 13
    assert value["actual_executed_task_count"] == 1
    assert len(value["rows"]) == len(value["next_evidence_conditions"]) == 14
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert cli("--cold-replay", output).returncode == 0
    for key in ["rows", "H1", "paper_ready", "validation_receipts", "cascade_skip_count"]:
        changed = deepcopy(value)
        if key == "rows":
            changed[key][0]["disposition"] = "qualified"
        elif key == "H1":
            changed[key]["qualified"] = True
        elif key == "validation_receipts":
            changed[key][0]["passed"] = False
        else:
            changed[key] = 1
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        target = tmp_path / (key + ".json")
        atomic_json(target, changed)
        assert cli("--cold-replay", target).returncode == 1
    assert cli("--date", "wrong").returncode == 2
    assert cli("--private-control").returncode == 2
    assert not e.replay(tmp_path / "missing")


def test_actual_failed_child_and_authority(tmp_path):
    """SCENARIO-VERIFY-8387-REPLAY: owned failure differs from external incompleteness."""
    root = private_root(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["validation_receipts"][0]["exit_code"] == 1
    assert e.replay(output)
    work = e.measure(root, tmp_path / "raw")
    changed = deepcopy(work)
    changed["tasks"][0]["title"] += " wrong authority"
    with pytest.raises(ValueError, match="authority"):
        e.reduce(changed, [dict(passed=True)])
    work["memory_measurements"]["parent_growth_mb"] = 501
    assert e.reduce(work, [dict(passed=True)])["verdict_class"] == "disqualified"
    assert runner.manifest(tmp_path)[0]["name"] == "owned_tests"


@pytest.fixture(scope="module")
def natural_work(tmp_path_factory):
    """One captured natural closure supports several independent tamper controls."""
    raw = tmp_path_factory.mktemp("natural")
    return e.measure(e.ROOT, raw), raw


def test_current_accounting_and_boundaries(tmp_path, natural_work):
    """SCENARIO-REPORT-8387-ACCOUNTING: direct deployment cannot inherit table or utility credit."""
    work, raw = natural_work
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8374, 8388))
    assert value["actual_executed_task_count"] == 10
    assert value["pre_gate_count"] == 3
    assert value["cascade_skip_count"] == value["missing_output_count"] == 1
    assert value["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"] == -0.00390625
    assert value["H2"]["block_bootstrap_summary"]["mean_gain"] == 0
    assert value["historical_table_boundary"]["fast_path_fraction"] == 0
    assert value["deployment_results"]["actual_continuous_training"] is None
    assert value["deployment_results"]["native_service_costs"] is None
    assert value["rows"][5]["verdict_class"] == "disqualified"
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert (
        sum(
            value[k]
            for k in ["completed_count", "failed_count", "censored_count", "excluded_count"]
        )
        == 14
    )
    assert all(not gap["closed"] for gap in value["three_prd_gaps"])


def test_natural_cold_reduction_and_primitive_tamper(tmp_path, natural_work):
    """SCENARIO-VERIFY-8387-REPLAY: rehashed primitives cannot replace task or utility meaning."""
    work, raw = natural_work
    path = tmp_path / "natural.json"
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    atomic_json(path, value)
    assert e.replay(path)
    for control in ["absent", "task", "utility", "history"]:
        changed = deepcopy(work)
        if control == "absent":
            changed["inputs"][6]["summary"]["selected"] = dict(invented_cost=0)
        elif control == "task":
            changed["inputs"][6]["task"] = deepcopy(changed["inputs"][6]["task"])
            changed["inputs"][6]["task"]["id"] = "exp8380-foreign"
        elif control == "utility":
            changed["history"][0]["selected"]["H1"]["bootstrap_summary"]["all_intended"][
                "mean_gain"
            ] = 1
        else:
            changed["history"][0]["honest_verdict"] = "complete_positive_foreign"
        destination = tmp_path / control
        candidate = e.build(
            changed, [dict(passed=True)], destination, tmp_path / (e.NAME + ".json")
        )
        atomic_json(path, candidate)
        assert not e.replay(path)


def test_actual_worker_error_and_flat_terminal_binding(tmp_path, natural_work):
    """SCENARIO-VERIFY-8387-REPLAY: real child errors and altered schema adapters reject."""
    work, _ = natural_work
    item = work["inputs"][1]
    request = tmp_path / "request.json"
    atomic_json(
        request,
        dict(
            task=item["task"],
            task_sha256="wrong",
            primary=item["primary"],
            closure=item["summary"]["closure"],
        ),
    )
    output = tmp_path / "worker.json"
    result = cli("--worker-request", request, "--worker-output", output)
    assert result.returncode == 1
    assert json.loads(output.read_bytes())["memory"]["peak_bound_mb"] == 1500
    closure = deepcopy(item["summary"]["closure"])
    terminal = e.load(closure[1])
    wrong = tmp_path / "foreign_origin.json"
    atomic_json(wrong, dict(primary_path="foreign"))
    terminal["schema_adapter_from"] = e.freeze(wrong, tmp_path / "wrong")
    path = tmp_path / "changed_adapter.json"
    atomic_json(path, terminal)
    closure[1] = dict(e.freeze(path, tmp_path / "changed"), path=closure[1]["path"])
    with pytest.raises(ValueError, match="terminal_schema_adapter_binding"):
        e.outcome(item["task"], item["primary"], tmp_path / "capture", closure)


def test_exact_authority_rejection_and_note(tmp_path, monkeypatch):
    """REQ-REPORT-8387: wrong task authority rejects; every continuation reaches the note."""
    root = private_root(tmp_path / "root")
    actual = e.contract.authority(root, tmp_path / "assessment")
    actual["tasks"] = actual["tasks"][:-1]
    with monkeypatch.context() as context:
        context.setattr(e.contract, "authority", lambda *args: actual)
        with pytest.raises(ValueError, match="authority"):
            e.measure(root, tmp_path / "foreign")
    work = e.measure(root, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", output)
    atomic_json(output, value)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    (tmp_path / "docs/research-notes").mkdir(parents=True)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n")
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: 0)
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    note = (tmp_path / "docs/research-notes/v722-capstone.md").read_text()
    assert "Three PRD gaps remain open" in note
    assert all(t["id"] in note for t in work["tasks"])
