"""REQ-REPORT-8346 / REQ-VERIFY-8346: frozen input custody, never new inference."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v720_frozen_input_contract as e
from carnot.reporting import v720_frozen_input_runner as r
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def cli(*args):
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    private = tmp_path_factory.mktemp("v720")
    output = private / (e.NAME + ".json")
    result = cli("--date", "20261009", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    return output, value, work


def test_natural_reuse(current):
    """SCENARIO-REPORT-8346-INPUTS: qualified nulls supply custody, not H1/H2."""
    output, value, work = current
    assert e.replay(output)
    assert value["current_contract_ready_score"] == 1
    assert value["frozen_heads_ready_score"] == value["frozen_predictions_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert len(value["rows"]) == 14 and len(value["future_dependencies"]) == 13
    assert value["protocol_sha256"] == inputs.PIN
    assert value["input_receipts"]["reserved_labels_parsed"] == 0
    assert value["role_counts"]["fit"]["feature_rows"] == 105
    assert {k: v["usable"] for k, v in value["role_counts"].items()} == inputs.USABLE
    assert value["methods_manifest"]["science_changed"] is False
    assert value["frozen_policy"]["selected_comparator"] == "RBF34"
    assert all(p["operand_root"] and p["milestone"] for p in value["historical_consumers"])
    assert all(k in value["field_principles"] for k in value if k != "field_principles")
    assert work["inputs"]["heads"] and work["inputs"]["predictions"]


def test_independent_gates(current, tmp_path):
    """SCENARIO-REPORT-8346-INPUTS: history and activation are separate gates."""
    _, value, work = current
    changed = deepcopy(work)
    changed["contract"]["activated"] = False
    changed["failures"].append(e.failure(tmp_path / "upstream", "activation", True, None))
    raw = tmp_path / "independent"
    atomic_json(raw / "measurement.json", changed)
    result = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    assert result["verdict_class"] == "blocked"
    assert result["current_contract_ready_score"] == 0
    assert result["frozen_heads_ready_score"] == result["frozen_predictions_ready_score"] == 1
    failed = e.build(changed, [dict(passed=False)], raw, tmp_path / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified"
    assert not failed["frozen_heads_ready_score"] and not failed["frozen_predictions_ready_score"]


@pytest.mark.parametrize("mutation", ["prompt", "delete"])
def test_authority_mutations(tmp_path, mutation):
    """SCENARIO-REPORT-8346-INPUTS: full task objects own execution authority."""
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    plan = yaml.safe_load((root / e.ACTIVE).read_bytes())
    if mutation == "prompt":
        plan["tasks"][0]["prompt"] += " changed"
    else:
        plan["tasks"].pop()
    (root / e.ACTIVE).write_text(yaml.safe_dump(plan))
    assert not e.authority(root, tmp_path / "authority")["activated"]


def test_cli_boundaries(tmp_path):
    """SCENARIO-VERIFY-8346-REPLAY: actual historical CLI uses preserved authority."""
    for milestone in ["2026.10.716", "2026.10.717", "2026.10.718", "2026.10.719", e.MILESTONE]:
        result = cli("--inspect-milestone", milestone)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout.splitlines()[-1])["count"] == 14
    assert cli("--inspect-milestone", e.MILESTONE, "--check-authority").returncode == 0
    assert (
        cli("--root", tmp_path, "--inspect-milestone", e.MILESTONE, "--check-authority").returncode
        == 1
    )
    assert cli("--date", "wrong").returncode != 0
    assert cli("--date").returncode != 0
    assert cli("--private-fixture").returncode != 0


@pytest.mark.parametrize(
    "field", ["role_counts", "frozen_heads_ready_score", "canonical_tasks_sha256"]
)
def test_rehashed_summary(current, tmp_path, field):
    """SCENARIO-VERIFY-8346-REPLAY: recomputed hashes cannot authorize summaries."""
    _, value, _ = current
    changed = deepcopy(value)
    changed[field] = {}
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "tamper.json"
    atomic_json(path, changed)
    assert cli("--cold-replay", path).returncode == 1


@pytest.mark.parametrize("field", ["support", "protocol", "inputs", "methods", "contract"])
def test_primitive_recompute(current, tmp_path, field):
    """SCENARIO-VERIFY-8346-REPLAY: even a freshly reduced forgery must fail."""
    _, value, work = current
    changed = deepcopy(work)
    if field == "support":
        changed[field]["counts"]["fit"]["usable"] += 1
    elif field == "protocol":
        changed[field]["policy"]["accept_below"] = 0.9
    elif field == "inputs":
        changed[field]["heads"]["manifest"]["heads"][0]["coefficients"][0] += 0.125
    elif field == "methods":
        changed[field]["science_changed"] = True
    else:
        changed[field]["tasks"][0]["prompt"] += " forged"
    raw = tmp_path / field
    atomic_json(raw / "measurement.json", changed)
    forged = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    path = tmp_path / "forged.json"
    atomic_json(path, forged)
    assert not e.replay(path)


def test_missing_external(tmp_path):
    """SCENARIO-VERIFY-8346-REPLAY: real missing inputs produce a checked block."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", tmp_path / "absent", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["frozen_heads_ready_score"] == value["frozen_predictions_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is None
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent.json")


def test_exact_hash_gates(tmp_path):
    """SCENARIO-REPORT-8346-INPUTS: missing bytes remain distinct from zero."""
    ref = dict(path=str(tmp_path / "operand"), sha256="sha256:" + "0" * 64)
    with pytest.raises(inputs.CustodyError) as missing:
        inputs.bind(ref, tmp_path / "missing", [], parse=False)
    assert missing.value.gate["observed"] is None
    Path(ref["path"]).write_bytes(b"drifted")
    with pytest.raises(inputs.CustodyError) as drift:
        inputs.bind(ref, tmp_path / "drift", [])
    assert drift.value.gate["observed"] == sha256_file(Path(ref["path"]))


def test_primary_terminal_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8346-INPUTS: a failed terminal cannot supply readiness."""
    source = json.loads((e.ROOT / inputs.SOURCE).read_bytes())

    def failed(*args):
        args[-1].append(dict(problem="terminal"))
        return {}

    monkeypatch.setattr(inputs.custody, "authenticate", failed)
    with pytest.raises(ValueError, match="terminal"):
        inputs.primary(e.ROOT, inputs.SOURCE, "fit_support_ready_score", tmp_path, [])
    for mutation in [
        {"fit_support_ready_score": 0},
        {"verdict_class": "disqualified"},
        {"required_checks_passed": False},
        {"flagged_adversarial": True},
    ]:
        monkeypatch.setattr(inputs.custody, "authenticate", lambda *args: dict(source, **mutation))
        with pytest.raises(ValueError, match="qualified_terminal"):
            inputs.primary(e.ROOT, inputs.SOURCE, "fit_support_ready_score", tmp_path, [])


def test_recount_and_label_barrier(current, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8346-INPUTS: no reserved-target JSON reaches a decoder."""
    _, _, work = current
    source = json.loads((e.ROOT / inputs.SOURCE).read_bytes())
    reserved = Path(source["evaluator_shards"]["reserved"]["path"]).read_bytes()
    decode = inputs.json.loads

    def guarded(value, *args, **kwargs):
        assert value != reserved
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(inputs.json, "loads", guarded)
    assert (
        inputs.recount(source, work["protocol"], tmp_path / "good", [])["counts"]
        == work["support"]["counts"]
    )
    for role, field in [("fit", "usable"), ("reserved", "feature_rows"), ("later", "usable")]:
        changed = deepcopy(source)
        changed["class_support_by_role"][role][field] += 1
        with pytest.raises(ValueError):
            inputs.recount(changed, work["protocol"], tmp_path / role, [])


def test_frozen_seal_guards(current, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8346-INPUTS: changed checkpoint and label flags fail closed."""
    _, _, work = current
    source = json.loads((e.ROOT / inputs.SOURCE).read_bytes())
    head = json.loads((e.ROOT / inputs.HEAD).read_bytes())
    pred = json.loads((e.ROOT / inputs.PRED).read_bytes())
    changed = deepcopy(head)
    changed["frozen_head_manifest"]["heads"][0]["coefficients"][0] += 0.125
    with pytest.raises(ValueError, match="checkpoint_seal"):
        inputs.frozen_heads(
            changed, source, work["protocol"], work["inputs"]["support"], tmp_path, []
        )
    changed = deepcopy(head)
    seal = changed["frozen_head_manifest"]
    seal["source_manifest_reference"]["sha256"] = "wrong"
    seal["whole_model_sha256"] = canonical_hash(
        {k: v for k, v in seal.items() if k != "whole_model_sha256"}
    )
    with pytest.raises(ValueError, match="head_source_hash"):
        inputs.frozen_heads(
            changed, source, work["protocol"], work["inputs"]["support"], tmp_path, []
        )
    changed = deepcopy(pred)
    changed["prediction_rows"].pop()
    with pytest.raises(ValueError, match="prediction_seal"):
        inputs.frozen_predictions(changed, work["inputs"]["heads"], tmp_path, [])
    bound = inputs.bind

    def wrong_labels(ref, *args, **kwargs):
        value = bound(ref, *args, **kwargs)
        if ref["path"].endswith("stream_predictors.json"):
            value["evaluator_labels_opened"] = True
        return value

    monkeypatch.setattr(inputs, "bind", wrong_labels)
    with pytest.raises(ValueError, match="sealed_label_barrier"):
        inputs.frozen_predictions(pred, work["inputs"]["heads"], tmp_path, [])


def test_rehashed_methods_snapshot(current, tmp_path):
    """SCENARIO-VERIFY-8346-REPLAY: rewriting and hashing method bytes is rejected."""
    _, value, work = current
    changed = deepcopy(work)
    changed["methods"]["science_changed"] = True
    snapshot = tmp_path / "methods.json"
    atomic_json(snapshot, changed["methods"])
    ref = next(r for r in changed["refs"] if r["path"].endswith(e.METHODS))
    ref.update(
        snapshot_path=str(snapshot),
        sha256=sha256_file(snapshot),
        snapshot_sha256=sha256_file(snapshot),
    )
    raw = tmp_path / "forged"
    atomic_json(raw / "measurement.json", changed)
    candidate = tmp_path / "forged.json"
    atomic_json(
        candidate,
        e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json")),
    )
    assert not e.replay(candidate)


def test_actual_failed_owned_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8346-REPLAY: a real failed child publishes disqualification."""
    from carnot.reporting.v709_execution import child

    def fail(plan, logs):
        return [
            child(
                "deliberate_failure",
                [sys.executable, "-u", "-c", "raise SystemExit(3)"],
                logs,
                deadline=10,
            )
        ]

    monkeypatch.setattr(r.qualified, "execute", fail)
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--date", "20261009", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["validation_receipts"][0]["actual_exit"] == 3
    assert not value["required_checks_passed"]
    assert value["frozen_heads_ready_score"] == value["frozen_predictions_ready_score"] == 0
    assert e.replay(output)
