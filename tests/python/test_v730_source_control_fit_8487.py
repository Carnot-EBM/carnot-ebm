"""REQ-REPORT-8487, REQ-VERIFY-8487: authenticate and fit scalar controls only."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import v730_source_control_fit as e
from carnot.reporting import v730_source_fit_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Sealed real producers provide meaningful tests of custody and numerical work."""
    raw = tmp_path_factory.mktemp("source-fit")
    return e.measure(e.ROOT, raw), raw


def test_frozen_heads_and_predictions(measured):
    """SCENARIO-REPORT-8487-FREEZE: only the new scalar head is trained."""
    work, raw = measured
    assert not work["failures"]
    result = work["result"]
    assert set(result["heads"]) == set(e.ARMS)
    assert set(result["temperature_choices"]) == set(result["coefficient_changes"]) == set(e.ARMS)
    upstream, view = e.operands(e.ROOT)
    old = view["original_heads"]
    for arm, original in e.INHERITED.items():
        assert result["heads"][arm] == old["heads"][original]
    scalar = result["heads"]["scalar_cubic_spline"]
    assert {h["parameter_count"] for h in scalar} == {13}
    assert {h["optimizer_steps"] for h in scalar} == {200}
    assert all(h["final_loss"] < h["initial_loss"] for h in scalar)
    assert all(h["duration_s"] > 0 for h in scalar)
    assert all(e.numeric_checks(h, np.array([0.01, 0.14, 0.5, 0.95]))["passed"] for h in scalar)
    assert len(work["decisions"]) == 7500
    assert not any("y" in row for row in work["decisions"])
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["head_controls_ready_score"] == 1
    assert value["head_parameter_counts"] == dict(zip(e.ARMS, [0, 2, 13, 109, 109], strict=True))
    assert value["test_label_reads"] == 0
    assert value["independent_count"] == 200
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["verdict_class"] == "circular_positive"
    assert upstream["source_controls_ready_score"] == 1


def test_thresholds_ties_and_information_boundary():
    """REQ-VERIFY-8487: exact fixed thresholds and source blindness prevent adaptive tuning."""
    assert [e.decision(p)["decision"] for p in [0, 0.2, 0.21, 0.79, 0.8, 1]] == [
        "accept",
        "accept",
        "escalate",
        "escalate",
        "reject",
        "reject",
    ]
    assert e.old.choose_temperature({"0.5": 1, "1.0": 1, "2.0": 1}) == 1
    assert e.old.choose_temperature({"0.5": 1, "1.0": 2, "2.0": 1}) == 0.5
    _, view = e.operands(e.ROOT)
    changed = deepcopy(view)
    for row in changed["rows"]:
        row["x"][1:] = [999] * 8
    a, b = e.train(view, []), e.train(changed, [])
    assert e.semantic(a) == e.semantic(b)
    changed["rows"][0]["role"] = "test"
    with pytest.raises(ValueError, match="test_label_boundary"):
        e.train(changed, [])


def test_missing_and_owned_failures(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8487-CUSTODY: missing inputs differ from failed owned work."""
    work, raw = measured
    value = e.build(work, [dict(passed=False)], raw, raw / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and value["head_controls_ready_score"] == 0
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "missing", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 25
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    monkeypatch.setattr(e, "numeric_checks", lambda *_: dict(passed=False))
    failed = e.measure(e.ROOT, tmp_path / "failed")
    assert failed["owned_failure"]
    assert not e.replay(tmp_path / "absent.json")


def test_private_cli_replay_and_rehashed_controls(tmp_path):
    """SCENARIO-VERIFY-8487-CLI: real children prove bitwise replay and refusal paths."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261013", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", argv + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["head_controls_ready_score"] == 1
    assert value["fresh_process_bitwise_agreement"] is True
    changed = deepcopy(value)
    changed["head_controls_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)


def test_manifest_exact_union(tmp_path):
    """REQ-VERIFY-8487: collected historical assertions execute once within finite bounds."""
    plan = r.manifest(tmp_path)
    for collection in [p for p in plan if p["name"].startswith("collect_")]:
        nodes = collection["collected_node_ids"]
        if nodes[0].startswith(e.TEST):
            continue
        actual = [
            n
            for p in plan
            if not p["name"].startswith("collect_")
            for n in p.get("collected_node_ids", [])
            if n.split("::")[0] == nodes[0].split("::")[0]
        ]
        assert actual == nodes and len(actual) == len(set(actual))


@pytest.mark.parametrize("fault", ["qualification", "authority", "method", "scaler", "trainer"])
def test_independent_operand_rejections(fault, monkeypatch):
    """SCENARIO-REPORT-8487-CUSTODY: independent seals reject changed method and fit operands."""
    original_read, original_hash = e.read, e.sha256_file
    protocol = e.methods.protocol()
    authority_paths = {ref["path"] for ref in protocol["historical_authorities"]["V726"]}

    def read(ref):
        value = deepcopy(original_read(ref))
        if fault == "scaler" and "operators" in value and "rows" in value:
            value["operators"]["scaler"]["maximum"][0] = 100
        if fault == "trainer" and "reference" in value and "label_reads" in value:
            value["reference"]["source_path"] = "forged"
        return value

    def hashed(path):
        if fault == "authority" and str(path) in authority_paths:
            return "sha256:forged"
        if fault == "method" and str(path).endswith("sparse_energy_7996.py"):
            return "sha256:forged"
        return original_hash(path)

    monkeypatch.setattr(e, "read", read)
    monkeypatch.setattr(e, "sha256_file", hashed)
    if fault == "qualification":
        monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    expected = dict(
        qualification="source_controls_not_qualified",
        authority="original_authority_pin",
        method="original_method_hash",
        scaler="original_scaler_knots",
        trainer="original_fit_tune_pin",
    )
    with pytest.raises(ValueError, match=expected[fault]):
        e.operands(e.ROOT)


@pytest.mark.parametrize(
    "fault",
    [
        "operand",
        "snapshot",
        "authority",
        "upstream",
        "head",
        "decision",
        "trainer",
        "missing",
        "receipt",
        "bits",
    ],
)
def test_rehashed_primitive_rejections(fault, measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8487: repaired envelope hashes cannot hide changed primitive arithmetic."""
    source_work, _ = measured
    work = deepcopy(source_work)
    raw = tmp_path / "raw"
    raw.mkdir()
    for ref in work["primitive_files"]:
        target = raw / Path(ref["path"]).name
        target.write_bytes(Path(ref["path"]).read_bytes())
    work["primitive_files"] = [
        e.reference(raw / Path(ref["path"]).name) for ref in work["primitive_files"]
    ]
    receipt = child("operand", [sys.executable, "-c", "print('sealed log')"], raw / "logs")
    if fault == "operand":
        work["source_refs"][0]["sha256"] = "sha256:forged"
    if fault == "snapshot":
        work["authority"]["authority_snapshots"]["design"]["sha256"] = "sha256:forged"
    if fault == "authority":
        work["authority"]["activated"] = not work["authority"]["activated"]
    if fault == "upstream":
        work["upstream_reference"]["sha256"] = "sha256:forged"
    if fault == "head":
        work["result"]["heads"]["scalar_cubic_spline"][0]["parameters"][0] += 0.01
        atomic_json(raw / "frozen_heads.json", work["result"])
    if fault in {"decision", "bits"}:
        work["decisions"][0]["probability"] = 0.0 if fault == "bits" else 0.7
        saved = deepcopy(work["decisions"])
        if fault == "bits":
            saved[0]["probability"] = -0.0
            monkeypatch.setattr(e, "predictions", lambda *_: work["decisions"])
        atomic_json(raw / "typed_decisions.json", dict(rows=saved, test_label_reads=0))
    if fault == "trainer":
        operands = json.loads((raw / "trainer_operands.json").read_text())
        operands["view"]["rows"][0]["y"] = 1 - operands["view"]["rows"][0]["y"]
        atomic_json(raw / "trainer_operands.json", operands)
    if fault == "missing":
        work["result"] = None
    if fault == "receipt":
        Path(receipt["stdout_path"]).write_text("changed log")
    work["primitive_files"] = [e.reference(Path(ref["path"])) for ref in work["primitive_files"]]
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], raw, output))
    assert not e.replay(output)


def test_temperature_and_regularization_negative_controls(measured, monkeypatch):
    """REQ-VERIFY-8487: omitted temperature and L2 produce independently detected derivatives."""
    work, _ = measured
    head = work["result"]["heads"]["scalar_cubic_spline"][0]
    original = e.objective
    probe = np.asarray([0.01, 0.14, 0.5, 0.95])
    monkeypatch.setattr(e, "objective", lambda h, q, y: original(dict(h, temperature=1.0), q, y))
    assert not e.numeric_checks(head, probe)["passed"]
    monkeypatch.setattr(
        e,
        "objective",
        lambda h, q, y: original(h, q, y) - 0.001 * np.sum(np.asarray(h["parameters"]) ** 2),
    )
    assert not e.numeric_checks(head, probe)["passed"]


def test_private_blocked_cli(tmp_path):
    """SCENARIO-VERIFY-8487-CLI: external absence produces validated blocked terminal bytes."""
    output = tmp_path / "results" / (e.NAME + ".json")
    receipt = child(
        "blocked",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--private-control",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["head_controls_ready_score"] == 0
    assert e.replay(output)
