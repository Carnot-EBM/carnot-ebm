"""REQ-REPORT-8513, REQ-VERIFY-8513: historical custody must survive cold replay."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v732_strongest_baseline_audit as e
from carnot.reporting import v732_strongest_baseline_runner as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Real sealed operands avoid replacing the science with favorable fixtures."""
    raw = tmp_path_factory.mktemp("strongest-baseline")
    return e.measure(e.ROOT, raw), raw


def test_original_losses_roles_and_manifest(measured):
    """SCENARIO-REPORT-8513-CUSTODY: all old losses and original roles agree."""
    work, raw = measured
    assert not work["failures"] and not work["owned_failure"]
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["comparator_inputs_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["intended_count"] == value["completed_count"] == 7500
    assert value["independent_count"] == 100
    assert len(value["raw_vs_full_question_rows"]) == 100
    original = e.read(dict(path=str(e.ROOT / e.AUDIT), sha256=e.AUDIT_PIN))
    assert value["historical_arm_metrics"]["A1"] == original["A1_metrics"]
    assert value["historical_arm_metrics"]["A2"] == original["A2_sensitivity"]
    assert value["identity_offset_already_present"] is True
    gates = {g["field"]: g for g in value["gate_check_summary"]}
    assert gates["methods_ready_score"]["observed"] == 1
    assert gates["source_inputs_ready_score"]["observed"] == 1
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    manifest = e.read(value["comparator_manifest_reference"])
    assert manifest["arms"] == list(e.COMPARATORS)
    assert manifest["answers"] == 900 and manifest["questions"] == 300
    assert {k: len(v) for k, v in manifest["question_roles"].items()} == dict(
        fit=150, tune=50, test=100
    )
    assert len(manifest["slots"]) == 22500
    assert manifest["identity_brier"]["temperature"] == 1
    assert manifest["identity_brier"]["alpha_grid"][0] == 0
    assert manifest["optimization_diagnostics"]["steps"] == [0, 25, 50, 100, 200]
    assert all(
        r["status"] == "future_dependency"
        for r in manifest["slots"]
        if r["arm"].startswith("identity_")
    )


def test_question_differences_are_fixed():
    """SCENARIO-REPORT-8513-CUSTODY: losses average siblings and seeds within question."""
    rows = [
        dict(
            unit_id=f"a{i}",
            question_hash=f"q{i // 2}",
            arm=arm,
            seed=s,
            probability=p,
            y=0,
            y_sensitivity=0,
        )
        for i in range(4)
        for s in e.n.SEEDS
        for arm, p in [("raw_p2", 0.2), (e.n.FULL, 0.1)]
    ]
    paired = e.question_differences(rows)
    assert len(paired) == 2
    assert paired[0]["A1"]["full_minus_raw_brier"] == pytest.approx(-0.03)
    assert paired[0]["A1"]["full_minus_raw_typed_loss"] == 0
    rows[0]["probability"] = None
    assert e.question_differences(rows)[0]["A1"]["missing_reason"]
    assert e.question_differences([]) == []


def test_missing_and_owned_failure(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8513-CLI: absence blocks, failed owned checks disqualify."""
    work, raw = measured
    value = e.build(work, [dict(name="failed", passed=False)], raw, raw / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["comparator_inputs_ready_score"] == 0
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "missing", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 7500
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert any(g.get("upstream") == e.fit.TASK for g in value["gate_check_summary"])
    monkeypatch.setattr(e, "evaluate", lambda *_: (_ for _ in ()).throw(ValueError("owned")))
    assert e.measure(e.ROOT, tmp_path / "failed")["owned_failure"]


def test_rehashed_manifest_rejected(measured, tmp_path):
    """SCENARIO-VERIFY-8513-COLD: local rehashing cannot change the original donor map."""
    work, raw = measured
    w = deepcopy(work)
    manifest = e.read(work["comparator_manifest_reference"])
    manifest["slots"][0]["question_hash"] = "tampered"
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    w["comparator_manifest_reference"] = e.reference(path)
    w["primitive_files"] = [
        r for r in w["primitive_files"] if r["path"] != str(raw / "comparator_manifest.json")
    ]
    w["primitive_files"].append(e.reference(path))
    atomic_json(tmp_path / "measurement.json", w)
    value = e.build(w, [], tmp_path, tmp_path / (e.NAME + ".json"))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)


def test_literature_mapping(measured):
    """SCENARIO-REPORT-8513-METHODS: adoption and deferred theorem claims stay explicit."""
    work, _ = measured
    mapping = work["literature_mapping"]
    assert {r["arxiv_id"] for r in mapping} == {
        "2609.39229",
        "2610.10627",
        "2603.22167",
        "2602.02056",
    }
    assert all(r["adopt"] and r["defer"] and r["inspected_scope"] for r in mapping)
    assert all(r["references"] for r in mapping)


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8513-CLI: real children publish, replay and reject invalid inputs."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261014", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261013"], 2),
        (["--private-control"], 2),
        (
            [
                "--private-control",
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "blocked" / (e.NAME + ".json")),
            ],
            0,
        ),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(
            f"cli{i}", [*argv, *args], tmp_path / "logs", expected=expected, deadline=120
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["comparator_inputs_ready_score"] == 1
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert terminal["consumer"]["gates"][0]["artifact_field"] == "comparator_inputs_ready_score"
    bound = e.read_bound_sidecar(output, Path(terminal["publication"]["sidecar_path"]))
    assert bound["report"]["passed"]
    value["raw_vs_full_question_rows"][0]["A1"]["full_minus_raw_brier"] = 99
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(tmp_path / "changed.json", value)
    assert child(
        "tampered",
        [*argv, "--cold-replay", str(tmp_path / "changed.json")],
        tmp_path / "logs",
        expected=1,
        deadline=120,
    )["passed"]


def test_validation_manifest(tmp_path, monkeypatch):
    """REQ-VERIFY-8513: inherit exact E2E node unions and genuine child coverage."""
    calls = []

    def collect(private, suite):
        calls.append(suite)
        return [suite + "::test_one"], dict(name="collect", argv=["true"])

    monkeypatch.setattr(runner.collected, "collect", collect)
    plan = runner.manifest(tmp_path)
    assert calls[0] == e.TEST
    historical = [p for p in plan if p["name"].startswith("historical_")]
    assert historical and all("--no-cov" in p["argv"] for p in historical)
    assert any(p["name"] == "coverage_report" and "--fail-under=100" in p["argv"] for p in plan)
    assert all(p["deadline"] <= 1200 for p in plan)


@pytest.mark.parametrize(
    "field", ["question_roles", "donor_mappings", "operator_manifest", "class_support"]
)
def test_repaired_control_binding_rejected(field, monkeypatch):
    """SCENARIO-REPORT-8513-CUSTODY: the manifest cannot substitute original bindings."""
    controls = e.read(dict(path=str(e.ROOT / e.fit.UPSTREAM), sha256=e.fit.UPSTREAM_PIN))
    controls[field] = {}
    original_read = e.read

    def changed(ref):
        if ref.get("sha256") == e.fit.UPSTREAM_PIN:
            return controls
        return original_read(ref)

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.comparator_manifest(e.ROOT)


def test_original_prediction_and_source_drift(monkeypatch):
    """SCENARIO-VERIFY-8513-COLD: raw prediction and source hashes retain independent seals."""
    with monkeypatch.context() as m:
        m.setattr(e.history, "predict_rows", lambda *_: [])
        with pytest.raises(ValueError, match="original_prediction_custody"):
            e.operands(e.ROOT)
    original_sha = e.sha256_file
    with monkeypatch.context() as m:
        m.setattr(
            e,
            "sha256_file",
            lambda p: "wrong" if p.name == "v730-methods-protocol.json" else original_sha(p),
        )
        with pytest.raises(ValueError, match="original_source_or_method_custody"):
            e.operands(e.ROOT)


def test_feature_basis_and_loss_drift(monkeypatch):
    """SCENARIO-REPORT-8513-CUSTODY: repaired features, basis and metrics remain rejected."""
    original_freeze = e.fit.controls.freeze
    with monkeypatch.context() as m:

        def changed(*args):
            value = original_freeze(*args)
            value["rows"][0]["q"] = 0.99
            return value

        m.setattr(e.fit.controls, "freeze", changed)
        with pytest.raises(ValueError, match="raw_feature_donor_binding"):
            e.comparator_manifest(e.ROOT)
    original_sha = e.sha256_file
    cached_operands = e.operands(e.ROOT)
    with monkeypatch.context() as m:
        m.setattr(e, "operands", lambda *_: cached_operands)
        m.setattr(
            e,
            "sha256_file",
            lambda p: "wrong" if p.name == "sparse_energy_7996.py" else original_sha(p),
        )
        with pytest.raises(ValueError, match="original_basis_binding"):
            e.comparator_manifest(e.ROOT)
    with pytest.raises(ValueError, match="historical_loss_reproduction"):
        e.evaluate([], [])


def test_comparator_and_literature_failure(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8513: failed owned custody cannot retain input readiness."""
    original_sha = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "wrong" if p.name == "raim_paper.txt" else original_sha(p)
    )
    work = e.measure(e.ROOT, tmp_path / "bad-literature")
    assert work["owned_failure"]
    value = e.build(
        work, [dict(passed=True)], tmp_path / "bad-literature", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "disqualified" and value["comparator_inputs_ready_score"] == 0


def test_rehashed_literature_rejected(measured, tmp_path):
    """SCENARIO-VERIFY-8513-COLD: a local method description cannot replace inspected sources."""
    work, _ = measured
    w = deepcopy(work)
    w["literature_mapping"][0]["adopt"] = "new threshold from test labels"
    atomic_json(tmp_path / "measurement.json", w)
    value = e.build(w, [], tmp_path, tmp_path / (e.NAME + ".json"))
    atomic_json(tmp_path / "candidate.json", value)
    assert not e.replay(tmp_path / "candidate.json")


def test_current_methods_publication_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8513: a present but unqualified methods receipt differs from absence."""
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    result = e.authority(e.ROOT, tmp_path / "authority")
    assert not result["activated"]
    assert any(r["observed"] == "methods_publication" for r in result["gate_check_summary"])
