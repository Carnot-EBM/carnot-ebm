"""REQ-REPORT-8528, REQ-VERIFY-8528: fixed optimization and immutable predictions."""

from copy import deepcopy
from pathlib import Path
import sys
import shutil

import numpy as np
import pytest

from carnot.verify import v733_identity_brier as n
from carnot.verify import v733_identity_brier_fit as e
from carnot.reporting import v733_identity_brier_runner as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Actual sealed operands prevent a convenient fixture from replacing historical custody."""
    raw = tmp_path_factory.mktemp("identity8528")
    return e.measure(e.ROOT, raw), raw


def test_numeric_contract():
    """SCENARIO-REPORT-8528-FIT: identity endpoints and ties stay exact."""
    raw = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    assert n.mix(raw, 1 - raw, 0).tobytes() == raw.tobytes()
    assert n.choose_alpha({str(a): 1 for a in n.ALPHAS}) == 0
    assert n.energy(0)["limiting_energy"] == "E1=+infinity"
    assert n.energy(1)["limiting_energy"] == "E0=+infinity"
    assert n.energy(0.5)["limiting_energy"] is None
    assert n.actions(raw).tolist() == ["accept", "accept", "escalate", "reject", "reject"]


def test_frozen_fit(measured):
    """SCENARIO-REPORT-8528-FIT: fixed full-batch budgets produce complete qualified rows."""
    work, raw = measured
    assert not work["failures"] and not work["owned_failure"]
    result = work["result"]
    assert result["optimizer_steps"] == 3000
    assert result["test_label_reads"] == 0
    assert result["gradient_checks"]["passed"]
    assert result["parameter_counts"] == dict(
        raw_p2=0, legacy_full_source=109, brier_scalar=13, brier_full=109, brier_source_permuted=109
    )
    for arm in n.TRAINABLE:
        for h in result["heads"][arm]:
            assert h["temperature"] == 1 and h["optimizer_steps"] == 200
            assert [r["step"] for r in h["prefixes"]] == [0, 25, 50, 100, 200]
            assert len(h["prefixes"][0]["fit"]["per_question_brier"]) == 150
            assert len(h["prefixes"][0]["tune"]["per_question_brier"]) == 50
            assert h["prefixes"][0]["coefficient_norm"] == 0
        assert result["duplicate_seed_heads"][arm]["unique_heads"] == 1
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["identity_heads_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["completed_count"] == value["intended_count"] == 7500
    assert value["independent_count"] == 100
    assert value["test_label_reads"] == 0
    assert value["seal_clock"]["heads_sealed_ns"] <= value["seal_clock"]["predictions_sealed_ns"]
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])


def test_rejections(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8528-CONTROLS: absence blocks; failed owned work disqualifies."""
    work = e.measure(tmp_path / "missing", tmp_path / "absent")
    value = e.build(work, [dict(passed=True)], tmp_path / "absent", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 7500
    assert any(r["observed"] is None for r in value["gate_check_summary"])
    original, raw = measured
    assert (
        e.build(original, [dict(passed=False)], raw, tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )
    upstream, view = e.operands(e.ROOT)
    view["rows"][0]["role"] = "test"
    with pytest.raises(ValueError, match="test_label_boundary"):
        e.train(view, [])
    monkeypatch.setattr(e, "train", lambda *_: (_ for _ in ()).throw(ValueError("owned")))
    assert e.measure(e.ROOT, tmp_path / "owned")["owned_failure"]


@pytest.mark.parametrize(
    "mode", ["valid", "aggregate", "head", "prediction", "saved_prediction", "source"]
)
def test_replay(measured, tmp_path, mode):
    """SCENARIO-VERIFY-8528-CONTROLS: repairing local hashes cannot alter numeric evidence."""
    original, _ = measured
    work = deepcopy(original)
    for ref in work["primitive_files"]:
        shutil.copyfile(ref["path"], tmp_path / Path(ref["path"]).name)
    work["primitive_files"] = [
        e.reference(tmp_path / Path(ref["path"]).name) for ref in work["primitive_files"]
    ]
    if mode == "head":
        work["result"]["heads"]["brier_scalar"][0]["parameters"][0] += 0.01
        atomic_json(tmp_path / "frozen_heads.json", work["result"])
    if mode == "prediction":
        work["decisions"][0]["probability"] = 0.123
        atomic_json(tmp_path / "predictions.json", dict(rows=work["decisions"]))
    if mode == "saved_prediction":
        saved = deepcopy(work["decisions"])
        saved[0]["probability"] = 0.123
        atomic_json(tmp_path / "predictions.json", dict(rows=saved))
    if mode == "source":
        work["source_refs"][0]["sha256"] = "wrong"
    work["primitive_files"] = [e.reference(Path(ref["path"])) for ref in work["primitive_files"]]
    atomic_json(tmp_path / "measurement.json", work)
    receipt = child("receipt", [sys.executable, "-c", "print('real')"], tmp_path / "logs")
    value = e.build(work, [receipt], tmp_path, tmp_path / (e.NAME + ".json"))
    if mode == "aggregate":
        value["completed_count"] -= 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert e.replay(tmp_path / "candidate.json") == (mode == "valid")


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8528-CONTROLS: actual children publish, replay and reject bad invocations."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for i, (args, expected) in enumerate(
        [
            (["--private-control", "--output", str(output)], 0),
            (["--cold-replay", str(output)], 0),
            (["--cold-replay", str(tmp_path / "missing")], 1),
            (["--deliberate-error"], 7),
            (["--date", "20261014"], 2),
            (["--private-control"], 2),
            (
                [
                    "--private-control",
                    "--root",
                    str(tmp_path / "missing"),
                    "--output",
                    str(tmp_path / "blocked" / (e.NAME + ".json")),
                ],
                0,
            ),
        ]
    ):
        receipt = child(
            f"cli{i}", [*argv, *args], tmp_path / "cli", expected=expected, deadline=120
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()


def test_manifest(tmp_path):
    """REQ-VERIFY-8528: original node union remains exact and disjoint before execution."""
    plan = runner.manifest(tmp_path)
    collected = [
        n for r in plan if r["name"].startswith("collect_") for n in r["collected_node_ids"]
    ]
    executed = [
        n for r in plan if r["name"].startswith("historical_") for n in r["collected_node_ids"]
    ]
    assert len(executed) == len(set(executed))
    assert set(executed) == {n for n in collected if not n.startswith(e.TEST + "::")}


@pytest.mark.parametrize("fault", ["qualification", "registration", "authority", "scaler"])
def test_operand_rejections(fault, monkeypatch):
    """SCENARIO-VERIFY-8528-CONTROLS: independent seals reject unqualified or altered operands."""
    original_read = e.read

    def read(ref):
        value = deepcopy(original_read(ref))
        if fault == "registration" and "identity_brier" in value:
            value["identity_brier"]["steps"] = 199
        if fault == "scaler" and "heads" in value:
            value["scaler"]["maximum"][0] = 99
        return value

    monkeypatch.setattr(e, "read", read)
    if fault == "qualification":
        monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    if fault == "authority":
        monkeypatch.setattr(e, "sha256_file", lambda _: "wrong")
    with pytest.raises(ValueError):
        e.operands(e.ROOT)


def test_gradient_failure_and_identity(measured, monkeypatch):
    """SCENARIO-REPORT-8528-FIT: tune fallback is admissible and derivative failure rejects fitting."""
    _, view = e.operands(e.ROOT)
    scaler = view["legacy"]["scaler"]
    row = dict(question_hash="q", x=[0.2] * 9, y=1)
    h = n.fit("brier_scalar", 17, scaler, [row], [dict(row, y=0)])
    assert h["alpha"] == 0
    work, raw_path = measured
    identity = deepcopy(work["result"])
    for arm in n.TRAINABLE:
        for head in identity["heads"][arm]:
            head["alpha"] = 0
    rows = e.predictions(identity, view["panel"])
    raw = {(r["answer_id"], r["seed"]): r["probability"] for r in rows if r["arm"] == "raw_p2"}
    assert all(
        r["probability"] == raw[(r["answer_id"], r["seed"])]
        for r in rows
        if r["arm"] in n.TRAINABLE
    )
    identity_work = dict(work, result=identity, decisions=rows)
    assert (
        e.build(identity_work, [dict(passed=True)], raw_path, raw_path / (e.NAME + ".json"))[
            "identity_heads_ready_score"
        ]
        == 1
    )
    monkeypatch.setattr(n, "gradient_checks", lambda _: dict(passed=False))
    with pytest.raises(ValueError, match="brier_gradient_check"):
        e.train(view, [])


def test_original_authority_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8528: private historical tests cannot substitute locally changed authority."""
    monkeypatch.setattr(runner, "sha256_file", lambda _: "wrong")
    with pytest.raises(ValueError, match="original_fit_authority"):
        runner.original_workspace(tmp_path)
