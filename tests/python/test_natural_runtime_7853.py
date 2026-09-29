"""REQ-VERIFY-7853: byte predicates, disjoint fitting and blocked custody."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.verify import evidence_views, training_qualification, training_runtime
from carnot.verify import natural_predicates as predicates
from carnot.verify import natural_training as natural
from carnot.verify.natural_bank import NaturalBank
from carnot.reporting.current_work_receipt import canonical_hash
from scripts.experiments import experiment_7853_v682_natural_runtime as cli


def _rows(role: str, group: str, label: int, answer: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"opaque-{role}-{group}",
            "group": group,
            "role": role,
            "source": "Lumen has 12 apples. Orion is present.",
            "answer": answer,
            "label": label,
            "known": [1 - label],
        }
    ]


def test_public_byte_grammar_and_mutations() -> None:
    """SCENARIO-VERIFY-7853-BYTES: each signal has a real byte toggle."""
    base = (b"Lumen has 12 apples. Orion is present.", b"Lumen has 12 apples.")
    assert predicates.signals(*base) == (0, 0, 0, 0)
    assert predicates.signals(base[0], b"Lumen has 13 apples.")[0] == 1
    assert predicates.signals(base[0], b"Lumen has not 12 apples.")[1] == 1
    assert predicates.signals(base[0], b"Quantum zebra violin nebula.")[2] == 1
    assert predicates.signals(base[0], b"Lumen meets Vega.")[3] == 1
    assert predicates.signals(b"12, STRASSE", "１２ Straße".encode())[0] == 0
    names = predicates.NAMES
    assert len(names) == len(set(names)) == 16
    assert predicates.features(*base)[names[0]] == 1.0
    assert sum(predicates.features(*base).values()) == 1
    row = _rows("fit", "saffron", 0, "Lumen has 12 apples.")[0]
    before = natural.prepare([row], "complete_static_constrained_set")[0]
    changed = {**row, "id": "opaque-else", "label": 1, "annotation": "false"}
    after = natural.prepare([changed], "complete_static_constrained_set")[0]
    np.testing.assert_array_equal(before["a"]["x"], after["a"]["x"])


def test_historical_fixture_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7853-BYTES: retain the old fixture defect as evidence."""
    original = training_qualification.fixture_records()[0]
    changed = {**original, "id": "fixture-1"}
    left, _ = training_qualification.make_batch(
        [original], "complete_static_constrained_set", list(predicates.NAMES)
    )
    right, _ = training_qualification.make_batch(
        [changed], "complete_static_constrained_set", list(predicates.NAMES)
    )
    np.testing.assert_array_equal(left["a"]["x"][..., :132], right["a"]["x"][..., :132])
    assert not np.array_equal(left["a"]["x"][..., 132:], right["a"]["x"][..., 132:])
    seen: dict[str, object] = {}
    real_fit = training_runtime.fit
    real_calibrate = training_runtime.calibrate

    def observed_fit(*args: object, **kwargs: object) -> dict[str, object]:
        seen["same_batch"] = args[1] is args[2] is left
        seen["seed"] = args[3]
        return real_fit(*args, **kwargs)

    def observed_calibrate(*args: object, **kwargs: object) -> float:
        seen["calibration_batch"] = args[1] is left
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(training_runtime, "fit", observed_fit)
    monkeypatch.setattr(training_runtime, "calibrate", observed_calibrate)
    training_qualification.fit_arm("complete_static_constrained_set", left, epochs=1)
    assert seen == {
        "same_batch": True,
        "seed": training_runtime.SEEDS[0],
        "calibration_batch": True,
    }


def test_local_mask_response_objective_and_aggregation() -> None:
    """SCENARIO-VERIFY-7853-SPLIT: unknown local labels stay masked."""
    rows = _rows("fit", "saffron", 0, "Lumen has 12 apples.")
    batch, _ = natural.prepare(rows, "local_set")
    params = training_runtime.init_params("energy_local", 67801)
    unknown = {**batch, "known": batch["known"].at[0, 0].set(-1)}
    response = training_runtime.init_params("response_set", 67801)
    assert float(
        training_runtime.loss(response, batch, "response_set", "canonical", (0, 0))
    ) == pytest.approx(
        float(training_runtime.loss(response, unknown, "response_set", "canonical", (0, 0)))
    )
    assert float(
        training_runtime.loss(params, batch, "energy_local", "canonical", (0, 0))
    ) != pytest.approx(
        float(training_runtime.loss(params, unknown, "energy_local", "canonical", (0, 0)))
    )
    aggregate = training_runtime.temperature_risk((0.2 + 0.6) / 2, 0.5)
    premature = (
        training_runtime.temperature_risk(0.2, 0.5) + training_runtime.temperature_risk(0.6, 0.5)
    ) / 2
    assert aggregate != pytest.approx(premature)


@pytest.mark.parametrize("arm", tuple(evidence_views.ARMS))
@pytest.mark.parametrize("seed", natural.SEEDS)
def test_natural_fit_split_and_source_erasure(tmp_path: Path, arm: str, seed: int) -> None:
    """SCENARIO-VERIFY-7853-SPLIT: actual interfaces enforce split and view rules."""
    fit = _rows("fit", "saffron", 0, "Lumen has 12 apples.")
    tune = _rows("tune", "cobalt", 1, "Lumen has 13 apples.")
    with pytest.raises(ValueError, match="overlap"):
        natural.fit(fit, fit, "response_set", 67801, 0.01, 1)
    batch, excluded = natural.prepare(fit, arm)
    assert not excluded
    assert batch["a"]["x"].shape[-1] == (148 if arm == "complete_static_constrained_set" else 132)
    head = natural.fit(fit, tune, arm, seed, 0.01, 1)
    assert head["seed"] == seed and len(head["curve"]) == 1
    assert head["gradient_error"] < 1e-3
    assert head["parameter_count"] <= 4096
    assert 0 <= natural.predict(head, tune)[0]["probability_unsupported"] <= 1
    changed_metadata = {**tune[0], "id": "opaque-renamed", "label": 0, "annotation": "changed"}
    assert natural.predict(head, tune) == natural.predict(head, [changed_metadata])
    training_runtime.save(tmp_path / f"{arm}.json", head)
    cold = training_runtime.load(tmp_path / f"{arm}.json")
    assert natural.predict(head, tune) == natural.predict(cold, tune)
    erased = [{**fit[0], "source": "Completely different source 987."}]
    a = natural.prepare(fit, "source_erased_constrained_set")[0]
    b = natural.prepare(erased, "source_erased_constrained_set")[0]
    np.testing.assert_array_equal(a["a"]["x"], b["a"]["x"])
    if arm == "source_erased_constrained_set":
        assert natural.predict(head, fit) == natural.predict(head, erased)
    if arm == "complete_static_constrained_set":
        changed = {**head, "params": {**head["params"], "w": head["params"]["w"].at[132, 0].add(5)}}
        assert natural.predict(head, tune) != natural.predict(changed, tune)


def test_bank_delayed_read_only_and_corruption(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7853-BANK: pending and released state survives restart."""
    path = tmp_path / "bank.json"
    bank = NaturalBank(path)
    feature = predicates.features(b"Lumen has 12.", b"Lumen has 13.")
    forecast = bank.predict("opaque-one", 0, feature, 0.4)
    assert forecast["probability"] == pytest.approx(0.4)
    assert bank.pending == ["opaque-one"]
    with pytest.raises(ValueError, match="early"):
        bank.release("opaque-one", 0, 1)
    assert NaturalBank(path).pending == ["opaque-one"]
    bank.release("opaque-one", 1, 1)
    assert NaturalBank(path).pending == []
    before = path.read_bytes()
    bank.predict("opaque-read", 2, feature, 0.4, read_only=True)
    assert path.read_bytes() == before
    state = json.loads(path.read_text())
    state["coefficients"][0] = 0.99
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="corrupt"):
        NaturalBank(path)


def test_blocked_cli_from_disqualified_source(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7853-BLOCK: external failure is terminal and explicit."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/experiments/experiment_7853_v682_natural_runtime.py"
    command = [
        sys.executable,
        str(script),
        "--date",
        "20260929",
        "--private-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "candidate.json"),
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=90)
    assert run.returncode == 0, run.stderr
    value = json.loads((tmp_path / "candidate.json").read_text())
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["verdict_class"] == "blocked"
    assert value["natural_training_ready_score"] == 0
    assert any(
        row["artifact_field"] == "source_boundary_ready_score"
        for row in value["gate_check_summary"]
    )
    cold = subprocess.run(
        [*command[:2], "--cold-replay", str(tmp_path / "candidate.json")],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert cold.returncode == 0, cold.stderr


def test_adapter_rejects_invalid_roles_and_ineligible_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7853-SPLIT: bad roles and abstentions cannot train."""
    fit = _rows("fit", "saffron", 0, "Lumen has 12 apples.")
    tune = _rows("tune", "cobalt", 1, "Lumen has 13 apples.")
    with pytest.raises(ValueError, match="unknown arm"):
        natural.prepare(fit, "absent")
    empty = {**fit[0], "answer": ""}
    with pytest.raises(ValueError, match="no eligible"):
        natural.prepare([empty], "response_set")
    batch, excluded = natural.prepare([*fit, empty], "response_set")
    assert batch["a"]["x"].shape[0] == 1 and len(excluded) == 1
    byte_row = {**fit[0], "source": b"Lumen has 12 apples.", "answer": b"Lumen has 12 apples."}
    assert natural.prepare([byte_row], "response_set")[1] == []
    with pytest.raises(ValueError, match="budget"):
        natural.fit(fit, tune, "response_set", 4, 0.01, 1)
    with pytest.raises(ValueError, match="role"):
        natural.fit([{**fit[0], "role": "evaluation"}], tune, "response_set", 67801, 0.01, 1)
    with pytest.raises(ValueError, match="labels"):
        natural.fit(
            [{key: value for key, value in fit[0].items() if key != "label"}],
            tune,
            "response_set",
            67801,
            0.01,
            1,
        )
    with pytest.raises(ValueError, match="ineligible"):
        natural.fit(fit, [*tune, {**tune[0], "answer": ""}], "response_set", 67801, 0.01, 1)
    head = natural.fit(fit, tune, "response_set", 67801, 0.01, 1)
    with pytest.raises(ValueError, match="ineligible"):
        natural.predict(head, [*fit, empty])


def test_bank_rejects_invalid_transitions_and_admits_once(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7853-BANK: admission requires released positive evidence."""
    path = tmp_path / "bank.json"
    bank = NaturalBank(path)
    feature = predicates.features(b"Lumen has 12.", b"Lumen has 13.")
    with pytest.raises(ValueError, match="duplicate"):
        bank.predict("", 0, feature, 0.4)
    with pytest.raises(ValueError, match="features"):
        bank.predict("wrong", 0, {}, 0.4)
    with pytest.raises(ValueError, match="probability"):
        bank.predict("wrong", 0, feature, float("nan"))
    bank.predict("one", 0, feature, 0.4)
    with pytest.raises(ValueError, match="duplicate"):
        bank.predict("one", 1, feature, 0.4)
    with pytest.raises(ValueError, match="order"):
        bank.predict("two", 0, feature, 0.4)
    with pytest.raises(ValueError, match="unknown"):
        bank.release("missing", 2, 1)
    with pytest.raises(ValueError, match="label"):
        bank.release("one", 1, 2)
    with pytest.raises(ValueError, match="admission"):
        bank.admit("one", "constant")
    bank.release("one", 1, 1)
    bank.admit("one", "constant")
    with pytest.raises(ValueError, match="block"):
        bank.admit("one", "constant")
    assert NaturalBank(path).state["admitted"]["0"]["name"] == "constant"
    state = json.loads(path.read_text())
    state["names"] = []
    state["checksum"] = canonical_hash(
        {key: value for key, value in state.items() if key != "checksum"}
    )
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="grammar"):
        NaturalBank(path)


def test_cli_preflight_errors_and_cold_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7853-BLOCK: parse errors and forged receipts fail closed."""
    bad_source = tmp_path / cli.SOURCES[0][1]
    bad_source.parent.mkdir(parents=True)
    bad_source.write_text("{invalid")
    checks, _ = cli.preflight(tmp_path)
    assert any(row["artifact_field"] == "schema" and not row["passed"] for row in checks)
    with monkeypatch.context() as patch:
        patch.setattr(cli, "preflight", lambda root: ([{"passed": True}], []))
        with pytest.raises(RuntimeError, match="qualified source"):
            cli.blocked_record("20260929")
    with pytest.raises(SystemExit) as invalid_date:
        cli.main(["--date", "20260928", "--output", str(tmp_path / "unused.json")])
    assert invalid_date.value.code == 2
    candidate = tmp_path / "candidate.json"
    assert cli.main(["--date", "20260929", "--output", str(candidate)]) == 0
    assert cli.cold_replay(candidate)["valid"]
    value = json.loads(candidate.read_text())
    value["preconditions_checked"] = []
    candidate.write_text(json.dumps(value))
    assert not cli.cold_replay(candidate)["valid"]
    assert cli.main(["--cold-replay", str(candidate)]) == 1
