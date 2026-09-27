"""Focused checks for REQ-REPORT-7730 and REQ-ENERGY-7730."""

import json
from pathlib import Path

import pytest

from carnot import experiment_7730_v673_set_energy_fit as exp


def test_advisory_dictionary_is_fixed_and_label_free() -> None:
    """SCENARIO-ENERGY-7730-LEAKAGE: conjunctions come from public bytes."""
    source = b"Alpha is 12. Beta is not 30."
    answer = b"Alpha is 13. Beta is 30."
    values = exp.advisory_features(source, answer)
    assert len(exp.FEATURE_DICTIONARY) == 16
    assert len(values) == 16
    assert set(values) <= {0.0, 1.0}
    assert exp.advisory_features(source, answer) == values
    assert any(values)


def test_policy_cost_and_ties() -> None:
    """REQ-ENERGY-7730: freeze the stated costs and escalation ties."""
    assert exp.policy_action(0.0) == "accept"
    assert exp.policy_action(1.0) == "reject"
    assert exp.policy_action(0.05) == "escalate"
    assert exp.policy_action(0.75) == "escalate"


def test_training_and_reload_from_original_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7730-E2E: fit, calibrate, save and replay decisions."""
    examples = [
        (b"Alpha is 12.", b"Alpha is 12.", 0),
        (b"Alpha is 12.", b"Alpha is 13.", 1),
        (b"Beta is 30.", b"Beta is 30.", 0),
        (b"Beta is 30.", b"Beta is 31.", 1),
    ]
    head, curve = exp.fit_head("complete_static", examples, 67301, 0.05, 0.0, 12)
    assert len(curve) == 12
    assert all(row["gradient_norm"] >= 0 for row in curve)
    assert head["parameter_count"] > 0
    head["temperature"] = exp.fit_temperature(head, examples)
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head))
    loaded = json.loads(path.read_text())
    for source, answer, _ in examples:
        before = exp.typed_decision(source, answer, head)
        after = exp.typed_decision(source, answer, loaded)
        assert abs(before["probability_unsupported"] - after["probability_unsupported"]) < 1e-8
        assert before["action"] == after["action"]


def test_precondition_denies_changed_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7730-CUSTODY: modified upstream bytes fail closed."""
    role = {"public_path": "fit_public.jsonl", "public_sha256": "sha256:wrong"}
    (tmp_path / "fit_public.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        exp.verified_role_file(tmp_path, role, "public")


def test_private_producer_and_cold_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7730-E2E: producer and cold reducer use original bytes."""
    import shutil

    root = Path(__file__).resolve().parents[2]
    source = root / "results/raw/experiment_7727_v673_development_corpus"
    target = tmp_path / "results/raw/experiment_7727_v673_development_corpus"
    target.mkdir(parents=True)
    for role in exp.ROLES:
        for kind in ("public", "evaluator"):
            shutil.copy2(source / f"{role}_{kind}.jsonl", target)
    shutil.copy2(source / "development_manifest.json", target)
    for name in exp.UPSTREAM:
        shutil.copy2(root / "results" / f"{name}.json", tmp_path / "results" / f"{name}.json")
    module = tmp_path / "python/carnot" / f"{exp.NAME}.py"
    module.parent.mkdir(parents=True)
    shutil.copy2(Path(exp.__file__), module)
    monkeypatch.setattr(exp, "ARMS", ("complete_static",))
    monkeypatch.setattr(exp, "SEEDS", (67301,))
    monkeypatch.setattr(exp, "CONFIGS", ((0.05, 0.0),))
    monkeypatch.setattr(exp, "EPOCHS", 2)

    def fake_commands(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": item.name,
                "passed": True,
                "exit_code": 0,
                "timed_out": False,
                "log_sha256": "sha256:private-test",
            }
            for item in commands
        ]

    monkeypatch.setattr(exp, "run_commands", fake_commands)
    artifact = exp.run_experiment(tmp_path, "20260926")
    assert artifact["verdict_class"] == "null"
    assert artifact["set_heads_ready_score"] == 1
    raw = tmp_path / exp.RAW
    replay = exp.reduce_raw(tmp_path, raw, raw / "terminal_candidate.json")
    assert replay["families"] == 384
    assert replay["arms"] == 1
    assert (tmp_path / exp.OUTPUT).is_file()
    rows = (raw / "rows.jsonl").read_text().splitlines()
    changed = json.loads(rows[0])
    changed["probability_unsupported"] = 0.12345
    rows[0] = json.dumps(changed)
    (raw / "rows.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="prediction"):
        exp.reduce_raw(tmp_path, raw)
