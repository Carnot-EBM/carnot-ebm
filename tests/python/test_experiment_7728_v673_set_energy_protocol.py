"""REQ-VERIFY-7728 and REQ-REPORT-7728 fixture qualification."""

import json
import math
from pathlib import Path
import runpy
import sys

import pytest

from carnot.verify import source_alignment as old
from carnot.verify import source_set_energy as subject
from carnot import experiment_7728_v673_set_energy_protocol as experiment


def test_independent_locations_and_enumeration():
    """SCENARIO-VERIFY-7728-MATH: claims may use different locations."""
    view = subject.prepare(b"Alpha is 12. Beta is 30.", b"Alpha is 12. Beta is 30.")
    params = subject.zero_parameters()
    params["weights"][0][128] = 8.0
    params["weights"][1][128] = -8.0
    result = subject.distribution(view, params)
    assert result["response_support"] == pytest.approx(subject.enumerate_response(view, params))
    assert result["response_support"] > 0.5
    assert len(result["sentence_support"]) == 2
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in result["sentence_support"])
    assert result["response_support"] + result["response_unsupported"] == pytest.approx(1)
    assert subject.shared_control(view, old.zero_parameters()) == pytest.approx(
        old.distribution(view, old.zero_parameters())["marginal"]
    )


def test_duplicates_reordering_and_erasure():
    """SCENARIO-VERIFY-7728-MATH: grouped priors do not reward copies."""
    params = subject.zero_parameters()
    params["weights"][0][128] = 3.0
    params["weights"][1][128] = -3.0
    answer = b"Alpha is 12."
    one = subject.prepare(b"Alpha is 12.", answer)
    duplicate = subject.prepare(b"Alpha is 12. Alpha is 12.", answer)
    reordered = subject.prepare(b"Beta is 30. Alpha is 12.", answer)
    assert subject.distribution(one, params)["response_support"] == pytest.approx(
        subject.distribution(duplicate, params)["response_support"], abs=1e-8
    )
    assert subject.distribution(reordered, params)["response_support"] == pytest.approx(
        subject.distribution(subject.prepare(b"Alpha is 12. Beta is 30. ", answer), params)[
            "response_support"
        ],
        abs=1e-8,
    )
    assert subject.distribution(subject.prepare(b"", answer), params)["null_mass"][0] == 1


def test_bounds_bytes_and_extreme_energies():
    """SCENARIO-VERIFY-7728-BOUNDS: every original byte survives."""
    source = "Café may close. It is not closed!".encode()
    answer = "Café may close. It is not closed!".encode()
    view = subject.prepare(source, answer)
    assert b"".join(view["source_sentences"]) == source
    assert b"".join(view["answer_units"]) == answer
    params = subject.zero_parameters()
    params["bias"] = [1000.0, -1000.0]
    assert math.isfinite(subject.distribution(view, params)["response_support"])
    assert subject.prepare(b"", b"")["abstention"] == "empty_answer"
    assert subject.prepare(b"A. " * 66, b"B.")["abstention"] == "source_windows_over_budget"
    assert subject.prepare(b"A.", b"B. " * 17)["abstention"] == "answer_units_over_budget"


def test_fixture_manifest_and_cold_replay(tmp_path: Path):
    """SCENARIO-REPORT-7728-REPLAY: raw rows bind the independent fixtures."""
    manifest, rows = experiment.build_protocol(tmp_path)
    assert len(rows) == 96
    assert len({r["family_id"] for r in rows}) == 96
    assert sum(r["role"] == "held" for r in rows) == 32
    assert manifest["max_parameters"] == 4096
    assert manifest["aggregation_assumption"] == "conditional_independence"
    reduced = experiment.reduce_raw(tmp_path)
    assert reduced["ready"]
    assert reduced["observed_families"] == 96
    assert reduced["normalization_rows"]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"rows": rows, "normalization_rows": reduced["normalization_rows"]})
    )
    experiment.cold_check(tmp_path, candidate)
    altered = json.loads(candidate.read_text())
    altered["rows"][0]["response_support"] = 0
    candidate.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="candidate rows"):
        experiment.cold_check(tmp_path, candidate)


def test_preconditions_and_scope(tmp_path: Path):
    """SCENARIO-REPORT-7728-SCOPE: absent producers have exact operands."""
    checks, failures, hashes = experiment.check_preconditions(tmp_path)
    assert checks
    assert failures
    assert failures[0]["operator"] == "=="
    assert failures[0]["artifact_path"].startswith("results/")
    assert hashes["absent_sources"]
    scope = experiment.frozen_scope(tmp_path)
    assert scope["test_paths"] == [experiment.TEST_PATH]
    assert "changed_module_coverage" in scope["required_names"]


def test_invalid_energies_and_matched_pooled_heads():
    """REQ-VERIFY-7728: invalid heads fail and pooled controls share inputs."""
    view = subject.prepare(b"Alpha is 12.", b"Alpha is 12.")
    for bad_view, bad_params, message in (
        (subject.prepare(b"A.", b"B. " * 17), subject.zero_parameters(), "abstained"),
        (view, {"weights": [[]], "bias": [0, 0]}, "parameter shape"),
        (view, {"weights": [[0] * 132 for _ in range(2)], "bias": [0]}, "parameter shape"),
        (view, {"weights": [[0] * 132 for _ in range(2)], "bias": [math.inf, 0]}, "nonfinite"),
    ):
        with pytest.raises(ValueError, match=message):
            subject.distribution(bad_view, bad_params)
    with pytest.raises(ValueError, match="too large"):
        subject.enumerate_response(
            subject.prepare(b"A. B. C. D.", b"A."), subject.zero_parameters()
        )
    assert subject.pooled_control(view, old.zero_parameters(), "logistic") == pytest.approx(
        [0.5, 0.5]
    )
    mlp = subject.zero_mlp_parameters()
    assert mlp["parameter_count"] <= 4096
    assert subject.pooled_control(view, mlp, "mlp") == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="unknown control"):
        subject.pooled_control(view, mlp, "bad")
    mlp["parameter_count"] = 5000
    with pytest.raises(ValueError, match="MLP budget"):
        subject.pooled_control(view, mlp, "mlp")


def test_raw_reducer_rejects_each_corruption(tmp_path: Path):
    """SCENARIO-REPORT-7728-REPLAY: raw input and metric corruption fail closed."""
    experiment.build_protocol(tmp_path)
    path = tmp_path / "rows.json"
    baseline = path.read_text()
    for key, replacement, message in (
        ("source_hex", "00", "fixture input"),
        ("response_support", 0, "fixture probability"),
        ("byte_retained", False, "fixture validity"),
    ):
        rows = json.loads(baseline)
        rows[0][key] = replacement
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError, match=message):
            experiment.reduce_raw(tmp_path)
    path.write_text(baseline)
    manifest = tmp_path / "manifest.json"
    old_manifest = manifest.read_text()
    value = json.loads(old_manifest)
    value["fixture_roles"]["held"] = 31
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixture count"):
        experiment.reduce_raw(tmp_path)
    manifest.write_text(old_manifest)
    reduced = experiment.reduce_raw(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "rows": reduced["rows"],
                "normalization_rows": reduced["normalization_rows"],
                "set_protocol_manifest_path": {"sha256": "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="manifest hash"):
        experiment.cold_check(tmp_path, candidate)


def _fake_inputs(root: Path) -> None:
    for relative, field, expected in experiment.INPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    field: expected,
                    "honest_verdict": "complete_null_fixture",
                    "flagged_adversarial": False,
                }
            )
        )
    module = root / experiment.CHANGED_MODULES[1]
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_bytes(Path(experiment.__file__).read_bytes())


def test_owned_run_and_terminal_disqualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7728-SCOPE: publication follows affected and terminal receipts."""
    _fake_inputs(tmp_path)
    calls = 0

    def fake_commands(_root, commands, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            {
                "name": command.name,
                "passed": calls != 2 or command.name != "adversarial_verify",
                "exit_code": int(calls == 2 and command.name == "adversarial_verify"),
                "command": "mock",
                "log_sha256": "sha256:mock",
            }
            for command in commands
        ]

    monkeypatch.setattr(experiment, "run_commands", fake_commands)
    artifact = experiment.run_experiment(tmp_path, "20260926")
    assert calls == 3
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is True
    assert (tmp_path / experiment.RESULT_PATH).read_bytes() == (
        tmp_path / experiment.RAW_PATH / "terminal_candidate.json"
    ).read_bytes()
    assert len(artifact["phase_spans"]) == 3
    checks, failures, hashes = experiment.check_preconditions(tmp_path)
    assert not failures and len(hashes["eligible_producers"]) == 2
    assert all(check["passed"] for check in checks[:2])


def test_main_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7728-REPLAY: CLI chooses cold and live paths."""
    experiment.build_protocol(tmp_path)
    reduced = experiment.reduce_raw(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"rows": reduced["rows"], "normalization_rows": reduced["normalization_rows"]})
    )
    assert experiment.main(["--cold-reduce", str(tmp_path), "--candidate", str(candidate)]) == 0
    with pytest.raises(SystemExit):
        experiment.main(["--cold-reduce", str(tmp_path)])
    monkeypatch.setattr(
        experiment, "run_experiment", lambda _root, _date: {"verdict_class": "blocked"}
    )
    assert experiment.main(["--date", "20260926"]) == 0
    monkeypatch.setattr(
        experiment, "run_experiment", lambda _root, _date: {"verdict_class": "disqualified"}
    )
    assert experiment.main(["--date", "20260926"]) == 1


def test_malformed_input_and_blocked_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7728-SCOPE: invalid external bytes remain blocked."""
    _fake_inputs(tmp_path)
    (tmp_path / experiment.INPUTS[0][0]).write_text("{")
    checks, failures, _hashes = experiment.check_preconditions(tmp_path)
    assert checks[0]["observed"] == "invalid_schema"
    assert failures[0]["field"] == experiment.INPUTS[0][1]
    monkeypatch.setattr(
        experiment,
        "run_commands",
        lambda _root, commands, **_kwargs: [
            {
                "name": command.name,
                "passed": True,
                "exit_code": 0,
                "command": "mock",
                "log_sha256": "sha256:mock",
            }
            for command in commands
        ],
    )
    artifact = experiment.run_experiment(tmp_path, "20260926")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["set_protocol_ready_score"] == 0
    assert artifact["gate_check_summary"][0]["observed"] == "invalid_schema"


def test_module_entry_cold_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SCENARIO-REPORT-7728-REPLAY: module entry checks saved raw bytes."""
    experiment.build_protocol(tmp_path)
    reduced = experiment.reduce_raw(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"rows": reduced["rows"], "normalization_rows": reduced["normalization_rows"]})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [experiment.__file__, "--cold-reduce", str(tmp_path), "--candidate", str(candidate)],
    )
    with pytest.raises(SystemExit) as done:
        runpy.run_module("carnot.experiment_7728_v673_set_energy_protocol", run_name="__main__")
    assert done.value.code == 0
