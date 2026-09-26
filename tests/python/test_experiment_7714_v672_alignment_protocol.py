"""REQ-VERIFY-7714 and REQ-REPORT-7714 protocol checks."""

import math
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.verify import source_alignment as sa
from carnot import experiment_7714_v672_alignment_protocol as experiment


def test_complete_bytes_and_windows():
    """SCENARIO-VERIFY-7714-BYTES retains UTF-8, qualifiers and context."""
    source = "Café is open. It is not open on Sunday. Hours vary by season."
    answer = "Café is open. It may close on Sunday."
    view = sa.prepare(source.encode(), answer.encode())
    assert b"".join(view["source_sentences"]) == source.encode()
    assert b"".join(view["answer_units"]) == answer.encode()
    assert any(b"not open on Sunday" in w for w in view["windows"])
    assert any(b"Hours vary by season" in w for w in view["windows"])
    assert len(view["windows"]) == 4
    assert len(view["pair_features"][0]) == sa.FEATURE_DIM
    assert view["source_bytes"] == source.encode()
    assert view["answer_bytes"] == answer.encode()


def test_bounds_are_abstentions():
    """SCENARIO-VERIFY-7714-BYTES never silently truncates families."""
    view = sa.prepare(b"One. ", b"Answer. " * 17)
    assert view["abstention"] == "answer_units_over_budget"
    view = sa.prepare(b"Source. " * 67, b"Answer.")
    assert view["abstention"] == "source_windows_over_budget"


def test_normalization_matches_enumeration_and_duplicate_invariance():
    """SCENARIO-VERIFY-7714-NORMALIZE checks joint and marginal math."""
    view = sa.prepare(b"Alpha is 12. Beta is not 13. Gamma may be 14.", b"Alpha is 13.")
    params = sa.zero_parameters()
    params["weights"][0][0] = 3.0
    params["weights"][1][0] = -2.0
    result = sa.distribution(view, params)
    assert math.isclose(sum(sum(pair) for pair in result["joint"]), 1.0, abs_tol=1e-12)
    for y in (0, 1):
        assert result["marginal"][y] == pytest.approx(sum(row[y] for row in result["joint"]))
    assert result["marginal"] == pytest.approx(sa.enumerate_marginal(view, params))
    duplicated = sa.prepare(b"Alpha is 12. Alpha is 12.", b"Alpha is 12.")
    original = sa.prepare(b"Alpha is 12.", b"Alpha is 12.")
    assert sa.distribution(duplicated, params)["marginal"] == pytest.approx(
        sa.distribution(original, params)["marginal"]
    )


def test_null_state_and_extreme_energy():
    """SCENARIO-VERIFY-7714-NORMALIZE keeps missing evidence undecided."""
    view = sa.prepare(b"", b"Maybe 7.")
    params = sa.zero_parameters()
    assert sa.distribution(view, params)["marginal"] == pytest.approx([0.5, 0.5])
    params["bias"] = [1000.0, -1000.0]
    assert all(math.isfinite(x) for x in sa.distribution(view, params)["marginal"])


def test_negative_controls_and_same_input_baselines():
    """REQ-VERIFY-7714 registers controls without source or label leakage."""
    view = sa.prepare(b"The count is 4. Not 5.", b"The count is 5.")
    erased = sa.prepare(b"", b"The count is 5.")
    assert view["pair_features"] != erased["pair_features"]
    assert sa.pooled_features(view) != sa.pooled_features(erased)
    assert sa.control_scores(view, sa.zero_parameters()) == pytest.approx([0.5, 0.5])
    assert sa.control_scores(view, sa.zero_parameters(), kind="mlp") == pytest.approx([0.5, 0.5])


def test_fixture_plan_has_independent_held_groups():
    """REQ-REPORT-7714 counts families once across overlapping views."""
    fixtures = sa.fixture_groups()
    assert len(fixtures) == 96
    assert len({item["family_id"] for item in fixtures}) == 96
    assert sum(item["role"] == "held" for item in fixtures) == 32
    assert all(item["source"] and item["answer"] for item in fixtures)


def test_missing_upstream_names_exact_operand(tmp_path):
    """SCENARIO-REPORT-7714-TERMINAL blocks missing external custody."""
    checks, failures, _hashes = experiment.check_preconditions(tmp_path)
    assert checks[0]["exists"] is False
    assert failures[0]["artifact_path"] == "results/experiment_7700_v671_record_span_protocol.json"
    assert failures[0]["field"] == "exists"
    assert failures[0]["operator"] == "=="


def test_raw_cold_reduction_detects_changed_byte(tmp_path):
    """SCENARIO-REPORT-7714-CUSTODY reopens every fixture and hash."""
    manifest, rows = experiment.build_protocol(tmp_path)
    assert manifest["fixture_count"] == 96
    assert len(rows) == 96
    assert len(experiment.reduce_raw(tmp_path)["normalization_rows"]) == 96
    rows_path = tmp_path / "rows.json"
    saved = json.loads(rows_path.read_text())
    saved[0]["source_hex"] = "00"
    rows_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="raw rows hash"):
        experiment.reduce_raw(tmp_path)


def _upstream(root: Path) -> None:
    """Make only the exact historical operands used by this protocol."""
    for relative, field, value in experiment.UPSTREAM:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({field: value, "flagged_adversarial": False, "verdict_class": "null"})
        )


def _fake_commands(root, commands, **_kwargs):
    return [
        {
            "name": command.name,
            "passed": True,
            "exit_code": 0,
            "command": "test",
            "log_sha256": "sha256:test",
        }
        for command in commands
    ]


def test_cpu_run_and_terminal_artifact(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7714-TERMINAL seals exact fixture evidence."""
    _upstream(tmp_path)
    code = tmp_path / experiment.CHANGED_MODULES[1]
    code.parent.mkdir(parents=True)
    code.write_bytes(Path(experiment.__file__).read_bytes())
    (tmp_path / "results/experiment_7712_v671_capstone.json").write_text("{}")
    (tmp_path / "results/experiment_7713_v672_contract_methods.json").write_text("{}")
    monkeypatch.setattr(experiment, "run_commands", _fake_commands)
    result = experiment.run_experiment(tmp_path, "20260926")
    assert result["verdict_class"] == "circular_positive"
    assert (tmp_path / experiment.RAW_PATH / "tmp").is_dir()
    assert result["alignment_protocol_ready_score"] == 1
    candidate = tmp_path / experiment.RAW_PATH / "terminal_candidate.json"
    experiment.cold_check(tmp_path / experiment.RAW_PATH, candidate)
    assert candidate.read_bytes() == (tmp_path / experiment.RESULT_PATH).read_bytes()
    assert len(result["phase_spans"]) == 3
    assert result["source_artifact_hashes"]["pre_gate_receipts"]


def test_block_and_terminal_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7714-TERMINAL distinguishes missing inputs and failed readers."""
    code = tmp_path / experiment.CHANGED_MODULES[1]
    code.parent.mkdir(parents=True)
    code.write_bytes(Path(experiment.__file__).read_bytes())
    monkeypatch.setattr(experiment, "run_commands", _fake_commands)
    blocked = experiment.run_experiment(tmp_path, "20260926")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["alignment_protocol_ready_score"] == 0
    _upstream(tmp_path)
    calls = 0

    def fail_once(root, commands, **kwargs):
        nonlocal calls
        calls += 1
        rows = _fake_commands(root, commands, **kwargs)
        if calls == 2:
            rows[1].update({"passed": False, "exit_code": 1})
        return rows

    monkeypatch.setattr(experiment, "run_commands", fail_once)
    failed = experiment.run_experiment(tmp_path, "20260926")
    assert failed["verdict_class"] == "disqualified"
    assert failed["flagged_adversarial"] is True
    assert failed["alignment_protocol_ready_score"] == 0
    assert calls == 3


def test_invalid_schema_and_empty_answer(tmp_path):
    """SCENARIO-VERIFY-7714-BYTES abstains and upstream schemas fail closed."""
    assert sa.sentence_spans(b"unfinished") == [b"unfinished"]
    assert sa.prepare(b"Source.", b"")["abstention"] == "empty_answer"
    with pytest.raises(ValueError, match="abstained family"):
        sa.distribution(sa.prepare(b"Source.", b""), sa.zero_parameters())
    with pytest.raises(ValueError, match="unknown control"):
        sa.control_scores(sa.prepare(b"Source.", b"Answer."), sa.zero_parameters(), kind="bad")
    _upstream(tmp_path)
    path = tmp_path / experiment.UPSTREAM[0][0]
    path.write_text("[]")
    checks, failures, _ = experiment.check_preconditions(tmp_path)
    assert checks[0]["schema_object"] is False
    assert len(failures) == 1


def _rewrite_raw(raw_dir: Path, rows):
    raw_bytes = experiment._json_bytes(rows)
    (raw_dir / "rows.json").write_bytes(raw_bytes)
    manifest = json.loads((raw_dir / "protocol.json").read_text())
    manifest["rows_sha256"] = experiment._digest(raw_bytes)
    (raw_dir / "protocol.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows.pop(), "fixture count"),
        (lambda rows: rows[0].update(source_hex="00"), "fixture source"),
        (lambda rows: rows[0].update(byte_reconstruction=False), "byte reconstruction"),
        (lambda rows: rows[0].update(source_sha256="bad"), "source hash"),
        (lambda rows: rows[0].update(marginal=[0.0, 1.0]), "normalization row"),
    ],
)
def test_cold_reducer_rejects_mutations(tmp_path, change, message):
    """SCENARIO-REPORT-7714-CUSTODY rejects changed raw rows even with a new hash."""
    _, rows = experiment.build_protocol(tmp_path)
    change(rows)
    _rewrite_raw(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        experiment.reduce_raw(tmp_path)


def test_nonfinite_guard_and_candidate_reader(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7714-NORMALIZE rejects nonfinite or false candidate claims."""
    _, rows = experiment.build_protocol(tmp_path)
    original_finite = experiment.math_isfinite
    monkeypatch.setattr(experiment, "math_isfinite", lambda _value: False)
    with pytest.raises(ValueError, match="nonfinite"):
        experiment.reduce_raw(tmp_path)
    monkeypatch.setattr(experiment, "math_isfinite", original_finite)
    candidate = tmp_path / "candidate.json"
    value = {
        "rows": rows,
        "normalization_rows": experiment.reduce_raw(tmp_path)["normalization_rows"],
        "alignment_protocol_path": {"sha256": experiment.sha256_file(tmp_path / "protocol.json")},
        "verdict_class": "circular_positive",
        "verifier_is_oracle": True,
    }
    candidate.write_text(json.dumps(value))
    experiment.cold_check(tmp_path, candidate)
    value["rows"] = []
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="candidate rows"):
        experiment.cold_check(tmp_path, candidate)
    value["rows"] = rows
    value["alignment_protocol_path"]["sha256"] = "bad"
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol hash"):
        experiment.cold_check(tmp_path, candidate)
    value["alignment_protocol_path"]["sha256"] = experiment.sha256_file(tmp_path / "protocol.json")
    value["verifier_is_oracle"] = False
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixture verdict"):
        experiment.cold_check(tmp_path, candidate)


def test_cli_modes_and_disqualified_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7714-TERMINAL handles CLI and failed validation."""
    _upstream(tmp_path)
    code = tmp_path / experiment.CHANGED_MODULES[1]
    code.parent.mkdir(parents=True)
    code.write_bytes(Path(experiment.__file__).read_bytes())
    experiment.build_protocol(tmp_path / experiment.RAW_PATH)
    checks, failures, hashes = experiment.check_preconditions(tmp_path)
    reduction = experiment.reduce_raw(tmp_path / experiment.RAW_PATH)
    artifact = experiment.build_artifact(
        tmp_path, "20260926", checks, failures, hashes, reduction, [], []
    )
    assert artifact["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        experiment, "run_experiment", lambda _root, _date: {"verdict_class": "blocked"}
    )
    assert experiment.main(["--date", "20260926"]) == 0
    monkeypatch.setattr(
        experiment, "run_experiment", lambda _root, _date: {"verdict_class": "disqualified"}
    )
    assert experiment.main(["--date", "20260926"]) == 1
    with pytest.raises(SystemExit):
        experiment.main(["--cold-reduce", str(tmp_path)])
    candidate = tmp_path / "candidate.json"
    value = {
        "rows": reduction["rows"],
        "normalization_rows": reduction["normalization_rows"],
        "alignment_protocol_path": {
            "sha256": experiment.sha256_file(tmp_path / experiment.RAW_PATH / "protocol.json")
        },
        "verdict_class": "circular_positive",
        "verifier_is_oracle": True,
    }
    candidate.write_text(json.dumps(value))
    assert (
        experiment.main(
            ["--cold-reduce", str(tmp_path / experiment.RAW_PATH), "--candidate", str(candidate)]
        )
        == 0
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            experiment.__file__,
            "--cold-reduce",
            str(tmp_path / experiment.RAW_PATH),
            "--candidate",
            str(candidate),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("carnot.experiment_7714_v672_alignment_protocol", run_name="__main__")
    assert caught.value.code == 0
