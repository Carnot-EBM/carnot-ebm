"""REQ-REPORT-7644: source witnesses are byte-bound structural facts only."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7644_v667_source_witness_prototype as experiment
from carnot.verify.source_claim_witness import parse_numbered_blocks, verify_claim


FIXTURES = Path(__file__).parent / "fixtures" / "experiment_7644_source_witness.jsonl"


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    """Independent annotations are fixed data, not verifier outputs."""

    return [json.loads(line) for line in FIXTURES.read_text().splitlines()]


def test_independent_fixture_roster(cases: list[dict]) -> None:
    """SCENARIO-REPORT-7644-STRUCTURE: all adversarial categories are represented."""

    assert len(cases) >= 48
    categories = {case["category"] for case in cases}
    assert {
        "line",
        "scope",
        "comment",
        "alias",
        "omitted",
        "syntax",
        "multiple",
        "decorator",
        "unicode",
        "malicious",
        "prose",
    } <= categories
    assert all(case["expected"] in {"supported", "contradicted", "unknown"} for case in cases)


def test_fixture_truth(cases: list[dict]) -> None:
    """SCENARIO-VERIFY-7644-EXACT: truth is checked against fixed annotations."""

    for case in cases:
        result = verify_claim(case["source"], case["claim"], closed_files=case["closed_files"])
        assert result["status"] == case["expected"], case["id"]
        assert result["source_sha256"].startswith("sha256:")
        assert isinstance(result["residual_unverified_span"], bool)
        assert result["parser_completeness"] in {"complete", "incomplete", "missing"}
        if result["status"] == "supported":
            assert isinstance(result["source_offset"], int)
            assert result["proposition_checked"]


def test_unicode_offsets_and_lossless_blocks() -> None:
    """SCENARIO-REPORT-7644-STRUCTURE: line spans map to source UTF-8 bytes."""

    source = "é intro\n```python file=a.py\n1 | # café\n2 | def f():\n3 |     return 1\n```\n"
    blocks = parse_numbered_blocks(source)
    assert len(blocks) == 1
    assert blocks[0]["lines"][1]["byte_start"] == source.encode().index(b"def f():")
    assert blocks[0]["lines"][1]["exact_bytes"] == b"def f():".hex()
    witness = verify_claim(source, "In `a.py`, `f` is defined at line 2.", closed_files=["a.py"])
    assert witness["source_offset"] == source.encode().index(b"def f():")


def test_source_swap_and_label_sidecar_rejection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7644-ABSTAIN: input binding and label firewall."""

    source = "```python file=a.py\n1 | def f(): pass\n```"
    changed = source.replace("f()", "g()")
    a = verify_claim(source, "In `a.py`, `f` exists.", closed_files=["a.py"])
    b = verify_claim(changed, "In `a.py`, `f` exists.", closed_files=["a.py"])
    assert a["source_sha256"] != b["source_sha256"]
    assert (a["status"], b["status"]) == ("supported", "contradicted")
    label = tmp_path / "labels.json"
    label.write_text('{"secret": true}')
    with pytest.raises(ValueError, match="label"):
        experiment.validate_predictor_input({"source": source, "labels": str(label)})


def test_raw_to_cold_reduction(tmp_path: Path, cases: list[dict]) -> None:
    """SCENARIO-REPORT-7644-TERMINAL: raw witness bytes survive cold reduction."""

    rows = experiment.build_fixture_rows(cases)
    assert len(rows) == len(cases)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    result = experiment.cold_reduce_rows(raw, cases)
    assert result["passed"] is True
    rows[0]["witness"]["source_sha256"] = "sha256:changed"
    raw.write_text(json.dumps(rows))
    assert experiment.cold_reduce_rows(raw, cases)["passed"] is False
    raw.write_text(json.dumps(rows[:-1]))
    assert experiment.cold_reduce_rows(raw, cases)["passed"] is False
    with pytest.raises(ValueError, match="predictor_input_fields"):
        experiment.validate_predictor_input({"source": "x"})


def test_incomplete_and_open_scopes() -> None:
    """SCENARIO-VERIFY-7644-OPEN: malformed and open sources abstain."""

    for source in (
        "```python file=a.py\nnot numbered\n```",
        "```python file=a.py\n2 | def f(): pass\n```",
        "```python file=a.py\n1 | def f(): pass",
        "```python file=a.py\n```",
    ):
        assert (
            verify_claim(source, "In `a.py`, `f` exists.", closed_files=["a.py"])["status"]
            == "unknown"
        )
    source = "```python file=a.py\n1 | def f(): pass\n```"
    assert verify_claim(source, "In `a.py`, `g` exists.")["status"] == "unknown"
    assert verify_claim(source, "In `a.py`, `f-g` exists.")["status"] == "unknown"
    assert verify_claim(source, "In `a.py`, `return 1` appears at line 1.")["status"] == "unknown"


def test_artifact_gates_and_cold_reader(tmp_path: Path, cases: list[dict]) -> None:
    """SCENARIO-REPORT-7644-TERMINAL: hashes and fixture truth govern publication."""

    root = tmp_path
    fixture = root / experiment.FIXTURES
    fixture.parent.mkdir(parents=True)
    fixture.write_text("".join(json.dumps(case) + "\n" for case in cases))
    pilot = root / experiment.PILOT
    pilot.parent.mkdir(parents=True)
    pilot.write_text(
        json.dumps(
            {
                "complete_source": "no code",
                "answer_sentences": [{"text": "No claim."}],
                "source_sha256": "test",
            }
        )
        + "\n"
    )
    rows = experiment.build_fixture_rows(cases)
    raw = root / experiment.RAW
    raw.mkdir(parents=True)
    (raw / "fixture_rows.json").write_text(json.dumps(rows))
    (root / experiment.SCHEMA_PATH).write_text(
        json.dumps(
            {
                "schema_version": experiment.SCHEMA_VERSION,
                "feature_names": list(experiment.FEATURE_NAMES),
            }
        )
    )
    hashes = {
        "producer_files": {experiment.FIXTURES.as_posix(): experiment.sha256_file(fixture)},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [],
    }
    receipt = {"name": "reader", "passed": True}
    artifact = experiment.build_artifact(
        date="20260925",
        root=root,
        fixture_rows=rows,
        pilot_rows=experiment._pilot_rows(experiment._read_jsonl(pilot)),
        preconditions=[],
        hashes=hashes,
        receipts=[receipt],
        spans=[],
        duration=0.1,
        terminal_ready=True,
    )
    assert artifact["verdict_class"] == "circular_positive"
    path = root / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert experiment.cold_replay(path, root)["passed"]
    assert experiment.independent_reduce(path, root)["passed"]
    fixture.write_text(fixture.read_text().replace("\n", " \n", 1))
    assert experiment.cold_replay(path, root)["passed"] is False
    blocked = [
        {
            "passed": False,
            "check": "input",
            "upstream": "pilot",
            "path": str(pilot),
            "field": "is_file",
            "operator": "eq",
            "expected": True,
            "observed": False,
        }
    ]
    assert (
        experiment.build_artifact(
            date="20260925",
            root=root,
            fixture_rows=rows,
            pilot_rows=[],
            preconditions=blocked,
            hashes=hashes,
            receipts=[receipt],
            spans=[],
            duration=0,
            terminal_ready=True,
        )["verdict_class"]
        == "blocked"
    )
    assert (
        experiment.build_artifact(
            date="20260925",
            root=root,
            fixture_rows=rows,
            pilot_rows=[],
            preconditions=[],
            hashes=hashes,
            receipts=[{"name": "reader", "passed": False}],
            spans=[],
            duration=0,
            terminal_ready=False,
        )["verdict_class"]
        == "disqualified"
    )
    assert (
        experiment.build_artifact(
            date="20260925",
            root=root,
            fixture_rows=rows,
            pilot_rows=[],
            preconditions=[],
            hashes=hashes,
            receipts=[],
            spans=[],
            duration=0,
            terminal_ready=False,
        )["verdict_class"]
        == "null"
    )


def test_bounded_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cases: list[dict]
) -> None:
    """SCENARIO-REPORT-7644-TERMINAL: completed units publish after readers."""

    fixture = tmp_path / experiment.FIXTURES
    fixture.parent.mkdir(parents=True)
    fixture.write_text("".join(json.dumps(case) + "\n" for case in cases))
    pilot = tmp_path / experiment.PILOT
    pilot.parent.mkdir(parents=True)
    pilot.write_text(
        json.dumps(
            {
                "complete_source": "plain text",
                "answer_sentences": [{"text": "No structural claim."}],
                "source_sha256": "sha256:test",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        experiment, "NAMED_INPUTS", (experiment.FIXTURES.as_posix(), experiment.PILOT.as_posix())
    )
    monkeypatch.setattr(experiment.checks, "build_scoped_commands", lambda *args, **kwargs: [])
    calls = []

    def fake_commands(root, commands, **kwargs):
        calls.append(kwargs["log_dir"])
        return [
            {
                "name": "focused" if len(calls) == 1 else "terminal",
                "passed": True,
                "exit_code": 0,
                "command": "fake bounded reader",
                "log_sha256": "sha256:test",
            }
        ]

    monkeypatch.setattr(experiment.checks, "run_commands", fake_commands)
    out = tmp_path / "out.json"
    final = experiment.run_experiment(tmp_path, "20260925", out)
    assert out.is_file() and len(calls) == 3
    assert final["verdict_class"] == "circular_positive"
    assert final["witness_ready_score"] == 1
    assert len(final["phase_spans"]) >= 5
    assert (tmp_path / experiment.RAW / "exact_terminal_reader_outcomes.json").is_file()
    schema = tmp_path / experiment.SCHEMA_PATH
    schema.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="frozen_witness_schema"):
        experiment.run_experiment(tmp_path, "20260925", out)
    schema.unlink()
    pilot.unlink()
    blocked = experiment.run_experiment(tmp_path, "20260925", out)
    assert blocked["verdict_class"] == "blocked"
    pilot.write_text(
        json.dumps(
            {
                "complete_source": "plain text",
                "answer_sentences": [{"text": "No claim."}],
                "source_sha256": "sha256:test",
            }
        )
        + "\n"
    )
    call_count = 0

    def failed_terminal(root, commands, **kwargs):
        nonlocal call_count
        call_count += 1
        return [
            {
                "name": "focused" if call_count == 1 else "terminal",
                "passed": call_count == 1,
                "exit_code": 0 if call_count == 1 else 1,
            }
        ]

    monkeypatch.setattr(experiment.checks, "run_commands", failed_terminal)
    assert experiment.run_experiment(tmp_path, "20260925", out)["verdict_class"] == "disqualified"
    call_count = 0

    def changed_terminal(root, commands, **kwargs):
        nonlocal call_count
        call_count += 1
        return [
            {
                "name": "focused" if call_count == 1 else "terminal",
                "passed": call_count != 3,
                "exit_code": 0 if call_count != 3 else 1,
            }
        ]

    monkeypatch.setattr(experiment.checks, "run_commands", changed_terminal)
    with pytest.raises(RuntimeError, match="exact_terminal_reader_outcomes_changed"):
        experiment.run_experiment(tmp_path, "20260925", out)


def test_cli_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7644-TERMINAL: CLI modes select read-only readers."""

    monkeypatch.setattr(experiment, "cold_replay", lambda path: {"passed": True})
    monkeypatch.setattr(experiment, "independent_reduce", lambda path: {"passed": False})
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda root, date, output: {
            "honest_verdict": "complete_null_fixture",
            "verdict_class": "null",
        },
    )
    assert experiment.main(["--cold-replay", "candidate.json"]) == 0
    assert experiment.main(["--independent-reduce", "candidate.json"]) == 1
    assert experiment.main(["--date", "20260925"]) == 0
    with pytest.raises(SystemExit):
        experiment.main(["--date", "wrong"])
    monkeypatch.setattr(sys, "argv", ["exp7644", "--date", "wrong"])
    with pytest.raises(SystemExit) as exit_state:
        runpy.run_module(
            "carnot.experiment_7644_v667_source_witness_prototype", run_name="__main__"
        )
    assert exit_state.value.code == 2
