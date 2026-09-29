"""REQ-REPORT-7839: current import custody and byte intervention tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands


def test_scenario_report_7839_imports_reproduces_old_failure(tmp_path: Path) -> None:
    """A successful process with the exact old line remains a failed receipt."""
    root = Path(__file__).resolve().parents[2]
    old = run_commands(
        root,
        [
            CommandSpec(
                "worktree_imports",
                (sys.executable, "-c", "print('worktree_imports_ok')"),
                "required",
                5,
            )
        ],
        log_dir=tmp_path / "old",
    )[0]
    assert old["exit_code"] == 0
    assert old["resolved_imports"] == {}
    assert old["passed"] is False


def test_scenario_report_7839_imports_rejects_forgery(tmp_path: Path) -> None:
    """A JSON object cannot claim a site-package or missing module as local."""
    root = Path(__file__).resolve().parents[2]
    for name, mapping in (
        ("empty", {}),
        ("foreign", {"carnot.verify.source_interventions": "/tmp/site-packages/fake.py"}),
        (
            "missing",
            {"carnot.verify.source_interventions": str(root / "python/carnot/verify/no.py")},
        ),
    ):
        code = f"import json; print(json.dumps({{'resolved_imports': {mapping!r}}}))"
        result = run_commands(
            root,
            [CommandSpec("worktree_imports", (sys.executable, "-c", code), "required", 5)],
            log_dir=tmp_path / name,
        )[0]
        assert result["exit_code"] == 0
        assert result["passed"] is False


def test_scenario_report_7839_bytes_differential() -> None:
    """Pure operations preserve the historical UTF-8 span behavior."""
    from carnot import experiment_7814_v679_counter_evidence_protocol as old
    from carnot.verify import source_interventions as new

    source = "Café one. Delta two. Echo trio. ".encode()
    answer = "Café works. Later words differ.".encode()
    offsets = new.sentence_offsets(source)
    assert offsets == old.sentence_offsets(source)
    assert new.target_span(answer) == old.target_span(answer)
    assert new.visible_after(source, offsets, 0) == old.visible_after(source, offsets, 0)
    assert new.remove_sentence(source, offsets, 1) == old.remove_sentence(source, offsets, 1)
    assert new.select_control(
        source, offsets, 0, "fixture", lambda _: 4, seed=67815
    ) == old.select_control(source, offsets, 0, "fixture", lambda _: 4)
    assert (
        new.select_control(
            source, offsets, 0, "fixture", lambda text: 100 if "Café" in text else 1, seed=68101
        )
        is None
    )
    edited, visible = new.visible_after(source, offsets, 0)
    assert edited == source[offsets[0]["end_byte"] :]
    assert [item["source_sentence_id"] for item in visible] == [1, 2]
    with pytest.raises(ValueError, match="invalid_witness"):
        new.remove_sentence(source, offsets, 99)


def test_scenario_report_7839_request_and_parser() -> None:
    """The strict parser and bounded builder reject the fixture edge cases."""
    from carnot.verify import source_interventions as new

    row = {"complete_response": "First. Second."}
    source = b"One here. Two there."
    offsets = new.sentence_offsets(source)
    frozen = new.protocol([], seed=68101)
    payload = new.make_request(row, source, offsets, "intact", frozen, lambda _: 5)
    assert payload["max_tokens"] == 256
    assert payload["seed"] == 68101
    assert "family_id" not in payload["messages"][1]["content"]
    assert "original_label" not in payload["messages"][1]["content"]
    good = '{"unsupported_probability": 0.25, "source_sentence_id": 0}'
    assert new.parse_reply(good, "stop", [0, 1])["disposition"] == "completed"
    for text, finish in ((good, "length"), ("{", "stop"), ('{"wrong":1}', "stop")):
        assert new.parse_reply(text, finish, [0, 1])["disposition"] == "invalid_parse"
    assert new.parse_reply(good, "stop", [1])["disposition"] == "invalid_witness"
    frozen["context_ceiling_tokens"] = 1
    with pytest.raises(ValueError, match="context_budget"):
        new.make_request(row, source, offsets, "intact", frozen, lambda _: 5)


def test_scenario_report_7839_fixture_and_direct_cli(tmp_path: Path) -> None:
    """The independent fixture and real CLI run without loading a model."""
    from carnot.verify import source_interventions as new

    row = {
        "family_id": "fixture",
        "complete_source": "One here. Two here. Three here.",
        "complete_response": "One here.",
    }
    row["source_sha256"] = new.digest(row["complete_source"].encode())
    row["response_sha256"] = new.digest(row["complete_response"].encode())
    reply = {
        "model": new.MODEL_ID,
        "choices": [
            {
                "message": {"content": '{"unsupported_probability":0.2,"source_sentence_id":0}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 12},
    }

    def visible_reply(payload: dict) -> dict:
        body = json.loads(payload["messages"][1]["content"])
        witness = body["source_sentence_offsets"][0]["source_sentence_id"]
        return {
            **reply,
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"unsupported_probability": 0.2, "source_sentence_id": witness}
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    rows = new.capture_fixture(row, new.protocol([], seed=68101), visible_reply, lambda _: 4)
    assert [item["status"] for item in rows] == ["completed"] * 3
    assert rows[1]["deleted_sentence_id"] == 0
    assert rows[2]["deleted_sentence_id"] in {1, 2}
    wrong = new.capture_fixture(
        row, new.protocol([], seed=68101), lambda payload: {**reply, "model": "wrong"}, lambda _: 4
    )
    assert wrong[0]["status"] == "wrong_model"
    assert all(item["status"].startswith("unstarted") for item in wrong[1:])
    timed = new.capture_fixture(
        row,
        new.protocol([], seed=68101),
        lambda payload: (_ for _ in ()).throw(TimeoutError()),
        lambda _: 4,
    )
    assert timed[0]["status"] == "timeout"
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            str(root / "scripts/experiments/experiment_7839_v681_intervention_protocol.py"),
            "--date",
            "20260928",
            "--fixture-e2e",
            str(tmp_path / "fixture.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((tmp_path / "fixture.json").read_text())["independent_families"] == 24
