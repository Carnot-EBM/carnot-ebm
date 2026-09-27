"""REQ-REPORT-7768: source custody and complete evidence views."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_7768_v676_source_view_qualification as exp
from carnot.verify import evidence_views as views
from test_experiment_7740_v674_sentence_label_protocol import _fixture


def test_scenario_report_7768_views_joint_premise_and_limits() -> None:
    """Keep both premises visible while leaving semantic success unclaimed."""
    source = b"Ada owns the key. The key opens the vault."
    answer = b"Ada can open the vault."
    pair = exp.qualify_views(source, answer)
    assert all(
        value["probability_unsupported"] is None for value in exp.arm_decisions(pair, None).values()
    )
    assert b"Ada owns the key." in pair["a"]["windows"][0]
    assert any(
        b"Ada owns the key." in w and b"key opens the vault" in w for w in pair["b"]["windows"]
    )
    assert all(b"".join(v["source_sentences"]) == source for v in pair.values())
    assert all(v["source_bytes"] == source and v["answer_bytes"] == answer for v in pair.values())
    assert views.deserialize_pair(views.serialize_pair(pair)) == pair
    assert pair["a"]["group_ids"] == [0, 1]
    assert sum(views.location_prior(pair["a"])) == pytest.approx(1.0)
    reversed_pair = exp.qualify_views(b"The key opens the vault. Ada owns the key.", answer)
    assert reversed_pair["a"]["source_bytes"] != source
    duplicate = exp.qualify_views(b"A. A.", b"A.")
    assert duplicate["a"]["group_ids"] == [0, 0]
    assert views.location_prior(duplicate["a"]) == pytest.approx([0.25, 0.25, 0.5])
    empty = exp.qualify_views(b"", b"A.")
    assert empty["a"]["windows"] == []
    assert views.location_prior(empty["a"]) == [1.0]
    assert exp.arm_decisions(exp.qualify_views(b"A. " * 65, b"A."), None) == {
        arm: {"action": "escalate", "probability_unsupported": 0.5} for arm in views.ARMS
    }
    assert exp.qualify_views(b"A.", b"B. " * 17)["a"]["abstention"] == "answer_units_over_budget"


def test_scenario_report_7768_custody_offsets_and_overlap() -> None:
    """Exact duplicates are visible; conflicting spans and corrupt offsets fail."""
    answer = "Café. Go!".encode()
    mark = {"start": 3, "end": 4, "text": "é", "implicit_true": False}
    mapped = exp.checked_targets(answer, [mark, mark])
    assert mapped["sentence_byte_offsets"] == [[0, 7], [7, len(answer)]]
    assert mapped["annotation_byte_offsets"] == [[3, 5], [3, 5]]
    assert mapped["duplicate_span_count"] == 1
    assert mapped["targets"] == [1, 0]
    assert exp.checked_targets(answer, None)["targets"] == [None, None]
    assert exp.checked_targets(
        answer, [{"start": 0, "end": 0, "text": "", "implicit_true": False}]
    )["targets"] == [None, None]
    with pytest.raises(ValueError, match="overlapping_annotation_spans"):
        exp.checked_targets(
            answer, [mark, {"start": 2, "end": 4, "text": "fé", "implicit_true": False}]
        )
    with pytest.raises(ValueError, match="annotation_offset_or_text"):
        exp.checked_targets(answer, [{**mark, "end": 200}])
    with pytest.raises(UnicodeDecodeError):
        exp.checked_targets(b"\xff", [])


def test_scenario_report_7768_custody_public_before_labels(tmp_path: Path) -> None:
    """Role assignment and view bytes come only from the authenticated public row."""
    manifest = _fixture(tmp_path)
    meta = json.loads(manifest.read_text())
    role = "fit"
    public = json.loads((manifest.parent / meta["roles"][role]["public_path"]).read_text())
    evaluator = json.loads((manifest.parent / meta["roles"][role]["evaluator_path"]).read_text())
    frozen = exp.prepare_public(public)
    assert frozen["role"] == role
    assert frozen["prior_exposure"] is True
    assert frozen["fresh_generalization_eligible"] is False
    changed = copy.deepcopy(evaluator)
    changed["annotations"][0]["implicit_true"] = True
    changed["label"] = 0
    assert exp.prepare_public(public) == frozen
    assert (
        exp.map_evaluator(public, evaluator)["sentence_targets"]
        != exp.map_evaluator(public, changed)["sentence_targets"]
    )
    with pytest.raises(ValueError, match="evaluator_join"):
        exp.map_evaluator(public, {**evaluator, "family_id": "other"})
    with pytest.raises(ValueError, match="response_label_mapping"):
        exp.map_evaluator(public, {**evaluator, "label": 7})
    with pytest.raises(ValueError, match="unauthorized"):
        exp.prepare_public({**public, "label": 1})
    with pytest.raises(ValueError, match="source_hash"):
        exp.prepare_public({**public, "complete_source": "changed"})


def test_scenario_report_7768_terminal_private_replay(tmp_path: Path) -> None:
    """A fresh reader rejects changed bytes, missing premises and missing rows."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in exp.COUNTS}
    raw = tmp_path / "raw"
    raw.mkdir()
    prepared = exp.prepare_corpus(manifest, raw, counts)
    assert len(prepared["rows"]) == 7
    assert len(prepared["annotation_coverage_rows"]) == 7
    assert exp.replay_corpus(manifest, raw, counts) == {"families": 7, "roles": counts}
    rows_path = raw / "rows.jsonl"
    original = rows_path.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    for mutate in (
        lambda row: row.update(source_sha256="sha256:changed"),
        lambda row: row.update(role="evaluation"),
        lambda row: row["view_b"].update(window_offsets=[]),
        lambda row: row["view_a"].update(windows=[]),
    ):
        changed = copy.deepcopy(rows)
        mutate(changed[0])
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in changed))
        with pytest.raises(ValueError):
            exp.replay_corpus(manifest, raw, counts)
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="family_count"):
        exp.replay_corpus(manifest, raw, counts)
    rows_path.write_bytes(original)
    label_path = (
        manifest.parent / json.loads(manifest.read_text())["roles"]["fit"]["evaluator_path"]
    )
    original_label = label_path.read_bytes()
    label = json.loads(original_label)
    label["annotations"][0]["text"] = "wrong"
    label_path.write_text(json.dumps(label) + "\n")
    with pytest.raises(ValueError):
        exp.replay_corpus(manifest, raw, counts)
    label_path.write_bytes(original_label)


def test_scenario_report_7768_terminal_real_child_basetemp(tmp_path: Path) -> None:
    """Pytest must work with a newly created nested parent in a child process."""
    parent = tmp_path / "nested" / "pytest"
    exp.prepare_basetemp(parent)
    target = parent / "child"
    test_file = tmp_path / "test_private.py"
    test_file.write_text("def test_real_child(tmp_path):\n    assert tmp_path.is_dir()\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={target}",
            str(test_file),
            "-q",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scenario_report_7768_terminal_preflight_missing_producer(tmp_path: Path) -> None:
    """Missing declared producer and conductor receipt remain distinct checks."""
    checks = exp.preflight(tmp_path)
    failures = [row for row in checks if not row["passed"]]
    assert {row["upstream_id"] for row in failures} >= {"exp7727", "exp7753_conductor_pre_gate"}
    assert all(
        set(row) >= {"artifact_path", "artifact_hash", "field", "expected", "observed", "operator"}
        for row in failures
    )


def test_scenario_report_7768_terminal_real_preflight() -> None:
    """Exact declared producer and all fourteen role shards authenticate."""
    assert all(row["passed"] for row in exp.preflight(exp.ROOT))


def test_scenario_report_7768_terminal_replay_inner_mismatch(tmp_path: Path) -> None:
    """A forged outer rows hash cannot conceal changed view or target data."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in exp.COUNTS}
    raw = tmp_path / "raw"
    raw.mkdir()
    exp.prepare_corpus(manifest, raw, counts)
    rows_path = raw / "rows.jsonl"
    sealed_path = raw / "source_view_manifest.json"
    original_rows = rows_path.read_bytes()
    original_sealed = sealed_path.read_bytes()
    rows = [json.loads(line) for line in original_rows.splitlines()]
    rows[0]["view_a"]["windows"] = []
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sealed = json.loads(original_sealed)
    sealed["rows_sha256"] = exp.sha256_file(rows_path)
    sealed_path.write_text(json.dumps(sealed))
    with pytest.raises(ValueError, match="public_view_replay_mismatch"):
        exp.replay_corpus(manifest, raw, counts)
    rows_path.write_bytes(original_rows)
    sealed_path.write_bytes(original_sealed)
    targets_path = raw / "targets.jsonl"
    targets = [json.loads(line) for line in targets_path.read_text().splitlines()]
    targets[0]["sentence_targets"] = [0]
    targets_path.write_text("".join(json.dumps(row) + "\n" for row in targets))
    with pytest.raises(ValueError, match="evaluator_replay_mismatch"):
        exp.replay_corpus(manifest, raw, counts)


def test_scenario_report_7768_terminal_counters_and_heartbeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long loops report completed units and bad roster sizes fail closed."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in exp.COUNTS}
    raw = tmp_path / "raw"
    raw.mkdir()
    original_authenticate = exp.prior.authenticate
    public, meta, checks = original_authenticate(manifest, counts)
    monkeypatch.setattr(exp.prior, "authenticate", lambda *_: (public[:-1], meta, checks))
    with pytest.raises(ValueError, match="family_count"):
        exp.prepare_corpus(manifest, raw, counts)
    monkeypatch.setattr(exp.prior, "authenticate", original_authenticate)
    ticks = iter(range(0, 10000, 31))
    monkeypatch.setattr(exp, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    exp.prepare_corpus(manifest, raw, counts)
    assert exp.replay_corpus(manifest, raw, counts)["families"] == 7
    monkeypatch.setattr(
        exp.prior,
        "authenticate",
        lambda *_: (
            public,
            {
                **meta,
                "roles": {
                    **meta["roles"],
                    "fit": {**meta["roles"]["fit"], "evaluator_path": "empty.jsonl"},
                },
            },
            checks,
        ),
    )
    (manifest.parent / "empty.jsonl").write_text("")
    with pytest.raises(ValueError, match="evaluator_count"):
        exp.prepare_corpus(manifest, raw, counts)


def test_scenario_report_7768_terminal_coverage_scope(tmp_path: Path) -> None:
    """All affected tests run; coverage traces only the new module's direct tests."""
    runner = runpy.run_path(str(exp.ROOT / exp.WRAPPER), run_name="exp7768_test")
    scope = json.loads(exp.SCOPE.read_text())
    commands = runner["validation_commands"](scope, tmp_path)
    focused = next(row for row in commands if row.name == "focused_pytest")
    coverage = next(row for row in commands if row.name == "changed_module_coverage")
    assert set(scope["test_paths"]) <= set(focused.argv)
    assert scope["test_paths"][0] in coverage.argv
    assert all(path not in coverage.argv for path in scope["test_paths"][1:])
