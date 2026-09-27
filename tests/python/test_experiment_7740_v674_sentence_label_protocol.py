"""REQ-REPORT-7740 and REQ-VERIFY-7740 sentence custody checks."""

from __future__ import annotations

import hashlib
from itertools import count
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.experiment_7740_v674_sentence_label_protocol import (
    authenticate,
    cold_reduce,
    map_targets,
    run_experiment,
)
from carnot import experiment_7740_v674_sentence_label_protocol as protocol


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    """Build seven isolated families so role custody is visible end to end."""
    raw = tmp_path / "input"
    raw.mkdir(parents=True)
    roles = {}
    for index, role in enumerate(
        ("fit", "tune", "policy", "online_update", "online_admission", "evaluation", "retention")
    ):
        source = f"Source {index} says blue."
        answer = f"Answer {index} is red."
        family = f"family-{index}"
        public = {
            "family_id": family,
            "role": role,
            "official_split": "test" if role == "evaluation" else "train",
            "source_id": f"source-{index}",
            "response_id": f"response-{index}",
            "complete_source": source,
            "complete_response": answer,
            "source_sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
            "response_sha256": "sha256:" + hashlib.sha256(answer.encode()).hexdigest(),
            "previously_exposed": True,
            "fresh_generalization_eligible": False,
        }
        evaluator = {
            "family_id": family,
            "response_id": public["response_id"],
            "label": 1,
            "annotations": [
                {
                    "start": answer.index("red"),
                    "end": answer.index("red") + 3,
                    "text": "red",
                    "implicit_true": False,
                }
            ],
        }
        for kind, row in (("public", public), ("evaluator", evaluator)):
            path = raw / f"{role}_{kind}.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        roles[role] = {
            "count": 1,
            "families": [family],
            "public_path": f"{role}_public.jsonl",
            "public_sha256": _hash(raw / f"{role}_public.jsonl"),
            "evaluator_path": f"{role}_evaluator.jsonl",
            "evaluator_sha256": _hash(raw / f"{role}_evaluator.jsonl"),
        }
    manifest = raw / "development_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.exp7727.development_manifest.v1",
                "counts": {role: 1 for role in roles},
                "roles": roles,
                "initially_open_evaluator_roles": ["fit"],
            },
            sort_keys=True,
        )
    )
    return manifest


def test_offsets_and_unknowns():
    """SCENARIO-VERIFY-7740-OFFSETS: half-open Unicode and sentence boundaries."""
    answer = "Café is open. It closes! Last?".encode()
    text = answer.decode()
    first = {
        "start": 0,
        "end": text.index(" It"),
        "text": text[: text.index(" It")],
        "implicit_true": False,
    }
    cross = {
        "start": text.index("open"),
        "end": text.index("closes") + 6,
        "text": text[text.index("open") : text.index("closes") + 6],
        "implicit_true": False,
    }
    assert map_targets(answer, [first])["targets"] == [1, 0, 0]
    assert map_targets(answer, [cross])["targets"] == [1, 1, 0]
    assert map_targets(answer, [{**first, "implicit_true": True}])["targets"] == [0, 0, 0]
    assert map_targets(answer, None)["targets"] == [None, None, None]
    assert map_targets(b"", [])["reason"] == "empty_answer"
    assert (
        map_targets(answer, [{"start": 0, "end": 0, "text": "", "implicit_true": False}])["reason"]
        == "unmappable_annotation"
    )
    with pytest.raises(ValueError, match="annotation_offset_or_text"):
        map_targets(answer, [{**first, "text": "wrong"}])


def test_authentication_fails_on_drift_and_duplicates(tmp_path):
    """SCENARIO-REPORT-7740-CUSTODY: both kinds of shard and all roles are fixed."""
    manifest = _fixture(tmp_path)
    assert (
        len(authenticate(manifest, {r: 1 for r in json.loads(manifest.read_text())["roles"]})[0])
        == 7
    )
    path = manifest.parent / "tune_public.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="public_sha256"):
        authenticate(manifest, {r: 1 for r in json.loads(manifest.read_text())["roles"]})
    manifest = _fixture(tmp_path / "second")
    data = json.loads(manifest.read_text())
    data["roles"]["tune"]["families"] = data["roles"]["fit"]["families"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="family"):
        authenticate(manifest, {r: 1 for r in data["roles"]})


def test_private_cli_and_cold_reduction(tmp_path):
    """SCENARIO-REPORT-7740-TERMINAL: a fresh process checks exact raw bytes."""
    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "global_suite_debt.json").write_text(json.dumps({"exit_code": 2, "scope": "unrelated"}))
    output = tmp_path / "result.json"
    result = run_experiment(manifest, raw, output, "20260927", fixture=True, validate=False)
    assert result["verdict_class"] == "circular_positive"
    assert result["sentence_protocol_ready_score"] == 1
    assert result["validation_receipts"]["global_suite_debt"]["exit_code"] == 2
    assert {row["family_id"]: row["role"] for row in result["rows"]} == {
        row["family_id"]: row["role"] for row in result["annotation_coverage_rows"]
    }
    assert cold_reduce(raw, raw / "terminal_candidate.json")["families"] == 7
    cli_raw = tmp_path / "cli_raw"
    cli_output = tmp_path / "cli_result.json"
    cli = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/experiments/experiment_7740_v674_sentence_label_protocol.py",
            "--fixture-manifest",
            str(manifest),
            "--raw",
            str(cli_raw),
            "--output",
            str(cli_output),
            "--date",
            "20260927",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli_output.read_text())["verdict_class"] == "circular_positive"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_7740_v674_sentence_label_protocol",
            "--cold-reduce",
            str(raw),
            "--candidate",
            str(raw / "terminal_candidate.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    feature = raw / "features.jsonl"
    assert "annotations" not in feature.read_text() and '"label"' not in feature.read_text()
    feature.write_bytes(feature.read_bytes() + b" ")
    with pytest.raises(ValueError, match="features_sha256"):
        cold_reduce(raw, raw / "terminal_candidate.json")


def test_invalid_input_and_annotations(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7740-ISOLATION: malformed authority never becomes a fit label."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in protocol.ROLES}
    data = json.loads(manifest.read_text())
    data["schema"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="development_manifest_contract"):
        authenticate(manifest, counts)
    data["schema"] = "carnot.exp7727.development_manifest.v1"
    data["roles"]["fit"]["count"] = 2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="role_count_or_roster"):
        authenticate(manifest, counts)
    data["roles"]["fit"]["count"] = 1
    manifest.write_text(json.dumps(data))
    public = manifest.parent / "tune_public.jsonl"
    fit = json.loads((manifest.parent / "fit_public.jsonl").read_text())
    tune = json.loads(public.read_text())
    tune["family_id"] = fit["family_id"]
    public.write_text(json.dumps(tune) + "\n")
    data["roles"]["tune"]["families"] = [fit["family_id"]]
    data["roles"]["tune"]["public_sha256"] = _hash(public)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate_family_or_source"):
        authenticate(manifest, counts)
    monkeypatch.setattr(protocol.alignment, "sentence_spans", lambda _: [])
    with pytest.raises(ValueError, match="sentence_partition"):
        map_targets(b"text", [])


def test_capture_rejects_corrupt_evaluator(tmp_path):
    """SCENARIO-REPORT-7740-CUSTODY: checked hashes do not excuse bad offsets."""
    for field, value, error in (
        ("annotations", [{"start": 99, "end": 100, "text": "x"}], "annotation_offset_or_text"),
        ("label", 0, "response_label_mapping"),
        ("response_id", "wrong", "evaluator_join"),
    ):
        manifest = _fixture(tmp_path / field)
        data = json.loads(manifest.read_text())
        path = manifest.parent / "fit_evaluator.jsonl"
        label = json.loads(path.read_text())
        label[field] = value
        path.write_text(json.dumps(label) + "\n")
        data["roles"]["fit"]["evaluator_sha256"] = _hash(path)
        manifest.write_text(json.dumps(data))
        public, parsed, _ = authenticate(manifest, {role: 1 for role in protocol.ROLES})
        raw = tmp_path / field / "raw"
        raw.mkdir()
        with pytest.raises(ValueError, match=error):
            protocol.capture(manifest, raw, public, parsed, 0.0)


def test_cold_reducer_rejects_summary_and_target_tamper(tmp_path):
    """SCENARIO-REPORT-7740-TERMINAL: changed summaries fail even with valid JSON."""
    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    run_experiment(manifest, raw, tmp_path / "out.json", "20260927", fixture=True, validate=False)
    candidate = raw / "terminal_candidate.json"
    value = json.loads(candidate.read_text())
    value["sample_size_budget"]["completed"] = 6
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="candidate_summary_mismatch"):
        cold_reduce(raw, candidate)
    value["sample_size_budget"]["completed"] = 7
    candidate.write_text(json.dumps(value))
    target = raw / "fit_targets.jsonl"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="target_sha256"):
        cold_reduce(raw, candidate)


def test_validation_and_terminal_reader_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7740-TERMINAL: current check exits control qualification."""
    source = _fixture(tmp_path / "source")
    root = tmp_path / "repo"
    manifest = (
        root / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    for path in source.parent.iterdir():
        shutil.copy2(path, manifest.parent / path.name)
    monkeypatch.setattr(protocol, "ROOT", root)
    monkeypatch.setattr(protocol, "COUNTS", {role: 1 for role in protocol.ROLES})
    validation = sys.modules["carnot.reporting.experiment_7303_validation_scope"]

    def scoped_commands(*args, **kwargs):
        assert kwargs["basetemp"].is_dir()
        return ["scoped"]

    monkeypatch.setattr(validation, "build_scoped_commands", scoped_commands)
    responses = iter(
        [
            [{"name": "scoped", "passed": True, "exit_code": 0}],
            [
                {"name": name, "passed": True, "exit_code": 0}
                for name in ("cold_replay", "adversarial_verify", "verdict_row_consistency_strict")
            ],
        ]
    )
    monkeypatch.setattr(validation, "run_commands", lambda *a, **k: next(responses))
    raw = tmp_path / "validated"
    result = run_experiment(manifest, raw, tmp_path / "validated.json", "20260927", validate=True)
    assert result["verdict_class"] == "null"
    assert (raw / "terminal_checks.json").is_file()
    responses = iter(
        [
            [{"name": "scoped", "passed": True, "exit_code": 0}],
            [
                {"name": "cold_replay", "passed": True, "exit_code": 0},
                {"name": "adversarial_verify", "passed": False, "exit_code": 1},
                {"name": "verdict_row_consistency_strict", "passed": True, "exit_code": 0},
            ],
        ]
    )
    raw = tmp_path / "failed"
    result = run_experiment(manifest, raw, tmp_path / "failed.json", "20260927", validate=True)
    assert result["verdict_class"] == "disqualified"
    assert result["flagged_adversarial"] is True
    assert result["sentence_protocol_ready_score"] == 0
    with pytest.raises(ValueError, match="run_date"):
        run_experiment(manifest, tmp_path / "bad_date", tmp_path / "bad_date.json", "20260926")
    with pytest.raises(ValueError, match="production_manifest_path"):
        run_experiment(source, tmp_path / "wrong_path", tmp_path / "wrong_path.json", "20260927")


def test_main_dispatch_and_bad_cold_arguments(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7740-TERMINAL: both CLI modes use the same reducer."""
    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    output = tmp_path / "result.json"
    assert (
        protocol.main(
            ["--fixture-manifest", str(manifest), "--raw", str(raw), "--output", str(output)]
        )
        == 0
    )
    assert (
        protocol.main(
            ["--cold-reduce", str(raw), "--candidate", str(raw / "terminal_candidate.json")]
        )
        == 0
    )
    with pytest.raises(SystemExit, match="2"):
        protocol.main(["--cold-reduce", str(raw)])
    monkeypatch.setattr(
        sys,
        "argv",
        ["module", "--cold-reduce", str(raw), "--candidate", str(raw / "terminal_candidate.json")],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("carnot.experiment_7740_v674_sentence_label_protocol", run_name="__main__")
    assert exit_info.value.code == 0


def test_missing_external_manifest_is_terminal_block(tmp_path):
    """SCENARIO-REPORT-7740-TERMINAL: missing source names its exact gate operand."""
    missing = tmp_path / "absent" / "development_manifest.json"
    output = tmp_path / "blocked.json"
    result = run_experiment(missing, tmp_path / "blocked_raw", output, "20260927")
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"][0]["artifact_path"] == str(missing)
    assert result["gate_check_summary"][0]["expected"] is True
    assert result["gate_check_summary"][0]["observed"] is False
    assert result["sample_size_budget"]["started"] == 0
    assert json.loads(output.read_text()) == result


def test_capture_guards_feature_and_evaluator_counts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7740-CUSTODY: capture never repairs unexpected rows."""
    manifest = _fixture(tmp_path)
    public, parsed, _ = authenticate(manifest, {role: 1 for role in protocol.ROLES})
    raw = tmp_path / "raw"
    raw.mkdir()
    tick = count(0, 31)
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: next(tick)))
    original_feature = protocol._feature_row
    monkeypatch.setattr(protocol, "_feature_row", lambda row: {**original_feature(row), "label": 1})
    with pytest.raises(ValueError, match="label_bearing_feature_row"):
        protocol.capture(manifest, raw, public, parsed, 0.0)
    monkeypatch.setattr(protocol, "_feature_row", original_feature)
    original_read = protocol._read_jsonl
    monkeypatch.setattr(
        protocol,
        "_read_jsonl",
        lambda path: [] if path.name == "fit_evaluator.jsonl" else original_read(path),
    )
    with pytest.raises(ValueError, match="evaluator_count"):
        protocol.capture(manifest, raw, public, parsed, 0.0)


def test_cold_reducer_checks_each_sealed_view(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7740-TERMINAL: hashes and independently reduced rows both matter."""
    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    run_experiment(manifest, raw, tmp_path / "out.json", "20260927", fixture=True, validate=False)
    candidate = raw / "terminal_candidate.json"
    protocol_path = raw / "sentence_protocol_manifest.json"
    frozen_protocol = protocol_path.read_bytes()
    protocol_path.write_bytes(frozen_protocol + b" ")
    with pytest.raises(ValueError, match="protocol_sha256"):
        cold_reduce(raw, candidate)
    protocol_path.write_bytes(frozen_protocol)
    coverage_path = raw / "annotation_coverage.jsonl"
    original_bytes = coverage_path.read_bytes()
    coverage_path.write_bytes(original_bytes + b" ")
    with pytest.raises(ValueError, match="coverage_hash_or_count"):
        cold_reduce(raw, candidate)
    coverage_path.write_bytes(original_bytes)
    original_read = protocol._read_jsonl

    def changed(path, name, edit):
        rows = original_read(path)
        if path.name == name:
            edit(rows)
        return rows

    variants = [
        ("annotation_coverage.jsonl", lambda rows: rows.clear(), "coverage_hash_or_count"),
        (
            "features.jsonl",
            lambda rows: rows[1].update(family_id=rows[0]["family_id"]),
            "duplicate_feature_or_coverage",
        ),
        ("features.jsonl", lambda rows: rows[0].update(label=1), "label_bearing_feature_or_hash"),
        ("fit_targets.jsonl", lambda rows: rows.clear(), "target_count"),
        (
            "fit_targets.jsonl",
            lambda rows: rows[0].update(sentence_targets=[0]),
            "target_mapping_mismatch",
        ),
        (
            "annotation_coverage.jsonl",
            lambda rows: rows[0].update(known_sentence_count=0),
            "coverage_summary_mismatch",
        ),
    ]
    for name, edit, error in variants:
        monkeypatch.setattr(
            protocol, "_read_jsonl", lambda path, n=name, e=edit: changed(path, n, e)
        )
        with pytest.raises(ValueError, match=error):
            cold_reduce(raw, candidate)
