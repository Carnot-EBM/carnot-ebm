"""REQ-REPORT-7754: qualify the exposed sentence protocol."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7754_v675_sentence_protocol as protocol
from test_experiment_7740_v674_sentence_label_protocol import _fixture


def test_child_basetemp_regression(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7754-CHILD: reproduce and repair the actual child setup error."""
    test_file = tmp_path / "test_private.py"
    test_file.write_text("def test_setup(tmp_path):\n    assert tmp_path.is_dir()\n")
    missing = tmp_path / "missing" / "nested" / "focused"
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        f"--basetemp={missing}",
        str(test_file),
        "-q",
    ]
    failed = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert "FileNotFoundError" in failed.stdout + failed.stderr
    protocol.prepare_basetemp(tmp_path / "missing" / "nested")
    passed = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stdout + passed.stderr


def test_unicode_mapping_and_unknowns() -> None:
    """SCENARIO-REPORT-7754-CUSTODY: byte and character positions stay distinct."""
    answer = "Café. Go!".encode()
    text = answer.decode()
    span = {"start": 3, "end": 4, "text": "é", "implicit_true": False}
    mapped = protocol.map_byte_targets(answer, [span])
    assert mapped["targets"] == [1, 0]
    assert mapped["sentence_byte_offsets"] == [[0, 7], [7, len(answer)]]
    assert mapped["annotation_byte_offsets"] == [[3, 5]]
    cross = {"start": 3, "end": len(text), "text": text[3:], "implicit_true": False}
    assert protocol.map_byte_targets(answer, [cross])["targets"] == [1, 1]
    assert protocol.map_byte_targets(answer, [])["targets"] == [0, 0]
    assert protocol.map_byte_targets(answer, None)["targets"] == [None, None]
    empty = {"start": 0, "end": 0, "text": "", "implicit_true": False}
    assert protocol.map_byte_targets(answer, [empty])["targets"] == [None, None]
    with pytest.raises(ValueError, match="annotation_offset_or_text"):
        protocol.map_byte_targets(answer, [{**span, "end": 100}])
    with pytest.raises(ValueError, match="annotation_offset_or_text"):
        protocol.map_byte_targets(answer, [{**span, "text": "x"}])


def test_label_mutation_does_not_change_features(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7754-CUSTODY: evaluator labels never enter public views."""
    manifest = _fixture(tmp_path)
    public = json.loads((manifest.parent / "fit_public.jsonl").read_text())
    baseline = protocol.feature_row(public)
    evaluator = manifest.parent / "fit_evaluator.jsonl"
    label = json.loads(evaluator.read_text())
    label["annotations"][0]["implicit_true"] = True
    evaluator.write_text(json.dumps(label) + "\n")
    assert protocol.feature_row(public) == baseline
    assert "label" not in baseline and "annotations" not in baseline


def test_private_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7754-TERMINAL: task CLI and reducer retain seven roles."""
    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    output = tmp_path / "result.json"
    result = protocol.run_experiment(
        manifest, raw, output, "20260927", fixture=True, validate=False
    )
    assert result["verdict_class"] == "circular_positive"
    assert result["sample_size_budget"]["completed"] == 7
    assert len(result["annotation_coverage_rows"]) == 7
    assert result["sentence_protocol_ready_score"] == 1
    assert protocol.cold_reduce(raw, raw / "terminal_candidate.json")["families"] == 7
    cli = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/experiments/experiment_7754_v675_sentence_protocol.py",
            "--fixture-manifest",
            str(manifest),
            "--raw",
            str(tmp_path / "cli_raw"),
            "--output",
            str(tmp_path / "cli_result.json"),
            "--date",
            "20260927",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert (
        json.loads((tmp_path / "cli_result.json").read_text())["verdict_class"]
        == "circular_positive"
    )
    feature = raw / "features.jsonl"
    feature.write_bytes(feature.read_bytes() + b" ")
    with pytest.raises(ValueError, match="features_sha256"):
        protocol.cold_reduce(raw, raw / "terminal_candidate.json")


def test_declared_producer_and_external_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7754-CUSTODY: producer and pre-gate failures are distinct."""
    checks, public, manifest, _ = protocol.preconditions(protocol.MANIFEST, False)
    assert all(item["passed"] for item in checks)
    assert len(public) == 640 and manifest["counts"]["fit"] == 256
    monkeypatch.setattr(protocol, "SOURCE", tmp_path / "missing_producer.json")
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    checks, public, manifest, shards = protocol.preconditions(tmp_path / "missing_manifest", False)
    assert not public and not manifest and not shards
    assert {item["upstream_id"] for item in checks if not item["passed"]} == {
        "exp7727",
        "exp7753_conductor_pre_gate",
    }
    result = protocol.run_experiment(
        tmp_path / "missing_manifest", tmp_path / "raw", tmp_path / "out.json", "20260927"
    )
    assert result["verdict_class"] == "blocked"
    assert result["sentence_protocol_ready_score"] == 0
    assert len(result["gate_check_summary"]) == 3
    with pytest.raises(ValueError, match="run_date"):
        protocol.run_experiment(
            tmp_path / "missing_manifest", tmp_path / "other", tmp_path / "x", "bad"
        )


def test_validation_success_and_failed_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7754-TERMINAL: all checks govern readiness."""
    manifest = _fixture(tmp_path)
    seen: list[bool] = []
    failing = False

    def fake_commands(
        _root: Path, commands: list, *, log_dir: Path, **_kwargs: object
    ) -> list[dict]:
        log_dir.mkdir(parents=True, exist_ok=True)
        seen.append(
            (Path("/tmp") / f"carnot-7754-{__import__('os').getpid()}" / "basetemp").is_dir()
        )
        rows = []
        for item in commands:
            bad = failing and item.name in {"ruff_format", "adversarial_verify"}
            log = log_dir / f"{item.name}.log"
            log.write_text("failed" if bad else "passed")
            rows.append(
                {
                    "name": item.name,
                    "passed": not bad,
                    "exit_code": int(bad),
                    "log_path": str(log),
                    "log_sha256": protocol.sha256_file(log),
                }
            )
        return rows

    monkeypatch.setattr(protocol, "run_commands", fake_commands)
    good = protocol.run_experiment(
        manifest,
        tmp_path / "good_raw",
        tmp_path / "good.json",
        "20260927",
        fixture=True,
        validate=True,
    )
    assert good["sentence_protocol_ready_score"] == 1
    assert all(seen)
    failing = True
    bad = protocol.run_experiment(
        manifest,
        tmp_path / "bad_raw",
        tmp_path / "bad.json",
        "20260927",
        fixture=True,
        validate=True,
    )
    assert bad["sentence_protocol_ready_score"] == 0
    assert bad["verdict_class"] == "disqualified"
    assert bad["flagged_adversarial"] is True
    assert bad["gate_check_summary"]


def test_cold_replay_rejects_changed_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7754-CUSTODY: even rehashed changed raw rows fail reduction."""
    for mode in ("hash", "count", "offset", "feature"):
        private = tmp_path / mode
        manifest = _fixture(private)
        raw = private / "raw"
        protocol.run_experiment(
            manifest, raw, private / "out.json", "20260927", fixture=True, validate=False
        )
        candidate_path = raw / "terminal_candidate.json"
        if mode == "feature":
            path = raw / "features.jsonl"
            rows = path.read_text().splitlines()
            row = json.loads(rows[0])
            row["sentence_count"] += 1
            rows[0] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n")
        else:
            path = raw / "byte_offsets.jsonl"
            rows = path.read_text().splitlines()
            if mode == "count":
                rows.pop()
            else:
                row = json.loads(rows[0])
                row["sentence_byte_offsets"][0][0] += 1
                rows[0] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n")
        if mode != "hash":
            evidence_path = raw / "evidence_manifest.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["features_sha256" if mode == "feature" else "byte_offsets_sha256"] = (
                protocol.sha256_file(path)
            )
            protocol.atomic_json(evidence_path, evidence)
            candidate = json.loads(candidate_path.read_text())
            candidate["source_artifact_hashes"]["pre_gate_receipts"]["evidence_manifest_sha256"] = (
                protocol.sha256_file(evidence_path)
            )
            protocol.atomic_json(candidate_path, candidate)
        with pytest.raises(
            ValueError,
            match={
                "hash": "byte_offsets_sha256",
                "count": "offset_count",
                "offset": "byte_offset_mapping_mismatch",
                "feature": "feature_recomputation_mismatch",
            }[mode],
        ):
            protocol.cold_reduce(raw, candidate_path)


def test_main_dispatch_and_module_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7754-TERMINAL: both CLI modes use the sealed candidate."""
    import runpy

    manifest = _fixture(tmp_path)
    raw = tmp_path / "raw"
    output = tmp_path / "result.json"
    assert (
        protocol.main(
            [
                "--fixture-manifest",
                str(manifest),
                "--raw",
                str(raw),
                "--output",
                str(output),
                "--date",
                "20260927",
            ]
        )
        == 0
    )
    assert (
        protocol.main(
            ["--cold-reduce", str(raw), "--candidate", str(raw / "terminal_candidate.json")]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        protocol.main(["--cold-reduce", str(raw)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment_7754_v675_sentence_protocol",
            "--cold-reduce",
            str(raw),
            "--candidate",
            str(raw / "terminal_candidate.json"),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("carnot.experiment_7754_v675_sentence_protocol", run_name="__main__")
    assert exit_info.value.code == 0


def test_cold_replay_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7754-CUSTODY: a long reducer reports completed units."""
    public = [
        {
            "family_id": str(i),
            "role": "fit",
            "complete_response": "A.",
            "response_sha256": "sha256:private",
        }
        for i in range(64)
    ]
    offsets = [
        {
            "family_id": str(i),
            "role": "fit",
            "response_sha256": "sha256:private",
            "sentence_byte_offsets": [[0, 2]],
            "annotation_byte_offsets": [],
        }
        for i in range(64)
    ]
    labels = [{"family_id": str(i), "annotations": []} for i in range(64)]
    path = tmp_path / "byte_offsets.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in offsets))
    (tmp_path / "evidence_manifest.json").write_text(
        json.dumps({"byte_offsets_sha256": protocol.sha256_file(path)})
    )
    (tmp_path / "sentence_protocol_manifest.json").write_text(
        json.dumps({"development_manifest_path": str(tmp_path / "source.json")})
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"sample_size_budget": {"role_counts": {"fit": 64}}}))
    monkeypatch.setattr(protocol.prior, "cold_reduce", lambda *_args: {"families": 64})
    monkeypatch.setattr(
        protocol.prior,
        "authenticate",
        lambda *_args: (public, {"roles": {"fit": {"evaluator_path": "fit_evaluator.jsonl"}}}, []),
    )
    monkeypatch.setattr(
        protocol.prior,
        "_read_jsonl",
        lambda p: (
            offsets
            if p.name == "byte_offsets.jsonl"
            else (
                [{"family_id": str(i)} for i in range(64)] if p.name == "features.jsonl" else labels
            )
        ),
    )
    monkeypatch.setattr(protocol, "feature_row", lambda row: {"family_id": row["family_id"]})
    assert protocol.cold_reduce(tmp_path, candidate)["families"] == 64
