"""REQ-REPORT-8908 / SCENARIO-REPORT-8908-CONTRACT: paired criteria retain custody limits."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting import natural_criterion_audit_8908 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(count: int = 26) -> tuple[list[dict[str, Any]], ...]:
    """Small Unicode pairs expose offset and refusal mistakes without natural-truth claims."""
    messages, spans, responses, sources = [], [], [], []
    for i in range(count):
        ident = str(i)
        text = "é🙂 claim " + ident
        meta = dict(original_id=ident, model="fixture", quality="good", temperature=1.0)
        messages.extend(
            [
                dict(
                    stable_id="u" + ident,
                    message_thread_stable_id=ident,
                    position="0",
                    role="user",
                    text="source " + ident,
                    meta=json.dumps(meta),
                ),
                dict(
                    stable_id="a" + ident,
                    message_thread_stable_id=ident,
                    position="1",
                    role="assistant",
                    text=text,
                    meta=json.dumps(meta),
                ),
            ]
        )
        sources.append(
            dict(
                source_id="s" + ident,
                task_type="QA" if i % 2 else "Summary",
                prompt="source " + ident,
                source_info="source " + ident,
            )
        )
        responses.append(
            dict(
                id=ident,
                source_id="s" + ident,
                response=text,
                model="fixture",
                quality="good",
                temperature=1.0,
                split="test",
                labels=[],
            )
        )
        if i == 0:
            spans.append(
                dict(
                    stable_id="span0",
                    message_stable_id="a0",
                    start_pos="0",
                    end_pos="2",
                    span_text="é🙂",
                    label="Incorrect Refusal",
                    reasoning="",
                )
            )
        if i == 1:
            responses[-1]["labels"] = [
                dict(start=3, end=8, text="claim", label_type="Evident Conflict")
            ]
            spans.append(
                dict(
                    stable_id="span1",
                    message_stable_id="a1",
                    start_pos="3",
                    end_pos="8",
                    span_text="claim",
                    label="Subtle Introduction Of Baseless Info",
                    reasoning="",
                )
            )
    return messages, spans, responses, sources


def test_paired_denominators_and_annotation_independence() -> None:
    """REQ-REPORT-8908: refusal alone is no factual-failure label; category changes remain visible."""
    reduced = e.audit(*fixture(), authority=True)
    assert reduced["completed_count"] == 26
    assert reduced["intended_count"] == 408 and reduced["censored_count"] == 382
    assert len(reduced["join_rows"]) == len(reduced["rows"]) == 408
    assert reduced["independent_count"] == reduced["independent_generalization_score"] == 0
    assert reduced["disagreement_rows"][0]["numerator"] == 0
    assert reduced["disagreement_rows"][1]["numerator"] == 2
    assert all(r["denominator"] == 26 for r in reduced["disagreement_rows"])
    assert reduced["join_rows"][0]["revised_refusal_count"] == 1
    assert reduced["join_rows"][1]["original_evidence_failure_count"] == 1
    assert reduced["natural_canary_ready_score"] == 1
    assert len(reduced["canary_manifest"]) == 24
    assert len({r["unit_id"] for r in reduced["canary_manifest"]}) == 24
    assert reduced["canary_manifest_hash"] == canonical_hash(reduced["canary_manifest"])
    assert {r["source_task"] for r in reduced["canary_manifest"]} == {"QA", "Summary"}
    shuffled = fixture()
    shuffled[0].reverse()
    shuffled[1].clear()
    assert e.audit(*shuffled, authority=True)["canary_manifest"] == reduced["canary_manifest"]
    for count in (0, 23, 24):
        result = e.audit(*fixture(count), authority=True)
        assert result["natural_canary_ready_score"] == int(count >= 24)
    assert not e.audit(*fixture(), authority=False)["canary_manifest"]


@pytest.mark.parametrize(
    "mutation",
    [
        "response",
        "prompt",
        "context",
        "model",
        "split",
        "offset",
        "label",
        "missing_response",
        "missing_source",
        "duplicate_response",
        "duplicate_message",
        "duplicate_source",
        "duplicate_span",
        "duplicate_original",
        "missing_user",
        "meta",
        "role",
        "position",
    ],
)
def test_join_rejects_ambiguous_or_changed_records(mutation: str) -> None:
    """SCENARIO-REPORT-8908-CONTRACT: bad cases stay in the intended denominator."""
    data = fixture()
    messages, spans, responses, sources = data
    if mutation == "response":
        responses[0]["response"] += "changed"
    elif mutation == "prompt":
        sources[0]["prompt"] += "changed"
    elif mutation == "context":
        sources[0]["source_info"] = ""
    elif mutation == "model":
        responses[0]["model"] = "other"
    elif mutation == "split":
        responses[0]["split"] = "train"
    elif mutation == "offset":
        spans[0]["end_pos"] = "3"
    elif mutation == "label":
        spans[0]["label"] = "Unsupported"
    elif mutation == "missing_response":
        responses.pop(0)
    elif mutation == "missing_source":
        sources.pop(0)
    elif mutation == "duplicate_response":
        responses.append(deepcopy(responses[0]))
    elif mutation == "duplicate_message":
        messages.append(deepcopy(messages[1]))
    elif mutation == "duplicate_source":
        sources.append(deepcopy(sources[0]))
    elif mutation == "duplicate_span":
        spans.append(deepcopy(spans[0]))
    elif mutation == "duplicate_original":
        messages[3]["meta"] = messages[1]["meta"]
    elif mutation == "missing_user":
        messages.pop(0)
    elif mutation == "meta":
        messages[1]["meta"] = "not json"
    elif mutation == "role":
        messages[0]["role"] = "tool"
    else:
        messages[0]["position"] = "1"
    result = e.audit(*data, authority=True)
    assert result["excluded_count"] >= 1
    assert len(result["join_rows"]) == 408
    assert result["completed_count"] + result["excluded_count"] + result["censored_count"] == 408
    assert all(r["unit_id"] != "0" for r in result["canary_manifest"])


def test_schema_separation_offsets_and_unknown_mappings(tmp_path: Path) -> None:
    """REQ-REPORT-8908: independent headers and Unicode slices are trust boundaries."""
    path = tmp_path / "messages.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fixture()[0][0]))
        writer.writeheader()
        writer.writerows(fixture()[0])
    assert len(e.read_csv(path, "messages")) == 52
    with pytest.raises(ValueError, match="csv_schema"):
        e.read_csv(path, "spans")
    assert (
        e.normalize_spans(
            [dict(start=0, end=2, text="é🙂", label_type="Evident Conflict")], "é🙂", revised=False
        )[0]["label"]
        == "EvidentConflict"
    )
    for change in [dict(start=-1), dict(end=3), dict(text="é"), dict(label_type="Unknown")]:
        span = dict(start=0, end=2, text="é🙂", label_type="Evident Conflict")
        span.update(change)
        with pytest.raises(ValueError):
            e.normalize_spans([span], "é🙂", revised=False)
    with pytest.raises(ValueError, match="orphan_span"):
        e.audit(fixture()[0], [dict(message_stable_id="missing")], *fixture()[2:], authority=True)
    with pytest.raises(ValueError, match="roster_overflow"):
        e.audit(*fixture(409), authority=True)


def inputs(tmp_path: Path) -> Path:
    """Preserve original fixture operands for the same replay path used by author data."""
    messages, spans, responses, sources = fixture()
    files = {}
    for kind, name, rows in [("messages", "messages.csv", messages), ("spans", "spans.csv", spans)]:
        path = tmp_path / name
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[kind] = path
    for kind, rows in [("responses", responses), ("sources", sources)]:
        path = tmp_path / (kind + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[kind] = path
    for kind, text in [
        ("card", "Use the dataset for benchmarking hallucination detection models."),
        ("method", "Original RAGTruth human annotations; Blue Guardrails automatic predictions"),
        ("license", "MIT License"),
    ]:
        files[kind] = tmp_path / kind
        files[kind].write_text(text)
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = tmp_path / "input_manifest.json"
    atomic_json(
        manifest,
        dict(
            scope="private_fixture",
            release_revision=e.REVISION,
            original_revision=e.ORIGINAL_REVISION,
            files={k: runner.ref(v) for k, v in files.items()},
            upstreams=[],
            retrievals=[],
        ),
    )
    return manifest


def cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Fresh script-path children check package imports outside the checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *args]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run"]
    print("before exp8908 private CLI", flush=True)
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    print("after exp8908 private CLI", result.returncode, result.stdout, result.stderr, flush=True)
    return result


def test_actual_cli_cold_replay_and_semantic_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8908-CONTRACT: publication and replay reject repaired aggregate hashes."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = inputs(tmp_path)
    receipts = tmp_path / "receipts.json"
    atomic_json(
        receipts,
        dict(
            checks=[dict(name="fixture_validation", passed=True)],
            owned_statement_coverage=dict(
                percent_covered=100, missing_lines=[], excluded_lines=[], scope="private_fixture"
            ),
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    result = cli(
        tmp_path,
        "--date",
        "20261107",
        "--inputs",
        str(manifest),
        "--receipts",
        str(receipts),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["natural_canary_ready_score"] == 1
    assert value["phase_spans"][0]["phase"] == "input_acquisition"
    assert "private_pairs" not in value
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    for field, replacement in [
        ("completed_count", 999),
        ("natural_canary_ready_score", 0),
        ("rights_ledger", []),
        ("canary_manifest", []),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    atomic_json(output, value)
    source = Path(json.loads(manifest.read_bytes())["files"]["messages"]["path"])
    source.write_text(source.read_text() + "changed")
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert cli(tmp_path, "--date", "bad").returncode == 1
    assert cli(tmp_path, "--deliberate-error").returncode == 1
    with pytest.raises(ValueError):
        runner.read_ref(dict(path=str(source), sha256="wrong"))


def test_external_absence_and_owned_failure_are_distinct(tmp_path: Path) -> None:
    """REQ-REPORT-8908: missing operands block; owned validation disqualifies; audit can still be ready."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = inputs(tmp_path)
    primitive = runner.measure(manifest, tmp_path / (e.NAME + ".json"), "20261107")
    passed = dict(
        checks=[dict(name="unit", passed=True)],
        owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
    )
    assert runner.build(primitive, passed)["corpus_audit_ready_score"] == 1
    assert runner.build(primitive, dict(checks=[]))["verdict_class"] == "disqualified"
    real = runner.read_ref(primitive)
    real["reduction"]["input_scope"] = "author_release"
    guarded = tmp_path / "owned_gate_primitive.json"
    atomic_json(guarded, real)
    assert runner.build(runner.ref(guarded), passed)["verdict_class"] == "disqualified"
    missing = runner.measure(tmp_path / "absent.json", tmp_path / (e.NAME + ".json"), "20261107")
    blocked = runner.build(missing, passed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["observed"] == "missing_file"
    assert blocked["natural_canary_ready_score"] == 0
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    data = json.loads(manifest.read_bytes())
    Path(data["files"]["card"]["path"]).write_text("No usage permission")
    data["files"]["card"] = runner.ref(Path(data["files"]["card"]["path"]))
    atomic_json(manifest, data)
    result = runner.inspect(manifest)
    assert result["corpus_audit_ready_score"] == 1 and result["natural_canary_ready_score"] == 0


def test_metadata_identity_and_canary_shortage() -> None:
    """REQ-REPORT-8908: user provenance must agree; task shortage cannot read disagreement labels."""
    data = fixture()
    meta = json.loads(data[0][0]["meta"])
    meta["original_id"] = "other"
    data[0][0]["meta"] = json.dumps(meta)
    assert e.audit(*data, authority=True)["excluded_count"] == 1
    data = fixture()
    for source in data[3]:
        source["task_type"] = "QA"
    assert len(e.audit(*data, authority=True)["canary_manifest"]) == 24


def test_inspection_authority_and_hash_failures(tmp_path: Path) -> None:
    """REQ-REPORT-8908: original files and crosswalk bytes outrank repaired report assertions."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    path = inputs(tmp_path)
    original = json.loads(path.read_bytes())
    for kind in ["revision", "file_hash", "producer", "lineage", "crosswalk"]:
        data = deepcopy(original)
        if kind == "revision":
            data["release_revision"] = "other"
        elif kind == "file_hash":
            data["files"]["card"]["sha256"] = "wrong"
        elif kind == "producer":
            data["scope"] = "author_release"
        elif kind == "lineage":
            Path(data["files"]["method"]["path"]).write_text("no lineage")
            data["files"]["method"] = runner.ref(Path(data["files"]["method"]["path"]))
        else:
            atomic_json(tmp_path / "annotation_crosswalk.json", dict(changed=True))
        atomic_json(path, data)
        with pytest.raises(ValueError):
            runner.inspect(path)
        Path(original["files"]["method"]["path"]).write_text(
            "Original RAGTruth human annotations; Blue Guardrails automatic predictions"
        )
    (tmp_path / "annotation_crosswalk.json").unlink()
    atomic_json(path, original)
    readme = Path(original["files"]["card"]["path"])
    readme.write_text("changed")
    primitive = runner.measure(path, tmp_path / (e.NAME + ".json"), "20261107")
    assert runner.build(primitive, dict(checks=[]))["verdict_class"] == "disqualified"


def test_retrieval_failures_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8908: missing HTTP200, byte drift and rejected upstream identity fail closed."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    log = tmp_path / "curl.stdout"

    def failed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        log.write_text("404")
        return dict(passed=False, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", failed)
    monkeypatch.setattr(runner, "URLS", dict(messages="https://example.invalid/pinned"))
    with pytest.raises(OSError):
        runner.fetch_inputs(tmp_path / "http")

    def corrupt(name: str, argv: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        log.write_text("200")
        Path(argv[argv.index("-o") + 1]).write_text("wrong bytes")
        return dict(passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", corrupt)
    with pytest.raises(ValueError, match="rejected_producer"):
        runner.fetch_inputs(tmp_path / "hash")
    monkeypatch.setattr(runner, "URLS", {})
    upstream = dict(experiment_id=1, task_id="wrong")
    monkeypatch.setattr(runner, "read_ref", lambda reference: upstream)
    with pytest.raises(ValueError, match="upstream_identity"):
        runner.fetch_inputs(tmp_path / "upstream")


def test_replay_original_authority_and_owned_receipts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8908-CONTRACT: repairing envelope hashes cannot repair changed authority."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = inputs(tmp_path)
    source = runner.measure(manifest, tmp_path / (e.NAME + ".json"), "20261107")
    validation = dict(
        checks=[dict(name="unit", passed=True)],
        owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
    )
    value = runner.build(source, validation)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    primitive = runner.read_ref(source)
    for change in ["code", "private", "reduction", "receipt", "log"]:
        forged = deepcopy(primitive)
        checks = deepcopy(validation)
        if change == "code":
            forged["code_config_hashes"][0]["sha256"] = "wrong"
        elif change == "private":
            payload = tmp_path / "forged_payload.json"
            atomic_json(payload, dict(pairs=[]))
            forged["reduction"]["local_payload_reference"] = runner.ref(payload)
        elif change == "reduction":
            forged["reduction"]["completed_count"] += 1
        elif change == "receipt":
            forged["current_work_receipt"]["event_count"] = 1
        else:
            log = tmp_path / "validation.stdout"
            log.write_text("original")
            checks["checks"][0].update(stdout_path=str(log), stdout_sha256="wrong")
        path = tmp_path / "forged_primitive.json"
        atomic_json(path, forged)
        atomic_json(output, runner.build(runner.ref(path), checks))
        with pytest.raises(ValueError):
            runner.replay(output)
    blocked = runner.measure(tmp_path / "absent.json", output, "20261107")
    atomic_json(output, runner.build(blocked, validation))
    assert runner.replay(output)["passed"]


def test_main_default_retrieval_and_owned_disqualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8908: absent releases and failed validation terminate with different verdict classes."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = inputs(tmp_path)
    receipts = tmp_path / "receipts.json"
    atomic_json(
        receipts,
        dict(
            checks=[dict(name="unit", passed=True)],
            owned_statement_coverage=dict(percent_covered=100, missing_lines=[], excluded_lines=[]),
        ),
    )
    monkeypatch.setattr(runner, "fetch_inputs", lambda directory: manifest)
    monkeypatch.setattr(
        runner,
        "validate",
        lambda candidate, logs: dict(passed=True, checks=[dict(name="terminal", passed=True)]),
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--receipts", str(receipts), "--output", str(output)]) == 0

    def missing(directory: Path) -> Path:
        raise OSError("external absent")

    monkeypatch.setattr(runner, "fetch_inputs", missing)
    assert runner.main(["--receipts", str(receipts), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    calls = []

    def fail_once(candidate: Path, logs: Path) -> dict[str, Any]:
        calls.append(candidate)
        return dict(passed=len(calls) != 1, checks=[dict(name="owned", passed=len(calls) != 1)])

    monkeypatch.setattr(runner, "validate", fail_once)
    assert (
        runner.main(
            ["--inputs", str(manifest), "--receipts", str(receipts), "--output", str(output)]
        )
        == 1
    )
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert runner.main(["--date", "2026117"]) == 1


def test_retrieval_headers_and_manifest_separation(tmp_path: Path) -> None:
    """REQ-REPORT-8908: retrieval metadata must authenticate independently of payload bytes."""
    from carnot.reporting import natural_criterion_runner_8908 as runner

    manifest = inputs(tmp_path)
    data = json.loads(manifest.read_bytes())
    headers = tmp_path / "headers"
    headers.write_text("HTTP/2 200")
    data["retrievals"] = [dict(body=data["files"]["messages"], headers=runner.ref(headers))]
    atomic_json(manifest, data)
    assert runner.inspect(manifest)["completed_count"] == 26
    headers.write_text("changed")
    with pytest.raises(ValueError, match="retrieval_hash"):
        runner.inspect(manifest)
