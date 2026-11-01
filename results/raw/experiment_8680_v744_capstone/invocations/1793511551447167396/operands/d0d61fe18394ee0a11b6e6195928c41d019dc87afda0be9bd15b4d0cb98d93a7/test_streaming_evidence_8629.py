"""REQ-VERIFY-8629: independent eager expectations protect historical semantics."""

import hashlib
import json
from pathlib import Path
import random

import pytest

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import evidence_features_7980 as legacy
from carnot.verify import source_alignment as alignment
from carnot.verify import streaming_evidence as s


def public(source=b"Birds have 12 eggs. ", answer=b"Birds have 13 eggs."):
    """Only complete original bytes and an opaque identity reach extraction."""
    return dict(family_id="fixture", source_bytes=source.hex(), answer_bytes=answer.hex())


def eager(row):
    """Use the original eager parser and pair arithmetic, independent of streaming."""
    source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
    sentences, units = alignment.sentence_spans(source), alignment.sentence_spans(answer)
    windows = [(sum(map(len, sentences[:i])), w) for i, w in enumerate(sentences)]
    windows += [
        (sum(map(len, sentences[:i])), b"".join(sentences[i : i + 3]))
        for i in range(max(0, len(sentences) - 2))
    ]
    unique, records, digest = {}, [], hashlib.sha256()
    for i, (start, w) in enumerate(windows):
        key = w.strip()
        fresh = key not in unique
        unique.setdefault(key, len(unique))
        record = dict(
            index=i,
            start=start,
            end=start + len(w),
            window_hash=hashlib.sha256(key).hexdigest(),
            group_id=unique[key],
            first_occurrence=fresh,
            answer_unit_ids=list(range(len(units))),
        )
        records.append(record)
        digest.update(s.record_bytes(record))
    reason = (
        "answer_units_over_budget"
        if len(units) > 16
        else ("empty_answer" if not units else "empty_source" if not source.strip() else None)
    )
    pairs = [alignment.pair_features(w, units)[-4:] for w in unique] if not reason else []
    values = (
        [
            v
            for i in range(4)
            for v in (sum(p[i] for p in pairs) / len(pairs), max(p[i] for p in pairs))
        ]
        if pairs
        else None
    )
    result = dict(
        family_id=row["family_id"],
        values=values,
        abstention=reason,
        window_count=len(unique),
        source_normalized_hash=legacy.normalized(source),
    )
    return dict(result, feature_hash=canonical_hash(result)), records, digest.hexdigest()


@pytest.mark.parametrize("chunk", [1, 32, 128])
@pytest.mark.parametrize(
    "source,answer",
    [
        (b"", b"Yes."),
        (b"  \t", b"Yes."),
        (b"Yes.", b""),
        ("Élan ３.\u2003猫?! \nEnd.tail".encode(), "猫 ３?".encode()),
        (b"Same. \nSame. Same. ", b"Same."),
        (b"Birds have 12 eggs. No birds remain. ", b"Birds have 13 eggs. Not 12. Birds? " * 5),
        (b"A. B. C. ", b"Yes. " * 17),
        (b"".join(f"Unit {i}. ".encode() for i in range(65)), b"Unit 4."),
        (b"".join(f"Unit {i}. ".encode() for i in range(66)), b"Unit 4."),
    ],
)
def test_eager_exact_parity(tmp_path, chunk, source, answer):
    """SCENARIO-VERIFY-8629-PARITY: offsets and floats agree without tolerance."""
    row, log = public(source, answer), tmp_path / "windows.jsonl"
    expected, records, digest = eager(row)
    actual = s.extract(row, log, chunk_size=chunk)
    assert actual["features"] == expected
    assert [json.loads(line) for line in log.read_text().splitlines()] == records
    assert actual["complete_window_digest"] == digest
    if actual["total_window_count"] <= 128:
        assert actual["features"] == legacy.extract(row)


def test_seeded_properties_and_129_window_count(tmp_path):
    """SCENARIO-VERIFY-8629-PARITY: variable punctuation and duplicates stay ordered."""
    rng = random.Random(8629)
    for i in range(12):
        source = "".join(
            rng.choice(["猫! ", "No 12.\n", "Tail?\t", "x.y "])
            for _ in range(rng.randrange(1, 105))
        ).encode()
        row = public(source)
        expected = eager(row)
        for chunk in [1, 32, 128]:
            actual = s.extract(row, tmp_path / "p.jsonl", chunk_size=chunk)
            assert actual["features"] == expected[0]
            assert actual["complete_window_digest"] == expected[2]
    # Sentence+triple totals are even above two sentences:129 unique windows
    # require one duplicate among130 original windows, never a dropped window.
    row = public(b"Same. Same. " + b"".join(f"Unit {i}. ".encode() for i in range(64)))
    actual = s.extract(row, tmp_path / "129.jsonl")
    assert actual["features"]["window_count"] == 129
    assert actual["total_window_count"] == 130


@pytest.mark.parametrize(
    "changed",
    [
        {"label": 1},
        {"family_id": ""},
        {"family_id": 1},
        {"source_bytes": "zz"},
        {"answer_bytes": None},
        {"source_bytes": "ff"},
        {"answer_bytes": "ff"},
    ],
)
def test_public_rejections(tmp_path, changed):
    """SCENARIO-VERIFY-8629-REPLAY: predictors cannot receive outcomes or bad UTF-8."""
    with pytest.raises(ValueError):
        s.extract(dict(public(), **changed), tmp_path / "bad")


@pytest.mark.parametrize(
    "limit,expected",
    [
        ({"source_bytes": 1}, "source_bytes_over_budget"),
        ({"answer_bytes": 1}, "answer_bytes_over_budget"),
        ({"working_bytes": 1}, "working_memory_over_budget"),
        ({"seconds": 0}, "response_time_over_budget"),
        ({"disk_bytes": 1}, "dedup_disk_over_budget"),
    ],
)
def test_resource_exits(tmp_path, limit, expected):
    """REQ-VERIFY-8629: resource refusal is explicit and never a partial feature."""
    value = s.extract(public(), tmp_path / "resource", limits=s.Limits(**limit))
    assert value["features"]["values"] is None
    assert value["features"]["abstention"] == expected
    assert not value["complete"]


def test_collision_safe_and_batch_validation(tmp_path, monkeypatch):
    """REQ-VERIFY-8629: hashes identify records; exact bytes determine equality."""
    row = public(b"Alpha. Beta. Gamma. ")
    expected = eager(row)[0]
    monkeypatch.setattr(s, "window_hash", lambda _: "collision")
    assert s.extract(row, tmp_path / "collision")["features"] == expected
    for chunk in [0, 129, True]:
        with pytest.raises(ValueError, match="chunk_size"):
            s.extract(row, tmp_path / "invalid", chunk_size=chunk)


def test_measure_replay_and_rehashed_tampering(tmp_path):
    """SCENARIO-VERIFY-8629-REPLAY: frozen fixtures defeat repaired derived hashes."""
    from copy import deepcopy
    from carnot.reporting import streaming_source_8629 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261018")
    validation = dict(receipts=[dict(name="private-test", passed=True)])
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, validation, output)
    e.atomic_json(output, value)
    assert e.replay(output)
    for field in ["streaming_extractor_ready_score", "rows", "complete_window_digest"]:
        changed = deepcopy(value)
        changed[field] = 999
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        e.atomic_json(output, changed)
        assert not e.replay(output)
    # Rewrite the original primitive and its manifest hash together.
    log = Path(work["chunk_rows"][9]["primitive"]["path"])
    original = log.read_bytes()
    log.write_bytes(original.replace(b'"start":0', b'"start":1', 1))
    assert log.read_bytes() != original
    work["chunk_rows"][9]["primitive"] = e.reference(log)
    e.atomic_json(Path(work["raw"]) / "measurement.json", work)
    changed = e.build(work, validation, output)
    e.atomic_json(output, changed)
    assert not e.replay(output)


def test_external_missing_and_owned_failure(tmp_path):
    """REQ-VERIFY-8629: absent branch inputs block and owned failures disqualify."""
    from carnot.reporting import streaming_source_8629 as e

    work = e.measure(tmp_path / "missing", tmp_path / "raw", "20261018")
    validation = dict(receipts=[dict(name="private-test", passed=True)])
    value = e.build(work, validation, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    validation["receipts"][0]["passed"] = False
    assert (
        e.build(work, validation, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    )


def test_real_cli_measure_and_replay(tmp_path):
    """SCENARIO-VERIFY-8629-REPLAY: actual children accept date and reject missing data."""
    import subprocess
    import sys
    from carnot.reporting import streaming_source_8629 as e

    cli = [sys.executable, str(e.ROOT / e.CLI)]
    output = tmp_path / "measurement.json"
    assert (
        subprocess.run(
            cli + ["--date", "20261018", "--measure-only", str(output)],
            cwd="/var/tmp",
            capture_output=True,
        ).returncode
        == 0
    )
    work = json.loads(output.read_bytes())
    candidate = tmp_path / (e.NAME + ".json")
    e.atomic_json(candidate, e.build(work, dict(receipts=[dict(passed=True)]), candidate))
    assert (
        subprocess.run(cli + ["--cold-replay", str(candidate)], capture_output=True).returncode == 0
    )
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(tmp_path / "missing")], capture_output=True
        ).returncode
        == 1
    )
    assert subprocess.run(cli + ["--date", "20261399"], capture_output=True).returncode == 2


def test_orchestrator_checked_publication(tmp_path, monkeypatch):
    """REQ-VERIFY-8629: exact candidate validation and both publication readers run."""
    from carnot.reporting import streaming_source_8629 as e
    from carnot.reporting import streaming_source_runner_8629 as runner
    from carnot.reporting import triviaplus_extraction_runner_8620 as shipped

    # Avoid a nested historical controller; actual outer validation owns the union.
    monkeypatch.setattr(shipped, "run_checks", lambda _: dict(receipts=[dict(passed=True)]))
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261018", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["streaming_extractor_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert e.replay(output)


def test_replay_owned_mutations_and_resource_tail(tmp_path):
    """SCENARIO-VERIFY-8629-REPLAY: every derived authority mutation fails closed."""
    from copy import deepcopy
    from carnot.reporting import streaming_source_8629 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261018")
    validation = dict(receipts=[dict(passed=True)])
    output = tmp_path / (e.NAME + ".json")
    for mode in ["code", "source", "chunk", "feature", "primitive"]:
        changed = deepcopy(work)
        if mode in ["code", "source"]:
            changed[mode + "_refs"][0]["sha256"] = "sha256:bad"
        elif mode == "chunk":
            changed["chunk_rows"].pop()
        elif mode == "feature":
            changed["chunk_rows"][0]["features"]["values"] = [999] * 8
        e.atomic_json(Path(work["raw"]) / "measurement.json", changed)
        value = e.build(changed, validation, output)
        if mode == "primitive":
            value["primitive_reference"]["sha256"] = "sha256:bad"
        e.atomic_json(output, value)
        assert not e.replay(output)
    for source, limits, reason in [
        (b"", s.Limits(seconds=0), "response_time_over_budget"),
        (b"Yes.", s.Limits(disk_bytes=8192), "dedup_disk_over_budget"),
    ]:
        result = s.extract(public(source), tmp_path / "tail", limits=limits)
        assert not result["complete"] and result["features"]["abstention"] == reason


def test_registered_byte_boundaries(tmp_path):
    """REQ-VERIFY-8629: caps refuse bytes explicitly and16 answer units still work."""
    answer = b"A. " * 16
    row = public(b"a" * (512 * 1024), answer)
    assert s.extract(row, tmp_path / "cap")["complete"]
    for changed, reason in [
        (public(b"a" * (512 * 1024 + 1)), "source_bytes_over_budget"),
        (public(answer=b"a" * (64 * 1024 + 1)), "answer_bytes_over_budget"),
    ]:
        assert s.extract(changed, tmp_path / "over")["features"]["abstention"] == reason


def test_original_coverage_retention(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8629-REPLAY: original shards survive combine and scratch cleanup."""
    from carnot.reporting import streaming_source_runner_8629 as runner
    from carnot.reporting import triviaplus_extraction_runner_8620 as shipped

    source = tmp_path / "coverage"
    source.mkdir()
    shard = source / ".coverage.original"
    shard.write_bytes(b"original shard bytes")
    receipt = dict(name="coverage_combine", argv=[str(source)], passed=True)
    monkeypatch.setattr(shipped, "run_checks", lambda _: dict(receipts=[receipt]))
    raw = tmp_path / "raw"
    value = runner.run_checks(raw)
    archive = value["receipts"][-1]
    assert archive["name"] == "original_coverage_retention" and archive["passed"]
    retained = Path(archive["retained_shards"][0]["path"])
    assert retained.read_bytes() == shard.read_bytes()
    assert shard.read_bytes() == b"original shard bytes"
