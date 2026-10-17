"""REQ-REPORT-8617; REQ-VERIFY-8617: human custody precedes readiness."""

from copy import deepcopy
import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import triviaplus_release_8617 as e
from carnot.reporting import triviaplus_release_runner_8617 as runner


@pytest.fixture
def private_release(tmp_path, monkeypatch):
    """Private byte pins exercise real parquet parsing without author-data dependence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "root"
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [record(), record(0, "train"), record(1, "valid")]
    pq.write_table(pa.Table.from_pylist(rows), cache / e.PARQUET)
    blobs = {}
    for name in e.BLOBS:
        path = cache / name
        if name != e.PARQUET:
            path.write_bytes(b"private source metadata")
        data = path.read_bytes()
        blobs[name] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    monkeypatch.setattr(e, "BLOBS", blobs)
    (root / "ops").mkdir(parents=True)
    (root / "ops/exclusion_manifest.yaml").write_text("private exclusions")
    for field, content in [
        (
            "METHOD",
            dict(
                triviaplus_coverage_plan={"internal_only": True},
                target_model={"name": "private numerical head"},
                methods_ready_score=1,
                required_checks_passed=True,
            ),
        ),
        ("PROTOCOL", {"private": True}),
    ]:
        path = root / (field + ".json")
        e.atomic_json(path, content)
        monkeypatch.setattr(e, field, str(path))
        monkeypatch.setattr(e, field + "_HASH", e.sha256_file(path))
    return root, cache


def artifact(private_release, tmp_path):
    """Bind private primitives and receipts exactly as a production invocation does."""
    root, cache = private_release
    output = tmp_path / "results" / (e.NAME + ".json")
    raw = output.parent / "raw" / output.stem
    work = runner.measure(root, raw, cache, "20261017")
    validation = dict(receipts=[dict(name="private_pass", passed=True)])
    e.atomic_json(raw / "validation.json", validation)
    value = runner.build(work, validation, raw, output)
    e.atomic_json(raw / "private_candidate.json", value)
    return work, validation, value, raw, output


def test_private_replay_and_rehashed_tampering(private_release, tmp_path):
    """SCENARIO-VERIFY-8617-CONTROLS: immutable original bytes defeat repaired hashes."""
    work, validation, value, raw, output = artifact(private_release, tmp_path)
    candidate = raw / "private_candidate.json"
    assert value["triviaplus_coverage_inputs_ready_score"] == 1
    assert e.replay(candidate)
    original = deepcopy(value)
    for key in ["rows", "overlap_rows", "primitive_reference", "code_config_hashes"]:
        changed = deepcopy(original)
        if key == "rows":
            changed[key][0]["source_length_bytes"] += 1
        elif key == "overlap_rows":
            changed[key][0]["status"] = "invented_disjointness"
        elif key == "primitive_reference":
            changed[key]["path"] = str(raw / "missing")
        else:
            changed[key][0]["sha256"] = "sha256:wrong"
        changed["reproducibility_checksum"] = e.canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        e.atomic_json(candidate, changed)
        assert not e.replay(candidate)
    e.atomic_json(candidate, original)
    saved = deepcopy(work)
    for mutate in ["refs", "codes", "plan"]:
        work = deepcopy(saved)
        if mutate == "refs":
            work["release_file_hashes"]["README.md"]["sha256"] = "wrong"
        elif mutate == "codes":
            work["code_refs"][0]["sha256"] = "wrong"
        else:
            work["plan"] = {"different": True}
        e.atomic_json(raw / "measurement.json", work)
        changed = runner.build(work, validation, raw, output)
        e.atomic_json(candidate, changed)
        assert not e.replay(candidate)


def test_missing_and_owned_failure(private_release, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8617-CUSTODY: external absence blocks; owned failure disqualifies."""
    work, validation, value, raw, output = artifact(private_release, tmp_path)
    validation["receipts"][0]["passed"] = False
    failed = runner.build(work, validation, raw, output)
    assert failed["verdict_class"] == "disqualified" and failed["flagged_adversarial"]
    root, cache = private_release
    monkeypatch.setattr(e, "acquire", lambda p: ({}, [e.gate(p, "missing", 1, None)]))
    missing = runner.measure(root, raw, cache, "20261017")
    blocked = runner.build(missing, dict(receipts=[dict(passed=True)]), raw, output)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["release_available_score"] == 0
    e.atomic_json(raw / "validation.json", dict(receipts=[dict(passed=True)]))
    blocked = runner.build(missing, dict(receipts=[dict(passed=True)]), raw, output)
    e.atomic_json(raw / "blocked.json", blocked)
    assert e.replay(raw / "blocked.json")
    method = e.read(e.reference(Path(e.METHOD)))
    method["methods_ready_score"] = 0
    e.atomic_json(Path(e.METHOD), method)
    monkeypatch.setattr(e, "METHOD_HASH", e.sha256_file(Path(e.METHOD)))
    assert any(
        g["field"] == "methods_ready_score"
        for g in runner.measure(root, raw, cache, "20261017")["gate_check_summary"]
    )
    with pytest.raises(ValueError, match="private_cache"):
        runner.measure(root, raw, root / "results/cache", "20261017")
    Path(e.METHOD).write_text("changed authority")
    assert runner.measure(root, raw, cache, "20261017")["gate_check_summary"]


def test_acquisition_bounds(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8617-CONTROLS: missing bytes, time and size limits are retained."""
    data = b"private download"
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    monkeypatch.setattr(e, "BLOBS", {"README.md": blob})
    monkeypatch.setattr(e.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    refs, gates = e.acquire(tmp_path / "success")
    assert refs and not gates

    class Oversize(bytes):
        def __len__(self):
            return 100 * 1024 * 1024 + 1

    class Stream(io.BytesIO):
        def read(self, *args):
            return Oversize(b"x")

    monkeypatch.setattr(e.urllib.request, "urlopen", lambda *a, **k: Stream())
    assert e.acquire(tmp_path / "oversize")[1]

    def missing(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(e.urllib.request, "urlopen", missing)
    assert e.acquire(tmp_path / "missing")[1]
    clocks = iter([0, 301])
    monkeypatch.setattr(e.time, "monotonic", lambda: next(clocks))
    assert e.acquire(tmp_path / "deadline")[1]


def test_predictor_custody_replay(private_release, tmp_path):
    """SCENARIO-VERIFY-8617-CONTROLS: replay checks private shards without rewriting them."""
    _, _, _, raw, _ = artifact(private_release, tmp_path)
    predictor = private_release[1] / "predictor.jsonl"
    predictor.write_text('{"target":1}\n')
    assert not e.replay(raw / "private_candidate.json")
    assert predictor.read_text() == '{"target":1}\n'


def test_roster_drift_controls(private_release, monkeypatch):
    """SCENARIO-VERIFY-8617-CONTROLS: sealed and newly read selection cannot drift."""
    cache = private_release[1]
    e.release_rows(cache)
    e.atomic_json(cache / "label_blind_roster.json", {"changed": True})
    with pytest.raises(ValueError, match="sealed_roster"):
        e.release_rows(cache, persist=False)
    original = e.seal_roster
    calls = 0

    def drift(rows):
        nonlocal calls
        calls += 1
        return original(rows) if calls == 1 else {"changed": True}

    monkeypatch.setattr(e, "seal_roster", drift)
    with pytest.raises(ValueError, match="roster_drift"):
        e.release_rows(cache)


def test_runner_publication(private_release, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8617-CONTROLS: publication readers agree on the checked primary."""
    root, cache = private_release
    output = tmp_path / "published" / (e.NAME + ".json")
    monkeypatch.setattr(runner, "run_checks", lambda raw: dict(receipts=[dict(passed=True)]))
    monkeypatch.setattr(
        runner,
        "validate_candidate",
        lambda path, raw: dict(
            passed=e.replay(path),
            receipts=[dict(name="row_lint", passed=True)],
            adversarial_findings=[],
        ),
    )

    def control(name, argv, logs, **kwargs):
        actual = int(not e.replay(Path(argv[-1])))
        return dict(name=name, passed=actual == kwargs["expected"], actual_exit=actual)

    monkeypatch.setattr(runner, "child", control)
    assert (
        runner.main(
            [
                "--date",
                "20261017",
                "--root",
                str(root),
                "--cache",
                str(cache),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert e.replay(output)
    assert runner.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert sidecar["reader_receipt"]["passed"]
    monkeypatch.setattr(runner, "run_checks", lambda raw: dict(receipts=[dict(passed=False)]))
    assert (
        runner.main(
            [
                "--date",
                "20261017",
                "--root",
                str(root),
                "--cache",
                str(cache),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


def test_validation_plan(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8617-CONTROLS: bounded children retain each original node once."""
    collected = []

    def collect(scratch, suite):
        nodes = [suite + "::test_one", suite + "::test_two"]
        collected.extend(nodes)
        return nodes, dict(name="collect_" + Path(suite).stem, passed=True)

    commands = []

    def child(name, argv, logs, **kwargs):
        commands.append((name, argv, kwargs))
        return dict(name=name, passed=True)

    monkeypatch.setattr(runner, "collect", collect)
    monkeypatch.setattr(runner, "child", child)

    def exact(plan):
        groups = [n for spec in plan if "junit_path" in spec for n in spec["collected_node_ids"]]
        return dict(passed=sorted(groups) == sorted(collected), name="exact_node_union")

    monkeypatch.setattr(runner, "exact_union", exact)
    for old in [None, "original-coverage"]:
        if old is None:
            monkeypatch.delenv("COVERAGE_FILE", raising=False)
        else:
            monkeypatch.setenv("COVERAGE_FILE", old)
        result = runner.run_checks(tmp_path / ("one" if old is None else "two"))
        assert all(r["passed"] for r in result["receipts"])
        assert all(k["deadline"] <= 180 for _, _, k in commands)
        collected.clear()
    for name, argv, _ in commands:
        if name.startswith("test_"):
            assert ["-n", "0", "-o", "addopts=", "--no-cov"] == argv[
                argv.index("-n") : argv.index("-n") + 5
            ]


def test_candidate_keeps_validator_findings(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8617-CONTROLS: one failed child defeats acceptance."""
    report = tmp_path / "adversarial.json"
    (tmp_path / "candidate").write_text("{}")
    report.write_text(json.dumps(dict(flagged_count=1, reports=[dict(flags=["actual finding"])])))

    def child(name, argv, logs, **kwargs):
        return dict(name=name, passed=name != "adversarial", stdout_path=str(report))

    monkeypatch.setattr(runner, "child", child)
    result = runner.validate_candidate(tmp_path / "candidate", tmp_path)
    assert not result["passed"] and result["adversarial_findings"]["flagged_count"] == 1


def record(index=0, split="test", labels=None):
    """Small private examples retain the author's real nested annotation shape."""
    votes = labels if labels is not None else [["supports"], ["contradicts"]]
    majority = "contradicts" if labels is None else "supports"
    return dict(
        article="Private complete source.",
        question=f"Question {index}?",
        answer="Private answer.",
        answer_sentence_list=["Private answer."],
        split=split,
        source="trivia",
        model="historical_generator",
        workerId=[str(i) for i in range(len(votes))],
        **{
            "sentence-level-word-labels": votes,
            "sentence_level_majority_vote": [majority],
            "response_level_label_binary": int(majority == "contradicts"),
            "response_level_label_binary_ws_llm_aaj": 0,
        },
    )


def test_human_aggregation():
    """SCENARIO-REPORT-8617-CUSTODY: strict ties, absent votes and noise stay distinct."""
    r = record(labels=[["supports"], ["not-mentioned"], [], [None]])
    r["sentence_level_majority_vote"] = ["not-mentioned"]
    r["response_level_label_binary"] = 1
    a = e.aggregate(r)
    assert a["binary_target"] == 1 and a["sentences"][0]["label"] == "not-mentioned"
    assert a["sentences"][0]["missing_count"] == 2
    assert a["sentences"][0]["disagreement_count"] == 1
    assert a["matches_release"] is True
    for labels in [["supplementary"], ["supports"], ["contradicts"], ["not-mentioned"]]:
        row = record(labels=[labels])
        row["sentence_level_majority_vote"] = labels
        row["response_level_label_binary"] = int(labels[0] in e.UNFAITHFUL)
        assert e.aggregate(row)["matches_release"]
    r["sentence-level-word-labels"] = [[]]
    assert e.aggregate(r)["binary_target"] is None
    r["sentence-level-word-labels"] = [["supports"], ["n/a"]]
    assert e.aggregate(r)["invalid_count"] == 0
    assert e.aggregate(r)["sentences"][0]["missing_count"] == 1
    r["sentence-level-word-labels"] = [["invalid"]]
    assert e.aggregate(r)["invalid_count"] == 1


def test_roster_and_predictor():
    """SCENARIO-REPORT-8617-CUSTODY: targets cannot change question selection."""
    rows = [record(i) for i in range(132)] + [record(0, "train")]
    roster = e.seal_roster(rows)
    assert len(roster["question_hashes"]) == 128
    changed = deepcopy(rows)
    for row in changed:
        row["response_level_label_binary"] = 1 - row["response_level_label_binary"]
    assert e.seal_roster(changed) == roster
    small = e.seal_roster([record(0), record(0, "train")])
    assert small["census"] and len(small["all_sibling_ids"]) == 2
    assert set(e.predictor_row(record(), "opaque")) == set(e.ALLOWLIST)
    assert e.question_hash("  Ａ B ") == e.question_hash("a b")


def test_reduce_and_missing_custody():
    """SCENARIO-REPORT-8617-CUSTODY: empty source and invalid votes close readiness."""
    rows = [record(), record(0, "train"), record(1, "valid")]
    value = e.reduce_rows(rows)
    assert value["annotation_authority_score"] == 1
    assert value["question_disjoint_score"] == 0
    assert value["rows"][0]["source_length_bytes"] > 0
    assert len(value["source_sibling_groups"]) == 1
    rows[0]["article"] = ""
    rows[1]["sentence_level_majority_vote"] = ["supports"]
    value = e.reduce_rows(rows)
    assert value["annotation_authority_score"] == 0
    assert not value["complete_custody"]


def test_authentication_and_replay_reject(tmp_path):
    """SCENARIO-VERIFY-8617-CONTROLS: a repaired local hash cannot replace a git blob."""
    path = tmp_path / "README.md"
    path.write_text("changed")
    with pytest.raises(ValueError, match="git_blob"):
        e.authenticate(path, "README.md")
    assert not e.replay(tmp_path / "absent.json")
    path.write_text("{}")
    assert not e.replay(path)


def test_cli_invalid_and_missing(tmp_path):
    """SCENARIO-VERIFY-8617-CONTROLS: real standalone CLI rejects bad date and evidence."""
    cli = str(e.ROOT / e.CLI)
    for args, code in [
        (["--date", "20260230"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        p = subprocess.run(
            [sys.executable, "-u", cli, *args], cwd=tmp_path, capture_output=True, timeout=60
        )
        assert p.returncode == code


def test_historical_overlap_unknown_and_known(tmp_path):
    """SCENARIO-REPORT-8617-CUSTODY: absent identity is unknown, never zero overlap."""
    p = tmp_path / "prior.json"
    p.write_text(
        json.dumps(
            {"question": "Question 0?", "rows": [{"question_hash": e.question_hash("Question 0?")}]}
        )
    )
    rows = [record()]
    inventory = e.overlap_inventory(rows, [p, tmp_path / "absent.json"])
    assert inventory[0]["overlap_question_count"] == 1
    assert inventory[1]["overlap_question_count"] is None
