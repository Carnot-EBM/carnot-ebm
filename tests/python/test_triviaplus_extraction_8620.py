"""REQ-REPORT-8620; REQ-VERIFY-8620: lexical reach precedes calibration."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import triviaplus_extraction_8620 as e


def public(source="There are 12 birds. ", answer="There are 13 birds.", uid="a"):
    """Only original text bytes and an opaque join may reach the predictor."""
    return dict(
        opaque_id=uid, source_bytes=source.encode().hex(), answer_bytes=answer.encode().hex()
    )


def test_complete_source_and_fixed_scores():
    """SCENARIO-REPORT-8620-CENSUS: oversized sources abstain without truncation."""
    r = e.predict(public())
    assert r["fixed_scores"]["numeric_mismatch maximum"] == 1
    assert r["source_length_bytes"] == len(bytes.fromhex(public()["source_bytes"]))
    assert r["values"] is not None
    long = e.predict(public("Sentence. " * 80))
    assert long["source_window_count"] == 158
    assert long["abstention"] == "source_windows_over_budget"
    assert long["fixed_scores"] is None
    assert e.predict(public(answer="Yes. " * 17))["answer_unit_count"] == 17
    assert e.predict(public(source=""))["abstention"] == "empty_source"
    assert e.predict(public(answer=""))["abstention"] == "empty_answer"
    for changed in [
        dict(public(), label=1),
        dict(public(), opaque_id=""),
        dict(public(), source_bytes="zz"),
    ]:
        with pytest.raises(ValueError):
            e.predict(changed)


def test_predictor_child_and_duplicate_rejection(tmp_path):
    """SCENARIO-VERIFY-8620-CONTROLS: a real child gets only allowlisted rows."""
    inp, out = tmp_path / "input.jsonl", tmp_path / "features.json"
    inp.write_text(json.dumps(public()) + "\n")
    argv = [sys.executable, str(e.ROOT / e.CLI), "--extract", str(inp), "--features", str(out)]
    assert subprocess.run(argv, cwd="/var/tmp", capture_output=True).returncode == 0
    assert json.loads(out.read_text())["rows"] == [e.predict(public())]
    inp.write_text((json.dumps(public()) + "\n") * 2)
    assert subprocess.run(argv, capture_output=True).returncode == 1


def test_cluster_intervals_and_strata():
    """SCENARIO-REPORT-8620-SUPPORT: siblings never multiply support clusters."""
    rows = []
    for i in range(32):
        r = dict(
            e.predict(public(uid=str(i))),
            question_hash=str(i),
            binary_target=i % 2,
            annotation_disagreement=bool(i % 2),
            unsupported_count=i % 2,
            contradicted_count=0,
            strict_aggregation={},
            raw_votes=[],
        )
        r["fixed_scores"] = dict.fromkeys(e.SCORES, float(i % 2))
        rows.append(r)
    report = e.describe(rows)
    assert report["usable_question_support"]["usable_clusters"] == 32
    for r in report["descriptive_auroc_intervals"]:
        assert r["auroc"] == 1
        assert r["interval95"] == [1, 1]
        assert r["bootstrap_draws"] == 10000
    assert e.describe(rows[:8])["descriptive_auroc_intervals"][0]["interval95"] is None
    assert e.describe([])["descriptive_auroc_intervals"][0]["auroc"] is None
    assert len(report["source_length_strata"]) == 6


def test_annotation_join_retains_raw_and_strict():
    """REQ-REPORT-8620: annotation sensitivity cannot rewrite the primary target."""
    raw = dict(
        answer_sentence_list=["A."],
        workerId=["x", "y"],
        **{"sentence-level-word-labels": [["supports"], ["contradicts"]]},
        sentence_level_majority_vote=["contradicts"],
        response_level_label_binary=1,
    )
    rows = e.join_labels([e.predict(public(uid="0"))], [raw], {"0": "q"})
    assert rows[0]["binary_target"] == 1
    assert rows[0]["annotation_disagreement"]
    assert rows[0]["raw_votes"] == raw["sentence-level-word-labels"]
    raw["response_level_label_binary"] = 0
    with pytest.raises(ValueError, match="human_aggregation"):
        e.join_labels([e.predict(public(uid="0"))], [raw], {"0": "q"})


def test_missing_inputs_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8620-CONTROLS: missing evidence blocks; failed checks disqualify."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    work = runner.measure(tmp_path / "missing", tmp_path / "raw", "20261017")
    validation = dict(receipts=[dict(name="pass", passed=True)])
    e.atomic_json(Path(work["raw"]) / "validation.json", validation)
    value = runner.build(work, validation, tmp_path / "results" / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["extraction_audit_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"]
    validation["receipts"][0]["passed"] = False
    e.atomic_json(Path(work["raw"]) / "validation.json", validation)
    assert (
        runner.build(work, validation, tmp_path / "results" / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_replay_missing_and_cli_date(tmp_path):
    """SCENARIO-VERIFY-8620-CONTROLS: missing primitives and invalid dates reject."""
    assert not e.replay(tmp_path / "absent")
    cli = [sys.executable, str(e.ROOT / e.CLI)]
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(tmp_path / "absent")], capture_output=True
        ).returncode
        == 1
    )
    assert subprocess.run(cli + ["--date", "20261399"], capture_output=True).returncode == 2


@pytest.fixture
def private_inputs(tmp_path, monkeypatch):
    """Pinned private parquet fixtures exercise custody without distributing author text."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, cache = tmp_path / "root", tmp_path / "cache"
    cache.mkdir()
    original = []
    for i in range(3):
        original.append(
            dict(
                question=f"Question {i % 2}",
                split="test",
                article="A bird has 12 eggs. ",
                answer="A bird has 13 eggs.",
                answer_sentence_list=["A bird has 13 eggs."],
                workerId=["x", "y"],
                **{"sentence-level-word-labels": [["supports"], ["contradicts"]]},
                sentence_level_majority_vote=["contradicts"],
                response_level_label_binary=1,
                model="historical_fixture",
            )
        )
    pq.write_table(pa.Table.from_pylist(original), cache / e.release.PARQUET)
    pins = {}
    refs = {}
    for name in e.release.BLOBS:
        path = cache / name
        if name != e.release.PARQUET:
            path.write_text("private metadata")
        data = path.read_bytes()
        pins[name] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    monkeypatch.setattr(e.release, "BLOBS", pins)
    for name in pins:
        refs[name] = e.release.authenticate(cache / name, name)
    e.release.release_rows(cache)
    primitive = cache / "primitive.json"
    e.atomic_json(
        primitive,
        dict(cache=str(cache), private_predictor_reference=e.reference(cache / "predictor.jsonl")),
    )
    checkpoint = cache / "head.json"
    checkpoint.write_text("private numerical checkpoint")
    method = dict(
        methods_ready_score=1,
        required_checks_passed=True,
        target_model=dict(name="nine-input energy head", checkpoint=e.reference(checkpoint)),
        triviaplus_coverage_plan=dict(question_clusters=2),
    )
    upstream = dict(
        triviaplus_coverage_inputs_ready_score=1,
        required_checks_passed=True,
        release_file_hashes=refs,
        primitive_reference=e.reference(primitive),
        label_blind_roster=e.release.seal_roster(original),
        historical_model_provenance=["historical_fixture"],
        private_cache_reference=dict(path=str(cache)),
    )
    for name, value in [
        (e.release.METHOD, method),
        (e.RELEASE, upstream),
        (e.release.PROTOCOL, {}),
    ]:
        path = root / name
        e.atomic_json(path, value)
    monkeypatch.setattr(e.release, "METHOD_HASH", e.sha256_file(root / e.release.METHOD))
    monkeypatch.setattr(e.release, "PROTOCOL_HASH", e.sha256_file(root / e.release.PROTOCOL))
    monkeypatch.setattr(e, "RELEASE_HASH", e.sha256_file(root / e.RELEASE))
    (root / "ops").mkdir()
    (root / "ops/exclusion_manifest.yaml").write_text("private exclusions")
    return root, cache


def private_artifact(private_inputs, tmp_path):
    """Derive a complete fixture census through the real predictor child."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    root, _ = private_inputs
    raw = tmp_path / "raw"
    output = tmp_path / "results" / (e.NAME + ".json")
    work = runner.measure(root, raw, "20261017")
    assert not work["gate_check_summary"]
    validation = dict(receipts=[dict(passed=True)])
    e.atomic_json(raw / "validation.json", validation)
    value = runner.build(work, validation, output)
    path = raw / "candidate.json"
    e.atomic_json(path, value)
    return work, validation, value, path


def test_original_byte_replay_and_rehashed_mutations(private_inputs, tmp_path):
    """SCENARIO-VERIFY-8620-CONTROLS: both manifest and derived tampering reject."""
    work, validation, value, path = private_artifact(private_inputs, tmp_path)
    assert value["extraction_audit_ready_score"] == 1
    assert e.replay(path)
    for field in ["extraction_audit_ready_score", "extraction_rows", "primitive_reference"]:
        changed = deepcopy(value)
        if field == "extraction_audit_ready_score":
            changed[field] = 0
        elif field == "extraction_rows":
            changed[field][0]["values"][0] += 0.1
        else:
            changed[field]["path"] = str(tmp_path / "absent")
        changed["reproducibility_checksum"] = e.canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        e.atomic_json(path, changed)
        assert not e.replay(path)
    e.atomic_json(path, value)
    feature_path = Path(work["features_reference"]["path"])
    sealed = json.loads(feature_path.read_bytes())
    for field in ["rows", "predictor_reference"]:
        changed = deepcopy(sealed)
        if field == "rows":
            changed[field][0]["source_length_bytes"] += 1
        else:
            changed[field]["sha256"] = "sha256:wrong"
        e.atomic_json(feature_path, changed)
        changed_work = dict(work, features_reference=e.reference(feature_path))
        e.atomic_json(Path(work["raw"]) / "measurement.json", changed_work)
        changed_value = dict(
            value, primitive_reference=e.reference(Path(work["raw"]) / "measurement.json")
        )
        e.atomic_json(path, changed_value)
        assert not e.replay(path)
    e.atomic_json(feature_path, sealed)
    changed_work = deepcopy(work)
    changed_work["report"]["usable_question_support"]["usable_rows"] += 1
    e.atomic_json(Path(work["raw"]) / "measurement.json", changed_work)
    e.atomic_json(
        path, dict(value, primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"))
    )
    assert not e.replay(path)
    e.atomic_json(Path(work["raw"]) / "measurement.json", work)
    e.atomic_json(path, value)
    Path(
        work["code_refs"][0]["path"]
    )  # Leave owned code untouched; change a source operand instead.
    (private_inputs[0] / "ops/exclusion_manifest.yaml").write_text("changed")
    assert not e.replay(path)


def test_authority_and_descriptor_rejections(private_inputs, monkeypatch):
    """REQ-VERIFY-8620: derived hashes cannot invent author bytes or roster authority."""
    root, cache = private_inputs
    method, upstream, primitive = e.operands(root)
    for field, target, pin, key in [
        (e.release.METHOD, method, e.release, "METHOD_HASH"),
        (e.RELEASE, upstream, e, "RELEASE_HASH"),
    ]:
        path = root / field
        for edit in ["qualification", "roster", "reference"]:
            changed = deepcopy(target)
            if edit == "qualification":
                changed["required_checks_passed"] = False
            elif edit == "roster" and field == e.RELEASE:
                changed["label_blind_roster"]["question_hashes"] = []
            elif edit == "reference" and field == e.RELEASE:
                changed["release_file_hashes"][e.release.PARQUET]["bytes"] += 1
            else:
                continue
            e.atomic_json(path, changed)
            with monkeypatch.context() as patch:
                patch.setattr(pin, key, e.sha256_file(path))
                with pytest.raises(ValueError):
                    e.content(root)
            e.atomic_json(path, target)
    predictor = cache / "predictor.jsonl"
    original = predictor.read_bytes()
    predictor.write_bytes(b"[]")
    with pytest.raises(ValueError, match="private_descriptor_hash"):
        e.operands(root)
    changed_primitive = dict(primitive, private_predictor_reference=e.reference(predictor))
    primitive_path = Path(upstream["primitive_reference"]["path"])
    e.atomic_json(primitive_path, changed_primitive)
    changed = dict(upstream, primitive_reference=e.reference(primitive_path))
    e.atomic_json(root / e.RELEASE, changed)
    with monkeypatch.context() as patch:
        patch.setattr(e, "RELEASE_HASH", e.sha256_file(root / e.RELEASE))
        with pytest.raises(ValueError, match="descriptor_custody"):
            e.content(root)
    predictor.write_bytes(original)


def test_measure_owned_child_failure(private_inputs, tmp_path, monkeypatch):
    """REQ-VERIFY-8620: a failed predictor is owned failure, never an external block."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=False))
    work = runner.measure(private_inputs[0], tmp_path / "failed", "20261017")
    validation = dict(receipts=[dict(passed=True)])
    e.atomic_json(Path(work["raw"]) / "validation.json", validation)
    value = runner.build(work, validation, tmp_path / "results" / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"


def test_frozen_check_commands_and_receipt_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8620: commands preserve serial flags and original collection sets."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    def fake_child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text("{}")
        return dict(name=name, argv=argv, passed=True, stdout_path=str(out))

    monkeypatch.setattr(runner, "child", fake_child)
    monkeypatch.setattr(
        runner,
        "collect",
        lambda scratch, suite: (
            [suite + "::test_private"],
            dict(name="collect_" + suite, passed=True),
        ),
    )
    monkeypatch.setattr(runner, "exact_union", lambda plan: dict(name="exact_union", passed=True))
    for previous in [None, "private_previous"]:
        if previous is None:
            monkeypatch.delenv("COVERAGE_FILE", raising=False)
        else:
            monkeypatch.setenv("COVERAGE_FILE", previous)
        report = runner.run_checks(tmp_path / "checks")
        assert all(r["passed"] for r in report["receipts"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    assert runner.validate_candidate(candidate, tmp_path)["passed"]


@pytest.mark.parametrize("passed", [True, False])
def test_private_main_publication_and_reader(tmp_path, monkeypatch, passed):
    """REQ-VERIFY-8620: real replay and validators preserve blocked versus disqualified."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    monkeypatch.setattr(
        runner,
        "run_checks",
        lambda raw: dict(receipts=[dict(name="private_owned_control", passed=passed)]),
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            ["--date", "20261017", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("blocked" if passed else "disqualified")
    assert e.replay(output)
    assert json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
        "reader_receipt"
    ]["passed"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    with pytest.raises(SystemExit):
        runner.main(["--extract", str(tmp_path / "input")])


def test_measure_checkpoint_and_feature_roster_failures(private_inputs, tmp_path, monkeypatch):
    """REQ-VERIFY-8620: numerical provenance and predictor joins must authenticate."""
    from carnot.reporting import triviaplus_extraction_runner_8620 as runner

    root, cache = private_inputs
    checkpoint = cache / "head.json"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    assert runner.measure(root, tmp_path / "checkpoint", "20261017")["gate_check_summary"]
    checkpoint.write_bytes(original)

    def wrong_roster(name, argv, logs, **kwargs):
        e.atomic_json(Path(argv[-1]), dict(rows=[]))
        return dict(passed=True)

    monkeypatch.setattr(runner, "child", wrong_roster)
    assert runner.measure(root, tmp_path / "roster", "20261017")["gate_check_summary"]
