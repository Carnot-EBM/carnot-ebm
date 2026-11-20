"""REQ-REPORT-8403 and REQ-VERIFY-8403: identities do not grant unseen labels."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import released_question_identity_8403 as e
from carnot.reporting.current_work_receipt import atomic_json


def panel():
    """Private controls retain each generator so question counts stay independent."""
    master = [
        dict(
            item_id=f"{i}:{m}",
            dataset="hotpotqa",
            qid=f"q{i}",
            model=m,
            question=f"Question {i}?",
            answer_text=f"Answer {i} {m}",
            annotator1_label=str(i % 2),
            annotator2_label="-1",
        )
        for i in range(300)
        for m in e.old.GENERATORS
    ]
    audit = deepcopy(master)
    for row in audit:
        for arm in e.ARMS:
            row.update(
                {
                    f"label_{arm}": "0",
                    f"conf_{arm}": "0.8",
                    f"raw_{arm}": '{"label":0,"confidence":0.8}',
                    f"error_{arm}": "",
                }
            )
    return master, audit


def test_complete_factorial_and_distinct_gates():
    """SCENARIO-REPORT-8403-FACTORIAL: exposed targets cannot become independent."""
    master, audit = panel()
    result = e.join_panel(master, audit, [], ["multi_hop_unknown"])
    assert len(result["rows"]) == 900 and len(result["question_identity_rows"]) == 300
    assert result["release_criterion_ready_score"] == 1
    assert result["disjoint_eval_ready_score"] == 0 and result["overlap_unknown_count"] == 300
    assert result["human_targets_previously_opened"]
    known = e.join_panel(master, audit, [], [])
    assert known["disjoint_cluster_count"] == 300 and known["disjoint_eval_ready_score"] == 1
    assert known["class_support"] == {"0": 150, "1": 150}


@pytest.mark.parametrize(
    "mutation", ["generator", "question", "answer", "item", "target", "output", "duplicate"]
)
def test_changed_joins(mutation):
    """SCENARIO-REPORT-8403-FACTORIAL: every joined field and factorial arm is bound."""
    master, audit = panel()
    if mutation == "duplicate":
        audit[1] = deepcopy(audit[0])
    else:
        field = {
            "generator": "model",
            "question": "question",
            "answer": "answer_text",
            "item": "item_id",
            "target": "annotator1_label",
            "output": "raw_p1",
        }[mutation]
        audit[0][field] = "changed"
    assert e.join_panel(master, audit, [], [])["release_criterion_ready_score"] == 0


def test_namespaces_unicode_missing_and_duplicates():
    """SCENARIO-REPORT-8403-IDENTITY: bare IDs and full source hashes are insufficient."""
    master, audit = panel()
    prior = [dict(dataset="triviaqa", original_id="q0", question_hash=e.old.text_hash("unrelated"))]
    value = e.join_panel(master, audit, prior, [])
    assert value["exact_overlap_count"] == 0
    prior[0]["question_hash"] = e.question_hash("  ＱＵＥＳＴＩＯＮ\u00a00? ")
    assert e.join_panel(master, audit, prior, [])["exact_overlap_count"] == 1
    for rows in (master, audit):
        for row in rows[3:6]:
            row["dataset"] = "triviaqa"
            row["question"] = "QUESTION 0?"
    result = e.join_panel(master, audit, [], [])
    assert result["cross_dataset_duplicates"] and result["question_cluster_count"] == 299
    assert not result["release_criterion_ready_score"]
    missing = e.join_panel([], [], [], ["missing"])
    assert len(missing["rows"]) == 900 and missing["censored_count"] == 900
    assert not missing["release_criterion_ready_score"]


def test_recover_sources_and_fail_closed():
    """SCENARIO-REPORT-8403-IDENTITY: source bytes authenticate before question access."""
    info = {"question": "  ＦＡＣＴ? ", "passages": "Evidence"}
    slot = dict(
        source_id="7",
        source_bytes=json.dumps(info, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode()
        .hex(),
        role="fit",
    )
    source = dict(source_id="7", source="MARCO", source_info=info)
    recovered = e.recover_sources(
        [slot, dict(source_id="absent", source_bytes="", role="evaluation")], [source]
    )
    assert recovered[0]["question_hash"] == e.question_hash("fact?")
    assert recovered[0]["source_bytes_sha256"] != recovered[0]["question_hash"]
    assert recovered[1]["question_hash"] is None and recovered[1]["missing_reason"]
    slot["source_bytes"] = b"changed".hex()
    with pytest.raises(ValueError, match="source_bytes"):
        e.recover_sources([slot], [source])
    with pytest.raises(ValueError, match="duplicate_source"):
        e.recover_sources([], [source, source])
    source["source_info"] = "non-question source"
    slot["source_bytes"] = source["source_info"].encode().hex()
    assert e.recover_sources([slot], [source])[0]["question_hash"] is None


def test_real_pins_and_rehashed_mapping(tmp_path):
    """SCENARIO-VERIFY-8403-REPLAY: independent release roots reject new hashes."""
    manifest = e.input_manifest(e.ROOT)
    reduced = e.reduce_manifest(manifest)
    assert reduced["release_criterion_ready_score"] == 1
    assert len(reduced["prior_source_custody"]["rows"]) == 320
    assert reduced["prior_source_custody"]["question_identities_recovered"] == 105
    assert reduced["disjoint_eval_ready_score"] == 0
    changed = deepcopy(manifest)
    changed["target_mapping"]["primary"] = "label_p1"
    with pytest.raises(ValueError, match="target_mapping"):
        e.reduce_manifest(changed)
    changed = deepcopy(manifest)
    changed["primaries"]["8389"]["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="primary_pin"):
        e.reduce_manifest(changed)
    absent = e.reduce_manifest(e.input_manifest(tmp_path))
    assert absent["censored_count"] == 900 and absent["gate_check_summary"]


def blocked_work(tmp_path):
    """REQ-VERIFY-8403: absent upstream bytes remain auditable through real children."""
    from carnot.reporting import released_question_identity_runner_8403 as runner

    raw = tmp_path / "raw"
    raw.mkdir()
    work = runner.measure(tmp_path / "absent-root", raw, tmp_path)
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    return work


def test_build_replay_real_children_and_publication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8403-REPLAY: checked private publication keeps actual failure paths."""
    from carnot.reporting import released_question_identity_runner_8403 as runner
    from carnot.reporting.primary_publication import read_bound_sidecar

    work = blocked_work(tmp_path)
    receipts = [dict(passed=True, scope="owned")]
    value = e.build(work, receipts)
    assert value["verdict_class"] == "blocked" and value["release_criterion_ready_score"] == 0
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    controls = runner.controls(candidate, tmp_path / "controls")
    assert len(controls) == 4 and all(r["passed"] for r in controls)
    assert runner.validate(candidate, tmp_path / "validation")["passed"]
    value["completed_count"] += 1
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    monkeypatch.setattr(runner, "measure", lambda *a: work.copy())

    def measured_private_plan(private):
        from coverage import Coverage

        data = Coverage(
            config_file=False,
            data_file=str(private / ".coverage.control"),
            include=[str(e.ROOT / e.OWNED[0])],
        )
        data.start()
        e.progress("private_measured_coverage")
        data.stop()
        data.save()
        data.json_report(outfile=str(private / "coverage.json"))
        return []

    monkeypatch.setattr(runner, "plan", measured_private_plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--date", "20261011", "--output", str(output)]) == 0
    terminal = json.loads(output.read_bytes())
    publication = json.loads(Path(terminal["terminal_validation_sidecar_path"]).read_bytes())
    assert read_bound_sidecar(output, Path(publication["sidecar_path"]))["report"]["passed"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--deliberate-error"]) == 1
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert runner.main(["--output", str(output)]) == 1
    monkeypatch.setattr(runner, "measure", lambda *a: (_ for _ in ()).throw(ValueError("owned")))
    assert runner.main(["--output", str(output)]) == 1
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_plan_and_date_contract(tmp_path):
    """REQ-VERIFY-8403: scoped coverage and both private E2E commands are frozen."""
    from carnot.reporting import released_question_identity_runner_8403 as runner

    plan = runner.plan(tmp_path)
    assert any(s["name"] == "coverage_report" and "--fail-under=100" in s["argv"] for s in plan)
    assert any(s["name"] == "private_E2E019" for s in plan)
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261010"])


def test_invalid_human_target():
    """SCENARIO-REPORT-8403-FACTORIAL: invalid target encodings cannot qualify."""
    master, audit = panel()
    master[0]["annotator1_label"] = audit[0]["annotator1_label"] = "2"
    with pytest.raises(ValueError, match="human_label_encoding"):
        e.join_panel(master, audit, [], [])


@pytest.mark.parametrize(
    "mutation", ["release_mapping", "release_absent", "source_pin", "source_absent"]
)
def test_typed_reference_failures(monkeypatch, mutation):
    """SCENARIO-VERIFY-8403-REPLAY: typed references fail independently before reduction."""
    manifest = e.input_manifest(e.ROOT)
    primary = {key: json.loads(e.read_reference(ref)) for key, ref in manifest["primaries"].items()}
    release_ref = primary["8389"]["manifest_reference"]
    actual_read = e.old.read_reference

    def changed(ref):
        data = actual_read(ref)
        if mutation == "release_mapping" and ref == release_ref:
            value = json.loads(data)
            value["target_mapping"] = {}
            return json.dumps(value).encode()
        if mutation == "release_absent" and ref == release_ref:
            raise FileNotFoundError(2, "missing private release", ref["path"])
        if mutation == "source_pin" and ref == manifest["primaries"]["8375"]:
            value = json.loads(data)
            value["source_overlap_rows"][0]["local_corpus_references"] = []
            return json.dumps(value).encode()
        if mutation == "source_absent" and ref == primary["8305"]["measurement_reference"]:
            raise FileNotFoundError(2, "missing private measurement", ref["path"])
        return data

    monkeypatch.setattr(e.old, "read_reference", changed)
    if mutation in ("release_mapping", "source_pin"):
        with pytest.raises(ValueError):
            e.reduce_manifest(manifest)
    else:
        value = e.reduce_manifest(manifest)
        assert value["gate_check_summary"] and value["disjoint_eval_ready_score"] == 0


def test_real_measurement_resources_and_rehashed_join(tmp_path, monkeypatch):
    """REQ-VERIFY-8403: real task authority and operand hashes precede public measurement."""
    from carnot.reporting import released_question_identity_runner_8403 as runner
    import os

    previous = {k: os.environ.get(k) for k in ("COVERAGE_RCFILE", "COVERAGE_FILE")}
    runner.plan(tmp_path)
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    raw = tmp_path / "real"
    raw.mkdir()
    work = runner.measure(e.ROOT, raw, tmp_path)
    assert work["authority"]["activated"] and not work["gate_check_summary"]
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    value = e.build(work, [dict(passed=True, scope="owned")])
    assert value["release_criterion_ready_score"] == 1 and value["disjoint_eval_ready_score"] == 0
    candidate = raw / (e.NAME + ".json")
    atomic_json(candidate, value)
    from carnot.reporting.v709_execution import child

    receipt = child(
        "real_positive",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(candidate),
        ],
        raw / "cold",
        deadline=60,
    )
    assert receipt["passed"]
    primitive = json.loads(e.read_reference(work["primitive_reference"]))
    primitive["rows"][0]["generator"] = "swapped"
    primitive["rows"][0]["join_hash"] = e.canonical_hash(primitive["rows"][0])
    changed = raw / "rehashed_join.json"
    atomic_json(changed, primitive)
    work["primitive_reference"] = e.reference(changed)
    atomic_json(Path(work["work_path"]), work)
    atomic_json(candidate, e.build(work, [dict(passed=True, scope="owned")]))
    assert not e.replay(candidate)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    stopped = runner.measure(e.ROOT, tmp_path / "resource-failure", tmp_path)
    assert stopped["gate_check_summary"][0]["check"] == "resource_preconditions"
    assert json.loads(e.read_reference(stopped["primitive_reference"]))["completed_count"] == 0
