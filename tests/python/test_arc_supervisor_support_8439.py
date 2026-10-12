"""REQ-REPORT-8439 / REQ-VERIFY-8439: primitive custody precedes support claims."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_supervisor_support_8439 as e
from carnot.reporting import arc_supervisor_support_runner_8439 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-REPORT-8439-SUPPORT: real cached operands never execute games."""
    parent = tmp_path_factory.mktemp("support")
    return parent, e.measure(e.ROOT, parent / "raw", parent / "private")


def test_primitive_counts_and_censoring(measured):
    """SCENARIO-REPORT-8439-SUPPORT: censoring cannot become a negative outcome."""
    parent, work = measured
    value = e.build(work, [dict(passed=True)], parent / "raw", parent / (e.NAME + ".json"))
    assert value["completed_count"] == 8 and value["independent_count"] == 2
    assert len(value["supervisor_outcome_rows"]) == 12
    assert value["censoring_counts"]["pending_redirects"] == 12
    assert value["censoring_counts"]["resolved_redirects"] == 0
    assert len(value["per_game_results"]) == 16
    assert value["overlap_by_arm"]["force_exploration_diversity"]["overlapping_games"] == 2
    assert value["overlap_by_arm"]["allow_reinduction"]["overlapping_games"] == 1
    assert value["arm_refinement_authorized"] is False
    assert value["duplicate_solve_credit"] == value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["llm_calls"] == 0
    assert set(value) <= set(value["field_principles"])
    candidate = parent / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for key in ["completed_count", "arc_support_ready_score", "arm_refinement_authorized"]:
        changed = deepcopy(value)
        changed[key] = 999
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(candidate, changed)
        assert not e.replay(candidate)
    assert not e.replay(parent / "missing")


def test_source_and_episode_mutations(measured, monkeypatch):
    """SCENARIO-VERIFY-8439-CONTROLS: independent seals reject local rehashes."""
    _, work = measured
    source = e.producer(e.ROOT)
    operands = deepcopy(work["operands"])
    assert e.reduce(source, operands) == work["rows"]
    for field, replacement in [("episode_id", "mismatched"), ("arm", "on"), ("seed", 99)]:
        changed = deepcopy(operands)
        changed[0]["unit"][field] = replacement
        with pytest.raises(ValueError, match="episode_assignment"):
            e.reduce(source, changed)
    changed = deepcopy(operands)
    changed[0]["steps"]["sha256"] = "sha256:rehashed"
    with pytest.raises(ValueError, match="primitive_seal"):
        e.reduce(source, changed)
    changed = deepcopy(source)
    changed["rows"][0]["redirect_count"] += 1
    with pytest.raises(ValueError, match="summary_drift"):
        e.reduce(changed, operands)
    monkeypatch.setattr(e, "CANDIDATE_PIN", "sha256:wrong")
    with pytest.raises(ValueError):
        e.producer(e.ROOT)


def test_absence_owned_failure_and_null(tmp_path, measured, monkeypatch):
    """REQ-REPORT-8439: external absence and owned failure have different verdicts."""
    missing = e.measure(tmp_path, tmp_path / "raw", tmp_path / "private")
    assert missing["failures"] and not missing["rows"]
    value = e.build(missing, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["completed_count"] == 0
    assert value["intended_count"] == 8 and value["censored_count"] == 8
    _, work = measured
    clean = dict(work, failures=[])
    atomic_json(tmp_path / "measurement.json", clean)
    value = e.build(clean, [dict(passed=True)], tmp_path, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "null" and value["arc_support_ready_score"] == 1
    value = e.build(clean, [dict(passed=False)], tmp_path, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and value["arc_support_ready_score"] == 0
    monkeypatch.setattr(e, "producer", lambda root: (_ for _ in ()).throw(ValueError("absent")))
    failed = e.measure(e.ROOT, tmp_path / "failed", tmp_path / "private")
    assert any(g["field"] == "qualified_terminal_and_primitives" for g in failed["failures"])


def test_manifest_and_real_cli(tmp_path):
    """SCENARIO-REPORT-8439-CLI: actual child publication and consumers qualify bytes."""
    specs = runner.manifest(tmp_path)
    assert any(s["name"] == "historical_wrapper_assertions" for s in specs)
    assert any(s["name"] == "private_E2E018_consumers" for s in specs)
    output = tmp_path / (e.NAME + ".json")
    cli = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    assert child(
        "cli", [*cli, "--private-control", "--output", str(output)], tmp_path / "logs", deadline=180
    )["passed"]
    value = json.loads(output.read_bytes())
    assert e.replay(output)
    read = reader_receipt(
        e.TASK, tmp_path, field="arc_support_ready_score", expected=value["arc_support_ready_score"]
    )
    assert read["passed"] and read["gate_sha256"] == sha256_file(output)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("error", ["--deliberate-error"], 7),
        ("date", ["--date", "20261011"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]


def test_bounded_optional_frontier_and_resources(tmp_path, monkeypatch):
    """REQ-VERIFY-8439: optional absence and host failures never invent evidence."""
    from types import SimpleNamespace

    source = e.producer(e.ROOT)
    assert e.frontier(None, source, tmp_path)["status"] == "not_supplied"
    assert (
        e.frontier(e.ROOT / e.evidence.CANDIDATE, source, tmp_path)["status"] == "already_consumed"
    )
    changed = tmp_path / "new.json"
    atomic_json(changed, dict(supervisor_outcome_rows=[dict(episode_id="unqualified")]))
    assert e.frontier(changed, source, tmp_path)["status"] == "unqualified_new_frontier"
    assert e.frontier(tmp_path / "absent", source, tmp_path)["sha256"] is None
    changed.write_bytes(b"x" * 2_000_001)
    assert e.frontier(changed, source, tmp_path)["status"] == "unavailable_or_invalid"
    monkeypatch.setattr(e.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path == e.ROOT / ".venv/bin/python" else is_file(path),
    )
    missing = e.measure(tmp_path, tmp_path / "raw", tmp_path / "private")
    assert any(g["field"] == "private_disk_memory_resources" for g in missing["failures"])
    assert any(g["field"] == "tool_available" for g in missing["failures"])


def test_terminal_roster_and_primitive_failure(tmp_path, measured, monkeypatch):
    """SCENARIO-VERIFY-8439-CONTROLS: ID substitution fails even after local rehash."""
    source = e.producer(e.ROOT)
    _, work = measured
    with pytest.raises(ValueError, match="episode_roster"):
        e.reduce(source, [])
    side = (
        tmp_path
        / Path(e.evidence.CANDIDATE).parent
        / "validators"
        / (e.CANDIDATE_PIN[7:] + ".json")
    )
    side.parent.mkdir(parents=True)
    e.shutil.copyfile(e.ROOT / e.evidence.CANDIDATE, side.parent.parent / "terminal_candidate.json")
    atomic_json(
        side,
        dict(
            primary_sha256=e.CANDIDATE_PIN,
            primary_path=str((tmp_path / e.evidence.UPSTREAM).absolute()),
            report=dict(passed=False),
        ),
    )
    with pytest.raises(ValueError, match="unqualified_terminal"):
        e.evidence.producer(tmp_path)
    changed = deepcopy(work["operands"])
    forged = tmp_path / "episode.json"
    terminal = e.evidence.read(changed[0]["terminal"])
    terminal["unit"]["episode_id"] = "mismatch"
    atomic_json(forged, terminal)
    changed[0]["terminal"] = e.reference(forged)
    source["raw_shard_hashes"][
        str(Path(source["rows"][0]["primitive_directory"]) / "episode.json")
    ] = sha256_file(forged)
    with pytest.raises(ValueError, match="terminal_assignment"):
        e.reduce(source, changed)
    hashing = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda path: "sha256:corrupt" if path.name == "steps.jsonl" else hashing(path),
    )
    missing = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "private")
    assert not missing["rows"] and missing["failures"]


def test_real_child_failure_and_serial_orchestration(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8439-CLI: real child failure and genuine shards are retained."""
    private = tmp_path / "private"
    specs = runner.manifest(private)
    config = private / "coverage.ini"
    cli = str(e.ROOT / e.CLI)
    assert child(
        "coverage_error",
        [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--rcfile=" + str(config),
            cli,
            "--deliberate-error",
        ],
        tmp_path / "logs",
        expected=7,
        deadline=60,
    )["passed"]
    for name, args in [
        ("combine", ["combine", "--keep"]),
        ("json", ["json", "-o", str(private / "coverage.json")]),
    ]:
        assert child(
            name,
            [str(e.ROOT / ".venv/bin/coverage"), *args[:1], "--rcfile=" + str(config), *args[1:]],
            tmp_path / "logs",
            deadline=60,
        )["passed"]
    actual_failure = dict(
        name="owned_failure",
        argv=[str(e.ROOT / ".venv/bin/python"), "-u", "-c", "raise SystemExit(3)"],
        expected=0,
        deadline=30,
        scope="owned",
    )
    monkeypatch.setattr(runner, "manifest", lambda path: [actual_failure])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(tmp_path, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["required_checks_passed"] is False
    assert e.replay(output)
    assert any(not r["passed"] and r["exit_code"] == 3 for r in value["validation_receipts"])
    assert specs[0]["name"] == "owned_tests"


def test_rehashed_operand_and_receipt_rejections(tmp_path, measured):
    """SCENARIO-VERIFY-8439-CONTROLS: rehashing work cannot invent qualified inputs."""
    parent, original = measured
    value = e.build(original, [dict(passed=True)], parent / "raw", parent / (e.NAME + ".json"))
    path = tmp_path / "candidate.json"
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "invalid"
    atomic_json(path, changed)
    assert not e.replay(path)
    for kind in ["raw", "producer", "rows", "authority", "receipt"]:
        work = deepcopy(original)
        changed = deepcopy(value)
        if kind == "raw":
            work["raw_shard_hashes"][str(Path(work["producer_reference"]["path"]))] = "wrong"
        if kind == "producer":
            work["producer_reference"]["sha256"] = "wrong"
        if kind == "rows":
            work["rows"][0]["action_count"] += 1
        if kind == "authority":
            work["authority"]["activated"] = False
        if kind == "receipt":
            log = tmp_path / "log"
            log.write_text("actual changed child log")
            changed["validation_receipts"] = [
                dict(passed=True, stdout_path=str(log), stdout_sha256="wrong")
            ]
        operand = tmp_path / "measurement.json"
        atomic_json(operand, work)
        changed["work_reference"] = e.reference(operand)
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(path, changed)
        assert not e.replay(path)


def test_retired_task_and_changed_registry(tmp_path, monkeypatch):
    """REQ-VERIFY-8439: registry exposure and task retirement are checked before outcomes."""
    import yaml

    source = e.producer(e.ROOT)
    candidate = tmp_path / e.evidence.CANDIDATE
    candidate.parent.mkdir(parents=True)
    e.shutil.copyfile(e.ROOT / e.evidence.CANDIDATE, candidate)
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "exclusion_manifest.yaml").write_text("retired:\n- experiment_id: 8439\n")
    registry = yaml.safe_load((e.ROOT / "ops/arc_solve_registry.yaml").read_bytes())
    next(r for r in registry["games"] if r["game"] == "tn36")["levels_reproduced"] = 0
    (ops / "arc_solve_registry.yaml").write_text(yaml.safe_dump(registry))
    monkeypatch.setattr(e, "producer", lambda root: source)
    hashing = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda path: "wrong" if path.name == "steps.jsonl" else hashing(path)
    )
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path / "private")
    fields = {g["field"] for g in work["failures"]}
    assert {
        "task_not_retired",
        "unchanged_registry_precheck",
        "qualified_terminal_and_primitives",
    } <= fields
