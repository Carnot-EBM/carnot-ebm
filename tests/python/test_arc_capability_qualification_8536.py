"""REQ-REPORT-8536: qualification cannot turn fixtures or history into new solves."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_capability_qualification_8536 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def checks():
    return [dict(name=n, passed=True) for n in e.REQUIRED_CHECKS]


def test_frozen_panel_and_original_history(tmp_path):
    """SCENARIO-REPORT-8536-PANEL: original hashes, games, budgets and censoring remain fixed."""
    source = e.authenticate(e.ROOT, tmp_path)
    assert source["motivation"] == dict(
        diversity_effects=8, disabled_latch_changes=8, censored_redirects=32, resolved_redirects=0
    )
    assert source["original_date"] == "20261014"
    panel = e.protocol()
    assert panel["games"] == ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
    assert panel["modes"] == ["baseline_applied", "capability_masked_applied"]
    assert len(panel["units"]) == 16
    assert all(u["max_actions"] == 512 and u["max_seconds"] == 120 for u in panel["units"])
    assert panel["measurement_task"] == "exp8537-arc-capability-panel"
    with pytest.raises(OSError):
        e.authenticate(tmp_path / "absent", tmp_path / "no")


def test_absent_inputs_owned_failure_and_replay(tmp_path):
    """SCENARIO-REPORT-8536-REPLAY: absence blocks; failed owned checks disqualify."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    path = tmp_path / "work.json"
    output = tmp_path / "results" / (e.NAME + ".json")
    atomic_json(path, work)
    value = e.build(path, checks(), output)
    assert value["verdict_class"] == "blocked"
    assert value["arc_capability_ready_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["headline_solve_credit"] == 0
    assert value["completed_count"] == 14
    assert all(
        {"check", "upstream", "path", "hash", "field", "operator", "expected", "observed"} <= set(g)
        for g in value["gate_check_summary"]
    )
    assert e.replay(value)
    failed = e.build(path, [dict(name="owned_tests", passed=False)], output)
    assert failed["verdict_class"] == "disqualified"
    assert not e.replay({})
    changed = deepcopy(value)
    changed["completed_count"] += 1
    assert not e.replay(changed)
    work["fixtures"]["selection_fixture_rows"][0]["selected_arms"] = ["invented"]
    atomic_json(path, work)
    changed = e.build(path, checks(), output)
    assert not e.replay(changed)


def test_historical_direct_rows_cannot_be_promoted(monkeypatch):
    """REQ-REPORT-8536: authenticate sealed operands without rerunning historical validation."""
    primary = e.checked(e.ROOT / e.UPSTREAM, e.UPSTREAM_PIN)
    work = e.checked(Path(primary["work_reference"]["path"]), primary["work_reference"]["sha256"])
    motivation = e.authenticate_operands(primary, work)
    assert motivation["censored_redirects"] == 32
    for field in ["effective_action_policy_change", "model_action_executed"]:
        changed = deepcopy(primary)
        event = next(
            r
            for r in changed["redirect_effect_rows"]
            if r["arm"]
            == (
                "force_exploration_diversity"
                if field == "effective_action_policy_change"
                else "allow_reinduction"
            )
        )
        event[field] = not event[field]
        with pytest.raises(ValueError, match="historical_effect"):
            e.authenticate_operands(changed, work)
    changed = deepcopy(work)
    changed["references"][0]["steps_sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="historical_primitive_hash"):
        e.authenticate_operands(primary, changed)


def test_replay_independent_rejection_paths(tmp_path):
    """SCENARIO-REPORT-8536-REPLAY: rehashing cannot remove independent authority checks."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    output = tmp_path / "results" / (e.NAME + ".json")
    for case in [
        "frozen",
        "protocol",
        "authority_hash",
        "task",
        "missing_source",
        "code",
        "log",
        "work_hash",
    ]:
        changed = deepcopy(work)
        receipts = checks()
        if case == "frozen":
            changed["frozen_panel"]["protocol"]["max_actions"] = 511
        elif case == "protocol":
            changed["frozen_panel"]["protocol"]["max_actions"] = 511
            frozen = tmp_path / "changed_panel.json"
            atomic_json(frozen, changed["frozen_panel"])
            changed["frozen_reference"] = dict(path=str(frozen), sha256=sha256_file(frozen))
        elif case in ["authority_hash", "task"]:
            active = tmp_path / "active.yaml"
            active.write_text("tasks: []\n")
            changed["activation_reference"] = dict(
                exists=True,
                snapshot_path=str(active),
                sha256="sha256:bad" if case == "authority_hash" else sha256_file(active),
            )
            changed["failures"] = []
        elif case == "missing_source":
            changed["failures"] = []
        elif case == "code":
            changed["code_config_hashes"] = dict(changed["code_config_hashes"])
            changed["code_config_hashes"][str(e.ROOT / e.POLICY)] = "sha256:bad"
        elif case == "log":
            log = tmp_path / "changed.stdout"
            log.write_text("actual")
            receipts.append(
                dict(name="log", passed=True, stdout_path=str(log), stdout_sha256="sha256:bad")
            )
        path = tmp_path / (case + "_work.json")
        atomic_json(path, changed)
        value = e.build(path, receipts, output)
        if case == "work_hash":
            value["work_reference"]["sha256"] = "sha256:bad"
        assert not e.replay(value), case


def test_ready_and_original_source_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8536: readiness requires actual task authority and original evidence."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")
    assert work["failures"] == []
    output = tmp_path / "results" / (e.NAME + ".json")
    path = tmp_path / "work.json"
    atomic_json(path, work)
    value = e.build(path, checks(), output)
    assert value["arc_capability_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["live_episodes_measured"] == 0
    assert value["unknown_capabilities"]
    assert e.replay(value)
    for field, replacement in [
        ("motivation", {}),
        ("original_date", "20261015"),
        ("historical_model_provenance", {}),
        ("authority_reference", {}),
    ]:
        changed = deepcopy(work)
        changed["source"][field] = replacement
        atomic_json(path, changed)
        assert not e.replay(e.build(path, checks(), output)), field
    changed = deepcopy(work)
    changed["activation_reference"]["exists"] = False
    atomic_json(path, changed)
    assert not e.replay(e.build(path, checks(), output))
    changed = deepcopy(work)
    changed["source"] = None
    atomic_json(path, changed)
    assert not e.replay(e.build(path, checks(), output))


def test_history_authority_and_deadline_refusals(tmp_path, monkeypatch):
    """REQ-REPORT-8536: original dates, authority and measured deadlines are strict."""
    original_checked = e.checked
    for case in ["validation", "authority_hash", "authority_task"]:

        def changed(path, digest):
            value = original_checked(path, digest)
            if case == "authority_hash" and "activation_reference" in value:
                value["activation_reference"]["sha256"] = "sha256:bad"
            return value

        with monkeypatch.context() as scoped:
            scoped.setattr(e, "checked", changed)
            if case == "validation":
                scoped.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
            if case == "authority_task":
                scoped.setattr(e, "canonical_hash", lambda _: "sha256:bad")
            with pytest.raises(ValueError, match="historical_"):
                e.authenticate(e.ROOT, tmp_path / case)
    from types import SimpleNamespace

    monkeypatch.setattr(e, "time", SimpleNamespace(monotonic=iter([0, 3601]).__next__))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")


def test_predicate_fields_keep_selection_and_effect_separate(tmp_path):
    """REQ-REPORT-8536: ledger eligibility, selected arm and actual effect are distinct."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    path = tmp_path / "work.json"
    atomic_json(path, work)
    value = e.build(path, checks(), tmp_path / "results" / (e.NAME + ".json"))
    for row in value["capability_predicate_rows"]:
        assert {
            "baseline_eligible",
            "capability_eligible",
            "capability_reason",
            "selected",
            "applied",
        } <= row.keys()
        assert not row["applied"] or row["selected"]
    assert all(r["masked"] for r in value["unknown_capabilities"])
