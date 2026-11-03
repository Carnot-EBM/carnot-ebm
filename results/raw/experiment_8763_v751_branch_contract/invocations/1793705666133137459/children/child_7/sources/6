"""REQ-REPORT-V751-PLAN / REQ-REPORT-8763 / REQ-VERIFY-8763."""

from copy import deepcopy
from datetime import datetime, UTC
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pytest
import yaml

from carnot.reporting import v751_branch_contract as api
from carnot.reporting import v751_branch_evidence as arc
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_complete_contract_and_mutations(tmp_path):
    """SCENARIO-REPORT-8763-BRANCHES: every complete object has independent custody."""
    before = (api.ROOT / api.ARCHIVE).read_bytes()
    value = api.authority(api.ROOT)
    assert value["ready"] and len(value["rows"]) == 13
    assert api.prior.tasks_digest([r["task_object"] for r in value["rows"]]) == api.PIN
    assert all(r["raw_denominator"] > 0 for r in value["rows"])
    controls = api.mutations(tmp_path / "mutations")
    assert len(controls) >= 13 * 16 + 10 and all(r["passed"] for r in controls)
    assert not api.authority(tmp_path / "missing")["ready"]
    assert (api.ROOT / api.ARCHIVE).read_bytes() == before
    text = (api.ROOT / api.DESIGN).read_text()
    tasks = api.parse_design(text, milestone=api.MILESTONE)[1]
    design = tmp_path / api.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    for task in tasks:
        task["prompt"] = api.prior.prior.encode(task["prompt"])
    (tmp_path / api.ACTIVE).write_text(yaml.safe_dump(dict(milestone=api.MILESTONE, tasks=tasks)))
    assert api.authority(tmp_path)["ready"]
    tasks[0]["prompt"] += " changed"
    (tmp_path / api.ACTIVE).write_text(yaml.safe_dump(dict(milestone=api.MILESTONE, tasks=tasks)))
    assert not api.authority(tmp_path)["ready"]


def test_private_cli_replay_and_branch_predicates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8763-TERMINAL: actual CLI children seal private failure bytes."""
    output = tmp_path / (api.NAME + ".json")
    command = [sys.executable, str(api.ROOT / api.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("replay", ["--cold-replay", str(output)], 0),
        ("absent", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("date", ["--date", "20260101"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        receipt = child(name, command + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8763 and value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == value["learning_protocol_ready_score"] == 0
    assert value["arc_new_outcomes_available_score"] == 0
    assert value["MODEL_SPECS"] == [] and api.runtime.replay(output)
    work = api.prior.load(value["primitive_reference"])
    validation = api.prior.load(value["validation_reference"])
    good = deepcopy(work)
    good.update(
        private_control=False,
        historical_fixture_rows=[dict(passed=True)] * 3,
        mutation_rows=[dict(passed=True)],
        owned_failure_controls=[dict(passed=True)] * 2,
    )
    checks = dict(
        validation,
        receipts=[dict(passed=True)],
        union=dict(passed=True, node_ids=[]),
        coverage=dict(complete=True),
    )
    positive = api.build(good, checks, output)
    assert positive["learning_protocol_ready_score"] == positive["contract_ready_score"] == 1
    assert positive["arc_new_outcomes_available_score"] == 0
    assert positive["arc_gate_check_summary"] and positive["verdict_class"] == "positive"
    bad = deepcopy(checks)
    bad["receipts"][0]["passed"] = False
    failed = api.build(good, bad, output)
    assert failed["learning_protocol_ready_score"] == failed["contract_ready_score"] == 0
    assert failed["arc_new_outcomes_available_score"] == 0
    good["learning"]["ready"] = False
    assert api.build(good, checks, output)["verdict_class"] == "blocked"
    value["full_task_objects"][0]["prompt"] += "forged"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not api.runtime.replay(output)


def test_actual_original_private_control_gap(tmp_path, monkeypatch):
    """REQ-VERIFY-8763: original CLI executes line354, publishes failure and exits zero."""
    output = tmp_path / (api.prior.NAME + ".json")
    monkeypatch.chdir(tmp_path)
    cli = api.ROOT / api.prior.CLI
    with TemporaryDirectory(prefix="v751-gap-") as directory:
        root = api.archive750(Path(directory) / "archive")
        receipt = child(
            "original_private",
            [
                sys.executable,
                str(cli),
                "--root",
                str(root),
                "--private-control",
                "--output",
                str(output),
            ],
            tmp_path / "logs",
            deadline=120,
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
        assert child(
            "original_replay",
            [sys.executable, str(cli), "--cold-replay", str(output)],
            tmp_path / "logs",
            deadline=120,
        )["passed"]
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["required_checks_passed"] is False
    assert value["contract_ready_score"] == value["learning_protocol_ready_score"] == 0
    checks = api.prior.load(value["validation_reference"])
    assert checks["receipts"] == [dict(name="private_control", passed=False)]


def test_production_dispatch_retains_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8763: failed owned checks cannot be relabeled as qualified work."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api.runtime, "Coverage", lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(api.scope, "mutations", lambda root: [dict(passed=True)])
    monkeypatch.setattr(
        api.runtime,
        "validate",
        lambda *args: dict(
            receipts=[dict(passed=False)] * 8,
            union=dict(passed=False),
            coverage=dict(complete=False),
            raw_references=[],
            ended_at_utc=datetime.now(UTC).isoformat(),
        ),
    )
    output = tmp_path / (api.NAME + ".json")
    assert api.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0


def test_arc_empty_and_missing_are_separate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8763-BRANCHES: an empty authoritative ledger grants no ARC call."""
    value = arc.freeze(tmp_path)
    assert not value["ready"] and not value["new_outcome_refs"]
    assert value["gate_check_summary"][-1]["observed"] == 0
    assert value["gate_check_summary"][-1]["field"] == "new_selected_applied_outcomes"
    assert value["frontier_reference"]["sha256"] == arc.FRONTIER_PIN
    monkeypatch.setattr(arc, "FRONTIER", "missing.json")
    missing = arc.freeze(tmp_path / "absent")
    assert not missing["ready"] and missing["gate_check_summary"]
    assert missing["gate_check_summary"][0]["observed"] == "missing_file"


@pytest.mark.parametrize(
    "change", ["copied", "before", "selected", "applied", "outcome", "authority"]
)
def test_arc_rejects_unqualified_events(change):
    """REQ-REPORT-8763: copied bytes and absent actual outcomes cannot open Exp8773."""
    frontier = dict(closed_at_utc="2026-11-01T00:00:00+00:00", ledger_receipt_identities=["old"])
    row = dict(
        receipt_id="new",
        status="completed",
        chronology="after_cutoff",
        event_timestamp="2026-11-03T00:00:00+00:00",
        fired=True,
        arm="allow_reinduction",
        source_path="/private/real.json",
        source_sha256="sha256:new",
        helped=False,
        outcome="no_progress_within_horizon",
        applied=True,
        selected=True,
        numerator=0,
    )
    checks = dict(failures=[], rows=[row])
    assert arc.admit(checks, frontier) == [row]
    if change == "copied":
        row["receipt_id"] = "old"
    elif change == "before":
        row["event_timestamp"] = "2026-10-31T00:00:00+00:00"
    elif change == "authority":
        checks["failures"] = [dict(passed=False)]
    elif change == "outcome":
        row["outcome"] = None
    else:
        row[change] = False
    assert arc.admit(checks, frontier) == []


def test_validation_adapter_preserves_original_bytes(tmp_path, monkeypatch):
    """REQ-VERIFY-8763: private bindings leave all original V750 assertions unchanged."""

    monkeypatch.chdir(tmp_path)

    def checked(raw, tracer, suites):
        assert len(suites) == 3
        assert Path(suites[-1]).read_bytes() == (api.ROOT / api.prior.TESTS[0]).read_bytes()
        return dict(passed=True)

    monkeypatch.setattr(api, "base_validate", checked)
    assert api.validate(tmp_path, None, ["original748", "original749"]) == dict(passed=True)


@pytest.mark.parametrize(
    "control", ["new", "producer", "baseline", "registry", "locator", "health"]
)
def test_arc_private_operand_controls(tmp_path, monkeypatch, control):
    """REQ-REPORT-8763: private constructed receipts test admission, never scientific credit."""
    source = tmp_path / "native.json"
    atomic_json(
        source, dict(rows=[dict(game="g", seed=1, event_timestamp="2026-11-03T00:00:00+00:00")])
    )
    row = dict(
        game="g",
        seed=1,
        source_reference=api.prior.reference(source),
        producer_reference=api.prior.reference(source),
        receipt_id="new",
        status="completed",
        selected=True,
        applied=True,
        outcome="level_progress_observed",
    )
    monkeypatch.setattr(arc.original, "delta", lambda *a: ([deepcopy(row)], {}))
    monkeypatch.setattr(arc.original, "receipt_id_for_row", lambda r: "new")
    if control == "producer":
        qualified = arc.original.qualified

        def rejected(path, pin):
            value, refs = qualified(path, pin)
            value["arc_sensitivity_ready_score"] = 0
            return value, refs

        monkeypatch.setattr(arc.original, "qualified", rejected)
    elif control in {"baseline", "registry"}:
        qualified = arc.original.qualified

        def altered(path, pin):
            value, refs = qualified(path, pin)
            if control == "baseline":
                value["frontier_after"]["ledger_receipt_identities"] = []
            else:
                value["registry_precheck"]["sha256"] = "forged"
            return value, refs

        monkeypatch.setattr(arc.original, "qualified", altered)
    elif control == "locator":
        discover = arc.locator.discover

        def incomplete():
            value = discover()
            value["sources"] = []
            return value

        monkeypatch.setattr(arc.locator, "discover", incomplete)
    result = arc.freeze(tmp_path / "raw")
    assert result["ready"] == (control in {"new", "health"})
    if result["ready"]:
        assert result["new_outcome_refs"][0]["selected"] is True


def test_archive_bytes_cannot_be_rebound(tmp_path, monkeypatch):
    """REQ-REPORT-V751-PLAN: a changed V750 archive cannot enter current custody."""
    original = api.prior.reference
    monkeypatch.setattr(
        api.prior,
        "reference",
        lambda p: (
            dict(original(p), sha256="forged") if p == api.ROOT / api.ARCHIVE else original(p)
        ),
    )
    with pytest.raises(ValueError, match="v750_archive_changed"):
        api.freeze(tmp_path, api.ROOT)
