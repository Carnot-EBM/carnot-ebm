"""REQ-REPORT-8789 / REQ-VERIFY-8789: exact custody and original failure coverage."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v753_branch_qualification as api
from carnot.reporting import v753_qualification_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_exact_authority_and_every_field(tmp_path):
    """SCENARIO-REPORT-8789-BRANCHES: complete fields and coordinated edits bind."""
    root = api.snapshot(api.ROOT, tmp_path / "authority")
    assert api.authority(root)["ready"]
    assert len(api.authority(root)["rows"]) == 12
    controls = api.mutations(root, tmp_path / "mutations")
    assert len(controls) > 200 and all(r["passed"] for r in controls)
    (root / api.ACTIVE).write_text("milestone: wrong\ntasks: []\n")
    assert not api.authority(root)["ready"]
    assert not api.authority(tmp_path / "absent")["ready"]


def test_memory_estimand_is_frozen():
    """REQ-REPORT-8789: streams are independent; seeds and arms are nested."""
    p = api.protocols([])["memory"]
    assert p["independent_streams"] == 30 and p["worlds_per_stream"] == 32
    assert len(p["seeds"]) == 3 and len(p["arms"]) == 5
    assert p["retention_worlds_per_stream"] == 12
    assert p["splits"] == dict(fit=40, tune=20)
    assert len(p["observed_features"]) == 6 and len(p["typed_decisions"]) == 3
    assert p["delay_queries"] == 8 and p["check_budget"] == 5 and p["template_limit"] == 32
    assert p["simultaneous_interval_percent"] == 99.5
    assert p["primary_Brier_upper_bound_lt"] == -0.01
    assert p["retention_and_error_upper_bound_le"] == 0.02
    assert p["reset_learned_state_between_streams"]
    assert p["old_learning_union_authorized"] is False
    assert api.memory_ready(p)
    p["independent_streams"] = 29
    assert not api.memory_ready(p)


def test_preflight_missing_resource_blocks(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8789-REPLAY: resources are checked before scientific work."""
    good = api.preflight(tmp_path)
    assert good["projection_fits"] and good["dependency_imports_passed"]
    assert good["measured_probe_s"] > 0 and good["peak_rss_bytes"] > 0
    assert len(good["original_source_checks"]) == 5
    assert all(r["passed"] for r in good["original_source_checks"])
    monkeypatch.setattr(api.shutil, "which", lambda name: None)
    assert not api.preflight(tmp_path)["projection_fits"]


def test_missing_import_and_source_are_explicit(tmp_path, monkeypatch):
    """REQ-VERIFY-8789: unavailable dependencies and missing bytes block before work."""

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(api.importlib, "import_module", missing)
    monkeypatch.setattr(api, "UPSTREAM_PINS", {"missing.json": "sha256:absent"})
    result = api.preflight(tmp_path)
    assert not result["dependency_imports_passed"] and not result["projection_fits"]
    assert result["upstream_checks"][0]["observed"] == "missing_file"


@pytest.mark.parametrize(
    ("setup_s", "parsing_s", "fits"),
    [(6.0, 0.2, True), (4200.0, 0.2, False), (0.2, 6.0, False)],
)
def test_preflight_charges_setup_once(tmp_path, monkeypatch, setup_s, parsing_s, fits):
    """SCENARIO-VERIFY-8789-PREFLIGHT-COST: imports are not repeated parsing work."""
    ticks = iter([0.0, setup_s, setup_s + parsing_s])
    monkeypatch.setattr(api.time, "monotonic", lambda: next(ticks))
    result = api.preflight(tmp_path)
    assert result["projection_fits"] is fits
    assert result["measured_setup_s"] == pytest.approx(setup_s)
    assert result["measured_probe_s"] == pytest.approx(parsing_s)
    assert result["projected_validation_s"] == pytest.approx(setup_s + parsing_s * 600 + 600)


def test_original_private_child_cannot_publish_readiness(tmp_path):
    """REQ-VERIFY-8789: real original failure statements execute without exclusions."""
    result = runner.original_failure(tmp_path)
    assert result["passed"] and result["receipt"]["normal_exit"]
    assert result["failed_pytest_receipt"]["normal_exit"]
    assert result["failed_pytest_receipt"]["exit_code"] == 1
    assert result["missing_statement_lines_executed"] == [88, 89, 90, 107]
    assert set(result["readiness_scores"].values()) == {0}


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-8789-REPLAY: CLI children retain failure and reject rehashing."""
    output = tmp_path / (api.NAME + ".json")
    command = [sys.executable, str(api.ROOT / api.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("replay", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("unsafe", ["--private-control"], 2),
        ("date", ["--date", "20260101"], 2),
    ]:
        receipt = child(name, command + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == value["memory_protocol_ready_score"] == 0
    assert api.replay(output)
    value["full_task_objects"][0]["prompt"] += "forged"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not api.replay(output)


def test_independent_scores_and_owned_failure(tmp_path):
    """REQ-REPORT-8789: optional absence does not suppress CPU qualification."""
    work = api.freeze(api.ROOT, tmp_path / "raw", "20261104")
    work["mutation_rows"] = [dict(passed=True)]
    work["original_failure"] = dict(passed=True)
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True, node_ids=[]),
        coverage=dict(complete=True),
        original_coverage=dict(complete=True),
        ended_at_utc=work["started_at_utc"],
        raw_references=[],
    )
    output = tmp_path / (api.NAME + ".json")
    atomic_json(tmp_path / "raw/work.json", work)
    atomic_json(tmp_path / "raw/validation.json", validation)
    value = api.build(work, validation, output)
    assert value["contract_ready_score"] == value["memory_protocol_ready_score"] == 1
    assert value["runtime_repair_ready_score"] == value["arc_new_outcomes_available_score"] == 0
    assert value["learning_execution_authorized"] is False
    validation["receipts"][0]["passed"] = False
    assert api.build(work, validation, output)["verdict_class"] == "disqualified"
    validation["receipts"][0]["passed"] = True
    work["contract"]["ready"] = False
    assert api.build(work, validation, output)["verdict_class"] == "blocked"


def test_production_dispatch_uses_real_validation(tmp_path, monkeypatch):
    """REQ-VERIFY-8789: dispatch cannot bypass owned validation or original custody."""
    monkeypatch.setattr(runner, "original_failure", lambda raw: dict(passed=True))
    monkeypatch.setattr(
        runner,
        "validate",
        lambda raw, original: dict(
            receipts=[dict(passed=False)],
            union=dict(passed=False, node_ids=[]),
            coverage=dict(complete=False),
            original_coverage=dict(complete=False),
            ended_at_utc=api.now(),
            raw_references=[],
        ),
    )
    output = tmp_path / (api.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_validation_merges_real_sqlite(tmp_path, monkeypatch):
    """REQ-VERIFY-8789: orchestration combines real SQLite and retains original nodes."""
    from types import SimpleNamespace

    database = tmp_path / "import.sqlite"
    receipt = child(
        "import_coverage",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--data-file=" + str(database),
            "-m",
            "carnot.reporting.v752_branch_methods",
        ],
        tmp_path / "logs",
    )
    assert receipt["passed"]
    ref = api.reference(database)
    result = dict(
        coverage=dict(combined_reference=ref),
        union=dict(node_ids=["original_family::test_original", "consumer::test_other"]),
        raw_references=[],
    )
    validator = SimpleNamespace(child=child)

    def validate(raw):
        """SCENARIO-VERIFY-8789-CANONICAL-LINT: relocation keeps original lint policy."""
        retained = runner.os.path.relpath(
            raw / "original_family/test_v752_branch_methods_8776.py", api.ROOT
        )
        receipt = validator.child(
            "ruff_check",
            [str(api.ROOT / ".venv/bin/ruff"), "check", retained],
            raw / "lint",
        )
        assert receipt["passed"], Path(receipt["stdout_path"]).read_text()
        assert receipt["argv"][-1] == api.old.TESTS[0]
        receipt = validator.child(
            "pytest_probe", [sys.executable, "-c", "pass", retained], raw / "lint"
        )
        assert receipt["passed"] and receipt["argv"][-1] == retained
        return result

    validator.validate = validate
    monkeypatch.setattr(runner, "namespace", lambda *args: validator)
    value = runner.validate(tmp_path, dict(sqlite_reference=ref, references=[ref]))
    wrapper = (tmp_path / "original_family/original_cli.py").read_text()
    assert "api.DESIGN =" in wrapper and "runner.terminal.CLI =" in wrapper
    assert value["original_node_inventory"] == ["original_family::test_original"]
    assert not value["original_coverage"]["complete"]
    assert value["original_coverage"]["combined_reference"]["sha256"]


def test_resource_block_publishes_without_measurement(tmp_path, monkeypatch):
    """REQ-VERIFY-8789: missing resources are blocked and never launch the controls."""
    preflight = api.preflight

    def unavailable(raw):
        return dict(preflight(raw), projection_fits=False)

    monkeypatch.setattr(api, "preflight", unavailable)
    monkeypatch.setattr(runner, "original_failure", lambda raw: pytest.fail("measurement launched"))
    output = tmp_path / (api.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["new_scientific_measurements"] == 0
