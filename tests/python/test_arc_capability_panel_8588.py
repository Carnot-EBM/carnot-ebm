"""REQ-REPORT-8588, REQ-VERIFY-8588: preserve live panel and terminal authority."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import arc_capability_panel_8588 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(tmp_path, *args):
    """Real children expose import and publication errors hidden by in-process calls."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=240)


def test_original_readiness_and_protocol(tmp_path, monkeypatch):
    """REQ-REPORT-8588: original hashes and independent readiness precede collection."""
    with e.adapters():
        source = e.authenticate(e.ROOT, tmp_path)
    assert source["publication"]["arc_panel_execution_ready_score"] == 1
    assert len(source["protocol"]["units"]) == 16
    monkeypatch.setattr(e, "PUBLICATION_PIN", "sha256:bad")
    with e.adapters(), pytest.raises(ValueError):
        e.authenticate(e.ROOT, tmp_path)


def test_upstream_fields_and_operand_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8588-REPLAY: self-consistent envelopes cannot hide bad gates."""
    publication = json.loads((e.ROOT / e.PUBLICATION).read_bytes())
    publication["arc_panel_execution_ready_score"] = 0
    monkeypatch.setattr(e, "checked", lambda *_: publication)
    with pytest.raises(ValueError, match="publication_readiness"):
        e.authenticate(e.ROOT, tmp_path)


def test_associations():
    """REQ-REPORT-8588: observed associations never claim causal counterfactuals."""
    assert e.association(dict(resolved_by_levelup=True)) == "progress_associated"
    assert e.association(dict(observed_level_delta=-1)) == "regression_associated"
    assert e.association({}) == "unresolved"


@pytest.mark.parametrize("failed", [False, True])
def test_private_cli_and_rehashed_tamper(tmp_path, failed):
    """SCENARIO-VERIFY-8588-CONTROLS: actual private CLI publishes blocked or failed bytes."""
    output = tmp_path / (e.NAME + ".json")
    arguments = ["--private-control", "--root", tmp_path / "absent", "--output", output]
    if failed:
        arguments.append("--owned-failure")
    result = cli(tmp_path, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert value["arc_panel_rows_ready_score"] == 0
    assert value["completed_count"] == 0 and value["intended_count"] == 16
    assert len(value["per_game_results"]) == 4
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] = 999
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--deliberate-error").returncode == 7
    assert cli(tmp_path, "--date", "20261015").returncode == 2


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8588: scoped checks preserve original E2E authority without recursion."""

    def validate(*_):
        assert e.validation.e is e
        assert e.validation.HISTORICAL == e.HISTORICAL
        return [dict(name="actual_scope", passed=True)]

    monkeypatch.setattr(e.validation, "validate_checks", validate)
    assert e.validate_checks(tmp_path, tmp_path)[0]["passed"]


def test_main_real_path_dispatch(monkeypatch):
    """REQ-VERIFY-8588: the normal invocation delegates to the shipped bounded controller."""
    monkeypatch.setattr(e.runner, "main", lambda argv: int(argv != ["--date", "20261015"]))
    assert e.main([]) == 0


@pytest.mark.parametrize("binding", [False, True])
def test_binding_and_registered_operand_rejections(tmp_path, monkeypatch, binding):
    """REQ-VERIFY-8588: valid outer hashes cannot authorize failed terminal or changed operands."""
    publication = json.loads((e.ROOT / e.PUBLICATION).read_bytes())
    monkeypatch.setattr(e, "checked", lambda *_: publication)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=binding)))
    for version in [735, 737]:
        path = tmp_path / f"openspec/change-proposals/v{version}-methods-protocol.json"
        atomic_json(path, dict(arc_capability_mask=version))
    with pytest.raises(
        ValueError, match="registered_protocol_drift" if binding else "publication_binding"
    ):
        e.authenticate(tmp_path, tmp_path)


def test_added_metrics_retain_genuine_effect_rows(tmp_path, monkeypatch):
    """REQ-REPORT-8588: report observed interventions and immutable action references."""
    original = json.loads(
        (e.ROOT / "results/experiment_8537_v733_arc_capability_panel.json").read_bytes()
    )
    work = dict(
        references=[dict(directory=str(tmp_path), steps_sha256="steps", terminal_sha256="terminal")]
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    monkeypatch.setattr(e, "BUILD", lambda *_: deepcopy(original))
    value = e.build(path, [], tmp_path / (e.NAME + ".json"))
    assert value["intervention_outcome_rows"]
    assert all(r["association"] == "unresolved" for r in value["intervention_outcome_rows"])
    assert len(value["per_action_rows_reference"]) == 1
