"""REQ-REPORT-8746 / REQ-VERIFY-8746: existing evidence cannot create new outcomes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_outcome_frontier_8746 as binding

e, runtime = binding.e, binding.runtime


def cli(*args):
    """SCENARIO-VERIFY-8746-SEAL: direct imports work outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-REPORT-8746-FRONTIER: only the production ledger is read."""
    path = tmp_path_factory.mktemp("arc8746") / (e.NAME + ".json")
    result = cli("--private-control", "--output", path, "--date", "20261102")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_empty(private):
    """SCENARIO-REPORT-8746-FRONTIER: no new evidence supports no policy change."""
    path, value = private
    assert value["experiment_id"] == 8746 and value["milestone"] == "2026.11.749"
    assert value["honest_verdict"] == "complete_null_no_new_supervisor_outcomes"
    assert value["new_event_rows"] == []
    assert value["frontier_before"] == value["frontier_after"]
    assert len(value["frontier_before"]["ledger_receipt_identities"]) == 14
    assert value["arc_frontier_ready_score"] == 0
    assert value["solve_provenance"] is None and value["new_level_credit"] == 0
    assert value["policy_decision"] == "no_supported_policy_change"
    assert value["policy_decision_rows"] == value["decision_rows"]
    assert value["unresolved_count"] == 0
    assert value["registry_precheck"]["loop_precheck_rows"]
    assert all(r["new_level_credit"] == 0 for r in value["registry_precheck"]["loop_precheck_rows"])
    assert value["authority"]["ready"] and len(value["authority"]["rows"]) == 12
    assert set(value) <= set(value["field_principles"])
    assert cli("--cold-replay", path).returncode == 0


@pytest.mark.parametrize("mode", ["source", "frontier", "ready", "registry"])
def test_retained_identity(monkeypatch, mode):
    """REQ-VERIFY-8746: rehashed mutable envelopes cannot replace retained identity."""
    value, refs = binding.legacy.qualified(e.ROOT / binding.FRONTIER, binding.FRONTIER_PIN)
    value = deepcopy(value)
    if mode == "source":
        value["code_config_hashes"][__file__] = "sha256:changed"
    elif mode == "frontier":
        value["frontier_after"]["ledger_receipt_identities"] = []
    elif mode == "registry":
        value["registry_precheck"]["sha256"] = "sha256:changed"
    else:
        value["arc_frontier_ready_score"] = 0
    qualified = binding.legacy.qualified
    monkeypatch.setattr(
        binding.legacy,
        "qualified",
        lambda path, pin: (
            (value, refs) if path == e.ROOT / binding.FRONTIER else qualified(path, pin)
        ),
    )
    with pytest.raises(ValueError, match="retained_frontier_identity"):
        binding.authenticate()


def test_rejection_paths(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8746-SEAL: owned failures and external absence stay distinct."""
    path, value = private
    changed = deepcopy(value)
    changed["new_level_credit"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    candidate = tmp_path / "tampered.json"
    e.atomic_json(candidate, changed)
    assert cli("--cold-replay", candidate).returncode == 1
    assert cli("--date", "bad").returncode == 1
    assert cli("--date", "20261302").returncode == 1
    assert cli("--private-control").returncode == 1
    failed = tmp_path / (e.NAME + ".json")
    assert cli("--private-control", "--failed-child", "--output", failed).returncode == 1
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(e, "UPSTREAM", "results/absent-8746.json")
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert runtime.run(blocked, "20261102", True, False) == 0
    result = json.loads(blocked.read_bytes())
    assert result["verdict_class"] == "blocked" and result["new_event_rows"] == []
    assert result["gate_check_summary"][-1]["operand_state"] == "missing_file"
    assert path.exists()


def test_authority_missing(tmp_path):
    """REQ-VERIFY-8746: absent task authority cannot silently grant readiness."""
    value = binding.authority(tmp_path)
    assert not value["ready"] and value["gate_check_summary"]


def test_duplicate_retained_identity():
    """SCENARIO-VERIFY-8746-SEAL: a known receipt cannot be rewritten as new."""
    baseline = e.read(dict(path=str(e.ROOT / e.BASELINE), sha256=e.LEDGER_PIN))
    changed = deepcopy(baseline)
    first = next(iter(changed["entries"]))
    changed["entries"][first] = dict(changed["entries"][first], arm="changed")
    with pytest.raises(ValueError, match="changed_retained_identity"):
        e.delta(baseline, changed, {})


def test_upstream_operand_states(tmp_path):
    """REQ-VERIFY-8746: absence and zero are different observations."""
    path = tmp_path / "upstream.json"
    for data, state in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"arc_frontier_ready_score": 0}, "observed_zero"),
    ]:
        if data is not None:
            e.atomic_json(path, data)
        observed = binding.observe(path, "unused", "unused", binding.FRONTIER_PIN)
        assert observed["operand_state"] == state and not observed["passed"]
