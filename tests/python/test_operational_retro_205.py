import os
import sys
import json

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts"))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import operational_retro_205 as module

# REQ-RETRO-205, SCENARIO-RETRO-205


def test_generate_retro_data(monkeypatch):
    """Verify retro data contains all required schema fields and correct milestone .205 values."""
    import subprocess

    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    def mock_run(cmd, *args, **kwargs):
        if "Activate milestone 2026.05.205" in cmd:
            return MockResult("abc1234 [conductor] Activate milestone 2026.05.205\n")
        elif "..HEAD" in cmd:
            return MockResult("def5678 Some subsequent commit\n")
        return MockResult("")

    monkeypatch.setattr(subprocess, "run", mock_run)

    data = module.generate_retro_data()

    assert data["schema"] == "carnot.operational_retro.v65"
    assert data["milestone"] == "2026.05.205"
    assert data["experiment"] == 2060
    assert data["acceptance_gate_passed"] is True
    assert data["honest_verdict"].startswith("complete:")
    assert "preconditions_checked" in data
    assert any("True" in s for s in data["preconditions_checked"])
    assert data["experiments_completed"] == 3
    assert data["experiments_blocked"] == 7
    assert data["experiments_total"] == 10
    assert data["compute_bound_experiments_count"] == 0
    assert "experiment_summary" in data
    assert len(data["experiment_summary"]) == 10
    assert "hardware_capability_gaps" in data
    assert "bottlenecks_identified" in data
    assert "improvements_suggested" in data
    assert "top_3_highest_leverage_actions" in data
    assert "estimated_time_savings_pct" in data
    assert "meta_reflection" in data
    assert "summary" in data


def test_generate_retro_data_preconditions_false(monkeypatch):
    """Verify preconditions_checked reflects False when no activation commit exists."""
    import subprocess

    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    def mock_run(cmd, *args, **kwargs):
        return MockResult("")

    monkeypatch.setattr(subprocess, "run", mock_run)

    data = module.generate_retro_data()
    assert any("False" in s for s in data["preconditions_checked"])


def test_check_preconditions_false(monkeypatch):
    """Verify check_preconditions returns False when git log finds no activation commit."""
    import subprocess

    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MockResult(""))
    assert module.check_preconditions() is False


def test_check_preconditions_true(monkeypatch):
    """Verify check_preconditions returns True when activation commit and subsequent commits exist."""
    import subprocess

    class MockResult:
        def __init__(self, stdout):
            self.stdout = stdout

    call_count = [0]

    def mock_run(cmd, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MockResult("abc1234 [conductor] Activate milestone 2026.05.205\n")
        return MockResult("def5678 Follow-up commit\n")

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert module.check_preconditions() is True


def test_experiment_summary_contains_all_ten(monkeypatch):
    """All 10 milestone .205 experiments must appear in the experiment_summary list."""
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: type("R", (), {"stdout": ""})(),
    )

    data = module.generate_retro_data()
    exp_ids = [e["experiment"] for e in data["experiment_summary"]]
    for expected_id in range(2050, 2060):
        assert expected_id in exp_ids, f"Exp {expected_id} missing from experiment_summary"


def test_main_execution(tmp_path, monkeypatch):
    """Verify main() writes the JSON to the correct path with required fields."""
    monkeypatch.chdir(tmp_path)

    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: type("R", (), {"stdout": ""})(),
    )

    module.main()

    out_path = os.path.join("results", "operational_retro_2026_05_205.json")
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["schema"] == "carnot.operational_retro.v65"
    assert saved["milestone"] == "2026.05.205"
    assert saved["experiment"] == 2060
    assert saved["acceptance_gate_passed"] is True
    assert saved["honest_verdict"].startswith("complete:")
