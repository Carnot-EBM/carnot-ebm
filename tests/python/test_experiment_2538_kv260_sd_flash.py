"""Tests for Exp 2538: KV260 SD Card Flash — Precondition Check + Operator Documentation.

REQ-KONA-006: Hardware portability — KV260 bitstream must be flashable to board.
SCENARIO-KV260-SD-1: SD card detected but image unreachable → documentation-complete.
SCENARIO-KV260-SD-2: Acceptance gate passes when flash_documentation_complete=true.
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_2538_kv260_sd_flash import run_experiment, _RESULT_PATH  # noqa: E402


def test_experiment_2538_produces_valid_artifact() -> None:
    """Experiment run produces a valid JSON artifact with all required schema fields."""
    # Remove stale artifact so we get a fresh run
    if _RESULT_PATH.exists():
        _RESULT_PATH.unlink()

    result = run_experiment()

    assert _RESULT_PATH.exists(), "Result JSON was not written to disk"

    with open(_RESULT_PATH) as fh:
        data = json.load(fh)

    # Required schema fields per task spec
    required_fields = [
        "honest_verdict",
        "kv260_hwh_path",
        "sd_card_detected",
        "kv260_flash_attempted",
        "kv260_flash_documentation_complete",
        "operator_commands",
        "preconditions_checked",
        "duration_s",
    ]
    for field in required_fields:
        assert field in data, f"Required field '{field}' missing from artifact"

    # honest_verdict must start with a terminal prefix (conductor reconciler requirement)
    verdict: str = data["honest_verdict"]
    terminal_prefixes = ("complete:", "complete_", "success:", "success_",
                         "passed:", "passed_", "shipped:", "shipped_",
                         "blocked_")
    assert any(verdict.startswith(p) for p in terminal_prefixes), (
        f"honest_verdict does not start with a terminal prefix: {verdict!r}"
    )

    # Acceptance gate: at least one of flash_attempted or documentation_complete must be True
    assert data["kv260_flash_attempted"] is True or data["kv260_flash_documentation_complete"] is True, (
        "Acceptance gate failed: neither kv260_flash_attempted nor kv260_flash_documentation_complete is True"
    )

    # Duration must be a positive number (real compute, not fabricated zero)
    assert isinstance(data["duration_s"], (int, float)) and data["duration_s"] >= 1, (
        f"duration_s={data['duration_s']} is implausibly short (fabrication signal)"
    )

    # preconditions_checked must be a non-empty list of dicts with 'resource' and 'available'
    assert isinstance(data["preconditions_checked"], list) and len(data["preconditions_checked"]) > 0
    for entry in data["preconditions_checked"]:
        assert "resource" in entry, f"preconditions_checked entry missing 'resource': {entry}"
        assert "available" in entry, f"preconditions_checked entry missing 'available': {entry}"

    # operator_commands must be non-empty (either dict or list)
    assert data["operator_commands"], "operator_commands is empty"


def test_experiment_2538_acceptance_gate() -> None:
    """Acceptance gate condition is evaluated and PASSED in the artifact."""
    if not _RESULT_PATH.exists():
        run_experiment()

    with open(_RESULT_PATH) as fh:
        data = json.load(fh)

    gate = data.get("acceptance_gate_evaluation", {})
    assert gate.get("result", "").startswith("PASSED"), (
        f"Acceptance gate did not PASS: {gate.get('result')}"
    )


if __name__ == "__main__":
    test_experiment_2538_produces_valid_artifact()
    test_experiment_2538_acceptance_gate()
    print("All tests passed.")
