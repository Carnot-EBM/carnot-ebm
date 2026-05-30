"""Tests for experiment 3421 GateMate bootstrap 'unspecified' root-cause diagnostic.

References: REQ-HW-107, SCENARIO-HW-107.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts import experiment_3421_gatemate_bootstrap_rootcause_diagnostic_v1 as exp

MOD = "scripts.experiment_3421_gatemate_bootstrap_rootcause_diagnostic_v1"


# --- check_toolchain (a) -----------------------------------------------------


def test_check_toolchain_records_paths():
    """check_toolchain maps each required tool to its resolved path or None.
    References: SCENARIO-HW-107
    """
    with patch(f"{MOD}.shutil.which", side_effect=lambda t: "/usr/bin/" + t if t == "yosys" else None):
        result = exp.check_toolchain()
    assert result["yosys"] == "/usr/bin/yosys"
    assert result["nextpnr-himbaechel"] is None
    assert result["openFPGALoader"] is None


# --- check_board_detect (b) --------------------------------------------------


def test_check_board_detect_tool_absent():
    """When openFPGALoader is not on PATH, detect records exit 127, not-available.
    References: SCENARIO-HW-107
    """
    with patch(f"{MOD}.shutil.which", return_value=None):
        result = exp.check_board_detect()
    assert result["available"] is False
    assert result["exit_code"] == 127


def test_check_board_detect_idcode_present():
    """A clean detect that reports the GateMate IDCODE counts as available.
    References: SCENARIO-HW-107
    """
    with patch(f"{MOD}.shutil.which", return_value="/usr/bin/openFPGALoader"), patch(
        f"{MOD}.subprocess.run"
    ) as run:
        run.return_value = MagicMock(returncode=0, stdout="found GateMate GM1A device", stderr="")
        result = exp.check_board_detect()
    assert result["available"] is True
    assert result["exit_code"] == 0


def test_check_board_detect_no_idcode():
    """A detect that exits 0 but shows no IDCODE marker is not available.
    References: SCENARIO-HW-107
    """
    with patch(f"{MOD}.shutil.which", return_value="/usr/bin/openFPGALoader"), patch(
        f"{MOD}.subprocess.run"
    ) as run:
        run.return_value = MagicMock(returncode=0, stdout="no jtag devices", stderr="")
        result = exp.check_board_detect()
    assert result["available"] is False


# --- script_assigns_verdict (c) ----------------------------------------------


def test_script_assigns_verdict_true(tmp_path):
    """A script that assigns honest_verdict is detected as such."""
    p = tmp_path / "s.py"
    p.write_text("artifact = build_result(d, honest_verdict='success: ok')\n")
    assert exp.script_assigns_verdict(str(p)) is True


def test_script_assigns_verdict_false(tmp_path):
    """A script that never references honest_verdict returns False (the exp3404 case)."""
    p = tmp_path / "s.py"
    p.write_text("artifact = build_result(d, status='error')\n")
    assert exp.script_assigns_verdict(str(p)) is False


def test_script_assigns_verdict_missing_file(tmp_path):
    """An unreadable script path is treated as 'no verdict found' rather than raising."""
    assert exp.script_assigns_verdict(str(tmp_path / "does_not_exist.py")) is False


def test_exp3404_script_really_never_sets_verdict():
    """Regression anchor: the actual exp3404 script does NOT assign honest_verdict.
    This is the root cause being diagnosed.
    References: SCENARIO-HW-107
    """
    assert exp.script_assigns_verdict(exp.EXP3404_SCRIPT) is False


# --- classify_rootcause ------------------------------------------------------


def test_classify_script_never_sets_verdict_takes_priority():
    """Missing verdict dominates even when toolchain/board are also absent.
    References: SCENARIO-HW-107
    """
    assert (
        exp.classify_rootcause(script_sets_verdict=False, toolchain_ok=False, board_reachable=False)
        == "script_never_sets_verdict"
    )


def test_classify_toolchain_missing():
    assert (
        exp.classify_rootcause(script_sets_verdict=True, toolchain_ok=False, board_reachable=False)
        == "toolchain_missing"
    )


def test_classify_board_unreachable():
    assert (
        exp.classify_rootcause(script_sets_verdict=True, toolchain_ok=True, board_reachable=False)
        == "board_unreachable"
    )


def test_classify_all_present_defaults():
    """No defect present still returns a defined classification string."""
    assert (
        exp.classify_rootcause(script_sets_verdict=True, toolchain_ok=True, board_reachable=True)
        == "script_never_sets_verdict"
    )


# --- recommend_fix -----------------------------------------------------------


@pytest.mark.parametrize(
    "cls", ["script_never_sets_verdict", "toolchain_missing", "board_unreachable"]
)
def test_recommend_fix_nonempty(cls):
    assert isinstance(exp.recommend_fix(cls), str)
    assert len(exp.recommend_fix(cls)) > 10


# --- build_diagnosis ---------------------------------------------------------


def test_build_diagnosis_script_never_sets_verdict():
    """The headline scenario: terminal complete: verdict + script_never_sets_verdict.
    References: SCENARIO-HW-107
    """
    diag = exp.build_diagnosis(
        {"yosys": "/usr/bin/yosys", "nextpnr-himbaechel": None, "openFPGALoader": None},
        {"available": False, "exit_code": 127, "stdout": ""},
        script_sets_verdict=False,
    )
    assert diag["rootcause_classification"] == "script_never_sets_verdict"
    assert diag["honest_verdict"].startswith("complete:")
    assert diag["no_flash_attempted"] is True
    # preconditions: 3 tools + board + script-verdict = 5 entries
    assert len(diag["preconditions_checked"]) == 5
    resources = {p["resource"] for p in diag["preconditions_checked"]}
    assert "exp3404_script_assigns_honest_verdict" in resources


def test_build_diagnosis_toolchain_missing_branch():
    """When the script DOES set a verdict, missing toolchain becomes the operative cause."""
    diag = exp.build_diagnosis(
        {"yosys": "/usr/bin/yosys", "nextpnr-himbaechel": None, "openFPGALoader": "/x"},
        {"available": False, "exit_code": 127, "stdout": ""},
        script_sets_verdict=True,
    )
    assert diag["rootcause_classification"] == "toolchain_missing"
    assert diag["honest_verdict"].startswith("complete:")


def test_build_diagnosis_board_unreachable_branch():
    diag = exp.build_diagnosis(
        {"yosys": "/x", "nextpnr-himbaechel": "/x", "openFPGALoader": "/x"},
        {"available": False, "exit_code": 1, "stdout": ""},
        script_sets_verdict=True,
    )
    assert diag["rootcause_classification"] == "board_unreachable"
    assert diag["honest_verdict"].startswith("complete:")


# --- main --------------------------------------------------------------------


def test_main_writes_valid_artifact(tmp_path):
    """main() writes an artifact with all REQ-HW-107 required fields.
    References: SCENARIO-HW-107
    """
    out = tmp_path / "results" / "experiment_3421.json"

    def fake_build_result(data, status, **kw):
        return {**data, "status": status, **kw, "experiment": 3421, "duration_s": 0.5}

    with patch(f"{MOD}.ExperimentTemplate") as Tmpl:
        inst = Tmpl.return_value
        inst._output_path = out
        inst.build_result.side_effect = fake_build_result
        artifact = exp.main()

    inst.setup.assert_called_once()
    assert out.exists()
    written = json.loads(out.read_text())
    for field in (
        "honest_verdict",
        "inference_substrate",
        "preconditions_checked",
        "rootcause_classification",
        "recommended_fix",
        "duration_s",
    ):
        assert field in written, f"missing required field {field}"
    assert written["inference_substrate"] == "hardware_smoke"
    assert written["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped_"))
    assert artifact["rootcause_classification"] in {
        "script_never_sets_verdict",
        "toolchain_missing",
        "board_unreachable",
    }
