"""REQ-ARC-WMTE-6730: the holdout-equalized A/B can drive the selfparse transport.

The arm script pops `CARNOT_ARC_INDUCE_TOOL_LOOP` before importing the loop, because when it
was written the tool arm had exactly one transport. Setting the variable from outside was
therefore silently discarded, and a run believing it measured selfparse would have measured the
native transport instead -- the shape of failure this session already hit twice (a guard whose
env var nothing set; a fix verified through a path production never takes).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "experiments" / "experiment_6474_holdout_equalized_induction_ab_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("_ab_arm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ab_arm"] = module
    spec.loader.exec_module(module)
    return module


def test_selfparse_arm_sets_the_transport() -> None:
    assert _module().transport_env("tool", "selfparse") == {
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse"
    }


def test_native_is_the_default_so_banked_cells_keep_their_meaning() -> None:
    assert _module().transport_env("tool", "native") == {}


def test_the_single_arm_never_gets_the_variable() -> None:
    """Setting it on the control arm would compare a loop against a loop."""
    module = _module()
    assert module.transport_env("single", "selfparse") == {}
    assert module.transport_env("single", "native") == {}
