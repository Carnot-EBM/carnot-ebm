"""Tests for the retired-generator pin guard and its harness wiring.

REQ: REQ-ARC-WMTE-6621 (openspec/capabilities/arc-world-model-trust-energy/spec.md).
SCENARIOs: SCENARIO-ARC-WMTE-6621-REFUSE,
SCENARIO-ARC-WMTE-6621-OVERRIDE,
SCENARIO-ARC-WMTE-6621-MATCH,
SCENARIO-ARC-WMTE-6621-HARNESS-WIRED.

Origin: the frozen lever harness pins the retired Qwen3.5-9B-MTP under a
prose banner, and a 2026-08-20 supervisor A/B silently ran on it. The
guard turns the banner into a runtime refusal. No test writes tracked
state; the harness-wiring test exits at the guard, before any server or
model work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from carnot.agentic.arc_generator_pin_guard import (  # noqa: E402
    RetiredPinError,
    check_frozen_pin,
)


def test_mismatch_refuses_and_names_both_pins() -> None:
    # SCENARIO-ARC-WMTE-6621-REFUSE
    with pytest.raises(RetiredPinError) as err:
        check_frozen_pin("Qwen3.5-9B-MTP", live_pin="Qwen3.8-27B", harness_name="t")
    msg = str(err.value)
    assert "Qwen3.5-9B-MTP" in msg and "Qwen3.8-27B" in msg
    assert "--allow-retired-pin" in msg


def test_override_warns_and_returns(capsys) -> None:
    # SCENARIO-ARC-WMTE-6621-OVERRIDE
    live = check_frozen_pin(
        "Qwen3.5-9B-MTP", allow_retired=True, live_pin="Qwen3.8-27B", harness_name="t"
    )
    assert live == "Qwen3.8-27B"
    err = capsys.readouterr().err
    assert "NOT citable" in err


def test_matching_pin_is_silent(capsys) -> None:
    # SCENARIO-ARC-WMTE-6621-MATCH
    assert check_frozen_pin("Qwen3.8-27B", live_pin="Qwen3.8-27B") == "Qwen3.8-27B"
    assert capsys.readouterr().err == ""


def test_default_live_pin_is_the_canonical_constant() -> None:
    # REQ-ARC-WMTE-6621 rule 1: derive, do not duplicate — the guard's
    # default live pin IS the canonical constant, so the pin can never
    # drift between two literals.
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_REPO_SUBSTR,
    )

    assert check_frozen_pin(ARC_LIVE_GENERATOR_REPO_SUBSTR) == ARC_LIVE_GENERATOR_REPO_SUBSTR


def test_explicit_model_path_override_supersedes_the_pin(capsys) -> None:
    # SCENARIO-ARC-WMTE-6621-ENV-OVERRIDE: an explicit weights path wins
    # over the repo pin at load time, and the recorded label derives from
    # the loaded weights (512eca0e6b) — so the guard must not refuse, or
    # it would force --allow-retired-pin onto runs that already measure
    # the LIVE generator (the induce-failure-fix acceptance-run shape).
    check_frozen_pin(
        "Qwen3.5-9B-MTP",
        live_pin="Qwen3.8-27B",
        harness_name="t",
        model_path_override="/models/Qwen3.8-27B-q4.gguf",
    )
    err = capsys.readouterr().err
    assert "override" in err and "not applicable" in err
    # An empty/whitespace override is NOT an override — still refuses.
    import pytest as _pytest

    with _pytest.raises(RetiredPinError):
        check_frozen_pin(
            "Qwen3.5-9B-MTP", live_pin="Qwen3.8-27B", harness_name="t", model_path_override="  "
        )


def test_real_harness_refuses_before_any_model_work(tmp_path: Path, monkeypatch) -> None:
    # SCENARIO-ARC-WMTE-6621-HARNESS-WIRED: the actual frozen harness,
    # invoked exactly as a runner would, must exit at the guard. This is
    # the origin incident replayed with the guard in place: had this
    # refusal existed on 2026-08-20, no discarded run.
    # The env override must be ABSENT here: with it set, the guard
    # correctly stands down and main() would proceed to real model work.
    monkeypatch.delenv("CARNOT_ARC_GGUF_PATH", raising=False)
    import arc_scored_path_lever_harness as harness

    assert harness.FROZEN_GENERATOR_PIN == "Qwen3.5-9B-MTP"
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_REPO_SUBSTR,
    )

    assert harness.FROZEN_GENERATOR_PIN != ARC_LIVE_GENERATOR_REPO_SUBSTR, (
        "if the live pin ever equals the frozen pin again, this scenario "
        "no longer applies — re-scope the test, do not delete the guard"
    )
    with pytest.raises(RetiredPinError):
        harness.main(["--games", "lp85", "--out", str(tmp_path / "row.json")])
