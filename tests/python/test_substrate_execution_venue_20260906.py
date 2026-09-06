"""Where a run happened is a separate field from what compute ran.

REQ-SUBSTRATE-VENUE-1. `hardware_board` was a member of the closed class enum until
2026-09-06. It answered WHERE a run happened, while the other six answer WHAT COMPUTE
ran, and only the latter can carry a duration floor. Because the enum is closed and a
non-member is critical, a board run had to choose one truth. These tests hold the split:
the class keeps its floor, the venue never gets one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import adversarial_verify as av  # noqa: E402


def _kinds(flags: list[av.Flag]) -> list[str]:
    return [f.kind for f in flags]


def test_retired_hardware_board_is_critical_and_names_the_venue_field() -> None:
    # SCENARIO-SUBSTRATE-VENUE-1
    flags: list[av.Flag] = []
    av.check_substrate_class({"inference_substrate_class": "hardware_board"}, flags)
    assert _kinds(flags) == [av.SUBSTRATE_CLASS_MISMATCH_KIND]
    assert flags[0].severity == "critical"
    # The point of a separate branch is that the message tells a producer where the fact
    # moved. Assert the redirection, not just the refusal.
    assert av.EXECUTION_VENUE_FIELD in flags[0].detail
    assert "kv260" in flags[0].detail


def test_a_venue_outside_the_closed_set_is_critical() -> None:
    # SCENARIO-SUBSTRATE-VENUE-2
    flags: list[av.Flag] = []
    av.check_execution_venue({"execution_venue": "moon_lander"}, flags)
    assert _kinds(flags) == [av.EXECUTION_VENUE_INVALID_KIND]
    assert flags[0].severity == "critical"


def test_a_non_string_venue_is_critical() -> None:
    # SCENARIO-SUBSTRATE-VENUE-2
    flags: list[av.Flag] = []
    av.check_execution_venue({"execution_venue": {"value": "kv260"}}, flags)
    assert _kinds(flags) == [av.EXECUTION_VENUE_INVALID_KIND]


def test_every_named_board_is_accepted() -> None:
    # SCENARIO-SUBSTRATE-VENUE-2, the accepting direction. A set that rejects everything
    # would pass the two tests above, so name each member.
    for venue in ("host", "kv260", "gatemate", "polarfire"):
        flags: list[av.Flag] = []
        av.check_execution_venue({"execution_venue": venue}, flags)
        assert flags == [], f"{venue} should be accepted"


def test_a_venue_never_produces_a_duration_flag() -> None:
    # SCENARIO-SUBSTRATE-VENUE-3. A venue is floor-free by construction: a run of 0.0s on
    # a board draws nothing from the venue check.
    flags: list[av.Flag] = []
    av.check_execution_venue({"execution_venue": "kv260", "duration_s": 0.0}, flags)
    assert flags == []


def test_absent_venue_is_silent() -> None:
    # SCENARIO-SUBSTRATE-VENUE-4
    flags: list[av.Flag] = []
    av.check_execution_venue({"duration_s": 1.0}, flags)
    assert flags == []


def test_a_board_venue_does_not_exempt_a_compute_class_from_its_floor() -> None:
    # SCENARIO-SUBSTRATE-VENUE-5. This is the regression the split must not introduce:
    # recording the board must not become a way to shed the 60s floor.
    artifact = {
        "inference_substrate_class": "model_full_generation",
        "execution_venue": "kv260",
        "duration_s": 1.0,
    }
    flags: list[av.Flag] = []
    av.check_substrate_class(artifact, flags)
    av.check_execution_venue(artifact, flags)
    details = " ".join(f.detail for f in flags)
    assert any(f.severity == "critical" for f in flags)
    assert "60" in details


def test_both_fields_are_declarable_together_on_a_clean_board_run() -> None:
    # SCENARIO-SUBSTRATE-VENUE-5, the passing direction: the whole repair exists so that
    # a board run can state both facts and draw nothing.
    artifact = {
        "inference_substrate_class": "model_full_generation",
        "execution_venue": "kv260",
        "duration_s": 120.0,
    }
    flags: list[av.Flag] = []
    av.check_substrate_class(artifact, flags)
    av.check_execution_venue(artifact, flags)
    assert flags == []


def test_the_venue_check_is_wired_into_the_verifier(tmp_path: Path) -> None:
    # A check nothing calls is the bug class this project names most often. Bite the CALL
    # SITE: run the real entrypoint on a real file, not the helper in isolation.
    # tmp_path, never the repo tree -- a test must not write tracked state.
    artifact = tmp_path / "experiment_9999_venue_wiring_probe.json"
    artifact.write_text(
        json.dumps(
            {
                "experiment": "venue_wiring_probe",
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "execution_venue": "not_a_real_venue",
                "duration_s": 1.0,
                "honest_verdict": "complete_probe",
            }
        ),
        encoding="utf-8",
    )
    report = av.verify_artifact(artifact)
    assert av.EXECUTION_VENUE_INVALID_KIND in [f["kind"] for f in report["flags"]]
