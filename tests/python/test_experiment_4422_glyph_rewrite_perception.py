"""Tests for Exp 4422 tr87 from-pixels glyph-rewrite verifier.

Spec refs: REQ-REPORT-4422, SCENARIO-REPORT-4422.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4422_glyph_rewrite_perception as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"

ON = mod.ON
A_FRAME = 10
B_FRAME = 7
C_FRAME = 11

PATTERNS = {
    1: np.array(
        [
            [ON, 0, 0, 0, ON],
            [ON, ON, 0, ON, 0],
            [0, ON, ON, 0, 0],
            [ON, 0, ON, ON, 0],
            [0, 0, 0, ON, ON],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [0, ON, ON, 0, 0],
            [ON, 0, 0, ON, 0],
            [0, ON, 0, ON, ON],
            [ON, ON, 0, 0, 0],
            [0, 0, ON, ON, ON],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [ON, ON, 0, ON, 0],
            [0, ON, 0, 0, ON],
            [ON, ON, ON, 0, 0],
            [0, 0, ON, 0, ON],
            [ON, 0, ON, ON, 0],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [ON, 0, ON, 0, ON],
            [0, 0, ON, ON, 0],
            [ON, ON, 0, 0, ON],
            [0, ON, 0, ON, 0],
            [ON, 0, 0, ON, ON],
        ],
        dtype=int,
    ),
    5: np.array(
        [
            [0, ON, 0, ON, ON],
            [ON, 0, ON, 0, 0],
            [ON, ON, 0, ON, 0],
            [0, ON, 0, 0, ON],
            [ON, 0, ON, ON, 0],
        ],
        dtype=int,
    ),
    6: np.array(
        [
            [ON, ON, ON, 0, ON],
            [0, ON, 0, ON, 0],
            [ON, 0, 0, ON, ON],
            [0, ON, ON, 0, 0],
            [ON, 0, ON, 0, ON],
        ],
        dtype=int,
    ),
}


def _paint(grid: np.ndarray, row: int, col: int, frame: int, value: int, *, rotate: int = 0, noise: bool = False) -> None:
    pat = np.rot90(PATTERNS[value], rotate)
    if noise:
        pat = pat.copy()
        pat[0, 0] = 0 if pat[0, 0] == ON else ON
    tile = np.full((5, 5), frame, dtype=int)
    tile[pat == ON] = ON
    grid[row : row + 5, col : col + 5] = tile


def _row(grid: np.ndarray, row: int, glyphs: list[tuple[int, int, int, bool]]) -> None:
    col = 4
    for frame, value, rotate, noise in glyphs:
        _paint(grid, row, col, frame, value, rotate=rotate, noise=noise)
        col += 7


def _rewrite_grid(*, winning: bool = True) -> np.ndarray:
    grid = np.full((64, 64), 3, dtype=int)
    _row(
        grid,
        5,
        [
            (A_FRAME, 1, 0, False),
            (A_FRAME, 2, 1, False),
            (B_FRAME, 3, 0, False),
            (A_FRAME, 4, 0, False),
            (B_FRAME, 5, 0, False),
            (B_FRAME, 6, 2, False),
        ],
    )
    _row(grid, 30, [(A_FRAME, 1, 2, True), (A_FRAME, 2, 0, False), (A_FRAME, 4, 0, False)])
    tail = 6 if winning else 5
    _row(grid, 42, [(B_FRAME, 3, 3, False), (B_FRAME, 5, 0, False), (B_FRAME, tail, 0, False)])
    return grid


def test_req_report_4422_spec_declares_pixel_verifier_contract() -> None:
    """REQ-REPORT-4422: OpenSpec declares the glyph perception artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4422" in spec
    assert "SCENARIO-REPORT-4422" in spec
    assert "segment_glyphs" in spec
    assert "Hamming-nearest" in spec
    assert "greedily rewriting" in spec
    assert "experiment_4422_glyph_rewrite_perception.json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4422_decodes_localized_tolerant_multi_glyph_rewrite() -> None:
    """REQ-REPORT-4422: rotated/noisy glyphs still decode through localized rule bands."""

    ok, debug = mod.decode_and_check_grid(_rewrite_grid(winning=True))

    assert ok is True
    assert debug["rules"] == 2
    assert debug["rewrite_passes"] == 1
    assert debug["localized"]["rule_bands"] == [(5, 9)]
    assert debug["localized"]["target_band"] == (30, 34)
    assert debug["localized"]["editable_band"] == (42, 46)
    assert debug["target_len"] == 3
    assert debug["editable_len"] == 3


def test_req_report_4422_rejects_nonwin_from_pixels() -> None:
    """REQ-REPORT-4422: the same pixel rule rejects non-winning editable sequences."""

    ok, debug = mod.decode_and_check_grid(_rewrite_grid(winning=False))

    assert ok is False
    assert debug["expected_sequence"] != debug["editable_sequence"]
    assert debug["rewrite_passes"] == 1


def test_req_report_4422_greedy_rewrite_supports_two_pass_rules() -> None:
    """REQ-REPORT-4422: greedy multi-glyph sequence rewrite handles chained rules."""

    a1 = (A_FRAME, 1)
    a2 = (A_FRAME, 2)
    b3 = (B_FRAME, 3)
    c4 = (C_FRAME, 4)
    rules = [
        ((a1, a2), (b3,)),
        ((b3,), (c4,)),
    ]

    ok, passes = mod.sequence_rewrite_matches([a1, a2], [c4], rules, max_passes=3)

    assert ok is True
    assert passes == [[b3], [c4]]
    assert mod.greedy_rewrite([a2, a1], rules) is None
    assert mod.sequence_rewrite_matches([a1], [c4], [((a1,), (b3,))], max_passes=1) == (False, [[b3]])


def test_req_report_4422_defensive_paths_are_honest(tmp_path: Path) -> None:
    """REQ-REPORT-4422: malformed pixels and failed gates produce non-success states."""

    blank = np.full((12, 12), 3, dtype=int)
    ok, debug = mod.decode_and_check_grid(blank)
    assert ok is False
    assert debug["rules"] == 0
    assert debug["rewrite_passes"] == 0
    assert mod._frame_color(np.full((5, 5), 3, dtype=int)) == 0

    false_positive_grounding = mod.evaluate_grounding(_rewrite_grid(winning=True), [_rewrite_grid(winning=True)])
    assert false_positive_grounding["false_positives"] == 1
    assert false_positive_grounding["grounded"] is False

    not_grounded = mod.build_artifact(
        root=tmp_path,
        grounding={"grounded": False, "fires_on_win": False, "false_positives": 0},
        preconditions_checked={},
        solution_labels=[],
        reproduction_result={"reproduced": True, "reached_level": 1},
        started_at=2.0,
        ended_at=1.0,
    )
    assert not_grounded["honest_verdict"] == "complete_glyph_rewrite_perception_not_grounded"
    assert not_grounded["duration_s"] == 0.0

    blocked = mod.build_artifact(
        root=tmp_path,
        grounding={"grounded": True, "fires_on_win": True, "false_positives": 0},
        preconditions_checked={},
        solution_labels=[],
        reproduction_result={"reproduced": False, "reached_level": 0},
        started_at=1.0,
        ended_at=2.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_reproduction_failed"

    with pytest.raises(ValueError, match="missing offline_reproduced"):
        mod.write_artifact(tmp_path, {"honest_verdict": "complete_bad"})


def test_scenario_report_4422_run_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4422: grounding plus reproduce() writes the required artifact."""

    result = mod.run(
        tmp_path,
        grounding_frames_fn=lambda _root: mod.GroundingFrames(
            win_grid=_rewrite_grid(winning=True),
            nonwin_grids=[_rewrite_grid(winning=False)],
            preconditions_checked={"offline_env_loads": {"tr87": True}, "banked_solution_labels": 2},
        ),
        solution_labels_fn=lambda _root: ['{"action": 2}', '{"action": 2}'],
        reproduce_fn=lambda _labels: {
            "game": "tr87",
            "reached_level": 1,
            "claimed_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        },
        now=lambda: 100.0,
    )
    artifact = json.loads(result.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "success_glyph_rewrite_perception_tr87_grounded_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["fires_on_win"] is True
    assert artifact["false_positives"] == 0
    assert artifact["grounded"] is True
    assert artifact["solution_label_count"] == 2
    assert len(artifact["reproducibility_checksum"]) == 64
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4422_schema_rejects_fabricated_success() -> None:
    """REQ-REPORT-4422: success artifacts must be grounded, oracle-backed, and reproduced."""

    artifact = {
        "honest_verdict": "success_glyph_rewrite_perception_tr87_grounded_reproduced",
        "offline_reproduced": False,
        "reproduced_levels": "1",
        "verifier_is_oracle": False,
        "fires_on_win": False,
        "false_positives": 1,
        "grounded": False,
        "reproducibility_checksum": "x",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "offline_reproduced must be true for success verdicts" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "success verdict requires fires_on_win true" in errors
    assert "success verdict requires zero false positives" in errors
    assert "success verdict requires grounded true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    malformed = {
        "honest_verdict": "not-terminal",
        "offline_reproduced": "yes",
        "reproduced_levels": 1,
        "verifier_is_oracle": True,
        "fires_on_win": "true",
        "false_positives": "0",
        "grounded": "true",
        "reproducibility_checksum": "a" * 64,
    }

    malformed_errors = mod.artifact_schema_errors(malformed)

    assert "offline_reproduced must be bare bool" in malformed_errors
    assert "fires_on_win must be bare bool" in malformed_errors
    assert "false_positives must be bare int" in malformed_errors
    assert "grounded must be bare bool" in malformed_errors
    assert "honest_verdict must start with terminal prefix" in malformed_errors
