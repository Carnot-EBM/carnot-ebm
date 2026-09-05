"""Spec: REQ-ARC-WMTE-7020, SCENARIO-ARC-WMTE-7020-A

The attempt scorer reads engines in the producer-evidence shape as well as the flat shape.

INCIDENT 2026-09-04. The producer evidence contract (exp6993, exp6994) changed where an
induction attempt is stored: from a flat `<game>/attempts/wm_<UTC>__<sha>.py` to a sealed
directory `<game>/attempts/evidence/<dir>/engine.py` carrying the engine plus its transitions
and envelope. The consumer was not moved with it.

The r11l eval that finished that night emitted four engines and every one landed in the new
shape. `scripts/arc_induction_quality.py` globbed only `world_model.py` and `attempts/wm_*.py`,
so it measured zero attempts from a run that produced four, and its report read clean while
doing it. A producer contract that ships without its consumer leaves every future engine
ungraded with every check green.

The path derivation matters as much as the glob: the new shape is two levels deeper, so reading
the game from the parent directory would file every engine under the game "evidence".
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_induction_quality as q  # noqa: E402

_ENGINE = "def step(s, a):\n    return s\n"


def _store(tmp_path: Path) -> Path:
    """A store holding one of each of the three shapes."""
    game = tmp_path / "r11l"
    (game / "attempts" / "evidence" / "arc-20260904T212401_1-abc").mkdir(parents=True)
    (game / "world_model.py").write_text(_ENGINE)
    (game / "attempts" / "wm_20260903T010101__deadbeef.py").write_text(_ENGINE)
    (game / "attempts" / "evidence" / "arc-20260904T212401_1-abc" / "engine.py").write_text(_ENGINE)
    return tmp_path


def test_an_evidence_shape_engine_is_discovered(tmp_path: Path) -> None:
    """The exact incident: four engines emitted, zero scored."""
    kept, _ = q.find_models([_store(tmp_path)], include_non_live=True)
    assert any(p.name == "engine.py" for p in kept)


def test_all_three_shapes_are_discovered_together(tmp_path: Path) -> None:
    kept, _ = q.find_models([_store(tmp_path)], include_non_live=True)
    assert sorted(p.name for p in kept) == [
        "engine.py",
        "wm_20260903T010101__deadbeef.py",
        "world_model.py",
    ]


def test_an_evidence_engine_is_filed_under_its_game_not_under_evidence(tmp_path: Path) -> None:
    """Reading the game from the parent would report every engine under the game 'evidence'."""
    p = tmp_path / "r11l" / "attempts" / "evidence" / "arc-20260904T212401_1-abc" / "engine.py"
    assert q._game_of(p) == "r11l"


def test_an_evidence_engine_counts_as_an_attempt_not_a_survivor(tmp_path: Path) -> None:
    """A survivor is the one engine a game kept; these are per-attempt emissions."""
    p = tmp_path / "r11l" / "attempts" / "evidence" / "arc-20260904T212401_1-abc" / "engine.py"
    assert q._is_attempt(p) is True


def test_the_two_older_shapes_still_classify_as_before(tmp_path: Path) -> None:
    """A widening must not move the population the earlier REQ measured."""
    flat = tmp_path / "r11l" / "attempts" / "wm_x.py"
    survivor = tmp_path / "r11l" / "world_model.py"
    assert (q._game_of(flat), q._is_attempt(flat)) == ("r11l", True)
    assert (q._game_of(survivor), q._is_attempt(survivor)) == ("r11l", False)


def test_score_model_reports_the_right_game_for_an_evidence_engine(tmp_path: Path) -> None:
    _store(tmp_path)
    p = tmp_path / "r11l" / "attempts" / "evidence" / "arc-20260904T212401_1-abc" / "engine.py"
    out = q.score_model(p)
    assert out["game"] == "r11l"
    assert out["population"] == "attempt"
