"""Tests for the entity + HUD perception detectors (python/carnot/agentic/arc_entity_hud_perception.py).

These detectors are the perception fix mandated by the REQ-ARC-WMTE-5831 diagnosis (perception is the ARC
wall) and confirmed valuable by the REQ-ARC-WMTE-5832 oracle-perception counterfactual (correct entities
flip goal induction 0/8 -> 7/8). They must recover, from the agent's OWN transitions, the two facts a
perfect detector would produce: which edge band is a status COUNTER (to ignore) and which color is the
PLAYER (the thing that moves).

Spec: REQ-ARC-WMTE-5833, SCENARIO-ARC-WMTE-5833-HUD-FILL, SCENARIO-ARC-WMTE-5833-MOVER,
SCENARIO-ARC-WMTE-5833-COMPOSE (openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_entity_hud_perception import (
    Transition,
    detect_hud_registers,
    detect_mover,
    perceive_entities,
)

SHAPE = (16, 16)


def _grid(cells: dict[tuple[int, int], int]) -> np.ndarray:
    g = np.zeros(SHAPE, dtype=np.int16)  # background 0
    for (r, c), v in cells.items():
        g[r, c] = v
    return g


def _scene(player_rc: tuple[int, int], hud_filled: int) -> np.ndarray:
    """A 16x16 frame: player=color5 at player_rc, a static target=color9 blob near (5,5), and the last
    row (15) filled left-to-right with `hud_filled` counter cells of color 7."""
    cells: dict[tuple[int, int], int] = {}
    cells[player_rc] = 5
    for (r, c) in [(5, 5), (5, 6), (6, 5), (6, 6)]:
        cells[(r, c)] = 9
    for k in range(hud_filled):
        cells[(15, k)] = 7
    return _grid(cells)


def _play() -> list[Transition]:
    """A short scripted play: the player moves R,R,D,U,L while the row-15 counter fills one cell/action."""
    moves = [(4, (0, 1)), (4, (0, 1)), (2, (1, 0)), (1, (-1, 0)), (3, (0, -1))]
    pos = (8, 8)
    trans: list[Transition] = []
    for hud, (action, (dr, dc)) in enumerate(moves):
        before = _scene(pos, hud)  # hud cells filled before this action
        pos = (pos[0] + dr, pos[1] + dc)
        after = _scene(pos, hud + 1)  # one more counter cell after
        trans.append(Transition(before=before, action=action, after=after))
    return trans


class TestHudDetection:
    def test_detects_fill_counter_on_last_row(self):
        bands = detect_hud_registers(_play())
        rows = [b for b in bands if b.axis == "row" and b.index == 15]
        assert rows, f"expected a HUD band on row 15, got {bands}"
        assert rows[0].direction == "fill"
        assert rows[0].monotone_ratio >= 0.99  # fills one cell per action, strictly monotone
        assert rows[0].changed_fraction >= 0.99  # changes on every action

    def test_does_not_flag_the_interactive_board(self):
        # the target blob at rows 5-6 is not an edge band and never changes -> not a HUD
        bands = detect_hud_registers(_play())
        assert all(not (b.axis == "row" and b.index in (5, 6)) for b in bands)

    def test_detects_deplete_counter(self):
        # a last-row band that starts full and empties one cell per action -> "deplete"
        trans = []
        for k in range(5):
            before = _grid({(15, j): 6 for j in range(10 - k)})
            after = _grid({(15, j): 6 for j in range(10 - k - 1)})
            trans.append(Transition(before=before, action=6, after=after, x=0, y=15))
        bands = detect_hud_registers(trans)
        row15 = [b for b in bands if b.axis == "row" and b.index == 15]
        assert row15 and row15[0].direction == "deplete"

    def test_no_bands_without_enough_transitions(self):
        assert detect_hud_registers(_play()[:2]) == []


class TestMoverDetection:
    def test_finds_the_player_color(self):
        m = detect_mover(_play())
        assert m is not None and m.color == 5
        assert m.alignment >= 0.9 and m.evidence >= 4

    def test_none_without_directional_transitions(self):
        # only clicks (action 6) -> no directional evidence
        clicks = [
            Transition(before=_scene((8, 8), k), action=6, after=_scene((8, 8), k + 1), x=k, y=15)
            for k in range(5)
        ]
        assert detect_mover(clicks) is None

    def test_static_color_is_not_the_mover(self):
        # color 9 (the target) never moves -> must not be picked as the player
        m = detect_mover(_play())
        assert m is None or m.color != 9


class TestComposePerception:
    def test_labels_player_hud_and_objects(self):
        trans = _play()
        current = _scene((8, 7), 5)  # final frame
        res = perceive_entities(current, trans)
        assert res.mover is not None and res.mover.color == 5
        assert any(b.axis == "row" and b.index == 15 for b in res.hud_bands)
        assert "PLAYER token (color 5)" in res.text
        # the row-15 counter is flagged to ignore, the color-9 target is a candidate object
        assert "IGNORE" in res.text and "counter" in res.text
        assert any(o["role"] == "object" and o["color"] == 9 for o in res.objects)
        assert any(o["role"] == "player" and o["color"] == 5 for o in res.objects)
        assert any(o["role"] == "status_bar" for o in res.objects)

    def test_graceful_when_no_transitions(self):
        res = perceive_entities(_scene((8, 8), 0), [])
        assert res.mover is None and res.hud_bands == []
        assert isinstance(res.text, str) and res.text
