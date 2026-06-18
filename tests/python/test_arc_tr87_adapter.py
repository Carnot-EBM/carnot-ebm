"""Unit tests for the tr87 GameAdapter (python/carnot/agentic/arc_game_adapters.py).

tr87 is a GLYPH-SUBSTITUTION configuration puzzle: 5 editable glyphs (series 'B', value 1-7) must be
set so that each, via a visible substitution rule, matches the target row (series 'A'). The adapter's
hand_verifier reads the game's internal config (rule map cifzvbcuwqe + target zvojhrjxxm + current
ztgmtnnufb) and returns the count of positions NOT yet at their rule-mapped value (0 == win) -- the
same internal-state-reading pattern as the lp85 adapter. These pin that verifier logic on a synthetic
game (game-independent); the end-to-end proof is solve_adaptered('tr87', 1) -> offline_reproduced=True.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the ARC solve infrastructure / per-game adapters).
"""
from carnot.agentic import arc_game_adapters as adapters


class _Sprite:
    def __init__(self, name):
        self.name = name


class _Game:
    """Synthetic tr87 game state: rule A4<->B3, A2<->B2, A3<->B6, A5<->B5, A1<->B1 (the L1 rule map)."""

    def __init__(self, current_b, target_a):
        self.cifzvbcuwqe = [
            ([_Sprite("nxA4")], [_Sprite("nxB3")]),
            ([_Sprite("nxA2")], [_Sprite("nxB2")]),
            ([_Sprite("nxA3")], [_Sprite("nxB6")]),
            ([_Sprite("nxA5")], [_Sprite("nxB5")]),
            ([_Sprite("nxA1")], [_Sprite("nxB1")]),
        ]
        self.ztgmtnnufb = [_Sprite(f"nxB{v}") for v in current_b]   # current editable values
        self.zvojhrjxxm = [_Sprite(f"nxA{v}") for v in target_a]    # target values


def test_tr87_registered_and_structural():
    assert "tr87" in adapters.adaptered_games()
    ad = adapters.get_adapter("tr87")
    assert ad is not None and ad.game == "tr87"
    for cb in (ad.action_labels, ad.apply, ad.state_key, ad.hand_verifier):
        assert callable(cb)
    assert ad.branch_mode == "replay"          # tr87 reset is idempotent + config is prefix-deterministic
    assert ad.featurize is None


def test_tr87_action_labels_are_four_keyboard_moves():
    ad = adapters.get_adapter("tr87")
    import json
    labels = ad.action_labels(None, frame=object())
    acts = sorted(json.loads(x)["action"] for x in labels)
    assert acts == [1, 2, 3, 4]                # ACTION1/2 cycle value, ACTION3/4 move selector


def test_tr87_hand_verifier_counts_rule_mapped_mismatches():
    ad = adapters.get_adapter("tr87")
    # target A4,A2,A3,A5,A1 -> required B3,B2,B6,B5,B1 (via the rule map)
    target = [4, 2, 3, 5, 1]
    solved = _Game(current_b=[3, 2, 6, 5, 1], target_a=target)
    assert ad.hand_verifier(solved) == 0.0     # all positions at the rule-mapped target -> win

    one_off = _Game(current_b=[3, 2, 6, 5, 7], target_a=target)   # last glyph wrong (7 != B1)
    assert ad.hand_verifier(one_off) == 1.0

    start = _Game(current_b=[1, 7, 2, 4, 6], target_a=target)     # the actual L1 start config
    assert ad.hand_verifier(start) == 5.0      # all five positions wrong at reset


def test_tr87_hand_verifier_never_crashes_on_malformed_game():
    ad = adapters.get_adapter("tr87")

    class _Bad:
        pass

    assert ad.hand_verifier(_Bad()) >= 1000.0  # guarded: a malformed level yields a large finite default
