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
    """Synthetic tr87 game state. `rules` is [([lhs_vals...], [rhs_vals...])] -- the visible rewrite grid,
    LHS over the TARGET series, RHS over the EDITABLE series (1-to-1 at L1, 1-to-many at L2, many-to-many
    at L3). `current`/`target` are the editable/target value lists."""

    def __init__(self, current, target, rules):
        self.cifzvbcuwqe = [([_Sprite(f"nxX{v}") for v in lhs], [_Sprite(f"nxY{v}") for v in rhs])
                            for lhs, rhs in rules]
        self.ztgmtnnufb = [_Sprite(f"nxY{v}") for v in current]     # current editable values
        self.zvojhrjxxm = [_Sprite(f"nxX{v}") for v in target]      # target values


# L1 rule map (1-to-1): 4->[3], 2->[2], 3->[6], 5->[5], 1->[1]
_L1_RULES = [([4], [3]), ([2], [2]), ([3], [6]), ([5], [5]), ([1], [1])]
# L2 rule map (1-to-many expansion): 1->[3], 3->[1,5,1], 5->[2,2], 7->[7]
_L2_RULES = [([1], [3]), ([3], [1, 5, 1]), ([5], [2, 2]), ([7], [7])]
# L3 rule map (many-to-many greedy): [6]->[4], [3,3]->[6,1], [4]->[7,7], [7,7]->[3], [1,5,1]->[6], [2]->[5]
_L3_RULES = [([6], [4]), ([3, 3], [6, 1]), ([4], [7, 7]), ([7, 7], [3]), ([1, 5, 1], [6]), ([2], [5])]


def test_tr87_registered_and_structural():
    assert "tr87" in adapters.adaptered_games()
    ad = adapters.get_adapter("tr87")
    assert ad is not None and ad.game == "tr87"
    for cb in (ad.action_labels, ad.apply, ad.state_key, ad.hand_verifier):
        assert callable(cb)
    assert ad.branch_mode == "fresh_env"       # gotcha #7: win-animation state leaks across reuse-one-env
    assert ad.featurize is None


def test_tr87_action_labels_are_four_keyboard_moves():
    ad = adapters.get_adapter("tr87")
    import json
    labels = ad.action_labels(None, frame=object())
    acts = sorted(json.loads(x)["action"] for x in labels)
    assert acts == [1, 2, 3, 4]                # ACTION1/2 cycle value, ACTION3/4 move selector


def test_tr87_hand_verifier_l1_rule_mapped_cyclic_distance():
    ad = adapters.get_adapter("tr87")
    # L1 (1-to-1): target A4,A2,A3,A5,A1 -> required editable 3,2,6,5,1 (via the rule map)
    target = [4, 2, 3, 5, 1]
    solved = _Game(current=[3, 2, 6, 5, 1], target=target, rules=_L1_RULES)
    assert ad.hand_verifier(solved) == 0.0     # all positions at the rule-mapped target -> win

    one_off = _Game(current=[3, 2, 6, 5, 7], target=target, rules=_L1_RULES)  # last 7, want 1
    assert ad.hand_verifier(one_off) == 1.0    # 7 -> 1 is one cyclic step on the 7-wheel

    start = _Game(current=[1, 7, 2, 4, 6], target=target, rules=_L1_RULES)    # the real L1 start
    # cyclic dists: |1->3|=2, |7->2|=2, |2->6|=3, |4->5|=1, |6->1|=2  => 10
    assert ad.hand_verifier(start) == 10.0


def test_tr87_hand_verifier_l2_one_to_many_expansion():
    ad = adapters.get_adapter("tr87")
    # L2 (1-to-many): target B1,B3,B5,B7 expands to [3]+[1,5,1]+[2,2]+[7] = [3,1,5,1,2,2,7]
    target = [1, 3, 5, 7]
    solved = _Game(current=[3, 1, 5, 1, 2, 2, 7], target=target, rules=_L2_RULES)
    assert ad.hand_verifier(solved) == 0.0     # editable == rule-expanded target -> win

    one_off = _Game(current=[3, 1, 5, 1, 2, 2, 6], target=target, rules=_L2_RULES)  # last 6, want 7
    assert ad.hand_verifier(one_off) == 1.0    # 6 -> 7 is one cyclic step


def test_tr87_hand_verifier_l3_greedy_multi_glyph_lhs():
    ad = adapters.get_adapter("tr87")
    # L3 (many-to-many): target [6,1,5,1,4,2,3,3] greedily parses as
    #   [6]->[4], [1,5,1]->[6], [4]->[7,7], [2]->[5], [3,3]->[6,1]  => required [4,6,7,7,5,6,1]
    target = [6, 1, 5, 1, 4, 2, 3, 3]
    solved = _Game(current=[4, 6, 7, 7, 5, 6, 1], target=target, rules=_L3_RULES)
    assert ad.hand_verifier(solved) == 0.0     # editable == greedy-rewritten target -> win

    start = _Game(current=[3, 4, 4, 5, 1, 4, 6], target=target, rules=_L3_RULES)   # the real L3 start
    # cyclic dists: |3->4|=1,|4->6|=2,|4->7|=3,|5->7|=2,|1->5|=3,|4->6|=2,|6->1|=2  => 15
    assert ad.hand_verifier(start) == 15.0


def test_tr87_hand_verifier_unmatchable_target_returns_large():
    ad = adapters.get_adapter("tr87")
    # a target position no rule LHS matches (an unmodelled tree/double/alter twist) -> large, search stops
    g = _Game(current=[1, 1], target=[9, 9], rules=_L1_RULES)
    assert ad.hand_verifier(g) >= 1000.0


def test_tr87_hand_verifier_length_gap_bounds_unmodelled_levels():
    ad = adapters.get_adapter("tr87")
    # L3+ (the editable ALSO expands): editable count < expansion count -> a >=7 residual the L1/L2
    # formula cannot zero, so the search stops rather than false-claiming the unmodelled twist.
    target = [1, 3]                            # expands to [3] + [1,5,1] = 4 required, but only 2 editable
    g = _Game(current=[3, 1], target=target, rules=_L2_RULES)
    assert ad.hand_verifier(g) >= 7.0


def test_tr87_hand_verifier_never_crashes_on_malformed_game():
    ad = adapters.get_adapter("tr87")

    class _Bad:
        pass

    assert ad.hand_verifier(_Bad()) >= 1000.0  # guarded: a malformed level yields a large finite default
