"""Unit tests for the tr87 GameAdapter (python/carnot/agentic/arc_game_adapters.py).

tr87 is a GLYPH-REWRITE configuration puzzle: editable glyphs (value 1-7) must be set so the editable
sequence equals the greedy REWRITE of the target sequence through the visible reference grid (rules
LHS->RHS). The adapter's hand_verifier reads the game's internal config (rules cifzvbcuwqe + target
zvojhrjxxm + current ztgmtnnufb + the level flags) and returns the summed CYCLIC distance of each
editable glyph to its rewritten target (0 == win) -- the lp85 _goal_key internal-state pattern. These
pin that verifier logic on a synthetic game (game-independent) across L1 (1-to-1), L2 (1-to-many), L3
(many-to-many) and L4 (double_translation: a 2-pass A->B->C chain). The end-to-end proof is
solve_adaptered('tr87', 4) -> offline_reproduced=True.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the ARC solve infrastructure / per-game adapters).
"""
from carnot.agentic import arc_game_adapters as adapters


class _Sprite:
    def __init__(self, tag):
        self.name = f"nx{tag}"          # tag is series+value, e.g. "A6"; value = trailing digit


class _Level:
    def __init__(self, flags):
        self._flags = flags

    def get_data(self, name):
        return self._flags.get(name)


class _Game:
    """Synthetic tr87 game state. `rules` is [([lhs_tags...], [rhs_tags...])] (the visible rewrite grid).
    `current`/`target` are tag lists. `flags` carries the level's tree_translation/double_translation."""

    def __init__(self, current, target, rules, flags=None):
        self.cifzvbcuwqe = [([_Sprite(t) for t in lhs], [_Sprite(t) for t in rhs]) for lhs, rhs in rules]
        self.ztgmtnnufb = [_Sprite(t) for t in current]
        self.zvojhrjxxm = [_Sprite(t) for t in target]
        self.current_level = _Level(flags or {})


# L1 (1-to-1): target X-series -> editable Y-series
_L1_RULES = [(["X4"], ["Y3"]), (["X2"], ["Y2"]), (["X3"], ["Y6"]), (["X5"], ["Y5"]), (["X1"], ["Y1"])]
# L2 (1-to-many): X3 -> [Y1,Y5,Y1], X5 -> [Y2,Y2]
_L2_RULES = [(["X1"], ["Y3"]), (["X3"], ["Y1", "Y5", "Y1"]), (["X5"], ["Y2", "Y2"]), (["X7"], ["Y7"])]
# L3 (many-to-many greedy): multi-glyph LHS
_L3_RULES = [(["X6"], ["Y4"]), (["X3", "X3"], ["Y6", "Y1"]), (["X4"], ["Y7", "Y7"]),
             (["X7", "X7"], ["Y3"]), (["X1", "X5", "X1"], ["Y6"]), (["X2"], ["Y5"])]
# L4 (double_translation): a two-level chain A->B (first pass) then B->C (second pass)
_L4_RULES = [(["A6"], ["B1"]), (["A1"], ["B3"]), (["A4"], ["B7"]), (["A7"], ["B6"]),
             (["B1"], ["C3"]), (["B3"], ["C2"]), (["B7"], ["C7"]), (["B6"], ["C1"])]


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
    # L1 (1-to-1): target X4,X2,X3,X5,X1 -> required editable 3,2,6,5,1
    target = ["X4", "X2", "X3", "X5", "X1"]
    solved = _Game(current=["Y3", "Y2", "Y6", "Y5", "Y1"], target=target, rules=_L1_RULES)
    assert ad.hand_verifier(solved) == 0.0     # all positions at the rewritten target -> win

    one_off = _Game(current=["Y3", "Y2", "Y6", "Y5", "Y7"], target=target, rules=_L1_RULES)  # last 7 want 1
    assert ad.hand_verifier(one_off) == 1.0    # 7 -> 1 is one cyclic step on the 7-wheel

    start = _Game(current=["Y1", "Y7", "Y2", "Y4", "Y6"], target=target, rules=_L1_RULES)    # real L1 start
    # cyclic dists: |1->3|=2, |7->2|=2, |2->6|=3, |4->5|=1, |6->1|=2  => 10
    assert ad.hand_verifier(start) == 10.0


def test_tr87_hand_verifier_l2_one_to_many_expansion():
    ad = adapters.get_adapter("tr87")
    # L2 (1-to-many): target X1,X3,X5,X7 -> [3]+[1,5,1]+[2,2]+[7] = [3,1,5,1,2,2,7]
    target = ["X1", "X3", "X5", "X7"]
    solved = _Game(current=["Y3", "Y1", "Y5", "Y1", "Y2", "Y2", "Y7"], target=target, rules=_L2_RULES)
    assert ad.hand_verifier(solved) == 0.0

    one_off = _Game(current=["Y3", "Y1", "Y5", "Y1", "Y2", "Y2", "Y6"], target=target, rules=_L2_RULES)
    assert ad.hand_verifier(one_off) == 1.0    # 6 -> 7 is one cyclic step


def test_tr87_hand_verifier_l3_greedy_multi_glyph_lhs():
    ad = adapters.get_adapter("tr87")
    # L3 (many-to-many): target X6,X1,X5,X1,X4,X2,X3,X3 greedily parses to required [4,6,7,7,5,6,1]
    target = ["X6", "X1", "X5", "X1", "X4", "X2", "X3", "X3"]
    solved = _Game(current=["Y4", "Y6", "Y7", "Y7", "Y5", "Y6", "Y1"], target=target, rules=_L3_RULES)
    assert ad.hand_verifier(solved) == 0.0

    start = _Game(current=["Y3", "Y4", "Y4", "Y5", "Y1", "Y4", "Y6"], target=target, rules=_L3_RULES)
    # cyclic dists: |3->4|=1,|4->6|=2,|4->7|=3,|5->7|=2,|1->5|=3,|4->6|=2,|6->1|=2  => 15
    assert ad.hand_verifier(start) == 15.0


def test_tr87_hand_verifier_l4_double_translation_two_pass_chain():
    ad = adapters.get_adapter("tr87")
    # L4: target A6,A1,A4,A7,A1,A6,A4 rewritten TWICE (A->B->C) -> required C3,C2,C7,C1,C2,C3,C7
    target = ["A6", "A1", "A4", "A7", "A1", "A6", "A4"]
    solved = ["C3", "C2", "C7", "C1", "C2", "C3", "C7"]
    g_double = _Game(current=solved, target=target, rules=_L4_RULES, flags={"double_translation": True})
    assert ad.hand_verifier(g_double) == 0.0   # 2-pass rewrite matched -> win

    # WITHOUT the flag the verifier rewrites ONCE (-> the B-series intermediate), so the C-series editable
    # is NOT at the required values: a non-zero distance. This pins the flag-driven pass count.
    g_single = _Game(current=solved, target=target, rules=_L4_RULES, flags={})
    assert ad.hand_verifier(g_single) > 0.0


def test_tr87_hand_verifier_l5_alter_rules_inverse_puzzle():
    ad = adapters.get_adapter("tr87")
    # L5 (alter_rules=True) INVERTS the puzzle: the RULES are editable, the target+editable are FIXED.
    # Use a UNIQUE-config instance so the parse-search result is deterministic: 1 rule [(1,1)],
    # target [X3], editable [Y5] -> the only winning rule is X3->Y5 (LHS must match target, RHS emit
    # editable). The verifier measures cyclic distance of the rule SIDES to that config.
    target, editable = ["X3"], ["Y5"]

    won = _Game(current=editable, target=target, rules=[(["X3"], ["Y5"])], flags={"alter_rules": True})
    assert ad.hand_verifier(won) == 0.0        # rules already at the winning config

    rhs_off = _Game(current=editable, target=target, rules=[(["X3"], ["Y1"])], flags={"alter_rules": True})
    assert ad.hand_verifier(rhs_off) == 3.0    # RHS 1 -> 5 is 3 cyclic steps; LHS already correct

    lhs_off = _Game(current=editable, target=target, rules=[(["X2"], ["Y5"])], flags={"alter_rules": True})
    assert ad.hand_verifier(lhs_off) == 1.0    # LHS 2 -> 3 is 1 cyclic step; RHS already correct


def test_tr87_hand_verifier_l5_unsolvable_rule_config_returns_large():
    ad = adapters.get_adapter("tr87")
    # editable longer than any single-rule rewrite can produce (RHS len 1, but 2 editable) -> no config
    g = _Game(current=["Y5", "Y6"], target=["X3"], rules=[(["X3"], ["Y5"])], flags={"alter_rules": True})
    assert ad.hand_verifier(g) >= 1000.0


def test_tr87_hand_verifier_l6_two_pass_alter_rules_solvable():
    ad = adapters.get_adapter("tr87")
    # L6-like: alter_rules + a 2-pass A->B->C chain (double_translation). A1->[B1,B2], B1->[C3], B2->[C5];
    # target [A1] rewritten twice = [C3,C5]. A winning rule config EXISTS, so the verifier (which searches
    # for one via the 2-level decomposition and routes by cyclic distance to it) returns a FINITE value,
    # not the no-config sentinel. The end-to-end proof is solve_adaptered('tr87', 6) -> reproduced.
    rules = [(["A1"], ["B1", "B2"]), (["B1"], ["C3"]), (["B2"], ["C5"])]
    g = _Game(current=["C3", "C5"], target=["A1"], rules=rules,
              flags={"alter_rules": True, "double_translation": True})
    d = ad.hand_verifier(g)
    assert 0.0 <= d < 1000.0
    assert ad.hand_verifier(g) == d            # deterministic (the parse-search result is cached)


def test_tr87_hand_verifier_l6_two_pass_unsolvable_returns_large():
    ad = adapters.get_adapter("tr87")
    # one A->B rule + one B->C rule produce a length-1 output, but editable has length 2 -> no config
    rules = [(["A1"], ["B1"]), (["B1"], ["C3"])]
    g = _Game(current=["C7", "C7"], target=["A1"], rules=rules,
              flags={"alter_rules": True, "double_translation": True})
    assert ad.hand_verifier(g) >= 1000.0


def test_tr87_hand_verifier_unmatchable_target_returns_large():
    ad = adapters.get_adapter("tr87")
    # a target position no rule LHS matches (an unmodelled alter_rules twist) -> large, search stops
    g = _Game(current=["Y1", "Y1"], target=["X9", "X9"], rules=_L1_RULES)
    assert ad.hand_verifier(g) >= 1000.0


def test_tr87_hand_verifier_length_gap_bounds_unmodelled_levels():
    ad = adapters.get_adapter("tr87")
    # editable count < expansion count -> a >=7 residual the formula cannot zero, so the search stops.
    target = ["X1", "X3"]                      # expands to [3] + [1,5,1] = 4 required, but only 2 editable
    g = _Game(current=["Y3", "Y1"], target=target, rules=_L2_RULES)
    assert ad.hand_verifier(g) >= 7.0


def test_tr87_hand_verifier_never_crashes_on_malformed_game():
    ad = adapters.get_adapter("tr87")

    class _Bad:
        pass

    assert ad.hand_verifier(_Bad()) >= 1000.0  # guarded: a malformed level yields a large finite default
