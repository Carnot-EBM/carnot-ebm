"""Tests for PoE-World weighted product-of-experts (arXiv:2505.10819).

The contract: PoEWorldModel.engine combines applying experts by a per-cell WEIGHTED CONSENSUS (the
product MAP), which is genuinely DIFFERENT from the nulled ProductWorldModel's highest-trust-wins max-vote
(exp4749) -- a quorum of weak agreeing experts can outvote one strong disagreeing expert. fit_poe_weights
fits non-negative log-odds weights from held-out accuracy and PRUNES experts no better than chance. Spec:
the PoE-World lever. verifier_is_oracle stays False.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    ProductWorldModel,
    ProgrammaticExpert,
    Transition,
)
from carnot.agentic.arc_poe_world_model import (
    PoEWorldModel,
    build_poe_world_model,
    fit_poe_weights,
)


def _cell_expert(name: str, action: int, r: int, c: int, value: int, trust: float = 0.0):
    """An expert that, for `action`, sets cell (r,c) to `value` (else identity)."""

    def _pre(grid, a, data):
        return int(a) == int(action)

    def _eff(grid, a, data):
        out = np.asarray(grid).copy()
        out[r, c] = value
        return out

    return ProgrammaticExpert(
        name=name, object_class="cell", precondition=_pre, effect=_eff, action=int(action), trust=trust
    )


def test_weighted_consensus_outvotes_one_strong_expert():
    """CORE differentiator: 3 weak experts (total weight 1.5) agree cell->2; 1 strong expert says cell->1.
    PoE weighted-consensus picks 2 (the quorum). The max-vote ProductWorldModel picks 1 (highest .trust).
    Same experts, different combination -> different cell -> proves PoE != max-vote."""
    grid = np.zeros((3, 3), dtype=int)
    strong = _cell_expert("strong", 1, 0, 0, value=1, trust=0.95)
    weak = [_cell_expert(f"weak{i}", 1, 0, 0, value=2, trust=0.6) for i in range(3)]
    experts = [strong, *weak]

    # PoE: strong weight 1.0, each weak 0.5 (sum 1.5) -> consensus 2 beats strong's 1 and no-change prior 0.5
    poe = PoEWorldModel(experts=experts, weights=[1.0, 0.5, 0.5, 0.5], no_change_prior=0.5)
    assert int(poe.engine(grid, 1, None)[0, 0]) == 2

    # max-vote ProductWorldModel: highest-trust applicable expert wins the cell -> strong's value 1
    maxvote = ProductWorldModel(experts)
    assert int(maxvote.engine(grid, 1, None)[0, 0]) == 1


def test_no_change_prior_blocks_a_lone_weak_expert():
    """A single weak expert (weight 0.4) cannot flip a cell against the no_change_prior (0.5): consensus
    is required, so the cell stays put. This is the guard against the identity-or-noise collapse."""
    grid = np.zeros((2, 2), dtype=int)
    lone = _cell_expert("lone", 1, 0, 0, value=7)
    poe = PoEWorldModel(experts=[lone], weights=[0.4], no_change_prior=0.5)
    assert int(poe.engine(grid, 1, None)[0, 0]) == 0  # prior 0.5 > 0.4 -> no flip


def test_pruned_zero_weight_expert_is_inert():
    """A pruned expert (weight 0) casts no vote -> the engine is identity when only pruned experts apply."""
    grid = np.zeros((2, 2), dtype=int)
    e = _cell_expert("e", 1, 0, 0, value=5)
    poe = PoEWorldModel(experts=[e], weights=[0.0], no_change_prior=0.5)
    assert np.array_equal(poe.engine(grid, 1, None), grid)


def test_engine_identity_when_no_expert_applies():
    grid = np.array([[3, 4], [5, 6]], dtype=int)
    e = _cell_expert("e", 2, 0, 0, value=9)  # only applies to action 2
    poe = PoEWorldModel(experts=[e], weights=[1.0])
    assert np.array_equal(poe.engine(grid, 1, None), grid)  # action 1 -> nobody applies -> identity


def _t(before, after, action=1):
    return Transition(np.array(before), action, None, np.array(after), 0, 0)


def test_fit_poe_weights_prunes_inaccurate_expert_and_sets_trust():
    """An expert that is RIGHT on held-out cells gets weight>0; one that is WRONG gets weight 0 (pruned).
    .trust is set to the held-out accuracy for the max-vote baseline."""
    # held-out: action 1 sets (0,0) from 0 to 1.
    heldout = [_t([[0, 0]], [[1, 0]]), _t([[0, 0]], [[1, 0]])]
    good = _cell_expert("good", 1, 0, 0, value=1)
    bad = _cell_expert("bad", 1, 0, 0, value=8)  # predicts wrong value
    weights = fit_poe_weights([good, bad], heldout)
    assert weights[0] > 0.0 and weights[1] == 0.0
    assert good.trust > 0.9 and bad.trust < 0.1


def test_build_poe_world_model_diagnostics():
    """build_poe_world_model assembles exact-delta experts from transitions, fits/prunes, and reports
    weighted_product_consensus combination with verifier_is_oracle False."""
    train = [_t([[0, 0]], [[1, 0]]), _t([[0, 0]], [[1, 0]])]
    heldout = [_t([[0, 0]], [[1, 0]])]
    model = build_poe_world_model(train, heldout)
    assert isinstance(model, PoEWorldModel)
    assert model.diagnostics_["combination"] == "weighted_product_consensus"
    assert model.diagnostics_["verifier_is_oracle"] is False
    assert model.diagnostics_["n_experts"] >= 1
    # the exact-delta expert reproduces the held-out transition
    assert int(model.engine(np.array([[0, 0]]), 1, None)[0, 0]) == 1
