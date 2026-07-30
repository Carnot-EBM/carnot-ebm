"""REQ-ARC-FCP-4629/4641: the live action-effect assets are UNTRACKED, so a git-worktree
arm of an A/B silently runs a DIFFERENT agent than the canonical checkout.

THE INCIDENT (2026-07-30). A treatment-activation pre-flight
(`results/outer_loop_arc_composite_treatment_activation_preflight_head_vs_8441055c0_20260730.json`)
reported a reproducible efficiency regression on ARC game vc33: `actions_to_first_solve`
10 -> 15, identical across SIX runs, TWO independently-built probes and TWO sampler seeds,
splitting exactly at commit f9a458e87 (the win-state / goal-predicate work). Six replicates
with zero variance is normally decisive, and the artifact reasoned from it that f9a458e87 had
cost ~56% of the competition score on that game.

It had not. Every one of those six runs also varied a SECOND factor perfectly aligned with the
treatment: the `pre`/`base` arms were served from a `/tmp` git worktree and the `post`/`head`
arms from the canonical checkout. Varying the commit with the ASSET STATE held fixed -- a 2x2
the pre-flight never ran -- gives (`actions_to_first_solve` / action-trace sha256 prefix; each
commit run in ONE tree, with the two untracked assets symlinked in and then removed):

    commit                          assets PRESENT      assets ABSENT
    8441055c0 (pre-flight's base)   15  (no trace)      10  (no trace)
    aa8a38e31 (f9a458e87's parent)  15  19ca5e74        10  e693e8c5
    f9a458e87 (the accused commit)  15  19ca5e74        10  e693e8c5
    HEAD (6fc2bd17b)                15  19ca5e74        10  e693e8c5

The commit axis is INERT in BOTH asset conditions; the asset axis carries the entire effect,
and the /tmp-worktree arms were simply the asset-absent condition in disguise. (8441055c0
predates `ARM_CONFIGS["frozen_gemma_pin"]`, the `policy_game_id` kwarg and `to_row`'s
`include_trace`, so its two cells were run with the older `frozen` arm and no trace capture;
both reductions were first shown INERT by re-running HEAD under them and getting the same
19ca5e74 trace.) The cause is that two live assets are gitignored, so `git worktree add` does
not materialise them:

    results/experiment_4629_live_frame_change_cnn.pt   (trained frame-change CNN, 2026-06-23)
    data/arc_transition_corpus/*.npz                   (25 games' action-effect rows, 2026-07-17)

Both predate the entire 25-commit span under test. Without them
`_load_submitted_frame_change_scorer()` returns None, `StepwiseExplorer` builds no
`ActionEffectExpansionPrior`, and the frontier is expanded in a different order -- which is how
an asset difference reaches the ACTION STREAM. The vc33 traces diverge at action 4, six actions
before the level-up that triggers vc33's only induction, so no goal-predicate change could have
reached the number at all.

WHAT THIS DOES *NOT* DISPROVE. Only the ACTIONS half of the reported regression is a confound.
The same pre-flight also recorded, for vc33, `n_plans_found` 1 -> 0 and `refinement_rounds_used`
1 -> 3 at the goal-satisfiability gate. That half survives: replaying a FIXED captured predicate
through each commit's gate (no path involved) shows the base gate calls the p2 probe's root-true
predicate satisfiable while f9a458e87 and HEAD reject it `goal_predicate_true_at_root`. Recovering
that plan would mean re-admitting a predicate that is true at the root, so it is not something to
revert -- but it is a real, commit-caused cost that has to be weighed, not dissolved. See
`results/outer_loop_arc_vc33_regression_attribution_correction_20260730.json`.

WHAT THESE TESTS PIN. Not the vc33 number itself: reproducing it needs `environment_files/`
(also gitignored, ~75 KB of game source per game), so a live-episode assertion would go red on
any clean checkout, and a test that cannot run in a clean checkout is a worse failure than the
skip it replaces. The live pin lives in `scripts/arc_vc33_explore_pin.py`, which hard-asserts
both numbers against a real episode and exits non-zero otherwise. What is pinned HERE is the
MECHANISM that made the measurement invalid, all of it asset-independent:

  1. the two assets are untracked (so a worktree arm lacks them),
  2. a tree without them yields NO scorer,
  3. a tree with them yields ONE (so 2 is not vacuous),
  4. the scorer's presence is what decides whether the expansion prior exists, i.e. the
     asset difference has a live route to the actions rather than to diagnostics only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic import arc_competition_agent as aca
from carnot.agentic.arc_competition_agent import StepwiseExplorer
from carnot.agentic.arc_frame_change_predictor import (
    LIVE_CNN_CHECKPOINT_RELATIVE_PATH,
    TRANSITION_CORPUS_RELATIVE_DIR,
    ActionEffectExpansionPrior,
    load_live_action_effect_scorer,
)


REPO = Path(__file__).resolve().parents[2]


def _git_tracked(relative: Path) -> bool:
    """True if git tracks `relative`. `git ls-files` prints nothing for an untracked or
    ignored path, and also prints nothing for a path that does not exist -- which is the
    same answer for this test's purpose: a `git worktree add` will not materialise it."""

    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--", str(relative)],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(out.stdout.strip())


def _write_corpus_shard(root: Path, game: str, n: int = 3) -> None:
    """Write a minimal but REAL `arc_transition_corpus` shard.

    `load_cached_transition_effect_rows` reads the arrays by name and computes a frame
    delta, so a shard whose `next_grids` differ from `grids` produces rows with
    `changed=True` -- the shape the live memory is built from. Fabricating the file rather
    than copying the real one keeps the test independent of the untracked corpus."""

    d = root / TRANSITION_CORPUS_RELATIVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    grids = np.zeros((n, 8, 8), dtype=np.int16)
    next_grids = np.zeros((n, 8, 8), dtype=np.int16)
    for i in range(n):
        next_grids[i, i % 8, i % 8] = 1  # a real, non-empty transition delta
    np.savez_compressed(
        d / f"{game}.npz",
        grids=grids,
        next_grids=next_grids,
        actions=np.full((n,), 6, dtype=np.int16),
        xs=np.arange(n, dtype=np.int16),
        ys=np.arange(n, dtype=np.int16),
        lb=np.zeros((n,), dtype=np.int16),
        la=np.zeros((n,), dtype=np.int16),
    )


class _StubScorer:
    """Minimal stand-in for the live action-effect scorer: the expansion-prior wiring only
    needs `candidate_score`, and using a stub keeps test 4 independent of torch weights."""

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        return 0.5


def test_live_action_effect_assets_are_untracked_so_a_worktree_arm_lacks_them() -> None:
    """REQ-ARC-FCP-4629: the CNN checkpoint and the transition corpus are not in git.

    This is the fact that invalidated the pre-flight. If either ever becomes tracked, a
    worktree arm WOULD carry it and this test should be revisited deliberately rather than
    silently -- the failure message says so."""

    assert not _git_tracked(LIVE_CNN_CHECKPOINT_RELATIVE_PATH), (
        f"{LIVE_CNN_CHECKPOINT_RELATIVE_PATH} is now git-tracked. A git-worktree arm would "
        "then carry it, which changes the confound this test documents."
    )
    assert not _git_tracked(TRANSITION_CORPUS_RELATIVE_DIR), (
        f"{TRANSITION_CORPUS_RELATIVE_DIR} is now git-tracked. Same reason."
    )


def test_scorer_is_none_in_a_tree_without_the_untracked_assets(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4629: no assets -> no live action-effect scorer.

    `tmp_path` stands in for a freshly-added git worktree: tracked files only."""

    assert load_live_action_effect_scorer(root=tmp_path) is None


def test_load_submitted_scorer_follows_the_repo_root_it_is_given(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4629: the SUBMITTED entry point resolves its assets off `REPO`.

    Pointing `arc_competition_agent.REPO` at an asset-free tree reproduces exactly what the
    pre-flight's worktree arms ran, without flipping `SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED`
    (a shipped default this test must not touch)."""

    original = aca.REPO
    try:
        aca.REPO = tmp_path
        assert aca._load_submitted_frame_change_scorer() is None
    finally:
        aca.REPO = original


def test_scorer_is_built_when_the_corpus_is_present(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4629: the absence assertions above are not vacuous.

    Without this, `load_live_action_effect_scorer` could return None unconditionally and
    every other test here would still pass."""

    _write_corpus_shard(tmp_path, "zz00")
    scorer = load_live_action_effect_scorer(root=tmp_path)
    assert scorer is not None
    assert scorer.memory is not None
    # The CNN half is genuinely absent in this tree, which is the asymmetry the real worktree
    # arms had for BOTH halves at once.
    assert scorer.cnn_scorer is None


def test_absent_scorer_removes_the_expansion_prior_that_reorders_the_frontier() -> None:
    """REQ-ARC-FCP-4641: the assets reach the ACTION STREAM, not just the diagnostics.

    `StepwiseExplorer` builds an `ActionEffectExpansionPrior` only when a frame-change scorer
    exists, and that prior decides which frontier state is expanded next. So an arm without
    the assets explores in a DIFFERENT ORDER -- which is why the vc33 traces diverged at
    action 4, upstream of every induction-side change the pre-flight was testing."""

    with_scorer = StepwiseExplorer(
        action_effect_expansion_prior=True, frame_change_scorer=_StubScorer()
    )
    without_scorer = StepwiseExplorer(action_effect_expansion_prior=True, frame_change_scorer=None)
    assert isinstance(with_scorer.action_effect_expansion_prior, ActionEffectExpansionPrior)
    assert without_scorer.action_effect_expansion_prior is None
