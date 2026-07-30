"""An A/A noise floor only certifies an A/B if it ran at the SAME configuration.

REQ-ARC-TAP-5901 / SCENARIO-TAP-5901-*: regression tests for the second-order failure found by
review on 2026-07-30.

THE INCIDENT, in two levels, because the second one is the subtle one.

LEVEL 1 -- the confound. A composite treatment-activation grid ran its control arm from a
`git worktree` and its treatment arm from the canonical checkout. Two live assets
(`results/experiment_4629_live_frame_change_cnn.pt`, `data/arc_transition_corpus/`) are
gitignored, and `git worktree add` does not materialise gitignored files. So the control arm's
`load_live_action_effect_scorer()` returned None, no `ActionEffectExpansionPrior` was built, the
search frontier expanded in a different order, and the action stream differed -- for reasons
that had nothing to do with the commits under test. Four cells were reported as
treatment-attributable, diverging at action indices 2-4. All four were the assets.

Crucially, NEITHER A/A floor could catch it: the treatment arm's replicate also ran from the
canonical checkout and the control arm's also from the worktree, so both floors held the asset
axis fixed by construction. That is why 10 of 10 replicate pairs came back IDENTICAL while the
A/B kept perturbing -- the floor was measuring a different axis than the one that differed.

LEVEL 2 -- the correction repeating the mistake. The fix was to re-run the control arm with the
assets restored. That worked. But the A/A floor run to certify the corrected result launched
AFTER the asset symlinks had been removed from the worktree again, so it replicated the
NO-ASSET configuration while the A/B it was certifying was measured WITH assets. It came back
byte-identical to the original (no-asset) control arm, and was read as "the residual difference
fails its own noise floor". It was not a floor at all -- it was a third A/B of the asset axis
wearing an A/A's name. Re-measured at the correct configuration, that same A/A came back
IDENTICAL across three replicates and the A/B difference stood.

Neither level is visible in the traces, in the commit shas, or in any completeness flag. The
only defence is to state the configuration and check it, which is what `a_config`/`b_config` and
`_config_witness` now do.
"""

from __future__ import annotations

from carnot.analysis.treatment_activation_preflight import (
    classify_trace_pair,
    preflight_verdict,
)

# One perturbed A/B cell: control arm ran WITH the assets, treatment arm is head.
_CONTROL = ["RESET", "A", "B", "C"]
_TREATMENT = ["RESET", "A", "B", "D"]


def _ab_pair() -> dict:
    return {
        "vc33": classify_trace_pair(
            _CONTROL, _TREATMENT, a_config="base+assets", b_config="head+assets"
        )
    }


def test_aa_floor_whose_own_arms_differ_in_config_cannot_certify() -> None:
    """SCENARIO-TAP-5901-1: the exact 2026-07-30 level-2 failure.

    The "replicate" ran at a different configuration from the arm it was replicating, so it is
    an A/B of the configuration axis, not an A/A. It must not license attribution -- and,
    equally important, it must not silently REFUSE attribution for the wrong stated reason, so
    the diagnostic names the real problem.
    """
    not_really_an_aa = {
        "vc33": classify_trace_pair(
            _CONTROL, _CONTROL, a_config="base+assets", b_config="base+NOassets"
        )
    }
    v = preflight_verdict(_ab_pair(), noise_pairs=not_really_an_aa)

    assert v["attributable_cells"] == []
    assert v["unattributable_cells_with_aa_class"]["vc33"] == "AA_FLOOR_IS_NOT_AN_AA"


def test_aa_floor_at_a_third_configuration_cannot_certify() -> None:
    """SCENARIO-TAP-5901-2: an internally consistent A/A that matches neither arm under test.

    It measures the determinism of some third setup. Whatever that tells us, it is not about
    either arm of this comparison.
    """
    unrelated = {
        "vc33": classify_trace_pair(
            _CONTROL, _CONTROL, a_config="some+other+config", b_config="some+other+config"
        )
    }
    v = preflight_verdict(_ab_pair(), noise_pairs=unrelated)

    assert v["attributable_cells"] == []
    assert v["unattributable_cells_with_aa_class"]["vc33"] == "AA_FLOOR_CONFIG_MISMATCH"


def test_aa_floor_at_the_matching_configuration_does_certify() -> None:
    """SCENARIO-TAP-5901-3: the positive control.

    Without this, a check that refuses everything would pass the two tests above while being
    useless. A correctly-configured floor must still license attribution, and must report the
    config check as HAVING RUN (empty unverified list) rather than skipped.
    """
    good = {
        "vc33": classify_trace_pair(
            _CONTROL, _CONTROL, a_config="base+assets", b_config="base+assets"
        )
    }
    v = preflight_verdict(_ab_pair(), noise_pairs=good)

    assert v["attributable_cells"] == ["vc33"]
    assert v["cells_whose_noise_floor_config_was_unverified"] == []
    assert v["noise_floor_config_unverified_warning"] is None


def test_absent_fingerprints_preserve_behaviour_but_say_the_check_did_not_run() -> None:
    """SCENARIO-TAP-5901-4: silence is not a pass.

    Every caller predating this parameter supplies no fingerprints. Those must keep working --
    but the result has to distinguish "the configuration was checked and matched" from "nobody
    said", because reading the second as the first is how the incident happened. So attribution
    is unchanged AND a warning names the cells whose floor is unverified.
    """
    legacy = {"vc33": classify_trace_pair(_CONTROL, _CONTROL)}
    v = preflight_verdict(_ab_pair(), noise_pairs=legacy)

    assert v["attributable_cells"] == ["vc33"], "legacy callers must not silently break"
    assert v["cells_whose_noise_floor_config_was_unverified"] == ["vc33"]
    assert "NOT VERIFIED" in v["noise_floor_config_unverified_warning"]


def test_config_witness_is_enforced_on_every_supplied_noise_arm() -> None:
    """SCENARIO-TAP-5901-5: a good floor on one arm cannot cover a bad floor on the other.

    The two-arm floor already refuses to let one arm's determinism speak for the other's. The
    configuration check has to compose with that the same way, or supplying a second arm would
    weaken the guarantee instead of strengthening it.
    """
    good_a = {
        "vc33": classify_trace_pair(
            _CONTROL, _CONTROL, a_config="base+assets", b_config="base+assets"
        )
    }
    bad_b = {
        "vc33": classify_trace_pair(
            _TREATMENT, _TREATMENT, a_config="head+NOassets", b_config="head+NOassets"
        )
    }
    v = preflight_verdict(_ab_pair(), noise_pairs=good_a, noise_pairs_b=bad_b)

    assert v["attributable_cells"] == []
    assert "AA_FLOOR_CONFIG_MISMATCH" in v["unattributable_cells_with_aa_class"]["vc33"]
