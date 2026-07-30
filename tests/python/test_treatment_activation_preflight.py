"""REQ-HARNESS-6050 / SCENARIO-HARNESS-6050-*: the treatment-activation pre-flight.

The load-bearing test in this file is the RETROSPECTIVE one
(`test_refuses_the_committed_retention_ab_grid`). A power pre-flight that cannot refuse a
grid we already KNOW was underpowered is worthless, so it is validated against the real
committed evidence of the 2026-07-29 engine-retention A/B -- 12 matched pairs of live-agent
action traces, on disk, unmodified -- rather than against synthetic fixtures alone. Synthetic
cases cover the branch logic; the committed grid proves the tool would have saved the day.
"""

from __future__ import annotations

import json
import os

import pytest

from carnot.analysis.treatment_activation_preflight import (
    BOTH_TRUNCATED,
    IDENTICAL,
    INCONCLUSIVE,
    MISSING,
    PASS,
    PERTURBED,
    REFUSE,
    TRUNCATION_ONLY,
    classify_trace_pair,
    format_report,
    min_one_way_discordant_pairs,
    preflight_verdict,
    two_sided_sign_test_p,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETENTION_CELLS = os.path.join(REPO, "results", "arc_engine_retention_20260729", "cells")


# ---- SCENARIO-HARNESS-6050-1: the threshold is derived from the test, not picked ----------
def test_sign_test_p_values_are_the_textbook_ones() -> None:
    # All discordant pairs one-way => p = 2 * (1/2)^d.
    assert two_sided_sign_test_p(1) == pytest.approx(1.0)
    assert two_sided_sign_test_p(2) == pytest.approx(0.5)
    assert two_sided_sign_test_p(4) == pytest.approx(0.125)
    assert two_sided_sign_test_p(5) == pytest.approx(0.0625)
    assert two_sided_sign_test_p(6) == pytest.approx(0.03125)
    # No discordant pairs means nothing to test: p must be 1.0, never something smaller.
    assert two_sided_sign_test_p(0) == pytest.approx(1.0)


def test_six_discordant_pairs_is_the_alpha_005_floor() -> None:
    """The arithmetic that makes >=6/12 the pre-flight's bar, stated as an assertion."""
    assert min_one_way_discordant_pairs(0.05) == 6
    assert two_sided_sign_test_p(5) >= 0.05  # 5 is NOT enough
    assert two_sided_sign_test_p(6) < 0.05  # 6 is
    # A stricter alpha demands more pairs; a looser one fewer. The threshold tracks alpha
    # rather than being a magic constant that silently disagrees with it.
    assert min_one_way_discordant_pairs(0.01) == 8
    assert min_one_way_discordant_pairs(0.10) == 5


# ---- SCENARIO-HARNESS-6050-2: the three classes are distinguished ------------------------
def test_identical_traces_are_identical_not_perturbed() -> None:
    rec = classify_trace_pair(["RESET", "a", "b"], ["RESET", "a", "b"])
    assert rec["cls"] == IDENTICAL
    assert rec["common_prefix_len"] == 3
    assert rec["first_divergence_index"] is None


def test_divergence_inside_both_traces_is_perturbed() -> None:
    rec = classify_trace_pair(["RESET", "a", "b"], ["RESET", "a", "c"])
    assert rec["cls"] == PERTURBED
    assert rec["first_divergence_index"] == 2


def test_truncated_incomplete_arm_is_a_missing_observation_not_a_perturbation() -> None:
    """The class that keeps a wall-clock cap from masquerading as a treatment effect."""
    rec = classify_trace_pair(
        ["RESET", "a", "b", "c"], ["RESET", "a"], a_complete=True, b_complete=False
    )
    assert rec["cls"] == TRUNCATION_ONLY
    assert rec["shorter_arm"] == "b"
    assert "MISSING" in rec["why"]


def test_truncated_but_COMPLETE_arm_is_a_real_behavioural_difference() -> None:
    """Same traces as above, but the short arm stopped ON PURPOSE -- that IS a difference.

    This is the mirror-image error of the previous test. Blanket-classifying every prefix
    relationship as a resource accident would throw away a genuine early-stop signal.
    """
    rec = classify_trace_pair(
        ["RESET", "a", "b", "c"], ["RESET", "a"], a_complete=True, b_complete=True
    )
    assert rec["cls"] == PERTURBED


def test_equal_traces_with_BOTH_arms_truncated_are_not_IDENTICAL() -> None:
    """The missing test, and the defect it would have caught (2026-07-30 review).

    `classify_trace_pair` checked `a == b` BEFORE consulting completeness, so a pair in which
    NEITHER arm got past action 27 of a 400-action budget came back IDENTICAL -- and the recorded
    reason asserted "no downstream endpoint can differ", which is unsupportable when neither arm
    was allowed to reach an endpoint. Live in the retention grid on cn04 and r11l.
    """
    rec = classify_trace_pair([1, 2, 3], [1, 2, 3], a_complete=False, b_complete=False)
    assert rec["cls"] == BOTH_TRUNCATED
    assert "MISSING observation" in rec["why"]
    # One arm incomplete at equal length is also not a measurement: whether the cut-short arm
    # would have gone on to diverge is unobserved.
    assert (
        classify_trace_pair([1, 2, 3], [1, 2, 3], a_complete=True, b_complete=False)["cls"]
        == TRUNCATION_ONLY
    )
    # And the genuine case still classifies as before.
    assert (
        classify_trace_pair([1, 2, 3], [1, 2, 3], a_complete=True, b_complete=True)["cls"]
        == IDENTICAL
    )


def test_a_truncated_AA_pair_cannot_license_attribution() -> None:
    """The FALSE PASS this fix closes -- the one direction that actually costs something.

    A cell counted as attributable only if its A/A class was IDENTICAL. Combined with the
    `a == b`-before-completeness bug, a double-truncated A/A pair -- which measures NOTHING --
    certified the harness as deterministic on that cell and licensed the A/B perturbation as
    treatment-attributable. Verified shape: 6 A/B-perturbed cells whose A/A replicates were each
    cut off after 2 actions produced `verdict=PASS`, `n_perturbed_attributable=6`,
    `best_reachable_p=0.03125` and no warning at all.

    A false REFUSE only wastes an experiment. A false PASS spends hours and then reports a number
    nobody can attribute, which is worse, so this is pinned in the PASS direction deliberately.
    """
    ab = {
        f"g{i}": classify_trace_pair([1, 2], [3, 4], a_complete=True, b_complete=True)
        for i in range(6)
    }
    assert all(r["cls"] == PERTURBED for r in ab.values())

    # A/A replicates that were BOTH cut short after 2 actions: no information whatsoever.
    aa_truncated = {
        f"g{i}": classify_trace_pair([1, 2], [1, 2], a_complete=False, b_complete=False)
        for i in range(6)
    }
    v = preflight_verdict(ab, noise_pairs=aa_truncated)
    assert v["verdict"] == REFUSE
    assert v["n_perturbed_attributable"] == 0
    assert set(v["unattributable_cells_with_aa_class"].values()) == {BOTH_TRUNCATED}

    # Defence in depth: even a hand-built A/A record that CLAIMS IDENTICAL is rejected when its
    # own completeness flags say an arm was cut short. `noise_pairs` is plain data a caller can
    # construct, so the rule is enforced at the point of use as well as at classification.
    aa_forged = {
        f"g{i}": {"cls": IDENTICAL, "a_complete": True, "b_complete": False} for i in range(6)
    }
    v2 = preflight_verdict(ab, noise_pairs=aa_forged)
    assert v2["verdict"] == REFUSE
    assert set(v2["unattributable_cells_with_aa_class"].values()) == {
        "IDENTICAL_BUT_AA_ARM_TRUNCATED"
    }

    # CONTROL: with a genuine complete A/A floor, the same A/B data DOES pass -- so the refusals
    # above are caused by the truncation, not by some unrelated tightening.
    aa_good = {
        f"g{i}": classify_trace_pair([1, 2], [1, 2], a_complete=True, b_complete=True)
        for i in range(6)
    }
    v3 = preflight_verdict(ab, noise_pairs=aa_good)
    assert v3["verdict"] == PASS
    assert v3["n_perturbed_attributable"] == 6


def test_planned_n_cells_actually_decides_when_probing_a_subset() -> None:
    """`planned_n_cells` was inert: the advertised probe-a-subset workflow could pass nothing.

    Its docstring said it "exists so a caller can probe 12 cells while planning 24", but only
    `ceiling_strict >= required` decided. Verified before the fix: 4 of 12 probed cells perturbing
    was REFUSED at planned_n_cells = 12, 24, 48 AND 1000 alike -- even though 1000 cells at that
    rate expects ~333 discordant-capable cells against the 6 needed. The tool computed
    `cells_needed` and then ignored it.
    """
    ab = {}
    for i in range(4):
        ab[f"p{i}"] = classify_trace_pair([1, 2], [3, 4], a_complete=True, b_complete=True)
    for i in range(8):
        ab[f"i{i}"] = classify_trace_pair([1, 2], [1, 2], a_complete=True, b_complete=True)
    aa = {k: classify_trace_pair([1, 2], [1, 2], a_complete=True, b_complete=True) for k in ab}

    # At the probed size the strict ceiling (4) is below 6, so REFUSE is correct.
    at_probed = preflight_verdict(ab, planned_n_cells=12, noise_pairs=aa)
    assert at_probed["verdict"] == REFUSE
    assert at_probed["decision_basis"] == "strict_attributable_ceiling_at_probed_n"

    # Probing 12 while planning 24 at a 1-in-3 rate expects 8 >= 6. That grid is worth running.
    at_24 = preflight_verdict(ab, planned_n_cells=24, noise_pairs=aa)
    assert at_24["verdict"] == PASS
    assert at_24["decision_basis"] == "projected_attributable_rate_at_planned_n"
    assert at_24["projected_attributable_cells_at_planned_n"] == pytest.approx(8.0)
    assert at_24["cells_needed_at_observed_perturbation_rate"] == 18

    # Planning 15 at the same rate expects 5 < 6 -- still refused, so the projection is a real
    # threshold and not a rubber stamp for any planned_n_cells above the probe size.
    assert preflight_verdict(ab, planned_n_cells=15, noise_pairs=aa)["verdict"] == REFUSE

    # And projection is NOT allowed without a noise floor: forecasting an unattributed rate just
    # predicts how much noise the bigger grid will contain.
    no_floor = preflight_verdict(ab, planned_n_cells=1000)
    assert no_floor["verdict"] == REFUSE
    assert no_floor["decision_basis"] == "strict_attributable_ceiling_at_probed_n"


def test_absent_or_empty_trace_is_MISSING() -> None:
    assert classify_trace_pair(None, ["RESET"])["cls"] == MISSING
    assert classify_trace_pair([], ["RESET"])["cls"] == MISSING
    assert classify_trace_pair(["RESET"], None)["cls"] == MISSING


# ---- SCENARIO-HARNESS-6050-3: the verdict refuses an underpowered grid --------------------
def _grid(n_perturbed: int, n_identical: int, n_trunc: int = 0) -> dict:
    pairs = {}
    for i in range(n_perturbed):
        pairs[f"p{i}"] = classify_trace_pair(["RESET", "a"], ["RESET", "b"])
    for i in range(n_identical):
        pairs[f"i{i}"] = classify_trace_pair(["RESET", "a"], ["RESET", "a"])
    for i in range(n_trunc):
        pairs[f"t{i}"] = classify_trace_pair(
            ["RESET", "a", "b"], ["RESET", "a"], a_complete=True, b_complete=False
        )
    return pairs


def test_refuses_when_fewer_than_six_cells_perturb() -> None:
    v = preflight_verdict(_grid(n_perturbed=5, n_identical=7))
    assert v["verdict"] == REFUSE
    assert v["discordance_ceiling_strict"] == 5
    assert v["best_reachable_p_strict"] == pytest.approx(0.0625)


def test_passes_at_exactly_six_perturbed_cells() -> None:
    v = preflight_verdict(_grid(n_perturbed=6, n_identical=6))
    assert v["verdict"] == PASS
    assert v["best_reachable_p_strict"] == pytest.approx(0.03125)


def test_truncations_do_not_buy_a_pass() -> None:
    """A missing observation must never be counted as a favourable one.

    Six cells' worth of ceiling exists only if truncations are (wrongly) credited. The
    strict ceiling is what decides, so this grid is refused -- while the charitable ceiling
    is still reported, because it makes the refusal argument stronger when it too falls short.
    """
    v = preflight_verdict(_grid(n_perturbed=2, n_identical=6, n_trunc=4))
    assert v["verdict"] == REFUSE
    assert v["discordance_ceiling_strict"] == 2
    assert v["discordance_ceiling_charitable_counts_truncations"] == 6


def test_missing_and_truncated_cells_are_excluded_from_the_perturbation_rate() -> None:
    """Only cells that YIELDED AN OBSERVATION belong in the rate's denominator.

    MISSING was always excluded. TRUNCATION_ONLY was NOT (2026-07-30 review): it sat in the
    denominator and never in the numerator, i.e. it was scored as a zero -- the exact thing this
    module's own class documentation forbids, with a direction, because the arm that gets
    truncated is systematically the slower one and a treatment that does more work is
    systematically slower. So the bias always ran against the treatment.
    """
    pairs = _grid(n_perturbed=3, n_identical=3)
    pairs["m0"] = classify_trace_pair(None, None)
    pairs["m1"] = classify_trace_pair(None, ["RESET"])
    pairs["t0"] = classify_trace_pair([1, 2, 3], [1, 2], a_complete=True, b_complete=False)
    v = preflight_verdict(pairs)
    assert v["n_probed_cells"] == 9
    assert v["n_comparable_cells"] == 7
    assert v["n_truncation_affected"] == 1
    assert v["n_usable_observations"] == 6
    # 3/6 -- MISSING and the truncation are both out of the denominator.
    assert v["perturbation_rate_over_usable_observations"] == pytest.approx(0.5)
    # The old, biased number is still emitted, under a name that says what it does: 3/7.
    assert v["perturbation_rate_over_comparable_counting_truncations_as_zero"] == pytest.approx(
        3 / 7, abs=1e-4
    )
    # At a 50% perturbation rate you need 12 cells to expect 6 perturbed ones.
    assert v["cells_needed_at_observed_perturbation_rate"] == 12


def test_zero_perturbation_reports_no_reachable_sample_size() -> None:
    v = preflight_verdict(_grid(n_perturbed=0, n_identical=12))
    assert v["verdict"] == REFUSE
    assert v["best_reachable_p_strict"] == pytest.approx(1.0)
    # Not 0, not a huge number -- None. No sample size fixes a treatment that never acts.
    assert v["cells_needed_at_observed_perturbation_rate"] is None


def test_a_pass_still_states_that_it_is_not_a_power_guarantee() -> None:
    """Guards against the single most likely misuse of this tool."""
    v = preflight_verdict(_grid(n_perturbed=8, n_identical=4))
    assert v["verdict"] == PASS
    assert "not sufficient" in v["interpretation"]
    assert "does NOT mean the experiment is powered" in v["interpretation"]


# ---- SCENARIO-HARNESS-6050-5: the A/A noise floor gates attribution ----------------------
def test_noise_floor_absent_is_reported_as_a_warning_not_silently_ignored() -> None:
    v = preflight_verdict(_grid(n_perturbed=8, n_identical=4))
    assert v["noise_floor_measured"] is False
    assert v["n_perturbed_attributable"] is None
    assert "NO A/A NOISE FLOOR" in v["noise_floor_warning"]
    assert "uninterpretable" in v["noise_floor_warning"]


def test_a_nondeterministic_harness_cannot_buy_a_pass() -> None:
    """Every cell perturbs under A/B -- but every cell also perturbs under A/A.

    This is the trivial-pass failure mode the noise floor exists to catch. Without it the
    verdict would be a confident PASS on a grid where nothing is attributable to the
    treatment at all.
    """
    labels = [f"g{i}" for i in range(12)]
    ab = {lab: classify_trace_pair(["RESET", "a"], ["RESET", "b"]) for lab in labels}
    aa = {lab: classify_trace_pair(["RESET", "a"], ["RESET", "c"]) for lab in labels}

    without = preflight_verdict(ab)
    assert without["verdict"] == PASS  # the wrong answer, reached honestly
    with_floor = preflight_verdict(ab, noise_pairs=aa)
    assert with_floor["verdict"] == REFUSE
    assert with_floor["n_perturbed_raw"] == 12
    assert with_floor["n_perturbed_attributable"] == 0
    assert with_floor["best_reachable_p_strict"] == pytest.approx(1.0)


def test_only_cells_deterministic_under_aa_count_as_attributable() -> None:
    labels = [f"g{i}" for i in range(8)]
    ab = {lab: classify_trace_pair(["RESET", "a"], ["RESET", "b"]) for lab in labels}
    aa = {}
    for i, lab in enumerate(labels):
        if i < 6:  # deterministic under A/A -> attributable
            aa[lab] = classify_trace_pair(["RESET", "a"], ["RESET", "a"])
        elif i == 6:  # noisy under A/A -> not attributable
            aa[lab] = classify_trace_pair(["RESET", "a"], ["RESET", "z"])
        else:  # A/A truncated -> determinism unestablished -> not attributable
            aa[lab] = classify_trace_pair(
                ["RESET", "a", "b"], ["RESET", "a"], a_complete=True, b_complete=False
            )
    v = preflight_verdict(ab, noise_pairs=aa)
    assert v["verdict"] == PASS
    assert v["n_perturbed_raw"] == 8
    assert v["n_perturbed_attributable"] == 6
    assert v["attributable_cells"] == [f"g{i}" for i in range(6)]
    assert v["unattributable_cells_with_aa_class"] == {"g6": PERTURBED, "g7": TRUNCATION_ONLY}


def test_missing_aa_cell_is_not_attributable() -> None:
    ab = {"g0": classify_trace_pair(["RESET", "a"], ["RESET", "b"])}
    aa = {"g0": classify_trace_pair(None, None)}
    v = preflight_verdict(ab, noise_pairs=aa)
    assert v["n_perturbed_attributable"] == 0
    assert v["unattributable_cells_with_aa_class"] == {"g0": MISSING}


def test_format_report_renders_every_cell_and_the_verdict() -> None:
    v = preflight_verdict(_grid(n_perturbed=1, n_identical=2))
    text = format_report(v, title="probe")
    assert "VERDICT: REFUSE" in text
    assert text.count("IDENTICAL") >= 2
    assert "probe" in text


# ---- SCENARIO-HARNESS-6050-4: RETROSPECTIVE VALIDATION on the committed grid --------------
def _load_committed_retention_pairs() -> dict:
    """The 12 matched pairs of the 2026-07-29 engine-retention A/B, read from `results/`.

    Read-only by design: `results/` is EVIDENCE. A run marked `timed_out` (or any non-`ok`
    status) is the incomplete arm -- exactly the input `classify_trace_pair` needs to tell a
    wall-clock cap apart from a behavioural difference.
    """
    games = sorted(
        {f.split("__")[1] for f in os.listdir(RETENTION_CELLS) if f.startswith("ret0__")}
    )
    pairs = {}
    for g in games:
        cells = {}
        for arm in ("ret0", "ret1"):
            with open(os.path.join(RETENTION_CELLS, f"{arm}__{g}__s1.json")) as fh:
                cells[arm] = json.load(fh)
        rec = {}
        for arm in ("ret0", "ret1"):
            res = cells[arm].get("result") or {}
            rec[arm] = {
                "trace": res.get("action_trace"),
                # "Complete" means the run ended on its own terms: the cell reported `ok` and
                # did not hit the wall-clock cap.
                "complete": cells[arm].get("status") == "ok" and not res.get("timed_out"),
            }
        pairs[g] = classify_trace_pair(
            rec["ret0"]["trace"],
            rec["ret1"]["trace"],
            a_complete=rec["ret0"]["complete"],
            b_complete=rec["ret1"]["complete"],
        )
    return pairs


def test_committed_retention_grid_is_present_and_complete() -> None:
    """Fail loudly rather than silently skipping if the evidence ever moves.

    A pytest.skip here would make the load-bearing retrospective validation invisible, which
    is the failure mode CLAUDE.md's "Tests Must Run and Assert" rule exists to prevent.
    """
    assert os.path.isdir(RETENTION_CELLS), f"committed evidence missing: {RETENTION_CELLS}"
    assert len(os.listdir(RETENTION_CELLS)) == 24


def test_refuses_the_committed_retention_ab_grid() -> None:
    """THE validation: the pre-flight must refuse a grid independently known to be dead.

    CORRECTED PARTITION (2026-07-30). The first published post-mortem read "8 byte-identical / 3
    truncation-only / 1 perturbed". Two of those eight were NOT measurements: on cn04 and r11l
    NEITHER arm got past action 27 of a 400-action budget, and the classifier reached its `a == b`
    branch before consulting completeness. The honest partition is 6 IDENTICAL / 3
    TRUNCATION_ONLY / 2 BOTH_TRUNCATED / 1 PERTURBED.

    With one perturbed cell the strict ceiling on discordant pairs is 1, so the best p-value the
    grid could ever have reported is 1.0 -- the null was guaranteed before it started. A
    pre-flight that does not refuse this does not work.

    HONEST NOTE ON THE CHARITABLE CEILING: with 5 truncation-affected cells rather than 3, the
    charitable ceiling is now 6, which just reaches alpha=0.05 (p=0.03125). So this refusal no
    longer holds "even under the most generous possible coding of the missing observations" -- it
    rests on the strict reading, in which a truncated cell is excluded rather than counted as a
    favourable discordant pair. That reading is the correct one (coding a missing observation as
    favourable is the exact error this module exists to prevent), but the weaker margin is stated
    rather than buried.
    """
    pairs = _load_committed_retention_pairs()
    v = preflight_verdict(pairs, planned_n_cells=12)

    assert v["verdict"] == REFUSE
    assert v["n_probed_cells"] == 12
    assert v["n_comparable_cells"] == 12
    assert v["counts"][IDENTICAL] == 6
    assert v["counts"][TRUNCATION_ONLY] == 3
    assert v["counts"][BOTH_TRUNCATED] == 2
    assert v["counts"][PERTURBED] == 1
    assert v["n_truncation_affected"] == 5
    assert v["discordance_ceiling_strict"] == 1
    assert v["discordance_ceiling_charitable_counts_truncations"] == 6
    assert v["best_reachable_p_strict"] == pytest.approx(1.0)
    assert v["best_reachable_p_charitable"] == pytest.approx(0.03125)
    # The rate is over USABLE observations (12 comparable minus 5 truncation-affected = 7), so
    # 1/7 = 0.1429 and 42 matched pairs would be needed. Counting the truncations as zeros instead
    # gives the old, biased-against-the-treatment 1/12 = 0.0833 and 72 pairs.
    assert v["n_usable_observations"] == 7
    assert v["perturbation_rate_over_usable_observations"] == pytest.approx(0.1429, abs=1e-4)
    assert v["cells_needed_at_observed_perturbation_rate"] == 42
    assert v["perturbation_rate_over_comparable_counting_truncations_as_zero"] == pytest.approx(
        0.0833, abs=1e-4
    )


def test_cli_refuses_the_committed_retention_grid_with_exit_code_1() -> None:
    """The CLI is the path a human actually uses, so its exit code is part of the contract.

    Exit 1 on REFUSE means a launch script can gate on it (`preflight ... && run_grid`) instead
    of relying on someone reading the report.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_tap_cli", os.path.join(REPO, "scripts", "treatment_activation_preflight.py")
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    rc = cli.main(
        [
            "--cells",
            RETENTION_CELLS,
            "--control",
            "ret0",
            "--treatment",
            "ret1",
            "--planned-n-cells",
            "12",
        ]
    )
    assert rc == 1

    # A missing directory is a usage error (2), NOT a refusal (1) -- a caller gating on exit 1
    # must not be told "underpowered" when the real problem is a typo'd path.
    assert (
        cli.main(
            [
                "--cells",
                os.path.join(REPO, "no_such_dir_xyz"),
                "--control",
                "ret0",
                "--treatment",
                "ret1",
            ]
        )
        == 2
    )
    # An arm with no cells at all is likewise usage, not refusal -- and it is checked PER ARM, so
    # a typo in either name is caught even though the cell census is the union of both. Reporting a
    # typo as "12 MISSING cells, underpowered" would send the caller after a measurement problem
    # that does not exist.
    assert (
        cli.main(["--cells", RETENTION_CELLS, "--control", "nosucharm", "--treatment", "ret1"]) == 2
    )
    assert (
        cli.main(["--cells", RETENTION_CELLS, "--control", "ret0", "--treatment", "nosucharm"]) == 2
    )


def test_cli_reads_completeness_from_the_cell_record_not_from_trace_length() -> None:
    """`--control ret0 --treatment ret1` must reproduce the library's own classification.

    Pinned because the CLI's `_load_arm_cell` is where the complete/incomplete flag is derived,
    and getting that wrong silently reclassifies every truncation.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_tap_cli2", os.path.join(REPO, "scripts", "treatment_activation_preflight.py")
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    pairs, provenance = cli._pairs(
        RETENTION_CELLS, "ret0", "ret1", ["cd82", "lp85", "vc33", "cn04"]
    )
    assert pairs["cd82"]["cls"] == IDENTICAL
    assert pairs["lp85"]["cls"] == TRUNCATION_ONLY  # ret1 timed out; ret0 ran to budget
    assert pairs["vc33"]["cls"] == PERTURBED
    assert pairs["cn04"]["cls"] == BOTH_TRUNCATED  # neither arm finished; equal traces
    # The verdict must say WHICH FILE each number came from, or it is not checkable by anyone who
    # was not at the keyboard.
    assert provenance["cd82"] == {"ret0": "ret0__cd82__s1.json", "ret1": "ret1__cd82__s1.json"}
    # And an arm that produced no file at all is MISSING, not an empty perturbation.
    assert cli._load_arm_cell(RETENTION_CELLS, "nosucharm", "cd82") == (None, False, None)


def test_cli_refuses_to_pick_silently_between_duplicate_records(tmp_path) -> None:
    """Two records for one arm+cell is a USAGE error, not a silent choice of one of them.

    The old code took `sorted(matches)[0]`. A second seed or a re-run replicate (`..__s2.json`)
    would then have been neither read, nor reported, nor counted as MISSING -- it would simply
    have vanished, and the verdict would rest on an arbitrary one of several available runs. That
    is the same silent-data-selection error class the pre-flight exists to catch, so it must exit
    2 rather than resolve itself.
    """
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location(
        "_tap_cli_dup", os.path.join(REPO, "scripts", "treatment_activation_preflight.py")
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    cells = tmp_path / "cells"
    cells.mkdir()
    for arm in ("ret0", "ret1"):
        src = os.path.join(RETENTION_CELLS, f"{arm}__cd82__s1.json")
        shutil.copy(src, cells / f"{arm}__cd82__s1.json")
    # A second run of the SAME arm+cell, which the old code would have discarded in silence.
    shutil.copy(cells / "ret1__cd82__s1.json", cells / "ret1__cd82__s2.json")

    with pytest.raises(cli.AmbiguousCellRecord):
        cli._load_arm_cell(str(cells), "ret1", "cd82")
    assert cli.main(["--cells", str(cells), "--control", "ret0", "--treatment", "ret1"]) == 2
    # --suffix is the explicit resolution, and it must then work.
    assert (
        cli.main(
            ["--cells", str(cells), "--control", "ret0", "--treatment", "ret1", "--suffix", "s1"]
        )
        == 1
    )


def test_the_one_perturbed_cell_is_vc33_and_the_truncations_are_named() -> None:
    """Per-cell attribution, so the refusal is diagnosable and not just a number.

    vc33 is also the cell that banked the MOST levels (2) in the whole grid, which is why
    losing it to an underpowered design was expensive.
    """
    pairs = _load_committed_retention_pairs()
    perturbed = sorted(g for g, r in pairs.items() if r["cls"] == PERTURBED)
    truncated = sorted(g for g, r in pairs.items() if r["cls"] == TRUNCATION_ONLY)
    assert perturbed == ["vc33"]
    assert truncated == ["lp85", "ls20", "su15"]
    assert pairs["vc33"]["first_divergence_index"] == 17


# ---- SCENARIO-HARNESS-6050-5: the A/A floor measured from committed cells ------------------
HELDOUT_CELLS = os.path.join(REPO, "results", "arc_heldout_31b_vs_9b_20260728", "cells")


def _state(path: str) -> tuple:
    if not os.path.exists(path):
        return None, False
    with open(path) as fh:
        d = json.load(fh)
    res = d.get("result") or {}
    return res.get("action_trace"), (d.get("status") == "ok" and not res.get("timed_out"))


# The SAME-COMMIT A/A control, committed 2026-07-30. `post` and `postb` are two runs of the
# IDENTICAL checkout (f9a458e87), same per-game sampler seed, same GGUF, same n_ctx, same action
# budget, same game -- differing only in session, GPU and port.
GOALSPEC_CELLS = os.path.join(REPO, "results", "arc_goalspec_f9a458e87_preflight_20260730", "cells")


def test_the_harness_is_not_deterministic_even_with_the_sampler_seed_set() -> None:
    """The A/A floor, measured on a PROVABLY same-code pair.

    RETRACTION AND REPLACEMENT (2026-07-30). The first version of this test paired the retention
    grid's `ret1` cells against a different experiment's `31b` cells and called it A/A, on the
    grounds that both were "retention ON at seed 1 on the same GGUF". That was WRONG, and it is
    retracted rather than reworded: the held-out cells ran BEFORE commit 11cd3c3a9 introduced
    engine retention at all, so those cells contain no retention code and carry no retention
    fields. Six-plus agentic commits separate the two sets. That pairing was therefore an A/B of
    the very treatment under test, and the conclusions drawn from it ("2 of 5 cells diverge under
    IDENTICAL code", and hence "vc33's attributable perturbation is ZERO") did not follow.

    This replacement uses the real thing: two runs of the same commit, asserted from the cells'
    own recorded `arm_commit`, `generator_sampler_seed_effective`, `observed_model_path`,
    `observed_n_ctx` and `budget` rather than from an argument about flags.

    The finding is STRONGER than the retracted one and points the same way: 1 of 2 pairs diverges
    even with `CARNOT_ARC_GENERATOR_SEED` set, and the two ft09 runs disagreed about whether a
    plan was found at all (1 vs 0). So seeding the sampler is necessary but NOT sufficient, and a
    noise floor must be measured per grid rather than assumed. n=2 supports "this happens"; it
    does not support a rate, and no rate is asserted here.
    """
    assert os.path.isdir(GOALSPEC_CELLS), f"committed evidence missing: {GOALSPEC_CELLS}"

    def _rec(arm: str, game: str) -> dict:
        with open(os.path.join(GOALSPEC_CELLS, f"{arm}__{game}__s1.json")) as fh:
            return json.load(fh)

    witnesses = (
        "arm_commit",
        "arm_repo",
        "generator_sampler_seed_effective",
        "observed_model_path",
        "observed_n_ctx",
        "budget",
        "explore_budget",
        "max_inductions",
        "seed",
        "anon_game_id",
    )
    aa = {}
    for game in ("ft09", "vc33"):
        a, b = _rec("post", game), _rec("postb", game)
        # The A/A claim rests on these being equal. Assert it instead of asserting it in prose.
        for key in witnesses:
            assert a[key] == b[key], (game, key, a[key], b[key])
        aa[game] = classify_trace_pair(
            (a["result"] or {}).get("action_trace"),
            (b["result"] or {}).get("action_trace"),
            a_complete=(a["status"] == "ok" and not (a["result"] or {}).get("timed_out")),
            b_complete=(b["status"] == "ok" and not (b["result"] or {}).get("timed_out")),
        )

    assert aa["vc33"]["cls"] == IDENTICAL, aa["vc33"]
    assert aa["ft09"]["cls"] == PERTURBED, aa["ft09"]
    assert aa["ft09"]["first_divergence_index"] == 26
    # The divergence reaches an outcome, not just the trace: same code, same seed, different
    # answer to "did the planner find a plan".
    assert _rec("post", "ft09")["result"]["n_plans_found"] == 1
    assert _rec("postb", "ft09")["result"]["n_plans_found"] == 0


def test_the_retention_grid_has_no_valid_noise_floor_so_its_one_cell_is_unattributable() -> None:
    """The retention A/B's single perturbed cell is UNATTRIBUTABLE -- not proven to be noise.

    The distinction matters and is the corrected claim. "Attributable perturbation is ZERO" says
    we measured the noise floor and vc33 fell inside it. We did not: its pre-registered `ret0b` /
    `ret1b` A/A arms were never run, and the cross-experiment substitute was invalid (see the
    retraction above). So the honest statement is that the grid contains one perturbed cell whose
    cause cannot be separated from harness noise.

    The VERDICT is unchanged either way -- REFUSE, ceiling 1, best reachable p = 1.0 -- which is
    why the retraction costs nothing operationally. It costs only a claim we could not support.
    """
    pairs = _load_committed_retention_pairs()
    v = preflight_verdict(pairs, planned_n_cells=12)

    assert v["verdict"] == REFUSE
    assert v["noise_floor_measured"] is False
    assert v["noise_floor_warning"] and "NO A/A NOISE FLOOR" in v["noise_floor_warning"]
    assert v["n_perturbed_raw"] == 1
    assert v["n_perturbed_attributable"] is None  # unmeasured, NOT zero
    assert v["best_reachable_p_strict"] == pytest.approx(1.0)


def test_the_cross_experiment_pairing_is_not_a_valid_aa_control() -> None:
    """Pin the retraction: `ret1` vs `31b` differ in CODE, so they cannot bound harness noise.

    Kept as an executable test rather than a comment so the invalid pairing cannot quietly come
    back. The witness is structural: the held-out cells predate engine retention, so they do not
    carry the retention diagnostic fields that every post-11cd3c3a9 cell carries.
    """
    with open(os.path.join(RETENTION_CELLS, "ret1__vc33__s1.json")) as fh:
        ret1 = json.load(fh)
    with open(os.path.join(HELDOUT_CELLS, "31b__vc33__s1.json")) as fh:
        heldout = json.load(fh)

    retention_fields = {k for k in ret1 if "retention" in k.lower()}
    assert retention_fields >= {
        "retention_requested",
        "retention_enabled_effective",
        "retention_support",
    }, sorted(retention_fields)
    assert not (retention_fields & set(heldout)), (
        "the held-out cells carry NO retention fields at all, i.e. they ran before the treatment "
        "existed -- so pairing them with a retention-ON run is an A/B of the treatment under "
        f"test, not an A/A. held-out top-level keys: {sorted(heldout)}"
    )


def test_vc33_divergence_index_is_stable_across_runs_and_models() -> None:
    """Four runs diverge at the SAME action index 17 -- a fact, reported without the overclaim.

    NARROWED 2026-07-30. This test previously concluded "no treatment effect can produce that
    pattern, so index 17 is a harness coin flip". That inference does not hold: the four runs are
    not matched (they differ in commit, and two of them differ in `generator_output_reached_policy`
    and in whether the run completed), so the partition crossing treatment lines has explanations
    other than a coin flip.

    What survives is the observation itself, which is still useful: divergence, when it happens on
    vc33, begins at a FIXED point rather than drifting -- consistent with a single branch point in
    the agent's decision sequence. That is a lead for locating the nondeterminism, not evidence
    about retention.
    """
    t = {
        "ret0_31B_off": _state(os.path.join(RETENTION_CELLS, "ret0__vc33__s1.json"))[0],
        "ret1_31B_on": _state(os.path.join(RETENTION_CELLS, "ret1__vc33__s1.json"))[0],
        "heldout_31b": _state(os.path.join(HELDOUT_CELLS, "31b__vc33__s1.json"))[0],
        "heldout_9b_other_model": _state(os.path.join(HELDOUT_CELLS, "9b__vc33__s1.json"))[0],
    }
    assert all(v for v in t.values())
    for a, b in (
        ("ret0_31B_off", "ret1_31B_on"),
        ("ret1_31B_on", "heldout_31b"),
        ("heldout_31b", "heldout_9b_other_model"),
    ):
        rec = classify_trace_pair(t[a], t[b])
        assert rec["cls"] == PERTURBED
        assert rec["first_divergence_index"] == 17, (a, b, rec)


def test_cli_census_is_the_union_of_both_arms_not_just_the_control() -> None:
    """A cell only the TREATMENT produced must be visible as MISSING, not silently dropped.

    A control-only census hides a drop asymmetry that favours the control -- the same
    missing-vs-zero error this tool exists to prevent, one level up. The held-out 31b-vs-9b grid
    is exactly that shape: 3 of 8 cells are absent on ONE arm.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_tap_cli3", os.path.join(REPO, "scripts", "treatment_activation_preflight.py")
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    # In this grid the 9b arm has 8 games and the 31b arm has 5. Enumerating 31b alone would see
    # 5 cells; the union sees all 8, three of which are MISSING on the 31b side.
    n_9b = len({f.split("__")[1] for f in os.listdir(HELDOUT_CELLS) if f.startswith("9b__")})
    n_31b = len({f.split("__")[1] for f in os.listdir(HELDOUT_CELLS) if f.startswith("31b__")})
    assert n_9b == 8 and n_31b == 5, (n_9b, n_31b)

    # Run with the SMALLER arm as the control: a control-only census would report 5 cells.
    rc = cli.main(["--cells", HELDOUT_CELLS, "--control", "31b", "--treatment", "9b", "--json"])
    assert rc == 1  # still a refusal
    pairs, _prov = cli._pairs(
        HELDOUT_CELLS,
        "31b",
        "9b",
        sorted({f.split("__")[1] for f in os.listdir(HELDOUT_CELLS) if "__" in f}),
    )
    v = preflight_verdict(pairs)
    assert v["n_probed_cells"] == 8
    assert v["counts"][MISSING] == 3


# ---- SCENARIO-HARNESS-6050-9: an unfinished grid is INCONCLUSIVE, never REFUSE ---------------
def test_an_incomplete_grid_is_INCONCLUSIVE_not_a_refusal() -> None:
    """The module's own "missing is not a zero" principle, applied at the VERDICT level.

    Every unrun cell classifies as MISSING, and MISSING is not PERTURBED, so a partially-run probe
    used to come back REFUSE. That is the same error the class docstrings forbid, one level up --
    and it bites hardest in the case that actually happened on 2026-07-30: a probe stopped after 2
    of 12 cells (because review found a bug in the arm under test) would have produced an artifact
    reading REFUSE, laundering a process decision into a finding that the treatment is inert.
    """
    ab = {
        "a": classify_trace_pair([1], [2], a_complete=True, b_complete=True),  # PERTURBED
        "b": classify_trace_pair([1], [1], a_complete=True, b_complete=True),  # IDENTICAL
    }
    for i in range(10):
        ab[f"pending{i}"] = classify_trace_pair(None, None)
    aa = {k: classify_trace_pair([1], [1], a_complete=True, b_complete=True) for k in ab}

    v = preflight_verdict(ab, planned_n_cells=12, noise_pairs=aa)
    assert v["verdict"] == INCONCLUSIVE
    assert v["grid_is_incomplete_and_still_arithmetically_open"] is True
    assert v["n_missing_cells_still_outstanding"] == 10
    # 1 attributable + 10 outstanding = 11, comfortably above the 6 required.
    assert v["max_reachable_ceiling_if_every_missing_cell_were_attributable"] == 11
    assert "NOT evidence that the treatment is inert" in v["interpretation"]
    assert "INCOMPLETE" in format_report(v)


def test_a_complete_grid_still_REFUSES() -> None:
    """The INCONCLUSIVE state must not soften a real refusal -- no missing cells, no reprieve."""
    ab = {
        f"g{i}": classify_trace_pair([1], [1], a_complete=True, b_complete=True) for i in range(11)
    }
    ab["p"] = classify_trace_pair([1], [2], a_complete=True, b_complete=True)
    aa = {k: classify_trace_pair([1], [1], a_complete=True, b_complete=True) for k in ab}
    v = preflight_verdict(ab, planned_n_cells=12, noise_pairs=aa)
    assert v["verdict"] == REFUSE
    assert v["grid_is_incomplete_and_still_arithmetically_open"] is False


def test_missing_cells_that_cannot_close_the_gap_still_REFUSE() -> None:
    """INCONCLUSIVE requires that the outstanding cells COULD change the answer.

    With 0 attributable so far and only 3 cells outstanding against 6 required, no outcome of the
    remaining work reaches alpha -- so this is a genuine refusal even though the grid is unfinished,
    and reporting it as INCONCLUSIVE would waste the three cells.
    """
    ab = {
        f"g{i}": classify_trace_pair([1], [1], a_complete=True, b_complete=True) for i in range(9)
    }
    for i in range(3):
        ab[f"m{i}"] = classify_trace_pair(None, None)
    aa = {k: classify_trace_pair([1], [1], a_complete=True, b_complete=True) for k in ab}
    v = preflight_verdict(ab, planned_n_cells=12, noise_pairs=aa)
    assert v["verdict"] == REFUSE
    assert v["grid_is_incomplete_and_still_arithmetically_open"] is False


def test_a_pass_is_never_downgraded_by_outstanding_cells() -> None:
    """An early PASS is final: more cells cannot un-perturb the ones already measured."""
    ab = {
        f"p{i}": classify_trace_pair([1], [2], a_complete=True, b_complete=True) for i in range(6)
    }
    for i in range(6):
        ab[f"m{i}"] = classify_trace_pair(None, None)
    aa = {k: classify_trace_pair([1], [1], a_complete=True, b_complete=True) for k in ab}
    v = preflight_verdict(ab, planned_n_cells=12, noise_pairs=aa)
    assert v["verdict"] == PASS
    assert v["grid_is_incomplete_and_still_arithmetically_open"] is False
