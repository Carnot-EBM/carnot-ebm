"""One arm's determinism must not license attribution for a comparison that spans two arms.

REQ-HARNESS-6052 (SCENARIO-HARNESS-6052-1 .. SCENARIO-HARNESS-6052-5).

THE HOLE, found by review 2026-07-30 on the first real grid to use this module. An A/B
comparison has a treatment arm and a control arm. ``preflight_verdict`` accepted ONE
``noise_pairs`` mapping -- one arm's A/A replicate -- and treated it as certifying the whole
comparison. So a grid whose CONTROL arm was the nondeterministic one would credit the control
arm's self-perturbation to the treatment, and the module would stamp it attributable.

That is not a theoretical hole. In the grid that exposed it the measured floor was
head-vs-headb 0/6, and every A/B pair held exactly ONE unreplicated base run. Worse, the
treatment under test changed a search DEDUP KEY -- precisely the kind of change that can
stabilise iteration order -- so "the treatment arm is deterministic" was not even weak evidence
that the control arm was. A noise floor whose validity rests on an argument is not a noise
floor; this module's own docstring says so about a different case, and then committed the error
in its own signature.

THE FIX. ``noise_pairs_b`` carries the second arm's A/A. Attribution now requires EVERY supplied
noise mapping to witness the cell as deterministic. With one mapping the behaviour is unchanged
(36 pre-existing tests still pass); with two, neither arm can speak for the other. A single-arm
call now also returns ``single_arm_noise_floor_warning``, because the realistic misuse is a
caller who reads ``noise_floor_measured: True`` and stops there.
"""

from __future__ import annotations

from carnot.analysis.treatment_activation_preflight import (
    IDENTICAL,
    MISSING,
    PERTURBED,
    TRUNCATION_ONLY,
    preflight_verdict,
    format_report,
)


def _rec(cls: str, *, a: bool = True, b: bool = True) -> dict:
    return {"cls": cls, "a_complete": a, "b_complete": b}


def _grid(n: int, cls: str = PERTURBED) -> dict:
    return {f"g{i}": _rec(cls) for i in range(n)}


def _clean_noise(n: int) -> dict:
    return {f"g{i}": _rec(IDENTICAL) for i in range(n)}


def test_second_arm_perturbing_removes_attribution() -> None:
    """The incident's shape: arm A repeats itself, arm B does not, every cell perturbs.

    Under the old single-arm rule all 8 cells were attributable and the grid PASSED. The
    perturbation was the control arm perturbing against itself.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-1.
    """
    pairs = _grid(8)
    quiet = _clean_noise(8)
    noisy = {f"g{i}": _rec(PERTURBED) for i in range(8)}

    single = preflight_verdict(pairs, noise_pairs=quiet)
    assert single["n_perturbed_attributable"] == 8
    assert single["verdict"] == "PASS"

    both = preflight_verdict(pairs, noise_pairs=quiet, noise_pairs_b=noisy)
    assert both["n_perturbed_attributable"] == 0
    assert both["verdict"] == "REFUSE"
    assert both["n_noise_arms_measured"] == 2


def test_a_cell_missing_from_the_second_arm_is_unwitnessed_not_a_pass() -> None:
    """A partial second-arm replicate must not certify the cells it never covered.

    This is the realistic case: the second arm is expensive, so a caller runs it on a SUBSET.
    The cells it covered are attributable; the cells it did not are missing observations.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-2.
    """
    pairs = _grid(4)
    quiet = _clean_noise(4)
    partial = {"g0": _rec(IDENTICAL), "g1": _rec(IDENTICAL)}

    v = preflight_verdict(pairs, noise_pairs=quiet, noise_pairs_b=partial)
    assert v["attributable_cells"] == ["g0", "g1"]
    assert v["cells_perturbed_but_lacking_a_second_arm_noise_witness"] == ["g2", "g3"]
    # The uncovered cells are reported as unattributable, never silently dropped.
    assert set(v["unattributable_cells_with_aa_class"]) == {"g2", "g3"}


def test_a_truncated_second_arm_replicate_does_not_certify() -> None:
    """A truncated A/A pair measured nothing; it must not stand in for a determinism witness.

    Same direction of error as the single-arm case the module already guards: a false PASS costs
    hours and produces a number nobody can attribute, a false REFUSE costs one experiment.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-3.
    """
    pairs = _grid(2)
    quiet = _clean_noise(2)
    truncated = {"g0": _rec(TRUNCATION_ONLY, b=False), "g1": _rec(IDENTICAL)}

    v = preflight_verdict(pairs, noise_pairs=quiet, noise_pairs_b=truncated)
    assert v["attributable_cells"] == ["g1"]
    assert "g0" in v["unattributable_cells_with_aa_class"]


def test_an_identical_but_truncated_second_arm_pair_does_not_certify() -> None:
    """IDENTICAL-with-an-incomplete-arm is the subtle one: agreement over a prefix is not a witness.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-3.
    """
    pairs = _grid(1)
    v = preflight_verdict(
        pairs,
        noise_pairs={"g0": _rec(IDENTICAL)},
        noise_pairs_b={"g0": _rec(IDENTICAL, b=False)},
    )
    assert v["n_perturbed_attributable"] == 0
    assert "IDENTICAL_BUT_AA_ARM_TRUNCATED" in v["unattributable_cells_with_aa_class"]["g0"]


def test_both_arms_clean_keeps_every_attribution() -> None:
    """The control. Two clean floors must not cost a single attributable cell.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-4.
    """
    pairs = _grid(6)
    v = preflight_verdict(pairs, noise_pairs=_clean_noise(6), noise_pairs_b=_clean_noise(6))
    assert v["n_perturbed_attributable"] == 6
    assert v["cells_perturbed_but_lacking_a_second_arm_noise_witness"] == []
    assert v["single_arm_noise_floor_warning"] is None


def test_single_arm_call_is_unchanged_but_now_warns() -> None:
    """Backward compatibility, plus the warning that makes the limitation visible.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-4.
    """
    pairs = _grid(6)
    v = preflight_verdict(pairs, noise_pairs=_clean_noise(6))
    assert v["n_perturbed_attributable"] == 6
    assert v["n_noise_arms_measured"] == 1
    assert v["per_cell_noise_b"] is None
    warning = v["single_arm_noise_floor_warning"]
    assert warning is not None and "ONLY ONE ARM" in warning
    # And it must reach a human reading the rendered report, not only the dict.
    assert "ONLY ONE ARM" in format_report(v)


def test_no_noise_floor_at_all_still_reports_the_original_warning() -> None:
    """The pre-existing no-floor path must not be shadowed by the new one.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-4.
    """
    v = preflight_verdict(_grid(3))
    assert v["noise_floor_measured"] is False
    assert v["single_arm_noise_floor_warning"] is None
    assert "NO A/A NOISE FLOOR" in v["noise_floor_warning"]


def test_a_missing_second_arm_map_entry_is_distinguished_from_a_perturbing_one() -> None:
    """ "The other arm never spoke" and "the other arm said no" are different findings.

    Conflating them is the missing-vs-present error this whole module exists to prevent, so the
    reason string has to keep them apart.

    REQ-HARNESS-6052 / SCENARIO-HARNESS-6052-5.
    """
    pairs = _grid(2)
    v = preflight_verdict(
        pairs,
        noise_pairs=_clean_noise(2),
        noise_pairs_b={"g0": _rec(PERTURBED)},
    )
    assert v["unattributable_cells_with_aa_class"]["g0"] == PERTURBED
    assert v["unattributable_cells_with_aa_class"]["g1"] == MISSING
    assert v["cells_perturbed_but_lacking_a_second_arm_noise_witness"] == ["g1"]
