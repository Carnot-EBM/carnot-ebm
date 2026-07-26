"""Tests for the scored-path lever A/B analyser.

Each test here reproduces a SPECIFIC measurement defect this project has already shipped once,
rather than a synthetic happy path. If one of these regresses, a wrong number gets published.

REQ-ARC-WMTE-5980 / SCENARIO-scored-path-lever-ab-analysis.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_scored_path_lever_ab",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_scored_path_lever_ab.py",
)
assert _SPEC and _SPEC.loader
AN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(AN)


def row(
    arm: str,
    game: str,
    seed: int,
    levels: int,
    *,
    lever1: bool = True,
    lever2: bool = False,
    lever3_verdict: str = "LEVER_OFF",
    valid: bool = True,
    responses: int = 4,
    states: int = 100,
    a2l: int | None = 50,
) -> dict:
    return {
        "arm": arm,
        "game": game,
        "seed": seed,
        "ran": True,
        "levels": levels,
        "budget": 400,
        "llm_on_row_valid": valid,
        "llm": {"responses": responses, "tokens_predicted": 100, "llm_wall_s": 1.0},
        # A realistic cell records generator health on BOTH sides; the analyser's recomputed validity
        # criterion reads these, so omitting them would make every synthetic LLM-on row invalid.
        "generator_healthy_before": True,
        "generator_healthy_after": True,
        "lever1_fired": lever1,
        "lever1_frontier_fire": {"tier_advances": 3 if lever1 else 0},
        # LEVER 2 IS DESCRIBED BY ITS DIAGNOSTICS, NOT BY A HAND-INJECTED FIRE FLAG. The analyser
        # RECOMPUTES `lever2_fired` from these digests (see `recomputed_lever2_fired`), because the
        # harness's original stamp was measured to be anti-correlated with the lever. A fixture that
        # set `lever2_fired` directly would test nothing about that computation -- which is exactly
        # how the broken stamp survived: every analyser test hand-injected the flag. The shape used
        # for a FIRING cell is the real one from r11l/tn36: the repaired detector resolves a mask
        # where the shipped classifier resolved none.
        "lever2_hud_fire": (
            {
                "hud_mask_resolved": True,
                "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
                "hud_mask_cell_count": 64,
                "hud_mask_digest": "fcbba0b6818499b6",
                "hud_shipped_mask_cell_count": 0,
                "hud_shipped_mask_digest": None,
            }
            if lever2
            else {
                "hud_mask_resolved": False,
                "hud_mask_source": "unresolved_no_bar_detected",
                "hud_mask_cell_count": 0,
                "hud_mask_digest": None,
                "hud_shipped_mask_cell_count": 0,
                "hud_shipped_mask_digest": None,
            }
        ),
        # The harness's own stamp, kept so the analyser can report disagreement. Set to what the
        # FIXED harness produces; the legacy-stamp disagreement has its own dedicated test.
        "lever2_fired": lever2,
        "lever3_verdict": lever3_verdict,
        "lever3_fired": lever3_verdict in ("FIRED_AND_PRUNED", "FIRED_NO_PRUNE"),
        "lever3_hazard_fire": {"rows_pruned": 0},
        "states_expanded": states,
        "actions_to_first_levelup": a2l,
        "wall_s": 100.0,
        "nodes_total": 10,
        "nodes_with_previous_frame": 10,
        "induction_attempts": 1,
        "induction_attempts_llm_reached": 1,
        "errors": 0,
    }


def _arms_for(game: str, seed: int, **kw) -> list[dict]:
    """One cell for all three task-1 arms, so the cell is MATCHED."""
    return [
        row("S_llmon", game, seed, kw.get("s", 1), lever2=kw.get("lever2_S", False)),
        row("S_minus_frontier_llmon", game, seed, kw.get("mf", 1)),
        row("S_minus_hud_llmon", game, seed, kw.get("mh", 1)),
    ]


def test_unmatched_cell_is_excluded_not_silently_compared() -> None:
    """A cell missing from one arm must not enter any delta.

    DEFECT REPRODUCED: comparing an arm's per-seed set against a control set built from a
    different cell population makes the control appear to fail against itself.
    """
    rows = _arms_for("aaaa", 1)
    rows.append(row("S_llmon", "bbbb", 1, 1))  # only the control has bbbb
    out = AN.analyse(rows)
    assert out["cells_matched_all_arms"] == 1
    assert out["cells_unmatched"] == [{"game": "bbbb", "seed": 1, "arms_present": ["S_llmon"]}]
    for v in out["lever_verdicts"].values():
        for per in v["per_seed"].values():
            assert "bbbb" not in per["control_win_set"]


def test_matching_is_pairwise_so_one_flaky_arm_does_not_delete_a_game_everywhere() -> None:
    """A cell missing for ONE arm must not remove that game from the OTHER levers' comparisons.

    DEFECT GUARDED: requiring all five arms to be present before a cell counted anywhere gave a
    single blipped cell five times the destructive reach -- and the cells that blip are the slow,
    interesting games. Here the hazard arm is missing on 'bbbb'; lever 1 must still be scored on
    both games, while the hazard comparison is scored on 'aaaa' alone and says so.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1) + _arms_for("bbbb", 1, s=1, mf=0, mh=1)
    rows.append(row("S_plus_hazard_llmon", "aaaa", 1, 1, lever3_verdict="FIRED_AND_PRUNED"))
    out = AN.analyse(rows)
    frontier = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]["1"]
    hazard = out["lever_verdicts"]["S_plus_hazard_llmon"]["per_seed"]["1"]
    assert frontier["n_games_measured"] == 2
    assert sorted(frontier["control_win_set"]) == ["aaaa", "bbbb"]
    assert hazard["n_games_measured"] == 1
    assert hazard["control_win_set"] == ["aaaa"]


def test_scoring_is_per_seed_never_the_any_seed_union() -> None:
    """A game won by the control on ONE seed must not be counted against it on another.

    DEFECT REPRODUCED: ANY-SEED UNION scoring. Under union scoring the control's seed-2 row for
    'aaaa' would read as a loss because 'aaaa' is in the union; per-seed matched scoring shows no
    loss on either seed.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1) + _arms_for("aaaa", 2, s=0, mf=0, mh=0)
    out = AN.analyse(rows)
    assert out["win_sets_per_seed"]["S_llmon"] == {"1": ["aaaa"], "2": []}
    per = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]
    assert per["1"]["lost_vs_control_movable"] == []
    # Seed 2: nobody wins, so the game cannot discriminate there at all.
    assert per["2"]["seed_verdict"] == "UNINTERPRETABLE_EMPTY_PASS_REGION"


def test_game_won_by_no_arm_is_forced_and_stamped_uninterpretable() -> None:
    """The forced-value defect: if no arm wins anything, the delta is arithmetically 0.

    DEFECT REPRODUCED: the C2_diag_roll condition, where the anchor games were won by no arm and
    a '0 difference' was first reported as a measurement.
    """
    rows = _arms_for("aaaa", 1, s=0, mf=0, mh=0)
    out = AN.analyse(rows)
    assert out["discriminating_games_per_seed"]["1"] == []
    assert out["nondiscriminating_games_per_seed"]["1"] == ["aaaa"]
    # The corpus summary must not disagree with the per-lever detail: a game won by SOME arm shows
    # up as discriminating even when it is not matched across every arm (the partial-run case).
    partial = AN.analyse(rows + [row("S_plus_hazard_llmon", "bbbb", 1, 1)])
    assert partial["discriminating_games_per_seed"]["1"] == ["bbbb"]
    for arm in ("S_minus_frontier_llmon", "S_minus_hud_llmon"):
        v = out["lever_verdicts"][arm]
        assert v["overall_verdict"] == "UNINTERPRETABLE_EMPTY_PASS_REGION"
        assert v["per_seed"]["1"]["witness_pass_region_nonempty"] is False


def test_witness_is_computed_at_the_gates_own_aggregation_level() -> None:
    """The witness must be a per-seed set, because the verdict is per-seed.

    DEFECT REPRODUCED: exp5835 computed the witness per-cell for a gate defined on an aggregate,
    so a structurally empty pass region was reported as passing.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1) + _arms_for("bbbb", 1, s=0, mf=0, mh=0)
    out = AN.analyse(rows)
    per = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]["1"]
    # bbbb is won by nobody -> not in the witness; aaaa is and the lever fires there.
    assert per["witness_movable_games"] == ["aaaa"]
    assert per["witness_pass_region_nonempty"] is True


def test_witness_is_not_vacuous_outcome_alone_cannot_certify_its_own_support() -> None:
    """A win difference must NOT be allowed to certify the support it is measured on.

    DEFECT REPRODUCED: a witness that reads the outcome cannot fail, so it is not a check. Here the
    HUD-off arm LOSES a win while every PROCESS observable (states expanded, action mix, modal
    action) is identical to the control and the lever's own counter never fired. The correct
    reading is 'this cell carries no evidence about the lever', not 'the lever caused the loss'.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=0, lever2_S=False)
    out = AN.analyse(rows)
    v = out["lever_verdicts"]["S_minus_hud_llmon"]
    per = v["per_seed"]["1"]
    assert per["witness_movable_games"] == []
    assert per["lost_on_nonfiring_game"] == ["aaaa"]
    assert per["lost_vs_control_movable"] == []
    assert v["overall_verdict"] == "UNINTERPRETABLE_EMPTY_PASS_REGION"


def test_process_divergence_certifies_support_even_when_the_counter_is_silent() -> None:
    """The under-detection guard: a lever can change the search without advancing a tier.

    DEFECT GUARDED: using `tier_advances > 0` as the only fire witness would exclude a cell where
    the frontier trio silently restricted the action vocabulary, diluting a real effect toward zero.
    Process divergence (states expanded / action mix) must certify the cell on its own.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1)
    for r in rows:
        r["lever1_fired"] = False  # the counter is silent in EVERY arm
        r["lever1_frontier_fire"] = {"tier_advances": 0}
        if r["arm"] == "S_minus_frontier_llmon":
            r["states_expanded"] = 250  # but the search demonstrably differs
    out = AN.analyse(rows)
    v = out["lever_verdicts"]["S_minus_frontier_llmon"]
    per = v["per_seed"]["1"]
    assert per["witness_movable_games"] == ["aaaa"]
    assert per["witness_by_behavioural_divergence_only"] == ["aaaa"]
    assert per["witness_by_direct_counter_only"] == []
    assert per["lost_vs_control_movable"] == ["aaaa"]
    assert v["overall_verdict"] == "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED"


def test_fully_inert_arm_is_reported_as_inert_on_every_game() -> None:
    """An arm byte-identical to the control everywhere proves the lever did nothing at all."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    out = AN.analyse(rows)
    per = out["lever_verdicts"]["S_minus_hud_llmon"]["per_seed"]["1"]
    assert per["arm_inert_vs_control_all_games"] == ["aaaa"]
    assert per["n_games_measured"] == 1


def test_effect_is_reported_when_the_lever_fired_on_the_moved_game() -> None:
    """The positive control for the machinery: a real, attributable effect must be detected."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=0, lever2_S=True)
    out = AN.analyse(rows)
    v = out["lever_verdicts"]["S_minus_hud_llmon"]
    assert v["per_seed"]["1"]["witness_movable_games"] == ["aaaa"]
    assert v["per_seed"]["1"]["lost_vs_control_movable"] == ["aaaa"]
    assert v["overall_verdict"] == "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED"


def test_llm_invalid_rows_are_excluded_and_counted() -> None:
    """An LLM-on row with a dead generator is not an LLM-on datum.

    DEFECT REPRODUCED: the server-storm / dead-server run that kept emitting rows LABELLED llm_on
    with no LLM in them.
    """
    rows = _arms_for("aaaa", 1)
    bad = row("S_llmon", "cccc", 1, 1, valid=False, responses=0)
    # An LLM-ON row whose generator died: zero completions AND unhealthy after the cell. (A row with
    # zero completions that never claimed the LLM is a perfectly valid LLM-off row, not this case.)
    bad.update(llm_enabled=True, generator_healthy_after=False)
    rows.append(bad)
    out = AN.analyse(rows)
    assert out["rows_total"] == 4
    assert out["rows_llm_valid"] == 3
    assert len(out["rows_excluded_llm_invalid"]) == 1
    assert out["rows_excluded_llm_invalid"][0]["game"] == "cccc"


def test_win_sets_are_sets_not_totals() -> None:
    """Equal win COUNTS over DISJOINT games must not read as 'no difference'.

    DEFECT REPRODUCED: comparing failure/win totals instead of sets.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1, lever2_S=True) + _arms_for(
        "bbbb", 1, s=0, mf=1, mh=0, lever2_S=True
    )
    out = AN.analyse(rows)
    assert out["win_counts_per_seed"]["S_llmon"]["1"] == 1
    assert out["win_counts_per_seed"]["S_minus_frontier_llmon"]["1"] == 1
    per = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]["1"]
    assert per["lost_vs_control_movable"] == ["aaaa"]
    assert per["gained_vs_control_movable"] == ["bbbb"]
    assert (
        out["lever_verdicts"]["S_minus_frontier_llmon"]["overall_verdict"]
        == "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED"
    )


def test_efficiency_is_paired_and_only_on_games_both_arms_win() -> None:
    """Efficiency deltas need both arms to have reached a level; otherwise there is no pair."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1) + _arms_for("bbbb", 1, s=1, mf=0, mh=1)
    for r in rows:
        if r["game"] == "aaaa" and r["arm"] == "S_minus_frontier_llmon":
            r["actions_to_first_levelup"] = 80
    out = AN.analyse(rows)
    pairs = out["lever_verdicts"]["S_minus_frontier_llmon"]["efficiency_paired_both_win"]
    assert [p["game"] for p in pairs] == ["aaaa"]
    assert pairs[0]["delta_arm_minus_control"] == 30


def test_fire_census_counts_lever3_verdict_classes_separately() -> None:
    """The four ways to get rows_pruned==0 must stay distinguishable in the census."""
    rows = _arms_for("aaaa", 1) + [
        row("S_plus_hazard_llmon", "aaaa", 1, 1, lever3_verdict="UNINTERPRETABLE_NOT_FITTED")
    ]
    out = AN.analyse(rows)
    census = out["fire_census_per_arm"]["S_plus_hazard_llmon"]
    assert census["lever3_verdicts"] == {"UNINTERPRETABLE_NOT_FITTED": 1}
    v = out["lever_verdicts"]["S_plus_hazard_llmon"]
    # The lever is ON in this arm and did not fire -> no evidence about it.
    assert v["overall_verdict"] == "UNINTERPRETABLE_EMPTY_PASS_REGION"


def _with_replicate(rows: list[dict], *, replicate_levels_by_game: dict[str, int]) -> list[dict]:
    """Add the same-config replicate arm so the noise floor is MEASURED rather than assumed."""
    out = list(rows)
    for r in rows:
        if r["arm"] != "S_llmon":
            continue
        rep = dict(r)
        rep["arm"] = "S_replicate_llmon"
        rep["levels"] = replicate_levels_by_game.get(r["game"], r["levels"])
        out.append(rep)
    return out


def test_effect_no_larger_than_the_same_config_noise_floor_is_not_an_effect() -> None:
    """With an LLM in the loop a seeded run is NOT deterministic, so a 1-game delta can be noise.

    DEFECT GUARDED: reporting a lever effect that the SAME configuration reproduces on its own. Here
    the treatment arm loses one game -- and so does a byte-identical replicate of the control. The
    delta therefore does not exceed the noise floor and must not be called an effect.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1, lever2_S=True)
    rows += _arms_for("bbbb", 1, s=1, mf=1, mh=1, lever2_S=True)
    # The replicate loses 'bbbb' with NO flag change -> noise floor = 1 win flip.
    rows = _with_replicate(rows, replicate_levels_by_game={"bbbb": 0})
    out = AN.analyse(rows)
    noise = out["noise_floor_same_config_replicate"]["1"]
    assert noise["win_flips_same_config"] == ["bbbb"]
    assert noise["n_win_flips_same_config"] == 1
    # Process observables are identical in this fixture, so determinism is recorded as True even
    # though the OUTCOME flipped -- which is exactly why the noise floor is measured on win flips
    # and not on process divergence alone.
    assert noise["run_is_deterministic_under_seed"] is True
    per = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]["1"]
    assert per["lost_vs_control_movable"] == ["aaaa"]
    assert per["n_games_moved_on_movable_support"] == 1
    assert per["same_config_noise_floor_win_flips"] == 1
    assert per["exceeds_same_config_noise_floor"] is False
    assert per["seed_verdict"] == "WIN_DIFFERENCE_WITHIN_SAME_CONFIG_NOISE_FLOOR"


def test_effect_larger_than_the_noise_floor_is_reported_as_an_effect() -> None:
    """The positive control for the noise gate: a delta bigger than the floor does count."""
    rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1, lever2_S=True)
    rows += _arms_for("bbbb", 1, s=1, mf=0, mh=1, lever2_S=True)
    rows = _with_replicate(rows, replicate_levels_by_game={})  # replicate matches control exactly
    out = AN.analyse(rows)
    assert out["noise_floor_same_config_replicate"]["1"]["n_win_flips_same_config"] == 0
    per = out["lever_verdicts"]["S_minus_frontier_llmon"]["per_seed"]["1"]
    assert per["lost_vs_control_movable"] == ["aaaa", "bbbb"]
    assert per["exceeds_same_config_noise_floor"] is True
    assert per["seed_verdict"] == "ATTRIBUTABLE_WIN_DIFFERENCE"


def test_llm_contribution_is_reported_as_a_comparison_not_as_a_lever() -> None:
    """Turning the LLM off is a REFERENCE, not one of the levers under test.

    It must appear as its own per-seed matched comparison (so 'the scored path wins k games with the
    LLM on' has something to be read against) and must NOT be folded into any lever verdict.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1) + _arms_for("bbbb", 1, s=1, mf=1, mh=1)
    rows.append(row("S_llmoff", "aaaa", 1, 0, responses=0))
    rows.append(row("S_llmoff", "bbbb", 1, 1, responses=0))
    out = AN.analyse(rows)
    cmp1 = out["llm_contribution_vs_llm_off"]["1"]
    assert cmp1["n_games_measured"] == 2
    assert cmp1["control_only_wins"] == ["aaaa"]  # the LLM won a game the LLM-off arm did not
    assert cmp1["other_only_wins"] == []
    assert "S_llmoff" not in out["lever_verdicts"]


def test_honest_verdict_is_terminal_prefixed_and_derived_from_the_verdicts() -> None:
    """The verdict string must carry a terminal prefix and must not be able to disagree.

    DEFECT GUARDED: a verdict lacking `complete_` gets substring-matched against partial tokens
    ('uninterpretable', 'no_effect') and a finished measurement is reclassified as a partial run.
    """
    rows = _arms_for("aaaa", 1, s=0, mf=0, mh=0)
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], 0.0)
    hv = art["honest_verdict"]
    assert hv.startswith("complete_")
    # Derived, not hand-written: the per-lever verdicts must appear inside it.
    assert "uninterpretable_empty_pass_region" in hv
    assert art["lever1_frontier_verdict"] == "UNINTERPRETABLE_EMPTY_PASS_REGION"


def test_duplicate_cells_are_detected_not_silently_overwritten() -> None:
    """Two row files containing the same cell must not silently pick a winner.

    DEFECT GUARDED: keying by (arm, game, seed) makes the LAST file loaded win, which is an
    invisible choice about which measurement counts. A divergent duplicate is a free independent
    replication and must be surfaced; an identical one confirms determinism.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    dup_same = dict(rows[0])
    dup_same["_source"] = "second_file.json"
    dup_diff = dict(rows[0])
    dup_diff["_source"] = "third_file.json"
    dup_diff["levels"] = 0
    dup_diff["states_expanded"] = 999
    out = AN.analyse(rows + [dup_same, dup_diff])
    assert len(out["duplicate_cells_identical"]) == 1
    assert len(out["duplicate_cells_divergent"]) == 1
    assert out["duplicate_cells_divergent"][0]["duplicate_states"] == 999
    # The FIRST occurrence is what is used, so the control still counts 'aaaa' as a win.
    assert out["win_sets_per_seed"]["S_llmon"]["1"] == ["aaaa"]


def test_storm_false_positive_is_rescued_but_a_real_dead_generator_is_not() -> None:
    """The validity witness must not discard a cell for activity belonging to other processes.

    DEFECT REPRODUCED (measured): the harness's stamp ANDs in a GLOBAL llama-server process count,
    which on this box also counts the conductor's own server and `[llama-server] <defunct>` zombies.
    Three cells whose generator was healthy on both sides and produced real completions were stamped
    invalid because that unrelated count rose from 2 to 5. Recomputing rescues them -- and must still
    reject a cell whose own generator really was dead.
    """
    good = row("S_llmon", "aaaa", 1, 1)
    good.update(
        llm_enabled=True,
        llm_on_row_valid=False,  # the harness's (wrong) stamp
        generator_healthy_before=True,
        generator_healthy_after=True,
        llama_servers_before=2,
        llama_servers_after=5,
        server_storm_suspected=True,
    )
    dead = row("S_minus_hud_llmon", "aaaa", 1, 1)
    dead.update(
        llm_enabled=True,
        llm_on_row_valid=False,
        llm={"responses": 0, "tokens_predicted": 0, "llm_wall_s": 0.0},
        generator_healthy_before=True,
        generator_healthy_after=False,
    )
    # A cell that STARTED while the server was restarting but then got real completions and ended
    # healthy IS a valid LLM-on datum. DEFECT REPRODUCED: requiring health on both sides deleted the
    # tn36 control -- 6 completions, 15514 predicted tokens -- which is the control for the only game
    # in the LLM-on design where a lever changes the outcome.
    started_mid_restart = row("S_llmon", "bbbb", 1, 0)
    started_mid_restart.update(
        llm_enabled=True,
        llm_on_row_valid=True,
        llm={"responses": 6, "tokens_predicted": 15514, "llm_wall_s": 120.0},
        generator_healthy_before=False,
        generator_healthy_after=True,
    )
    assert AN.llm_row_is_valid(started_mid_restart) is True

    assert AN.llm_row_is_valid(good) is True
    assert AN.llm_row_is_valid(dead) is False
    out = AN.analyse([good, dead])
    assert out["rows_llm_valid"] == 1
    rescued = out["rows_rescued_from_harness_storm_false_positive"]
    assert len(rescued) == 1 and rescued[0]["arm"] == "S_llmon"
    assert rescued[0]["llama_servers_after"] == 5
    assert [x["arm"] for x in out["rows_excluded_llm_invalid"]] == ["S_minus_hud_llmon"]


def test_companion_llm_off_analysis_does_not_become_the_headline() -> None:
    """The cheap LLM-OFF design must be embedded WITHOUT displacing the scored verdicts.

    DEFECT GUARDED: the companion is ~100x cheaper per cell, so it can be run at a power the scored
    design cannot afford -- which makes it the most dangerous number in the artifact. It must appear
    under its own key, and the headline lever verdicts must still be the LLM-ON ones.
    """
    on_rows = _arms_for("aaaa", 1, s=1, mf=0, mh=1, lever2_S=True)
    off_rows = [
        row(a.replace("_llmon", "_llmoff"), "aaaa", 1, 1) for a in ("S_llmon", "S_minus_hud_llmon")
    ]
    AN.set_condition("_llmoff")
    companion = AN.analyse(off_rows)
    AN.set_condition("_llmon")
    primary = AN.analyse(on_rows)
    art = AN.build_artifact(primary, on_rows, [Path("x.json")], 0.0, companion, off_rows)
    assert art["companion_llm_off_design"] is companion
    assert "S_minus_hud_llmoff" in companion["lever_verdicts"]
    # The headline verdicts come from the LLM-ON analysis, not the companion.
    assert (
        art["lever1_frontier_verdict"]
        == (primary["lever_verdicts"]["S_minus_frontier_llmon"]["overall_verdict"])
    )
    assert art["companion_rows"] == off_rows


def test_duration_s_is_measured_compute_not_analyser_runtime() -> None:
    """duration_s must be the compute the conclusions rest on, not how long the analyser took.

    DEFECT REPRODUCED (caught by running the real linter): the first draft set duration_s from the
    analyser's own clock, so the artifact declared a live-LLM substrate and a 0.01s duration --
    adversarial_verify raised CRITICAL DURATION_TOO_SHORT on a run that had used hours of GPU time.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    for r in rows:
        r["llm_enabled"] = True
        r["wall_s"] = 300.0
        r["llm"] = {"responses": 4, "tokens_predicted": 1000, "llm_wall_s": 270.0}
    # t0 is a real clock reading here; passing 0.0 would make analysis_duration_s the epoch.
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], time.time())
    assert art["duration_s"] == 900.0  # 3 cells x 300s of real measured compute
    assert art["measured_llm_wall_s"] == 810.0
    assert art["duration_s"] > 60.0, "must clear the live_llm_inference floor"
    assert art["analysis_duration_s"] < art["duration_s"]


def test_cost_is_split_by_condition_so_the_pooled_median_cannot_mislead() -> None:
    """An LLM-off cell costs seconds and an LLM-on cell costs minutes; pooling them is misleading.

    DEFECT REPRODUCED: the pooled median read 3.2s for a design whose LLM-on cells were ~300s each,
    and that pooled number is exactly what a next-run scoping decision would have used.
    """
    on = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    for r in on:
        r["llm_enabled"] = True
        r["wall_s"] = 300.0
    off = [row("S_llmoff", "aaaa", 1, 1)]
    off[0]["llm_enabled"] = False
    off[0]["wall_s"] = 3.0
    out = AN.analyse(on + off)
    assert out["cost"]["llm_on"]["wall_s_per_cell_median"] == 300.0
    assert out["cost"]["llm_off"]["wall_s_per_cell_median"] == 3.0
    assert out["cost"]["llm_on"]["n_cells"] == 3
    assert out["cost"]["llm_off"]["n_cells"] == 1
    assert out["cost"]["llm_on"]["projected_hours_25games_x_1seed_x_5arms_serial"] > 10


def test_acceptance_gates_are_computed_and_can_actually_fail() -> None:
    """The gates must be falsifiable, and must gate the MEASUREMENT'S validity, not the result.

    A gate that can only pass is not a check. Here: the noise-floor gate fails when no replicate arm
    was run, and the live-generator gate fails when an LLM-on row recorded zero completions.
    """
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    for r in rows:
        r["llm_enabled"] = True
        r["wall_s"] = 300.0
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], time.time())
    assert art["acceptance_gate_all_llm_on_rows_had_a_live_generator"] is True
    assert art["acceptance_gate_noise_floor_was_measured"] is False  # no replicate arm present
    assert art["acceptance_gate_no_effect_reported_without_a_witness"] is True

    # A dead generator on an LLM-on row must fail the liveness gate.
    dead = [dict(r) for r in rows]
    dead[0]["llm"] = {"responses": 0, "tokens_predicted": 0, "llm_wall_s": 0.0}
    art2 = AN.build_artifact(AN.analyse(dead), dead, [Path("x.json")], time.time())
    assert art2["acceptance_gate_all_llm_on_rows_had_a_live_generator"] is False

    # With a replicate arm the noise-floor gate passes.
    with_rep = rows + [dict(rows[0], arm="S_replicate_llmon")]
    art3 = AN.build_artifact(AN.analyse(with_rep), with_rep, [Path("x.json")], time.time())
    assert art3["acceptance_gate_noise_floor_was_measured"] is True


def test_artifact_declares_every_mandatory_field() -> None:
    """The artifact must carry the fields the fabrication linter and the disciplines require."""
    rows = _arms_for("aaaa", 1)
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], 0.0)
    for k in ("honest_verdict_placeholder_absent",):
        assert k not in art
    for k in (
        "inference_substrate",
        "verifier_is_oracle",
        "solve_provenance",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
        "preconditions_checked",
        "model_specs",
    ):
        assert k in art, k
    assert art["inference_substrate"] == "live_llm_inference"
    assert art["verifier_is_oracle"] is False
    assert art["solve_provenance"] == "development_proxy"
    assert art["offline_reproduced"] is False
    assert art["flags_flipped"] == []
    assert len(art["reproducibility_checksum"]) == 64


# ---------------------------------------------------------------------------------------------
# 2026-07-26 ADVERSARIAL-REVIEW REPAIRS. Every test below reproduces a defect the review found in
# the FIRST published version of this measurement, against the REAL recorded values.
# ---------------------------------------------------------------------------------------------

# HUD diagnostics COPIED VERBATIM from recorded rows of
# results/outer_loop_scored_path_lever_ab_llm_on_20260726.json -- only the six fields the fire
# predicate reads are kept, values unchanged. These are real detector output, not hand-authored
# shapes, and crucially NONE of them is a `lever2_fired` flag: the point is to exercise the
# computation that produces that flag, which no prior test did.
REAL_HUD_DIAGNOSTICS = {
    # r11l, shipped arm: the repaired detector resolves a 64-cell mask where the SHIPPED classifier
    # resolves none. states_expanded on this cell is 41; the S_minus_hud cell is 319.
    "r11l_shipped": {
        "hud_mask_resolved": True,
        "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
        "hud_mask_cell_count": 64,
        "hud_mask_digest": "fcbba0b6818499b6",
        "hud_shipped_mask_cell_count": 0,
        "hud_shipped_mask_digest": None,
    },
    # tn36, shipped arm: 61 cells, same shape. states_expanded 17 vs 49 with the lever off.
    "tn36_shipped": {
        "hud_mask_resolved": True,
        "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
        "hud_mask_cell_count": 61,
        "hud_mask_digest": "791a436c692cdbf8",
        "hud_shipped_mask_cell_count": 0,
        "hud_shipped_mask_digest": None,
    },
    # r11l with the HUD trio OFF: nothing resolves, so the lever cannot have fired.
    "r11l_hud_off": {
        "hud_mask_resolved": False,
        "hud_mask_source": "unresolved_no_bar_detected",
        "hud_mask_cell_count": 0,
        "hud_mask_digest": None,
        "hud_shipped_mask_cell_count": 0,
        "hud_shipped_mask_digest": None,
    },
    # lf52: the shipped classifier ALREADY resolves this mask and the repair adds nothing, so the
    # repaired detector's digest EQUALS the shipped digest -- a genuine non-firing cell. This is the
    # case that must stay False, and it is why the fix is "differs from shipped", not "resolved".
    "lf52_shipped": {
        "hud_mask_resolved": True,
        "hud_mask_source": "status_bar_classifier_req5583_no_repair_added_cell",
        "hud_mask_cell_count": 64,
        "hud_mask_digest": "e92122951bcd64e7",
        "hud_shipped_mask_cell_count": 64,
        "hud_shipped_mask_digest": "e92122951bcd64e7",
    },
    # cn04: neither resolves. Also non-firing.
    "cn04_shipped": {
        "hud_mask_resolved": False,
        "hud_mask_source": "unresolved_stage2_refused_and_no_shipped_bar",
        "hud_mask_cell_count": 0,
        "hud_mask_digest": None,
        "hud_shipped_mask_cell_count": 0,
        "hud_shipped_mask_digest": None,
    },
}


def test_lever2_fires_when_a_mask_appears_where_the_shipped_config_had_none() -> None:
    """THE BROKEN FIRE COUNTER. Driven by real recorded detector output, never a hand-set flag.

    DEFECT REPRODUCED (measured 2026-07-26): the harness's first predicate ANDed in
    `hud_shipped_mask_digest`, requiring the ALREADY-SHIPPED classifier to have produced a mask
    before the REPAIRED detector's mask could count as a difference. The shipped classifier returns
    None on exactly the two games (r11l, tn36) where the repaired detector resolves one, so the
    counter was anti-correlated with its own lever and read 0 in ALL 430 cells while the lever was
    demonstrably firing -- resolved mask None -> 64 cells on r11l with states_expanded 319 -> 41.
    A fire counter that is broken in the direction that HIDES an effect is worse than one that
    invents an effect, because the result looks like a clean null.

    Every prior analyser test set `lever2_fired` by hand, so nothing exercised this computation --
    the same hand-injection anti-pattern that let an exp5836 wiring test pass against a dead
    observe channel.
    """
    fired = AN.recomputed_lever2_fired
    assert fired({"lever2_hud_fire": REAL_HUD_DIAGNOSTICS["r11l_shipped"]}) is True
    assert fired({"lever2_hud_fire": REAL_HUD_DIAGNOSTICS["tn36_shipped"]}) is True
    # NEGATIVE CONTROLS -- the fix must not degrade into "resolved == fired".
    assert fired({"lever2_hud_fire": REAL_HUD_DIAGNOSTICS["lf52_shipped"]}) is False
    assert fired({"lever2_hud_fire": REAL_HUD_DIAGNOSTICS["cn04_shipped"]}) is False
    assert fired({"lever2_hud_fire": REAL_HUD_DIAGNOSTICS["r11l_hud_off"]}) is False
    # An unreadable diagnostics block proves nothing either way.
    assert fired({"lever2_hud_fire": {"error": "AttributeError:boom"}}) is False
    assert fired({}) is False


def test_harness_and_analyser_lever2_predicates_agree() -> None:
    """DRIFT GUARD. The predicate exists in two places -- the harness stamps it live, the analyser
    recomputes it from recorded rows -- so they must agree on every real case or a re-run would
    silently disagree with a re-analysis of the same cells."""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "arc_scored_path_lever_harness",
        _P(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py",
    )
    assert spec and spec.loader
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    for name, diag in REAL_HUD_DIAGNOSTICS.items():
        assert harness.hud_lever_fired(diag) == AN.recomputed_lever2_fired(
            {"lever2_hud_fire": diag}
        ), name
    assert harness.hud_lever_fired(REAL_HUD_DIAGNOSTICS["r11l_shipped"]) is True


def test_the_analyser_reports_where_it_disagrees_with_the_harness_stamp() -> None:
    """A silent correction is its own defect. A row recorded BEFORE the fire-counter repair carries
    `lever2_fired=False` on a firing cell; the census must name the game rather than quietly using
    the corrected value."""
    rows = _arms_for("r11l", 1, s=1, mf=1, mh=1, lever2_S=True)
    for r in rows:
        if r["arm"] == "S_llmon":
            r["lever2_fired"] = False  # the pre-repair harness stamp
    out = AN.analyse(rows)
    census = out["fire_census_per_arm"]["S_llmon"]
    assert census["lever2_fired_cells"] == 1
    assert census["lever2_fired_cells_per_harness_stamp"] == 0
    assert census["lever2_games_mask_differs"] == ["r11l"]
    assert census["lever2_games_where_recomputed_disagrees_with_harness_stamp"] == ["r11l"]


def test_a_firing_lever2_cell_is_no_longer_frozen_out_of_the_witness() -> None:
    """THE CONSEQUENCE of the broken counter, at the level that changes a verdict. With the counter
    reading 0, a game the HUD lever moves is excluded from lever 2's movable support and the lever
    is stamped UNINTERPRETABLE even when the win moved. With the counter fixed, the same cell enters
    the support."""
    # r11l: the control (HUD on) wins it, the HUD-off arm does not.
    rows = _arms_for("r11l", 1, s=1, mf=1, mh=0, lever2_S=True)
    out = AN.analyse(rows)
    per = out["lever_verdicts"]["S_minus_hud_llmon"]["per_seed"]["1"]
    assert per["witness_movable_games"] == ["r11l"]
    assert per["witness_pass_region_nonempty"] is True
    assert per["lost_vs_control_movable"] == ["r11l"]
    assert per["seed_verdict"] != "UNINTERPRETABLE_EMPTY_PASS_REGION"


def test_acceptance_gate_violations_enumerates_every_failing_gate() -> None:
    """DEFECT REPRODUCED: `acceptance_gate_violations` was assigned ONE gate's list, so an artifact
    could carry `acceptance_gate_all_llm_on_rows_had_a_live_generator=False` and
    `acceptance_gate_violations=[]` at the same time -- and any consumer reading the purpose-named
    machine-readable field saw a clean run. That is exactly what the first published artifact did."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    for r in rows:
        r["llm_enabled"] = True
        r["wall_s"] = 300.0
    rows[0]["llm"] = {"responses": 0, "tokens_predicted": 0, "llm_wall_s": 0.0}  # dead generator
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], time.time())
    assert art["acceptance_gate_all_llm_on_rows_had_a_live_generator"] is False
    assert (
        "acceptance_gate_all_llm_on_rows_had_a_live_generator" in art["acceptance_gate_violations"]
    )
    # And the noise-floor gate also fails here (no replicate arm), so BOTH must be listed -- a
    # single-gate copy could never produce two entries.
    assert art["acceptance_gate_noise_floor_was_measured"] is False
    assert "acceptance_gate_noise_floor_was_measured" in art["acceptance_gate_violations"]
    assert len(art["acceptance_gate_violations"]) >= 2
    # Every listed violation must be a real False gate, and every False gate must be listed.
    for name in art["acceptance_gate_violations"]:
        assert art[name] is False, name
    for k, v in art.items():
        if (
            k.startswith("acceptance_gate_")
            and not k.startswith("acceptance_gate_violations")
            and not k.endswith(("_principle", "_note", "_detail"))
            and v is False
        ):
            assert k in art["acceptance_gate_violations"], k


def test_violations_is_empty_only_when_every_gate_passes() -> None:
    """The positive control for the scan: a clean run must report an EMPTY list, or the field would
    be useless in the other direction."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    rows += [dict(r, arm="S_replicate_llmon") for r in rows if r["arm"] == "S_llmon"]
    for r in rows:
        r["llm_enabled"] = True
        r["wall_s"] = 300.0
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], time.time())
    assert art["acceptance_gate_violations"] == []


def test_game_unit_sign_test_is_exact_and_states_what_the_design_could_reach() -> None:
    """THE MISSING STATISTIC. The first artifact had zero occurrences of sign_test / p_value /
    binomial / significan, while this project's own top known-issues entry -- one day old -- names
    the exact one-sided sign test ON THE GAME UNIT as the standard and had just used it to WITHDRAW
    a sibling HUD claim.

    The values asserted here are the ones that actually decide the frontier question:
      * 2 movers for, 1 against (the measured budget-400 LLM-off result: lp85+tn36 vs cd82) -> 0.5
      * 2 for, 0 against (the best single seed) -> 0.25
      * 1 for, 0 against (the LLM-on support: tn36 alone) -> 0.5, and 0.5 is also the FLOOR, so no
        outcome on a one-game support can ever clear 0.05.
      * 5 for, 0 against -> 0.031, which is what the convention-transfer battery's frontier main
        effect reached and therefore what "clears the bar" looks like here.
    """
    t = AN.exact_one_sided_sign_test
    assert t(2, 1)["p_one_sided_exact"] == 0.5
    assert t(2, 1)["smallest_reachable_p_at_this_n"] == 0.125
    assert t(2, 1)["underpowered_support"] is True  # 3 movers can never reach 0.05
    assert t(2, 0)["p_one_sided_exact"] == 0.25
    assert t(1, 0)["p_one_sided_exact"] == 0.5
    assert t(1, 0)["smallest_reachable_p_at_this_n"] == 0.5
    assert t(5, 0)["p_one_sided_exact"] == 0.0312
    assert t(5, 0)["clears_p_0_05"] is True
    assert t(5, 0)["underpowered_support"] is False
    # No discordant game at all: undefined, never reported as a passing or failing p.
    none = t(0, 0)
    assert none["p_one_sided_exact"] is None
    assert none["undefined_because_no_discordant_game"] is True
    assert none["clears_p_0_05"] is False


def test_every_lever_verdict_carries_its_game_unit_test_and_its_scope() -> None:
    """A verdict string with no support size and no p-value invites a 1-game / 1-seed spot check to
    be read as a corpus result -- which is how a 3-game single-seed comparison ended up tabled as a
    co-equal column headed with the scored condition's name."""
    rows = _arms_for("aaaa", 1, s=0, mf=1, mh=1)
    rows += [dict(r, arm="S_replicate_llmon") for r in rows if r["arm"] == "S_llmon"]
    out = AN.analyse(rows)
    v = out["lever_verdicts"]["S_minus_frontier_llmon"]
    assert v["overall_verdict"] == "ATTRIBUTABLE_WIN_DIFFERENCE"
    st = v["game_unit_sign_test"]
    assert st["unit"] == "game"
    assert st["movers_favouring_arm"] == ["aaaa"]
    assert st["n_discordant_games"] == 1
    assert st["p_one_sided_exact"] == 0.5
    assert st["smallest_reachable_p_at_this_n"] == 0.5
    # The generalisation verdict must say the design could not have cleared the bar, NOT merely
    # that it failed to.
    assert v["generalisation_verdict"] == "UNDERPOWERED_BY_DESIGN_NO_OUTCOME_COULD_CLEAR_P05"
    assert v["scope"]["n_seeds_compared"] == 1
    assert v["scope"]["n_games_matched_max_across_seeds"] == 1
    assert v["scope"]["n_movable_games_union_over_seeds"] == 1
    art = AN.build_artifact(out, rows, [Path("x.json")], 0.0)
    assert art["acceptance_gate_every_lever_verdict_carries_a_game_unit_sign_test"] is True


def test_a_game_unit_win_clears_the_bar_when_the_support_is_wide_enough() -> None:
    """The positive control for the sign test: six movers all in one direction DOES clear 0.05, so
    the test is not a rubber stamp that always says underpowered."""
    rows: list[dict] = []
    for g in ("g1", "g2", "g3", "g4", "g5", "g6"):
        rows += _arms_for(g, 1, s=0, mf=1, mh=1)
    rows += [dict(r, arm="S_replicate_llmon") for r in rows if r["arm"] == "S_llmon"]
    out = AN.analyse(rows)
    st = out["lever_verdicts"]["S_minus_frontier_llmon"]["game_unit_sign_test"]
    assert st["n_games_favouring"] == 6
    assert st["p_one_sided_exact"] == 0.0156
    assert st["clears_p_0_05"] is True
    assert (
        out["lever_verdicts"]["S_minus_frontier_llmon"]["generalisation_verdict"]
        == "DIRECTION_ESTABLISHED_ON_GAME_UNIT"
    )


def test_cross_seed_noise_floor_is_measured_and_is_not_the_same_seed_replicate() -> None:
    """DEFECT REPRODUCED: the only noise floor was a SAME-SEED replicate, which is structurally
    zero -- so `exceeds_same_config_noise_floor` reduced to "did anything change at all" while the
    verdict was named EFFECT_ON_WINS. The variance a win-set claim generalises over is the
    ACROSS-SEED movement of the control arm, which is measurably non-zero (the control's own win set
    moved 3 -> 4 -> 4 with different membership in the real budget-400 design, and the
    convention-transfer battery records the shipped arm as measured_deterministic=false with 54 of
    75 cells varying across seeds)."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1) + _arms_for("bbbb", 1, s=0, mf=0, mh=0)
    # Seed 2: the CONTROL flips 'bbbb' to a win with no flag change whatsoever.
    rows += _arms_for("aaaa", 2, s=1, mf=1, mh=1) + _arms_for("bbbb", 2, s=1, mf=1, mh=1)
    out = AN.analyse(rows)
    cs = out["noise_floor_control_across_seeds"]
    assert cs["arm"] == "S_llmon"
    assert cs["n_games_measured_on_every_seed"] == 2
    assert cs["max_win_flips_across_any_seed_pair"] == 1
    assert cs["games_won_on_every_seed"] == ["aaaa"]
    assert cs["n_games_unstable_across_seeds"] == 1
    assert cs["control_is_stable_across_seeds"] is False
    # The same-seed floor stays 0 -- it measures something different and must not be conflated.
    assert "noise_floor_same_config_replicate_note" in out
    assert "noise_floor_control_across_seeds_note" in out


def test_llm_plan_channel_census_counts_the_rows_where_the_plan_channel_actually_opened() -> None:
    """DEFECT REPRODUCED: the first write-up said the induced world model is rejected "every single
    time" with planned=0. One row of thirty had `induction_planned=1`, and that row is the ONLY
    positive control the inertness null has -- without it, "the gate correctly rejects a weak model"
    and "the plan path never influences behaviour" are indistinguishable. It is computed here so it
    cannot be restated from memory, together with whether its matched control is a valid pairing."""
    rows = _arms_for("aaaa", 1, s=1, mf=1, mh=1)
    for r in rows:
        r["llm_enabled"] = True
        r["induction_reasons"] = {"stall": 1}
    # One arm's cell DID open the plan channel, on a re-induction after a level-up.
    rows[1]["induction_planned"] = 1
    rows[1]["induction_reasons"] = {"level_up_reinduction": 1}
    out = AN.analyse(rows)
    census = out["llm_plan_channel_census"]
    assert census["n_llm_on_rows_that_ran"] == 3
    assert census["induction_planned_distribution"] == {0: 2, 1: 1}
    assert census["n_rows_where_plan_channel_opened"] == 1
    assert census["induction_reason_counts"] == {"stall": 2, "level_up_reinduction": 1}
    opened = census["rows_where_plan_channel_opened"][0]
    assert opened["arm"] == "S_minus_frontier_llmon"
    assert opened["induction_reasons"] == {"level_up_reinduction": 1}
    assert opened["matched_control_row_is_valid"] is True


def test_the_budget_note_does_not_claim_400_is_an_eval_imposed_bound() -> None:
    """DEFECT REPRODUCED (fatal): the artifact asserted "400 is the scored agent's own MAX_ACTIONS
    cap, so it is the eval's condition". The comment above that constant says the opposite -- the
    real bound is the eval's <=12h wall clock and MAX_ACTIONS is an INTENDED OVERRIDE POINT. The
    misreading inverted the headline recommendation, because lever orderings reverse with the
    budget: at 2000 the shipped configuration is the best of four arms (median 12 wins), at 400 it
    wins 3-4 of 25."""
    rows = _arms_for("aaaa", 1)
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], 0.0)
    note = art["budget_note"]
    assert "self-imposed" in note
    assert "12h" in note
    assert "OVERRIDE POINT" in note
    assert "so it is the eval's condition" not in note
    # And the artifact must point at the budget-2000 measurement it has to be reconciled against,
    # so a reader cannot act on one budget in ignorance of the other.
    prior = art["prior_measurements_that_must_be_reconciled_against"]
    assert any("cptb_shipped_lever_convention_transfer" in p["artifact"] for p in prior)
    assert any("tn36" in p["what_it_already_recorded_that_this_run_replicates"] for p in prior)
    assert any("p=0.031" in p["what_it_measures_that_this_one_does_not"] for p in prior)


def test_no_flag_recommendation_can_be_made_on_an_underpowered_support() -> None:
    """DEFECT REPRODUCED (fatal): the first write-up recommended UN-FLIPPING the shipped frontier
    trio on a 2-game support at p=0.5 -- one day after this project used the same test to withdraw a
    sibling HUD claim at p=0.5 as "arithmetically forced rather than measured". The disposition is
    now DERIVED, so a recommendation cannot be written at a bar the evidence does not meet."""
    rows = _arms_for("aaaa", 1, s=0, mf=1, mh=1)
    rows += [dict(r, arm="S_replicate_llmon") for r in rows if r["arm"] == "S_llmon"]
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], 0.0)
    rec = art["flag_change_recommendation_per_lever"]["S_minus_frontier_llmon"]
    assert rec["recommendation"] == "NO_RECOMMENDATION_UNDERPOWERED_ON_GAME_UNIT"
    assert "p=0.5" in rec["reason"]
    assert art["acceptance_gate_no_flag_recommendation_without_game_unit_significance"] is True
    assert art["single_budget_measurement"] is True


def test_a_single_budget_blocks_a_recommendation_even_when_the_sign_test_clears() -> None:
    """Lever orderings REVERSE between budget 400 and budget 2000, so even a p<=0.05 direction
    measured at one budget only is not a basis for changing a shipped flag. Six movers in one
    direction clears the sign test; the single-budget condition must still block the advice."""
    rows: list[dict] = []
    for g in ("g1", "g2", "g3", "g4", "g5", "g6"):
        rows += _arms_for(g, 1, s=0, mf=1, mh=1)
    rows += [dict(r, arm="S_replicate_llmon") for r in rows if r["arm"] == "S_llmon"]
    art = AN.build_artifact(AN.analyse(rows), rows, [Path("x.json")], 0.0)
    rec = art["flag_change_recommendation_per_lever"]["S_minus_frontier_llmon"]
    assert rec["p_one_sided_exact_game_unit"] == 0.0156
    assert rec["recommendation"] == "NO_RECOMMENDATION_SINGLE_BUDGET"
    # With BOTH budgets present the disposition becomes a review recommendation -- never a flip.
    two_budgets = [dict(r) for r in rows] + [dict(r, budget=2000) for r in rows]
    art2 = AN.build_artifact(AN.analyse(two_budgets), two_budgets, [Path("x.json")], 0.0)
    rec2 = art2["flag_change_recommendation_per_lever"]["S_minus_frontier_llmon"]
    assert rec2["recommendation"] == "EVIDENCE_SUPPORTS_A_FLAG_REVIEW"
    assert "operator" in rec2["reason"].lower()
    assert art2["single_budget_measurement"] is False
    assert art2["flags_flipped"] == []
