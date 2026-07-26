"""Tests for the scored-path lever A/B analyser.

Each test here reproduces a SPECIFIC measurement defect this project has already shipped once,
rather than a synthetic happy path. If one of these regresses, a wrong number gets published.

REQ-ARC-WMTE-5980 / SCENARIO-scored-path-lever-ab-analysis.
"""

from __future__ import annotations

import importlib.util
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
        "lever1_fired": lever1,
        "lever1_frontier_fire": {"tier_advances": 3 if lever1 else 0},
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
    assert v["overall_verdict"] == "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED"


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
    assert v["overall_verdict"] == "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED"


def test_llm_invalid_rows_are_excluded_and_counted() -> None:
    """An LLM-on row with a dead generator is not an LLM-on datum.

    DEFECT REPRODUCED: the server-storm / dead-server run that kept emitting rows LABELLED llm_on
    with no LLM in them.
    """
    rows = _arms_for("aaaa", 1)
    rows.append(row("S_llmon", "cccc", 1, 1, valid=False, responses=0))
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
        == "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED"
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
    assert per["seed_verdict"] == "EFFECT_WITHIN_SAME_CONFIG_NOISE_FLOOR"


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
    assert per["seed_verdict"] == "EFFECT_ON_WINS"


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
