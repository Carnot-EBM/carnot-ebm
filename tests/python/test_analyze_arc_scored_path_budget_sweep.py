"""REGRESSION TESTS for the budget-sweep analyser's corrected measurement machinery.

Every test here reproduces a SPECIFIC defect that an adversarial review found in the published
2026-07-26 budget-sweep artifact. They are written against the counterexample that motivated each
correction, not against a synthetic happy path, per CLAUDE.md's QA-Layer Authenticity Discipline
("write the regression test for the exact incident/counterexample that motivated the check").

THE FOUR DEFECTS UNDER TEST:

  1. THE INVERTED SCORING CONCLUSION (fatal). The analyser asserted the gateway's action-charging
     rule was "NOT resolvable locally" and reported two models that "DISAGREE IN SIGN", one of which
     said raising the budget cut the score by 5.6x. `arc_agi.scorecard` is INSTALLED; the tail is
     score-irrelevant; and the losing model rested on a formula a 2026-06-20 review had already
     retracted. An operator following the old report would have LOWERED the budget.
  2. THE OMITTED max_score CLAMP. The per-game score is clamped by the index-weighted fraction of
     levels SOLVED, so the number of levels solved sets a CEILING on what per-level speed can earn.
     It was absent from the analysis entirely. NOTE: writing the test for it corrected the obvious
     reading -- because the clamp is applied as `min()`, depth is the dominant lever only at EQUAL
     per-level speed; depth bought by grinding can score BELOW a fast shallow solve, which is the
     mechanism behind this sweep's 3.3x win count producing only ~1.02x score.
  3. PSEUDO-REPLICATION. Every sign test used the CELL (game, seed) as its unit, but 3 seeds of one
     game are replicates. The inferential target is a hidden game, so the GAME is the unit.
  4. THE RETRACTED FORMULA MUST NOT COME BACK as a live reading of the metric.

Spec: REQ-ARC-WMTE-5981 / SCENARIO-ARC-WMTE-5981-TAIL-IS-SCORE-IRRELEVANT /
SCENARIO-ARC-WMTE-5981-CLAMP-IS-A-CEILING-NOT-A-FLOOR /
SCENARIO-ARC-WMTE-5981-GAME-UNIT-TEST-IS-THE-HEADLINE /
SCENARIO-ARC-WMTE-5981-MEMORY-ENVELOPE-IS-MEASURED /
SCENARIO-ARC-WMTE-5981-VERDICT-CANNOT-CONTRADICT-THE-MEASUREMENT.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "analyze_arc_scored_path_budget_sweep",
    REPO / "scripts" / "analyze_arc_scored_path_budget_sweep.py",
)
assert _SPEC and _SPEC.loader
MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MOD)


# =================================================================================================
# DEFECT 1 -- the charging rule IS resolvable locally, and the tail is free.
# =================================================================================================
def test_charging_rule_is_resolvable_locally_against_the_installed_scorer():
    """The old artifact said this was not resolvable locally. The scorer ships in the venv."""
    res = MOD._resolve_charging_rule()
    assert res["resolvable_locally"] is True
    assert res["scorer_module_path"].endswith("arc_agi/scorecard.py")


def test_post_solve_tail_is_score_irrelevant():
    """THE COUNTEREXAMPLE that inverted the recommendation.

    One solved level, then a tail of N unproductive actions. The retired 'total-action charge' model
    predicted the score falls as 1/N^2. The installed scorer charges those actions to the trailing
    INCOMPLETE level, which scores 0.0 regardless -- so the score is IDENTICAL from a 15-action tail
    to a 100,000-action one. If this test ever fails, the charge model changed upstream and the
    budget recommendation must be re-derived.
    """
    res = MOD._resolve_charging_rule()
    scores = set(res["tail_probe_same_solve_varying_tail"].values())
    assert len(scores) == 1, f"tail became score-relevant: {scores}"
    assert res["tail_is_score_relevant"] is False
    # And the specific value, so a silent upstream rescaling is caught rather than absorbed.
    assert scores == {2.7778}


def test_scorer_is_driven_not_reimplemented():
    """A superhuman solve must score ABOVE the retracted formula's ceiling of 1.0, which is the
    cheapest proof that the real (baseline/actions)^2*100-with-115-cap formula is in play."""
    base8 = [20, 30, 40, 50, 60, 70, 80, 90]
    # L1 solved in 15 actions against a baseline of 20 -> superhuman.
    assert MOD._drive_scorer(base8, [15], 15) > 1.0
    # Solving ALL levels superhumanly reaches the top of the scale.
    assert MOD._drive_scorer(base8, [10 * (i + 1) for i in range(8)], 100) == 100.0


# =================================================================================================
# DEFECT 2 -- the max_score clamp, which the old analysis omitted.
# =================================================================================================
def test_max_score_clamp_is_the_index_weighted_fraction_of_levels_solved():
    """The clamp is what caps a perfect-but-shallow agent: superhuman on L1 of 8 -> 2.78/100.

    Asserted by checking that, when the agent is superhuman on every level it solves, the reported
    score EQUALS the clamp -- i.e. the ceiling is what binds, not the per-level term. That equality
    is the direct evidence the clamp is being applied at all, which the previous analysis omitted.
    (Whether depth beats speed is a separate, conditional question -- see the next test.)
    """
    table = MOD._max_score_clamp_table()
    probe = table["probe_on_an_8_level_game"]
    for entry in probe.values():
        # Superhuman on every solved level means the clamp -- not the per-level term -- binds.
        assert math.isclose(
            entry["score_with_superhuman_speed_on_every_solved_level"],
            entry["clamp_index_weighted_fraction_times_100"],
            rel_tol=1e-6,
        )
    assert math.isclose(probe["1"]["clamp_index_weighted_fraction_times_100"], 2.7778, abs_tol=1e-3)
    assert math.isclose(
        probe["4"]["clamp_index_weighted_fraction_times_100"], 27.7778, abs_tol=1e-3
    )
    assert probe["8"]["clamp_index_weighted_fraction_times_100"] == 100.0


def test_the_clamp_is_a_CEILING_so_depth_does_not_unconditionally_beat_speed():
    """A CORRECTION FOUND BY WRITING THIS TEST, and the reason the +2% result makes sense.

    The intuitive reading of the clamp -- 'solving more levels is the dominant lever' -- is only true
    at equal per-level speed, because `to_score` applies the clamp as a `min()` ceiling. A deep but
    SLOW solve scores below a shallow but FAST one. Since grinding is exactly what a raised action
    budget buys, this is the mechanism behind the measured 3.3x win count producing only ~1.02x
    score. Asserting BOTH halves so neither over-claim can be reintroduced.
    """
    base8 = [20, 30, 40, 50, 60, 70, 80, 90]
    fast_shallow = MOD._drive_scorer(base8, [15], 4000)
    slow_deep = MOD._drive_scorer(base8, [400, 900, 1500, 2200], 4000)
    assert slow_deep < fast_shallow, "the clamp is a ceiling, not a floor"

    # ...but at EQUAL per-level speed, depth multiplies the score.
    def at_human(n: int) -> float:
        lua, cum = [], 0
        for i in range(n):
            cum += base8[i]
            lua.append(cum)
        return MOD._drive_scorer(base8, lua, 4000)

    assert at_human(1) < at_human(2) < at_human(4) < at_human(8)
    assert math.isclose(at_human(8), 100.0, abs_tol=1e-6)


# =================================================================================================
# DEFECT 3 -- pseudo-replication. The game is the unit; cells are within-game replicates.
# =================================================================================================
def test_sign_test_reports_both_tails_and_the_direction():
    """A one-sided test is how a REVERSAL comes back p=0.89 and reads as 'no effect'."""
    st = MOD.sign_test_two_sided(2, 7)
    assert st["direction_favoured"] == "lower_budget"
    assert st["p_one_sided_pos"] != st["p_one_sided_neg"]
    assert st["p_two_sided"] == round(
        min(1.0, 2 * min(st["p_one_sided_pos"], st["p_one_sided_neg"])), 6
    )


def test_zero_discordant_pairs_is_not_a_null_result():
    st = MOD.sign_test_two_sided(0, 0)
    assert st["interpretable"] is False
    assert "NOT a null result" in st["reason"]


def test_clustering_inflates_the_cell_level_p_relative_to_the_game_level_p():
    """THE COUNTEREXAMPLE, with the published design's own shape.

    The 400->1000 step gained 15 CELLS spread over only 7 distinct GAMES. Testing cells treats the
    3 seeds of a game as 3 independent draws, so the p collapses to 6.1e-05; testing games gives
    0.0156. Both are 'significant' at that step, but at 1000->2000 (7 cells over 3 games) the
    cell-level test passes at 0.0156 while the game-level test does NOT (0.25). Asserting the
    ordering here is what stops the cell-level value being quoted as the design's significance.
    """
    cells = MOD.sign_test_two_sided(15, 0)
    games = MOD.sign_test_two_sided(7, 0)
    assert cells["p_two_sided"] < games["p_two_sided"]
    assert math.isclose(games["p_two_sided"], 0.015625, abs_tol=1e-9)
    # The step that flips interpretation: 7 cells / 3 games.
    cells_1k2k = MOD.sign_test_two_sided(7, 0)
    games_1k2k = MOD.sign_test_two_sided(3, 0)
    assert cells_1k2k["p_two_sided"] < 0.05
    assert games_1k2k["p_two_sided"] > 0.05, "the game-unit test must NOT clear p<.05 on 3 games"


def test_smallest_reachable_p_is_reported_so_underpower_is_visible():
    """'Not significant' on a 2-game support is uninformative unless the design's floor is stated:
    2 games can never reach p<0.05 two-sided, so more SEEDS cannot fix it -- only more games."""
    games = MOD.sign_test_two_sided(2, 0)
    assert games["min_reachable_two_sided_p_at_this_support"] == 0.5
    assert games["min_reachable_two_sided_p_at_this_support"] > 0.05


# =================================================================================================
# DEFECT 4 -- the retracted formula and the retired charge model must not return.
# =================================================================================================
def _artifact_keys(obj, out=None):
    """Every dict KEY anywhere in a nested structure. Testing keys rather than raw text is what
    distinguishes 'the model is emitted as a live reading' from 'a comment explains why it was
    retired' -- scanning raw source for the name flags the retraction itself, which is the
    negation-blindness defect CLAUDE.md's QA-Layer discipline names explicitly."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(k)
            _artifact_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _artifact_keys(v, out)
    return out


def _run_analyser_on_synthetic_rows(tmp_path, with_memory: bool = False) -> dict:
    """Drive the analyser END-TO-END on a tiny synthetic sweep and return the emitted artifact.

    WHY THIS EXISTS. Tests that read the COMMITTED artifact cannot tell "the analyser emits this"
    from "the on-disk file happens to contain it". Proved by mutation three separate times: deleting
    the `max_score_clamp` wiring, and making `honest_verdict` re-assert BOTH struck claims, each left
    the committed-artifact tests GREEN. That is measurement-failure #9 (a stale artifact read as
    current) reproduced inside the test suite. Everything asserted about analyser BEHAVIOUR must go
    through this fixture.

    WHY THE DESIGN IS 3 SEEDS WITH A DISAGREEING GAME. A first version used ONE seed per game, which
    makes cell == game and renders the pseudo-replication defect UNDETECTABLE BY CONSTRUCTION:
    mutating `st_games = sign_test(len(games_gained), ...)` to `sign_test(len(gained), ...)` passed.
    The fixture below is deliberately shaped so the two units DISAGREE:

      ga -- won at both budgets on all 3 seeds        -> concordant, contributes no sign
      gb -- won only at b2000, on all 3 seeds         -> 3 CELLS gained, 1 GAME gained
      gc -- won only at b2000, on 1 of 3 seeds        -> 1 CELL gained,  1 GAME gained
      gd -- won at b400 on 1 seed, lost at b2000      -> 1 CELL lost,    1 GAME lost

    So the step has 4 cells gained / 1 cell lost but 2 games gained / 1 game lost. Any test that
    conflates the units now fails.
    """
    import json
    import subprocess
    import sys

    seeds, budgets = [1, 2, 3], [400, 2000]

    def won_at(game: str, seed: int, b: int) -> bool:
        if game == "ga":
            return True
        if game == "gb":
            return b == 2000
        if game == "gc":
            return b == 2000 and seed == 1
        return b == 400 and seed == 1  # gd: a REGRESSION at the higher budget

    rows = []
    for g in ["ga", "gb", "gc", "gd"]:
        for s in seeds:
            for b in budgets:
                won = won_at(g, s, b)
                rows.append(
                    {
                        "game": g,
                        "seed": s,
                        "budget": b,
                        "ran": True,
                        "levels": 1 if won else 0,
                        "reached": 1 if won else 0,
                        "actions": int(b * 0.98),
                        "wall_s": round(b * 0.005, 3),
                        "construct_s": 2.5,
                        "efficiency": 2.0 if won else 0.0,
                        "actions_to_first_levelup": 100 if won else None,
                        "states_expanded": b,
                        "nodes_total": b // 4,
                        "nodes_with_frame": b // 4,
                        "nodes_with_previous_frame": b // 4,
                        "unique_frames": b // 4,
                        "induction_attempts": 1,
                        "lever1_fired": False,
                        "lever2_fired": False,
                        "lever3_verdict": "LEVER_OFF",
                        "hud_mask_resolved": True,
                        "hud_mask_cell_count": 63,
                        "hud_diagnostics_readable": True,
                        "gated_flags": {"tier_exhaustion": True},
                        "arm": f"S_llmoff_b{b}",
                    }
                )
    rows_path = tmp_path / "synth_rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "sweep": "synthetic",
                "rows": rows,
                "budgets_requested": budgets,
                "seeds_requested": seeds,
                "games_requested": ["ga", "gb", "gc", "gd"],
                "arm": "S",
                "arm_flags": {},
                "llm_enabled": False,
                "scored_agent_max_actions": 400,
                "flag_parity_vs_live_globals": {"pinned_vs_live_drift": {}},
                "elapsed_s": 10.0,
                "rows_checksum": "synthetic",
            }
        )
    )
    extra: list[str] = []
    if with_memory:
        # Two games with DELIBERATELY DIFFERENT per-game deltas, so a projection that silently
        # substitutes the median for the worst case produces a different number and fails. With
        # equal deltas the `worst >= median` assertion holds with equality and the mutation survives.
        mem_path = tmp_path / "mem.jsonl"
        mem_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "game": g,
                        "seed": 1,
                        "budget": b,
                        "ran": True,
                        "levels": 1,
                        "actions": b,
                        "nodes_total": b // 4,
                        "nodes_with_frame": b // 4,
                        "wall_s": 5.0,
                        "shared_libs_rss_mib": 800.0,
                        "after_rss_mib": 800.0 + d,
                        "per_game_delta_mib": d,
                        "per_game_delta_peak_mib": d,
                    }
                )
                for b in budgets
                for g, d in (("ga", 40.0 * b / 400), ("gb", 100.0 * b / 400))
            )
        )
        extra = ["--memory-rows", str(mem_path)]
    out = tmp_path / "artifact.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "analyze_arc_scored_path_budget_sweep.py"),
            "--rows",
            str(rows_path),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text())


def test_analyser_emits_the_corrected_score_axis_sections(tmp_path):
    """END-TO-END: the corrected sections must come out of the ANALYSER, not merely exist on disk."""
    art = _run_analyser_on_synthetic_rows(tmp_path)
    e = art["efficiency_axis"]
    assert e["authoritative_scorer_resolution"]["tail_is_score_relevant"] is False
    assert "max_score_clamp" in e, "the omitted clamp analysis is missing again"
    assert (
        e["max_score_clamp"]["probe_on_an_8_level_game"]["1"][
            "clamp_index_weighted_fraction_times_100"
        ]
        == 2.7778
    )
    assert "scored_sum_authoritative" in e
    assert "scored_sum_under_both_charge_models" not in e
    assert "premise_coverage_budget_exhaustion" in e


def test_analyser_emits_the_game_level_headline_test(tmp_path):
    """END-TO-END, ON A DESIGN WHERE THE TWO UNITS DISAGREE. See the fixture docstring: the step has
    4 cells gained / 1 lost but 2 games gained / 1 lost, so a 'game-level' test secretly computed
    from cells produces different discordant counts and fails here. A 1-seed design cannot catch
    that, and the first version of this test used one -- the mutation passed."""
    art = _run_analyser_on_synthetic_rows(tmp_path)
    (step,) = art["marginal_return_per_step"]
    assert "clustering_note" in step
    assert "sign_test_on_cells_both_tails" not in step, "the unqualified name is back"

    games = step["HEADLINE_sign_test_on_GAMES_both_tails"]
    cells = step["sign_test_on_cells_WITHIN_GAME_REPLICATES_not_independent"]
    assert step["games_gained"] == ["gb", "gc"]
    assert step["games_lost"] == ["gd"]
    # THE LOAD-BEARING ASSERTIONS: the two units must carry DIFFERENT discordant counts.
    assert (games["n_pos"], games["n_neg"]) == (2, 1)
    assert (cells["n_pos"], cells["n_neg"]) == (4, 1)
    assert games["n_discordant"] != cells["n_discordant"], (
        "the game-level test is being computed from cells -- the pseudo-replication defect is back"
    )
    assert games["p_two_sided"] > cells["p_two_sided"], (
        "clustering must not inflate the game-level p"
    )
    assert (
        art["headline"]["HEADLINE_game_level_sign_test_p_two_sided_by_step"]["400->2000"]
        == (games["p_two_sided"])
    )
    assert art["headline"]["n_distinct_games_that_moved_by_step"]["400->2000"] == 3


def test_witness_is_computed_at_the_GAME_unit_the_headline_test_uses(tmp_path):
    """THE GAP THE ADVERSARIAL REVIEW OF THIS FIX FOUND. Moving the headline test to the game unit
    while leaving the witness at the cell unit reproduces CLAUDE.md's own named defect: 'a per-cell
    witness for a median gate is how that defect recurred'. The witness must state how many GAMES
    could have moved, since that is the unit the quoted p-value is computed on."""
    art = _run_analyser_on_synthetic_rows(tmp_path)
    (step,) = art["marginal_return_per_step"]
    w = step["WITNESS_pass_region_nonempty"]
    assert "n_games_that_could_gain" in w, "witness is still cell-only"
    assert "n_games_that_could_regress" in w
    assert w["nonempty_at_the_game_unit"] is True
    # gb/gc/gd could gain (not won at b400 on some seed); ga/gd could regress.
    assert w["n_games_that_could_gain"] >= len(step["games_gained"])
    assert w["n_games_that_could_regress"] >= len(step["games_lost"])
    # The cell-level counts stay, explicitly labelled, so the raw evidence is not lost.
    assert "n_cells_that_could_gain" in w


def test_retired_charge_model_is_not_emitted_as_a_live_artifact_field():
    """The retired model surfaced as artifact FIELDS (`pessimistic_*`, `scored_sum_under_both_charge_
    models`). Those must be gone from the emitted structure. The published artifact is the thing an
    operator reads, so that -- not the source text -- is what this asserts against."""
    import json

    art = json.loads(
        (REPO / "results" / "outer_loop_scored_path_budget_sweep_20260726.json").read_text()
    )
    keys = set(_artifact_keys(art))
    assert "scored_sum_under_both_charge_models" not in keys
    assert not [k for k in keys if k.lower().startswith("pessimistic")]
    assert not [k for k in keys if "pessimistic" in k.lower()]
    # And the replacement IS present, so this test cannot pass by the section being deleted.
    assert "scored_sum_authoritative" in keys
    assert "authoritative_scorer_resolution" in keys
    assert "max_score_clamp" in keys


def test_verdict_does_not_reassert_the_two_struck_claims(tmp_path):
    """END-TO-END. `scored_efficiency_term_degrades_quadratically` and `wall_clock_never_binding` both
    contradicted the artifact's own measured numbers, and honest_verdict is the FIRST thing the
    Reading-Results Discipline says to read.

    THIS RUNS THE ANALYSER rather than reading the committed file. Proved necessary by mutation: a
    verdict template rewritten to re-assert BOTH struck claims left the committed-file version of
    this test GREEN, because the on-disk artifact still held the corrected string.
    """
    art = _run_analyser_on_synthetic_rows(tmp_path)
    v = art["honest_verdict"]
    assert v.startswith("complete_")
    assert "degrades_quadratically" not in v
    # 'wall_clock_never_binding' unqualified is struck; the LLM-OFF-scoped form is what replaced it.
    assert "_wall_clock_never_binding" not in v
    assert "llm_off_wall_never_binding" in v
    assert "authoritative_score_sum_ROSE" in v


def test_committed_artifact_matches_the_current_analyser_on_these_invariants():
    """The COMMITTED artifact must also satisfy the invariants, or it is stale (failure #9: the
    published artifact predated its own analyser by 42 minutes, so every corrected number was
    missing from it). The end-to-end tests prove the analyser is right; this proves the published
    file was rebuilt after the analyser changed."""
    import json

    art = json.loads(
        (REPO / "results" / "outer_loop_scored_path_budget_sweep_20260726.json").read_text()
    )
    v = art["honest_verdict"]
    assert "degrades_quadratically" not in v and "_wall_clock_never_binding" not in v
    assert "llm_off_wall_never_binding" in v and "authoritative_score_sum_ROSE" in v
    keys = set(_artifact_keys(art))
    assert "max_score_clamp" in keys and "authoritative_scorer_resolution" in keys
    assert "scored_sum_under_both_charge_models" not in keys
    assert "n_games_that_could_gain" in keys, "witness is cell-only in the committed artifact"
    for step in art["marginal_return_per_step"]:
        assert "HEADLINE_sign_test_on_GAMES_both_tails" in step


def test_memory_envelope_is_measured_and_projected_by_the_concurrent_game_count(tmp_path):
    """SCENARIO-ARC-WMTE-5981-MEMORY-ENVELOPE-IS-MEASURED.

    Memory was previously a residual ('memory at 110 concurrent games untested') citing a 6.6 GiB
    estimate, while the report recommended the largest measured budget. It is the constraint that
    hard-fails.

    END-TO-END, WITH UNEQUAL PER-GAME DELTAS. The fixture's two games differ (40 vs 100 MiB at b400),
    which is load-bearing: an earlier version probed games with equal deltas, so substituting the
    MEDIAN for the WORST case satisfied `worst >= median` with equality and the mutation survived.
    """
    art = _run_analyser_on_synthetic_rows(tmp_path, with_memory=True)
    mem = art["memory_envelope"]
    assert mem["measured"] is True
    assert mem["n_probe_cells"] > 0
    assert "host_ram_is_UNCONFIRMED" in mem
    shared = mem["shared_libs_rss_mib_median"]
    for entry in mem["per_budget"].values():
        env = entry["C_110games_12h"]
        expected = (shared + env["n_games"] * entry["per_game_delta_mib_worst"]) / 1024.0
        assert math.isclose(
            env["projected_peak_gib_if_every_game_is_worst_case"], expected, abs_tol=0.02
        )
        # STRICTLY greater: the fixture's deltas differ, so a median-for-worst substitution fails.
        assert (
            env["projected_peak_gib_if_every_game_is_worst_case"]
            > env["projected_peak_gib_at_median_per_game"]
        ), "the worst-case projection is not using the worst per-game delta"
        assert entry["per_game_delta_mib_worst"] > entry["per_game_delta_mib_median"]
    # Memory must be ranked FIRST among binding constraints, not listed as a residual.
    keys = list(art["headline"]["BINDING_CONSTRAINT"])
    assert keys[0].startswith("1_memory")


def test_no_shipped_flag_or_action_cap_was_changed():
    """SCENARIO-ARC-WMTE-5981-NO-FLAG-IS-CHANGED. This is a MEASUREMENT task; the decision is the
    operator's. Asserted against the live module so a stray edit to the cap fails the suite."""
    import json

    # READ BY AST, NOT BY IMPORT. Importing arc_competition_agent costs ~590 MiB of RSS (the same
    # shared-library cost the memory envelope measures), which trips this suite's own memory-leak
    # guard at teardown. An AST read gets both cap sites without paying it, and is robust to
    # reformatting in a way a text grep is not.
    tree = ast.parse(
        (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    )
    module_level = [
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MAX_ACTIONS" for t in n.targets)
        and isinstance(n.value, ast.Constant)
    ]
    class_level = [
        stmt.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "CarnotAgent"
        for stmt in node.body
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MAX_ACTIONS" for t in stmt.targets)
        and isinstance(stmt.value, ast.Constant)
    ]
    assert class_level == [400], f"the SCORED cap must not be edited by this work: {class_level}"
    assert module_level == [200], f"the module-level cap must not be edited either: {module_level}"
    art = json.loads(
        (REPO / "results" / "outer_loop_scored_path_budget_sweep_20260726.json").read_text()
    )
    nc = art["what_was_NOT_changed"]
    assert nc["MAX_ACTIONS_class_attr_line_6230"] == 400
    assert nc["SUBMITTED_flags_touched"] == []
    assert nc["submission_made"] is False


def test_leaderboard_eval_docstring_no_longer_advertises_the_retracted_formula():
    """The stale header line is what a downstream analyser read as the definition before building the
    unsound model on it. It may still appear INSIDE the correction note that retracts it, so this
    asserts on the retraction being present rather than the phrase being absent."""
    src = (REPO / "scripts" / "arc_leaderboard_eval.py").read_text()
    head = src.split('"""')[1]
    assert "AUTHORITATIVE LEADERBOARD METRIC" in head
    assert "EnvironmentScoreCalculator" in head
    assert "docstring corrected 2026-07-26" in head
    # The retracted paraphrase must not be the metric's stated definition any more: the sentence
    # that introduces the metric must name the installed scorer, not the paraphrase.
    intro = head.split("NOTE (docstring corrected")[0]
    assert "min(baseline/agent_actions,1)^2" not in intro
