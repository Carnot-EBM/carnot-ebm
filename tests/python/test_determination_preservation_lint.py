"""A commit may not silently DROP a fabrication-gate determination from a results artifact.

REQ-ARC-WMTE-5995 / SCENARIOs: determination-strip-is-refused, fail-forward-numbers-are-allowed,
deliberate-clearing-requires-a-note, guard-fires-on-unstaged-changes

WHY THIS EXISTS (2026-07-27). A conductor re-run overwrote seven artifacts in place and dropped
``flagged_adversarial: True`` from all seven (six also lost their corrigendum records). That is
not an ordinary never-prune violation: every consumer of CLAUDE.md's fabrication gate keys off
the field being PRESENT, so losing it RE-ADMITS a quarantined artifact to capstone /
evidence-table / paper-v6 aggregation -- silently, with no human-read diff. All seven still
reported ``1 flagged`` after the overwrite, so the determinations were live, not stale.

THE TEST THAT MATTERS MOST IS `test_guard_fires_on_an_unstaged_strip`. The lint's first draft
listed filenames from ``git diff --cached`` while reading the new side from the working tree, so
an unstaged strip produced an EMPTY file list and the lint printed OK on a tree that had just
lost a determination -- it failed to fire on a faithful replay of its own origin incident. A
guard that cannot detect the thing it was written for is worse than no guard, because it
converts an open problem into a false sense of coverage. That regression is pinned here.

The must-NOT-fire controls are equally load-bearing. The operator's standing directive is
fail-forward ("always committing and never reverting so that we fail forward"), so a re-run
that legitimately changes MEASUREMENTS must pass untouched. A lint that refuses normal work
gets disabled, and then protects nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import determination_preservation_lint as dpl  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo with one committed, flagged artifact.

    Driving the REAL git plumbing rather than mocking it: the origin bug was in which git
    command the lint chose (`--cached` vs `HEAD`), so a mocked git would have reproduced the
    bug rather than caught it.
    """
    r = tmp_path / "repo"
    (r / "results").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    art = r / "results" / "experiment_1_thing.json"
    art.write_text(
        json.dumps(
            {
                "experiment": 1,
                "duration_s": 12.5,
                "auroc": 0.91,
                "flagged_adversarial": True,
                "corrigendum_pending": "DURATION_TOO_SHORT",
                "corrigendum_note": "flagged 2026-05-30",
            },
            indent=2,
        )
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")
    monkeypatch.setattr(dpl, "REPO", r)
    return r, art


def _write(art: Path, **changes) -> None:
    d = json.loads(art.read_text())
    for k, v in changes.items():
        if v is dpl:  # sentinel: delete the key
            d.pop(k, None)
        else:
            d[k] = v
    art.write_text(json.dumps(d, indent=2) + "\n")


def test_stripping_the_stamp_is_refused(repo):
    """THE ORIGIN INCIDENT: the stamp vanishes on an in-place overwrite."""
    _, art = repo
    _write(art, flagged_adversarial=dpl)
    v = dpl.check()
    assert v, "a dropped determination must be refused"
    assert any("flagged_adversarial True ->" in x for x in v)


def test_guard_fires_on_an_unstaged_strip(repo):
    """THE REGRESSION THAT MADE THE FIRST DRAFT USELESS -- pinned.

    The strip is written to the working tree and NOT staged. The first draft listed files from
    `git diff --cached`, found none, and returned clean.
    """
    _, art = repo
    _write(art, flagged_adversarial=dpl)
    assert _git(repo[0], "diff", "--cached", "--name-only") == "", "precondition: nothing staged"
    assert dpl.check(), "the guard must not depend on the change being staged"


def test_losing_a_corrigendum_record_is_refused(repo):
    """The corrigendum trail is the evidence behind the stamp; a re-run does not supersede it."""
    _, art = repo
    _write(art, corrigendum_pending=dpl, corrigendum_note=dpl)
    v = dpl.check()
    assert any("lost corrigendum record" in x for x in v)


def test_changing_measurements_while_keeping_the_stamp_passes(repo):
    """MUST-NOT-FIRE 1: fail-forward. A re-run's new numbers are normal, healthy work."""
    _, art = repo
    _write(art, duration_s=999.9, auroc=0.42, a_new_metric=0.1)
    assert dpl.check() == [], (
        "a lint that refuses normal re-runs gets disabled and protects nothing"
    )


def test_deliberate_clearing_with_a_note_passes(repo):
    """MUST-NOT-FIRE 2: a determination CAN be retracted -- auditably."""
    _, art = repo
    _write(
        art,
        flagged_adversarial=False,
        flagged_adversarial_cleared_note="Cleared: substrate taxonomy fixed; 0 flagged now.",
    )
    assert not [x for x in dpl.check() if "flagged_adversarial True ->" in x]


def test_clearing_without_a_note_is_refused(repo):
    """False-with-no-note is indistinguishable from the accident, so it is refused."""
    _, art = repo
    _write(art, flagged_adversarial=False)
    assert any("no *_cleared_note" in x for x in dpl.check())


def test_an_unflagged_artifact_is_never_implicated(repo):
    """Only artifacts that HELD a determination can lose one."""
    r, _ = repo
    other = r / "results" / "experiment_2_clean.json"
    other.write_text(json.dumps({"experiment": 2, "duration_s": 5.0}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "add clean")
    other.write_text(json.dumps({"experiment": 2, "duration_s": 6.0}, indent=2) + "\n")
    assert dpl.check() == []


def test_unparseable_or_absent_sides_do_not_crash_the_commit(repo):
    """A truncated artifact mid-write must not wedge every commit in the repo."""
    _, art = repo
    art.write_text('{"experiment": 1, "flagged')
    assert dpl.check() == [], "unparseable NEW side is not evidence of a dropped determination"


def test_the_real_repo_is_currently_clean():
    """The live tree must hold every determination it holds at HEAD.

    Guards the 7 restorations from 2026-07-27: if a later re-run strips one again, this fails
    in CI even if the commit-time hook was bypassed.
    """
    assert dpl.check() == [], "a determination has been dropped in the working tree"


# =========================================================================================
# 2026-07-29 WIDENING -- the three confirmed incidents from the test-suite-rewrites-the-record
# hazard, replayed as fixtures.
#
# Every one of these was produced by RUNNING THE TEST SUITE, not by a human edit: a class of
# `test_experiment_*.py` calls `runpy.run_path` on the real experiment script, which rewrites
# its own artifact at the historical path. Two of the three sailed straight past the
# pre-widening lint. They are pinned here so a future narrowing of MARKER_PATTERNS or
# STRENGTH_BANDS cannot silently re-open the hole.
# =========================================================================================


def test_incident_1_a_correction_note_is_a_corrigendum_even_without_that_word(repo):
    """exp3946: the guard sat in the path of this deletion and stayed silent.

    ``results/experiment_3946_r11l_first_solve.json`` lost FOUR fields when its test re-ran the
    experiment script. Two of them -- ``inference_substrate_correction_note`` and
    ``inference_substrate_original_invalid_value`` -- are a hand-written 2026-07-27 corrigendum:
    a dated retraction recording that the artifact's original substrate string was illegal, and
    what it used to say. The other two are the ARC Live-Path Reachability Discipline's own gate
    key (``solve_provenance``, which decides whether a solve is headline-eligible or
    CRITICAL-flagged) and its note.

    The pre-widening lint's only pattern was the literal string "corrigendum". None of these
    four names contain it, so all four vanished with the lint reporting OK. THAT is the bug this
    test pins: a guard that is trusted and does not fire is worse than no guard.
    """
    r, _ = repo
    art = r / "results" / "experiment_3946_r11l_first_solve.json"
    original = {
        "experiment": 3946,
        "game": "r11l",
        "duration_s": 4.21,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_original_invalid_value": "live_arc_agi3_api",
        "inference_substrate_correction_note": (
            "2026-07-27: the original declaration was illegal under the substrate taxonomy; "
            "this run stepped the OFFLINE arcade and loaded no model."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": "offline dev twin via arc_loop_solve + a hand-registered adapter",
    }
    art.write_text(json.dumps(original, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 3946")

    # What the re-run actually wrote: same measurement, four records gone.
    rerun = {
        k: v for k, v in original.items() if k in ("experiment", "game", "inference_substrate")
    }
    rerun["duration_s"] = 3.98
    art.write_text(json.dumps(rerun, indent=2) + "\n")

    violations = dpl.check()
    joined = " ".join(violations)
    assert violations, "the four dropped records must refuse the commit"
    for lost in (
        "inference_substrate_correction_note",
        "inference_substrate_original_invalid_value",
        "solve_provenance",
        "solve_provenance_note",
    ):
        assert lost in joined, f"{lost} was dropped but is not named in the refusal"


def test_incident_2_a_substrate_may_not_be_weakened_in_place(repo):
    """exp307: ``inference_mode`` flipped ``live_gpu`` -> ``cpu_training``.

    Nothing was DELETED here, so no amount of widening the dropped-field patterns would have
    caught it. The field survived and lied: an artifact that recorded a live-GPU training run
    now claims CPU. That retroactively rewrites what hardware a landed measurement ran on, and
    it defeats CLAUDE.md's Inference-Substrate Declaration Discipline, whose duration floors are
    applied per-substrate.

    This exact transition landed in the real history TWICE (commits a3af2404c, 0a856a300), and
    the sibling artifact exp911 took ``real_model`` -> ``synthetic_runner`` five more times --
    eight instances across 1,200 commits that nothing in the repo was watching for.
    """
    r, _ = repo
    art = r / "results" / "experiment_307_jepa_real_training.json"
    art.write_text(json.dumps({"experiment": 307, "inference_mode": "live_gpu"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 307")

    art.write_text(
        json.dumps({"experiment": 307, "inference_mode": "cpu_training"}, indent=2) + "\n"
    )

    violations = [v for v in dpl.check() if "WEAKENED" in v]
    assert violations, "live_gpu -> cpu_training must refuse the commit"
    assert "live_gpu" in violations[0] and "cpu_training" in violations[0]


def test_incident_2_variant_real_model_to_synthetic_runner(repo):
    """exp911's five landed instances of the same class, in the other direction of vocabulary.

    Pinned separately because it exercises a different strength band (REAL -> NOT-REAL-COMPUTE
    rather than LIVE -> REAL-BUT-CHEAP). A narrowing that kept only the `gpu`/`cpu` tokens would
    still pass the exp307 test above while re-opening this one.
    """
    r, _ = repo
    art = r / "results" / "experiment_911_drift_probe_tier0i.json"
    art.write_text(json.dumps({"experiment": 911, "inference_mode": "real_model"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 911")

    art.write_text(
        json.dumps({"experiment": 911, "inference_mode": "synthetic_runner"}, indent=2) + "\n"
    )
    assert [v for v in dpl.check() if "WEAKENED" in v], "real_model -> synthetic_runner must refuse"


def test_an_unrelated_change_note_does_not_excuse_a_substrate_downgrade(repo):
    """The escape hatch must name the field it excuses.

    A draft of ``_has_change_note`` accepted ANY key ending in ``_change_note``, regardless of
    what it referred to. That meant an unrelated ``corpus_change_note`` elsewhere in the same
    artifact would silently excuse a substrate downgrade -- the "guard that does not fire"
    failure mode, reintroduced inside the fix for it. Pinned here.
    """
    r, _ = repo
    art = r / "results" / "experiment_505_thing.json"
    art.write_text(json.dumps({"inference_substrate": "live_llm_inference"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 505")

    art.write_text(
        json.dumps(
            {
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "corpus_change_note": "swapped MuSR for GSM8K this round",
            },
            indent=2,
        )
        + "\n"
    )
    assert [v for v in dpl.check() if "WEAKENED" in v], (
        "a note about something else is not an explanation of THIS downgrade"
    )


def test_incident_3_timestamp_rewrites_are_deliberately_NOT_this_lint_s_job(repo):
    """exp1035: run_date / started_at / finished_at rewritten to today by a test re-run.

    This lint does NOT fire on it, ON PURPOSE, and the boundary is worth pinning so nobody
    "fixes" it later. Timestamps are MEASUREMENTS of a run, and the operator's standing
    fail-forward directive means a re-run legitimately producing new ones must not be refused.
    A lint that blocks ordinary re-runs gets disabled, and then protects nothing.

    The timestamp class is caught by the OTHER guard shipped alongside this widening --
    ``scripts/test_suite_mutation_check.py`` -- which reports every tracked file a test run
    touched without any opinion about content. The division of labour is deliberate: this lint
    is narrow and deep, that detector is broad and shallow.
    """
    r, _ = repo
    art = r / "results" / "experiment_1035_dualgpu_rocm_v3.json"
    art.write_text(
        json.dumps(
            {
                "experiment": 1035,
                "run_date": "20260727",
                "started_at": "2026-07-27T03:34:05.843330+00:00",
                "finished_at": "2026-07-27T03:34:05.881266+00:00",
                "duration_s": 0.038,
            },
            indent=2,
        )
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 1035")

    art.write_text(
        json.dumps(
            {
                "experiment": 1035,
                "run_date": "20260729",
                "started_at": "2026-07-29T05:17:00.000000+00:00",
                "finished_at": "2026-07-29T05:17:00.041000+00:00",
                "duration_s": 0.041,
            },
            indent=2,
        )
        + "\n"
    )
    assert dpl.check() == [], (
        "timestamp churn is fail-forward measurement churn; refusing it here would make the "
        "lint unusable. test_suite_mutation_check.py is the guard for this class."
    )


def test_a_substrate_downgrade_stated_on_purpose_is_allowed(repo):
    """The escape hatch: weakening is legitimate work when it is declared where a reader sees it.

    Same shape as the original ``*_cleared_note`` allowance. The point of the rule is not that
    substrates may never change -- experiments genuinely get re-scoped from live inference to
    aggregation -- it is that the change must be distinguishable from an accident.
    """
    r, _ = repo
    art = r / "results" / "experiment_500_thing.json"
    art.write_text(json.dumps({"inference_substrate": "live_llm_inference"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 500")

    art.write_text(
        json.dumps(
            {
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "inference_substrate_change_note": (
                    "2026-07-29: rescoped to an aggregation pass; the live numbers it cites now "
                    "come from exp5178, which is unchanged."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    assert [v for v in dpl.check() if "WEAKENED" in v] == []


def test_an_unrecognised_substrate_string_is_never_read_as_a_downgrade(repo):
    """Unknown vocabulary must rank as UNKNOWN, not as WEAK.

    The project invents substrate strings constantly -- 2,842 declarations across ~40 distinct
    vocabularies in the tree today. Ranking an unrecognised string as 0 would refuse a large
    fraction of honest commits, which is the failure mode that gets a guard switched off.
    """
    r, _ = repo
    art = r / "results" / "experiment_501_thing.json"
    art.write_text(json.dumps({"inference_substrate": "live_llm_inference"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 501")

    art.write_text(
        json.dumps({"inference_substrate": "some_brand_new_substrate_name"}, indent=2) + "\n"
    )
    assert [v for v in dpl.check() if "WEAKENED" in v] == []
    assert dpl._strength_rank("some_brand_new_substrate_name") is None


def test_a_principle_wrapped_substrate_is_unwrapped_before_ranking(repo):
    """CLAUDE.md lets ANY field be written ``{"principle": ..., "value": ...}``; 162 artifacts do.

    Origin bug #2 of the QA-Layer Authenticity Discipline was exactly this: a checker in
    ``adversarial_verify.py`` read such a field as a bare string and silently stopped
    recognising it on 176 artifacts. This guard must not repeat it.
    """
    r, _ = repo
    art = r / "results" / "experiment_502_thing.json"
    wrapped = {"principle": "declares what compute actually ran", "value": "live_llm_inference"}
    art.write_text(json.dumps({"inference_substrate": wrapped}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 502")

    art.write_text(
        json.dumps({"inference_substrate": {**wrapped, "value": "sota_gguf_mock"}}, indent=2) + "\n"
    )
    assert [v for v in dpl.check() if "WEAKENED" in v], (
        "a principle-wrapped substrate must be unwrapped before ranking, or the whole rule "
        "silently stops applying to 162 artifacts"
    )


def test_measurement_fields_containing_the_word_correct_are_not_protected(repo):
    """``energy_correct`` / ``n_correct`` / ``judge_correct`` are ACCURACY NUMBERS, not records.

    601 artifacts carry one. If ``correct`` were a marker pattern, every honest re-run that
    changed an accuracy count would be refused. Only the longer, prose-shaped ``correction``
    matches -- this test is what keeps that distinction from being casually widened away.
    """
    r, _ = repo
    art = r / "results" / "experiment_503_thing.json"
    art.write_text(
        json.dumps({"energy_correct": 41, "n_correct": 41, "judge_correct": 38}, indent=2) + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 503")

    art.write_text(json.dumps({"energy_correct": 44}, indent=2) + "\n")
    assert dpl.check() == [], (
        "dropping/changing accuracy measurements is fail-forward, not a violation"
    )


def test_an_empty_marker_field_carries_no_record_to_lose(repo):
    """A ``*_note`` that was ``null`` or ``""`` held nothing; refusing over it is pure noise."""
    r, _ = repo
    art = r / "results" / "experiment_504_thing.json"
    art.write_text(
        json.dumps({"methodology_note": "", "caveat": None, "notes": []}, indent=2) + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 504")

    art.write_text(json.dumps({"experiment": 504}, indent=2) + "\n")
    assert dpl.check() == []


def test_one_deletion_produces_one_refusal_line(repo):
    """Rule 3 must not re-report what rules 1 and 2 already own.

    Not cosmetic: a refusal message that repeats itself trains the reader to skim, and the
    whole value of this guard is that a human reads the line and understands what was lost.
    """
    _, art = repo  # the shared fixture holds flagged_adversarial + two corrigendum fields
    _write(art, flagged_adversarial=dpl, corrigendum_pending=dpl, corrigendum_note=dpl)
    violations = dpl.check()
    stamp_lines = [v for v in violations if "flagged_adversarial" in v]
    assert len(stamp_lines) == 1, f"the stamp was reported {len(stamp_lines)} times: {stamp_lines}"


def test_a_legitimate_analyser_rebuild_is_not_refused(repo):
    """MUST-NOT-FIRE, and the most load-bearing one in this file.

    Re-running an analyser is the single most COMMON write to ``results/`` in this project, so if
    the widening false-fires here it refuses the loop's own routine work every night -- and a lint
    that refuses honest work gets disabled, which is the same outcome as having no lint at all.

    This fixture is not invented. It is the exact field shape of the 12 analyser artifacts rebuilt
    on 2026-07-28 in commit ``8441055c0`` (9 ``outer_loop_*`` plus the card-ground-truth and
    reset-attribution passes). A rebuild moves FOUR things and nothing else:

      * ``build_timestamp_utc`` -- when the analyser ran
      * ``duration_s``          -- how long it took (0.594 -> 0.604 in the real diff)
      * ``provenance.git_head`` -- which commit it was built from
      * ``provenance.code[].sha256`` / ``.bytes`` -- the hashes of the scripts it read

    Every one of those is a MEASUREMENT of the rebuild, not a review output. Critically, the
    top-level ``provenance`` key SURVIVES -- it is rewritten in place, not dropped -- which is why
    rule 3 must key on a marker field DISAPPEARING rather than on it changing. And
    ``inference_substrate`` stays ``aggregation_from_upstream_artifacts`` throughout, so rule 4
    has nothing to compare downward.

    Verified against the real commit as well as this fixture: the widened lint examined all 28
    modified artifacts in ``8441055c0`` (12 of them analysers, both sides parsed) and reported no
    violation, while the same run refused four injected violations on that same real file.
    """
    r, _ = repo
    art = r / "results" / "outer_loop_arc_max_actions_answer_20260726.json"
    before = {
        "title": "The MAX_ACTIONS answer",
        "run_date": "2026-07-26",
        "build_timestamp_utc": "2026-07-27T05:05:38Z",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_principle": "An analyser pass inherits its methodology from the "
        "artifacts it cites.",
        "duration_s": 0.594,
        "random_seed": 20260724,
        "verifier_is_oracle": False,
        "provenance": {
            "git_head": "692bedfa8c8b9bfc95dd26b95631f01cfad9b996",
            "code": [
                {
                    "path": "scripts/arc_scored_path_lever_harness.py",
                    "sha256": "11098ad3",
                    "bytes": 57542,
                }
            ],
        },
    }
    art.write_text(json.dumps(before, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed analyser")

    after = json.loads(json.dumps(before))
    after["build_timestamp_utc"] = "2026-07-28T20:26:43Z"
    after["duration_s"] = 0.604
    after["provenance"]["git_head"] = "a2222320c629ae14699ca795a4de27abf4c4e296"
    after["provenance"]["code"][0].update(sha256="fe0dcc1e", bytes=59647)
    art.write_text(json.dumps(after, indent=2) + "\n")

    assert dpl.check() == [], (
        "a routine analyser rebuild moves only clocks, git_head and dependency hashes; refusing "
        "it would block the loop's own nightly work and get this lint switched off"
    )


def test_the_analyser_rebuild_fixture_is_not_silently_unprotected(repo):
    """The companion to the test above: prove that PASS is a real pass, not blindness.

    A must-not-fire test is worthless on its own -- it also passes when the lint cannot see the
    artifact at all (wrong path glob, unparseable side, empty file list). That is not a
    hypothetical: while validating the widening, a positive control was accidentally run against a
    worktree checked out BEFORE the widening landed, and all four injected violations came back
    silent. The result looked like a clean bill of health and was in fact a measurement of the old
    two-rule lint.

    So this test injects violations into the SAME fixture the rebuild test uses and requires each
    to be refused. If a future change makes the lint stop seeing this artifact shape, the test
    above keeps passing and THIS one fails.
    """
    r, _ = repo
    art = r / "results" / "outer_loop_arc_max_actions_answer_20260726.json"
    before = {
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_principle": "inherits methodology from cited artifacts",
        "verifier_is_oracle": False,
        "duration_s": 0.594,
        "provenance": {"git_head": "692bedfa", "code": []},
    }
    art.write_text(json.dumps(before, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed analyser 2")

    for drop, expect in (
        ("provenance", "provenance declaration"),
        ("inference_substrate_principle", "inference-substrate declaration"),
        ("verifier_is_oracle", "circularity declaration"),
    ):
        d = {k: v for k, v in before.items() if k != drop}
        art.write_text(json.dumps(d, indent=2) + "\n")
        assert any(expect in v for v in dpl.check()), f"dropping {drop} must be refused"

    # And the in-place weakening, which no dropped-field rule can see.
    d = json.loads(json.dumps(before))
    d["inference_substrate"] = "sota_gguf_mock"
    art.write_text(json.dumps(d, indent=2) + "\n")
    assert [v for v in dpl.check() if "WEAKENED" in v], "an in-place substrate downgrade must fire"


def test_a_correction_field_matched_by_no_other_pattern_is_still_protected(repo):
    """The ``correction`` marker pattern, exercised in isolation. Found by mutation testing.

    Deleting the ``correction`` pattern from ``MARKER_PATTERNS`` left the whole suite GREEN. The
    reason is that the incident-1 test's fields -- ``inference_substrate_correction_note`` and
    ``inference_substrate_original_invalid_value`` -- are ALSO matched by the
    ``^inference_substrate`` pattern, and its ``solve_provenance*`` fields by ``provenance``. So
    every field that test names was double-covered, and the pattern that the incident is actually
    NAMED for was never the thing being tested.

    That is the same shape as the bug this whole file exists for: coverage that looks real and is
    not. These two field names are matched by the ``correction`` pattern and nothing else, so this
    test fails the moment that pattern is weakened or removed.
    """
    r, _ = repo
    art = r / "results" / "experiment_507_thing.json"
    art.write_text(
        json.dumps(
            {
                "correction_note": "2026-07-27: retracted the AUROC; the corpus was contaminated.",
                "data_correction": "row 14 was double-counted in the original tally",
            },
            indent=2,
        )
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 507")

    # Precondition: these must be matched by the `correction` pattern ALONE, or this test would
    # pass for the wrong reason -- exactly the trap it was written to close.
    for key in ("correction_note", "data_correction"):
        assert dpl._marker_kind(key) == "correction record", f"{key} is covered by another pattern"

    art.write_text(json.dumps({"experiment": 507}, indent=2) + "\n")
    violations = dpl.check()
    assert any("correction record" in v for v in violations), (
        "a hand-written correction record must not be droppable"
    )


def test_a_dropped_corrigendum_is_reported_once_not_twice(repo):
    """Rule 3 must not re-report the corrigendum family that rule 2 already owns.

    Found by mutation testing. ``test_one_deletion_produces_one_refusal_line`` pins the dedup for
    ``flagged_adversarial`` only, so blanking the OTHER half of ``already_reported`` -- the part
    that suppresses rule 3 on fields rule 2 already reported -- left the suite green. A corrigendum
    deletion would then produce two refusal lines naming the same fields.

    Duplicate refusal text is not cosmetic here. The whole value of this guard is that a human
    reads the line and understands what was lost; a message that says the same thing twice in
    different words trains the reader to skim, and skimming is how the origin incident survived
    seven artifacts.
    """
    _, art = repo  # fixture holds flagged_adversarial + corrigendum_pending + corrigendum_note
    _write(art, corrigendum_pending=dpl, corrigendum_note=dpl)
    violations = dpl.check()
    corrigendum_lines = [v for v in violations if "corrigendum" in v.lower()]
    assert len(corrigendum_lines) == 1, (
        f"the corrigendum deletion was reported {len(corrigendum_lines)} times: {corrigendum_lines}"
    )


def test_a_dropped_substrate_declaration_is_refused_too(repo):
    """The loophole rule 4 cannot see: DELETING the declaration instead of weakening it.

    Rule 4 only compares a field present on both sides, and a bare ``inference_mode`` -- the exact
    field the exp307 incident touched -- matches no marker pattern. So an artifact could have had
    its declaration flipped `live_gpu -> cpu_training` and been refused, while deleting the field
    outright sailed through. Absent is strictly WORSE than weaker: `adversarial_verify.py` treats a
    missing declaration as the strict default `live_llm_inference` and applies the 60s duration
    floor to something that may have run in milliseconds.
    """
    r, _ = repo
    art = r / "results" / "experiment_506_thing.json"
    art.write_text(json.dumps({"experiment": 506, "inference_mode": "live_gpu"}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 506")

    art.write_text(json.dumps({"experiment": 506}, indent=2) + "\n")
    violations = dpl.check()
    assert any("inference_mode" in v for v in violations), (
        "deleting a substrate declaration must be at least as refused as weakening it"
    )


# =========================================================================================
# 2026-07-29 REVIEW FIXES. Each test below pins a defect found by reviewing the widening
# against REAL artifacts rather than fixtures. Every one of them was a case where the guard
# fired on honest work -- the mirror image of the original bug, and just as corrosive: a
# guard that refuses correct commits gets switched off, which leaves the record unprotected.
# =========================================================================================

_EXP5178_REAL = (
    "live_llm_embedding_extraction; Substrate corrected 2026-07-03: this task's dominant work "
    "is a single forward pass per candidate (llama_cpp.Llama(embedding=True, "
    "pooling_type=LAST).embed against the real local gemma-4-26B-A4B-it GGUF) to extract "
    "final-token hidden-state vectors for 48 candidates, then centroid-probe training on those "
    "vectors -- no iterative token-by-token generation. The original declaration "
    "('live_llm_inference') implied full generative inference (60s floor), which this workload "
    "genuinely is not; verifier_ensemble_against_cached_candidates also does not fit (its "
    "definition explicitly requires the LLM NOT be loaded, and this task did load it, for "
    "embeddings)."
)

_EXP5161_REAL = (
    "verifier_ensemble_against_cached_candidates; Substrate honesty, corrected 2026-07-02: this "
    "artifact's DOMINANT content (n=60 pilot statistics, cluster bootstrap, exact test) rescores "
    "GAP-4's EXISTING cached candidate pool -- it does not invoke a generative LLM."
)


def test_the_inline_note_convention_does_not_make_a_real_gguf_load_look_cheap():
    """exp5178: ranking the PROSE instead of the declared value inverted the rule's verdict.

    The project documents `<canonical value><separator><human note>` as the way to declare a
    substrate with an explanation, and `adversarial_verify.py` strips the note before matching.
    The first draft of `_strength_rank` scanned the WHOLE string and took the minimum band, so
    exp5178 -- whose note mentions `cached` and `verifier_ensemble_against_cached_candidates`
    only to explain why those do NOT apply -- was ranked REAL-BUT-CHEAP. The lint then refused
    the commit while asserting the exact opposite of what the artifact says: it declares, and
    performed, a real GGUF load.

    This is the negation-blindness failure class named in CLAUDE.md's QA-Layer Authenticity
    Discipline: a checker confusing "did X" with "explicitly did NOT do X".
    """
    f = "inference_substrate"
    assert dpl._strength_rank("live_llm_inference", f) == 3
    assert dpl._strength_rank(_EXP5178_REAL, f) == 3, (
        "the leading canonical token is live_llm_embedding_extraction (a real model load); "
        "words appearing in the trailing human note must not change the rank"
    )


def test_a_substrate_correction_explained_inline_is_an_accepted_downgrade():
    """exp5161: the downgrade note was INSIDE the value, and only siblings were inspected.

    `_has_change_note` originally looked at sibling keys only, so the most carefully documented
    corrections in the corpus -- which put a dated rationale directly in the declaration, per the
    project's own convention -- were refused, while a terse downgrade that happened to have some
    sibling note was waved through.
    """
    f = "inference_substrate"
    assert dpl._strength_rank(_EXP5161_REAL, f) == 2
    assert dpl._has_change_note({f: _EXP5161_REAL}, f) is True


def test_a_bare_downgrade_with_no_prose_is_still_refused():
    """The escape hatch must not be so wide that the origin incidents walk through it.

    exp307's `cpu_training` carries no trailing prose, so it must NOT count as self-explaining.
    Without this the inline-note fix would silently disable rule 4.
    """
    assert dpl._has_change_note({"inference_mode": "cpu_training"}, "inference_mode") is False
    assert dpl._has_change_note({"inference_mode": "cpu_training."}, "inference_mode") is False


def test_a_non_enum_substrate_name_is_unrankable_not_phantom_live():
    """exp5240: `arc_live_path_patch_synthesis` is the ARC live CODE path, not a substrate.

    A bare token scan reads its `live` token as LIVE/HARDWARE. That value was never legal under
    the Inference-Substrate Declaration Discipline's fixed enum, so when a later commit honestly
    corrected it to `aggregation_from_upstream_artifacts`, rule 4 saw band 3 -> band 2 and
    refused a taxonomy REPAIR as a downgrade.

    The fix is structural rather than another token-list patch: for the one field governed by a
    documented vocabulary, rank ONLY from that vocabulary. Unknown is unrankable on BOTH sides --
    the rule already refused to treat an unknown NEW value as weak, and now likewise refuses to
    treat an unknown OLD value as strong. The collision cannot recur for any future `arc_live_*`
    name, which a token-boundary tweak alone could not promise (that string genuinely contains
    `live` as a whole token).
    """
    f = "inference_substrate"
    assert dpl._strength_rank("arc_live_path_patch_synthesis", f) is None
    assert dpl._strength_rank("aggregation_from_upstream_artifacts", f) == 2


def test_the_exp5240_taxonomy_repair_is_not_refused(repo):
    """End-to-end form of the above, through `check()` rather than the ranking helper."""
    r, _ = repo
    art = r / "results" / "experiment_5240_arc_rubric.json"
    art.write_text(
        json.dumps({"experiment": 5240, "inference_substrate": "arc_live_path_patch_synthesis"})
        + "\n"
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 5240")

    art.write_text(
        json.dumps(
            {"experiment": 5240, "inference_substrate": "aggregation_from_upstream_artifacts"}
        )
        + "\n"
    )
    assert not [v for v in dpl.check() if "5240" in v], (
        "correcting an ILLEGAL substrate value to a legal enum value is a repair, not a downgrade"
    )


def test_the_two_origin_incidents_are_still_refused_after_all_of_the_above():
    """The whole point of the review fixes is to narrow the rule WITHOUT disarming it.

    Both real incidents live on free-form mode fields with no governing enum, which is exactly
    why the token scan is retained there.
    """
    assert dpl._strength_rank("live_gpu", "inference_mode") == 3
    assert dpl._strength_rank("cpu_training", "inference_mode") == 2
    assert dpl._strength_rank("real_model", "inference_mode") == 3
    assert dpl._strength_rank("synthetic_runner", "inference_mode") == 1


def test_band_tokens_match_at_token_boundaries_not_as_bare_substrings():
    """`dry_run` must not be dragged to NOT-RUN, and `not_run` must not be diluted.

    A draft of the token-anchoring fix split multi-word tokens (`not_run` -> `not`, `run`), which
    would have made the bare token `run` a NOT-RUN marker -- so `dry_run` (NOT-REAL-COMPUTE) and
    even `live_run` would have ranked 0. Caught before landing; pinned here so it cannot return.
    """
    assert dpl._strength_rank("dry_run", "inference_mode") == 1
    assert dpl._strength_rank("not_run", "inference_mode") == 0
    assert dpl._strength_rank("blocked_no_live_gpu", "inference_mode") == 0


def test_the_added_review_output_markers_are_protected(repo):
    """Review outputs a re-run does not supersede: sample-size, false-negative, forbidden-claims.

    `n_samples_justification` (56 artifacts) is the disclosure CLAUDE.md's Adversarial Artifact
    Verification rule REQUIRES for a distributional claim; losing it silently un-discloses the
    claim's statistical basis.
    """
    r, _ = repo
    art = r / "results" / "experiment_777_thing.json"
    seed = {
        "experiment": 777,
        "n_samples_justification": "n=10000 so std(empirical_KL) < 0.005",
        "false_negative_risk_checked": True,
        "paper_v6_forbidden_claims": ["KV260 speedup at d=128"],
    }
    art.write_text(json.dumps(seed, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 777")

    art.write_text(json.dumps({"experiment": 777}, indent=2) + "\n")
    violations = " ".join(dpl.check())
    for field in seed:
        if field == "experiment":
            continue
        assert field in violations, f"{field} must be protected as a review output"


def test_honest_verdict_is_deliberately_not_protected(repo):
    """The boundary of the marker list: a re-run legitimately produces a NEW verdict.

    Protecting `honest_verdict` (5,245 artifacts) would refuse the fail-forward behaviour the
    operator's standing directive requires. Pinned so a later widening cannot quietly cross it.
    """
    r, _ = repo
    art = r / "results" / "experiment_888_thing.json"
    art.write_text(json.dumps({"experiment": 888, "honest_verdict": "complete: a"}) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed 888")

    art.write_text(json.dumps({"experiment": 888, "honest_verdict": "complete: b"}) + "\n")
    assert not [v for v in dpl.check() if "888" in v]


def test_an_unknown_substrate_may_rank_weak_but_never_strong():
    """The asymmetry that resolves exp5240 WITHOUT disarming the fabrication case.

    A first attempt returned None for every non-enum `inference_substrate`. It fixed exp5240 and
    silently un-protected `sota_gguf_mock` -- CLAUDE.md's own fabrication exemplar, also not in
    the enum. Two pre-existing tests caught the regression, and the fix is to distinguish a CLAIM
    of strength (cheap to make by accident) from an ADMISSION of weakness (nobody accidentally
    says their run was mocked).
    """
    f = "inference_substrate"
    assert dpl._strength_rank("arc_live_path_patch_synthesis", f) is None, "claim: not trusted"
    assert dpl._strength_rank("sota_gguf_mock", f) == 1, "admission: always trusted"
    assert dpl._strength_rank("blocked_model_not_cached", f) == 0, "admission: always trusted"
    # ... and the asymmetry is scoped to the enum-governed field only. Free-form mode fields have
    # no vocabulary to fall back on, so they must keep ranking `live_gpu` strong -- otherwise the
    # exp307 origin incident stops being refused.
    assert dpl._strength_rank("live_gpu", "inference_mode") == 3


def test_a_negated_weakness_word_does_not_drag_the_rank_down():
    """Token anchoring, pinned. Found by mutation testing -- nothing exercised it.

    Reverting `_token_scan_band` to a bare `token in value` substring test left the whole suite
    GREEN, so the anchoring the docstring claims was entirely untested. It matters because the
    rank is the MINIMUM matching band: a spurious match on a WEAK token inside a longer word
    silently downgrades an honest declaration.

    Both cases below are the negation-blindness class CLAUDE.md's QA-Layer Authenticity
    Discipline names -- a checker confusing "did X" with "explicitly did NOT do X":
      * `uncached_...` contains `cached` (band 2) but asserts the opposite.
      * `unblocked_...` contains `blocked` (band 0) but asserts the opposite.
    Unanchored, both would rank a live GPU run as cheap or as never-run, and a subsequent honest
    edit would be refused as a "downgrade" from a rank the value never deserved.
    """
    assert dpl._token_scan_band("uncached_live_gguf_inference") == 3
    assert dpl._token_scan_band("unblocked_live_gpu_run") == 3
    # The anchoring must not break PREFIX matching, which the vocabulary depends on:
    # `simulat` has to catch both `simulated` and `simulation`, `analys` both spellings.
    assert dpl._token_scan_band("cpu_simulated_annealing") == 1
    assert dpl._token_scan_band("offline_analysis_only") == 2
