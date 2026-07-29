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


def test_a_corrupt_overwrite_of_an_UNPROTECTED_artifact_does_not_wedge_the_commit(repo):
    """A truncated artifact mid-write must not wedge commits over artifacts with nothing at stake.

    This is the surviving half of the original
    ``test_unparseable_or_absent_sides_do_not_crash_the_commit``, whose concern was real: an
    experiment writing its artifact while a commit runs leaves a momentarily-truncated file, and
    a guard that refuses on ANY unreadable artifact would block unrelated work at random.

    The blanket version of that assertion was DELIBERATELY NARROWED on 2026-07-29 -- see the
    test directly below for the half that was wrong and why. The measured split is that 11,352
    of 15,284 readable tracked artifacts (74.3%) carry no protected field at all, so this
    exemption still covers the large majority of the corpus.
    """
    r, _ = repo
    other = r / "results" / "experiment_2_clean.json"
    other.write_text(json.dumps({"experiment": 2, "duration_s": 5.0}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "add clean")
    other.write_text('{"experiment": 2, "dur')
    assert dpl.check() == [], (
        "a corrupt artifact that held no determination is not this lint's business"
    )


def test_a_corrupt_overwrite_of_a_PROTECTED_artifact_is_refused(repo):
    """Destroying the whole artifact must not score cleaner than editing one field out of it.

    THIS REVERSES A PRIOR ASSERTION, deliberately, and the reasoning is recorded here because
    silently flipping a test's expectation is how a guard gets hollowed out. The pre-2026-07-29
    test asserted flatly that "unparseable NEW side is not evidence of a dropped determination"
    and passed the whole class. But ``_load_now`` returned ``None`` for BOTH "deleted" and
    "unreadable", and ``check`` skipped ``None`` -- so overwriting a flagged artifact with
    truncated bytes destroyed its determination, its corrigendum trail and its substrate
    declaration all at once, and the lint printed OK.

    That is the same never-prune harm as deletion, which the operator named explicitly as
    "strictly worse than editing it". A guard whose cheapest bypass is `> file` is not a guard.

    The transient-write concern that motivated the original assertion is preserved by scoping
    the refusal to artifacts that actually CARRY protected content (the test above), and by
    naming the remedy -- `git checkout -- <path>` -- in the refusal message itself.
    """
    _, art = repo
    art.write_text('{"experiment": 1, "flagged')
    v = dpl.check()
    assert len(v) == 1, v
    assert "NO LONGER READABLE" in v[0]
    # The message must tell the reader what was at stake and how to undo it, or the refusal is
    # just an obstacle.
    assert "flagged_adversarial" in v[0]
    assert "git checkout --" in v[0]


def test_an_absent_or_unreadable_OLD_side_is_never_a_violation(repo):
    """Nothing readable at HEAD means this commit destroyed nothing. Must not fire, must not crash.

    The other half of the original test's intent: a brand-new artifact has no old side, and an
    artifact that was ALREADY corrupt at HEAD carried no recoverable record for this commit to
    have dropped.
    """
    r, _ = repo
    fresh = r / "results" / "experiment_3_new.json"
    fresh.write_text(json.dumps({"experiment": 3, "flagged_adversarial": True}, indent=2) + "\n")
    assert dpl.check() == [], "a newly added artifact has no old side to have lost anything from"

    corrupt = r / "results" / "experiment_4_was_already_broken.json"
    corrupt.write_text("{not json at all")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "commit an already-corrupt artifact")
    corrupt.write_text("{still not json")
    assert dpl.check() == [], "an artifact unreadable at HEAD had no record for this commit to drop"


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


# =========================================================================================
# 2026-07-29 FAIL-CLOSED HARDENING -- the two holes the Layer-2 QA audit found in the
# PRE-widening lint and which survived the widening unchanged, plus the exemption hole.
#
# All three were demonstrated live against REAL artifacts before being fixed (a disposable
# git repo seeded from results/experiment_1680_polarfire_smoke_v2.json and
# results/experiment_3946_r11l_first_solve.json): every scenario below scored a clean
# `determination-preservation-lint: OK` under the pre-fix code.
#
# The unifying principle: A GUARD MUST FAIL CLOSED. Each of these was a path where the lint
# could not see, or did not look, and reported OK anyway -- which is strictly worse than
# having no lint, because it converts an open hole into a false sense of coverage.
# =========================================================================================


def test_deleting_a_flagged_artifact_is_refused(repo):
    """`--diff-filter=M` considered MODIFIED files only, so `git rm` bypassed the guard entirely.

    Deleting the artifact destroys the determination, the corrigendum trail and the substrate
    declaration in one move. It is the cheapest possible bypass and it scored clean.
    """
    r, art = repo
    _git(r, "rm", "-q", "results/experiment_1_thing.json")
    v = dpl.check()
    assert len(v) == 1, v
    assert "DELETED" in v[0]
    assert "flagged_adversarial" in v[0], "the message must name what was destroyed"


def test_deleting_an_artifact_that_held_nothing_is_allowed(repo):
    """The charter boundary, stated as a test.

    This is the DETERMINATION-preservation lint, not a blanket ban on removing files from
    ``results/``. Refusing every deletion would be over-broad -- and an over-broad guard gets
    disabled, which is the same outcome as no guard.
    """
    r, _ = repo
    other = r / "results" / "experiment_2_clean.json"
    other.write_text(json.dumps({"experiment": 2, "duration_s": 5.0}, indent=2) + "\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "add clean")
    _git(r, "rm", "-q", "results/experiment_2_clean.json")
    assert dpl.check() == []


def test_an_unstaged_deletion_is_refused(repo):
    """Same reasoning as `test_guard_fires_on_an_unstaged_strip`: staging is not the trigger.

    `rm results/x.json` without `git rm` must be caught too -- the lint compares the working
    tree against HEAD precisely so that staging state cannot hide a loss.
    """
    _, art = repo
    art.unlink()
    v = dpl.check()
    assert len(v) == 1 and "DELETED" in v[0], v


def test_a_rename_that_drops_a_marker_is_refused(repo):
    """`git mv` + strip: invisible under BOTH the old path and the new one with `--diff-filter=M`.

    A rename is not itself a violation (see the control below), but it must not be usable as a
    laundering step for a field drop.
    """
    r, _ = repo
    _git(r, "mv", "results/experiment_1_thing.json", "results/experiment_1_renamed.json")
    moved = r / "results" / "experiment_1_renamed.json"
    d = json.loads(moved.read_text())
    d.pop("corrigendum_note")
    moved.write_text(json.dumps(d, indent=2) + "\n")
    v = dpl.check()
    assert len(v) == 1, v
    assert "corrigendum" in v[0]
    # The refusal must name BOTH paths or the reader cannot find the file that now holds it.
    assert "experiment_1_thing.json -> results/experiment_1_renamed.json" in v[0]


def test_a_rename_that_preserves_everything_is_allowed(repo):
    """Moving an artifact is legitimate work. The guard tracks content, not location."""
    r, _ = repo
    _git(r, "mv", "results/experiment_1_thing.json", "results/experiment_1_renamed.json")
    assert dpl.check() == []


def test_git_failure_refuses_rather_than_reporting_clean(repo, monkeypatch, tmp_path):
    """The fail-open that mattered most: `subprocess.run(...).stdout` with `returncode` ignored.

    Any git failure -- broken repo, missing binary, bad ref, index lock held by one of the
    concurrent workflows this repo runs -- produced an empty file list, which reads as "nothing
    changed" and printed OK. The tree could have lost every determination in it.
    """
    _, art = repo
    _write(art, flagged_adversarial=dpl)  # a REAL violation is present...
    # A directory that EXISTS but is not a git repository, so git actually runs and exits
    # non-zero. Pointing REPO at a NONEXISTENT path instead would make subprocess raise
    # OSError, which is the separate path covered by `test_missing_git_binary_refuses` -- it
    # would leave the `returncode != 0` branch untested. Mutation testing caught exactly that:
    # reverting the returncode check left the whole suite green.
    not_a_repo = tmp_path / "plain_directory"
    not_a_repo.mkdir()
    monkeypatch.setattr(dpl, "REPO", not_a_repo)
    with pytest.raises(dpl.GuardError):
        dpl.check()  # ...and the guard must say "I could not look", not "OK"

    # main() must translate that into a REFUSAL exit code, not an exception traceback that a
    # hook wrapper might swallow.
    assert dpl.main([]) == 1


def test_missing_git_binary_refuses(repo, monkeypatch):
    """OSError from subprocess is a separate path from a non-zero exit; both must fail closed."""

    def _boom(*a, **k):
        raise OSError("git: command not found")

    monkeypatch.setattr(dpl.subprocess, "run", _boom)
    with pytest.raises(dpl.GuardError):
        dpl.check()


def test_a_broken_adversarial_verify_import_refuses(repo, monkeypatch):
    """The `except Exception: return None` at the enum lookup was a fail-open, not defence.

    Swallowing it silently swapped rule 4's CANONICAL ENUM matcher for the weaker free-form
    prose scan -- the exact "drifted copy of a matcher" hazard `_canonical_substrate_band`'s own
    docstring is written to prevent. This cannot deadlock a repair of adversarial_verify.py
    itself: the pre-commit hook is scoped to `^results/.*\\.json$`.
    """
    import builtins

    real_import = builtins.__import__

    def _fail_on_av(name, *a, **k):
        if name == "adversarial_verify":
            raise ImportError("simulated: adversarial_verify is broken")
        return real_import(name, *a, **k)

    # Exercise the REAL function body. An earlier version of this test monkeypatched
    # `_adversarial_verify` itself, so the fail-closed code inside it was never executed --
    # mutation testing caught that by reverting the fix and watching the suite stay green.
    monkeypatch.setattr(dpl, "_AV_MODULE", None)
    monkeypatch.delitem(sys.modules, "adversarial_verify", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fail_on_av)

    with pytest.raises(dpl.GuardError):
        dpl._adversarial_verify()
    with pytest.raises(dpl.GuardError):
        dpl._canonical_substrate_band("live_llm_inference")


@pytest.mark.parametrize(
    "key,value",
    [
        # Every one of these key names is REAL and live in results/**.json today. Under the
        # pre-fix `any("cleared" in k and v)` test, each would have lifted a fabrication
        # quarantine while saying nothing whatever about it.
        ("cache_cleared", True),
        ("step1_vram_cleared", True),
        ("quota_gate_cleared", True),
        ("game_fully_cleared", True),
        ("zombie_already_cleared", True),
        ("drc_ioplanning_errors_cleared", ["a", "b"]),
        # A note-shaped NAME is still not enough if the value is not a written rationale.
        ("flagged_adversarial_cleared_note", True),
        ("flagged_adversarial_cleared_note", ""),
    ],
)
def test_an_unrelated_cleared_key_cannot_lift_a_quarantine(repo, key, value):
    """The clearing exemption required only a truthy key whose NAME contained "cleared"."""
    _, art = repo
    _write(art, flagged_adversarial=False, **{key: value})
    v = dpl.check()
    assert any("flagged_adversarial" in x and "cleared_note" in x for x in v), (
        f"{key}={value!r} must not satisfy the deliberate-clearing exemption; got {v}"
    )


def test_a_real_written_cleared_note_still_clears(repo):
    """The escape hatch must stay usable, or people route around the guard instead of using it."""
    _, art = repo
    _write(
        art,
        flagged_adversarial=False,
        flagged_adversarial_cleared_note=(
            "Cleared 2026-07-29: DURATION_TOO_SHORT was a false positive under the pre-2026-07-03"
            " substrate taxonomy; adversarial_verify.py now reports 0 flagged on this artifact."
        ),
    )
    assert dpl.check() == []


def test_load_at_refuses_on_a_bad_ref_rather_than_reporting_it_absent():
    """`_load_at` swallowed EVERY non-zero git exit as "the path is not in that revision".

    That conflated the routine case (a newly added artifact genuinely has no old side) with
    "your repository is broken" and "that ref does not exist". Under the old code a bad ref
    emptied the OLD side of every comparison, so nothing could ever be detected as lost and
    `--ref` audits of landed commits silently passed.
    """
    with pytest.raises(dpl.GuardError):
        dpl._load_at("definitely-not-a-ref-a1b2c3", "results/experiment_1_thing.json")


def test_a_genuinely_absent_path_at_a_ref_is_still_benign(repo):
    """The other side of the same coin: the routine case must NOT be turned into a refusal."""
    assert dpl._load_at("HEAD", "results/never_existed.json") is None


def test_the_hook_is_always_run_because_pre_commit_hides_deletions():
    """The SECOND half of the deletion hole, and it is not in this file's own source.

    `pre_commit.git.get_staged_files` builds its file list with `--diff-filter=ACMRTUXB`
    ("Everything except for D"), so a commit that only DELETES artifacts matches this hook's
    `files:` pattern zero times and pre-commit reports "(no files to check) Skipped" -- the lint
    is never invoked at all. Verified directly against a scratch repo: identical hook, identical
    staged deletion, `Skipped` without `always_run` and `Passed` with it.

    So fixing `--diff-filter=M` inside the lint was necessary but NOT sufficient; without this
    flag the fixed code never gets to run on the very case it was fixed for.
    """
    import yaml

    # PARSE THE YAML; DO NOT SUBSTRING-MATCH THE FILE. The first version of this test did
    # `assert "always_run: true" in block` -- which passed even with the KEY deleted, because
    # the phrase also occurs in the explanatory COMMENT directly above it. Mutation testing
    # caught it: removing the real key left this test green. That is precisely the
    # match-without-boundaries bug class CLAUDE.md's QA-Layer Authenticity Discipline exists
    # for, reproduced here in the test written to prevent it.
    cfg = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
    hooks = [h for r in cfg["repos"] for h in r.get("hooks", [])]
    hook = next(h for h in hooks if h["id"] == "determination-preservation-lint")
    assert hook.get("always_run") is True, (
        "without always_run, a deletion-only commit never invokes this hook at all"
    )


def test_a_curated_retirement_out_of_results_is_allowed(repo):
    """MUST-NOT-FIRE: moving an artifact to `legacy/fabricated/` preserves the record.

    FOUND BY CALIBRATING AGAINST REAL HISTORY, not by fixture-writing. Commit `bed0635b6`
    ("[outer-loop] Retire fabricated exp2823 TruthfulQA artifact to legacy/fabricated/") moved a
    `flagged_adversarial: True` artifact -- corrigendum TAUTOLOGY record and all -- out of
    results/ into legacy/fabricated/, with a README and an ops/exclusion_manifest.yaml entry.
    Nothing was lost; that IS the project's documented retirement path for a fabricated result.

    The first cut of the deletion fix refused it, because a `-- results` pathspec makes git
    report the source half of a cross-directory move as a plain deletion. Diffing the whole tree
    and filtering on the OLD path lets git pair it as R100 instead.
    """
    r, _ = repo
    (r / "legacy" / "fabricated").mkdir(parents=True)
    _git(r, "mv", "results/experiment_1_thing.json", "legacy/fabricated/experiment_1_thing.json")
    assert dpl.check() == [], "a documented retirement that preserves the record must pass"


def test_a_move_out_of_results_that_drops_a_marker_is_still_refused(repo):
    """The escape hatch above must not become a laundering route."""
    r, _ = repo
    (r / "legacy" / "fabricated").mkdir(parents=True)
    dest = "legacy/fabricated/experiment_1_thing.json"
    _git(r, "mv", "results/experiment_1_thing.json", dest)
    d = json.loads((r / dest).read_text())
    d.pop("corrigendum_note")
    (r / dest).write_text(json.dumps(d, indent=2) + "\n")
    v = dpl.check()
    assert len(v) == 1 and "corrigendum" in v[0], v


# =====================================================================================
# 2026-07-29, SECOND PASS. Three bypasses constructed against the file as it stood AFTER
# the widening + the delete/rename/fail-closed fixes -- i.e. against a version already
# believed clean. Each was demonstrated to score OK on real artifacts before being fixed,
# so each test below is a replay of a working bypass, not a hypothetical.
# =====================================================================================


def test_emptying_a_corrigendum_in_place_is_refused(repo):
    """Rule 2 compared key NAMES, so nulling the value kept the key and lost the record.

    `_corrigendum_keys` returned every name-matching key regardless of value, so
    `old_keys - new_keys` was empty when the key survived with its content gone. Confirmed
    against the real `experiment_1680_polarfire_smoke_v2.json`: setting `corrigendum_pending`
    to null scored a clean OK while destroying the TAUTOLOGY record that documents why the
    artifact is quarantined.
    """
    _, art = repo
    _write(art, corrigendum_pending=None)
    v = dpl.check()
    assert len(v) == 1, v
    assert "corrigendum" in v[0] and "corrigendum_pending" in v[0], v


def test_emptying_a_marker_in_place_is_refused(repo):
    """Rule 3's condition was a bare `key in new`, i.e. a pure name check.

    An emptied value cannot be caught by rule 4 either: `_strength_rank` returns None for a
    non-string, and None is deliberately read as "unrankable", never as a downgrade. So
    emptying was the one edit that defeated every rule at once.
    """
    _, art = repo
    _write(art, solve_provenance="live_agent_self_discovery")
    _git(repo[0], "add", "-A")
    _git(repo[0], "commit", "-q", "-m", "add provenance")
    _write(art, solve_provenance="")
    v = dpl.check()
    assert len(v) == 1, v
    assert "EMPTIED" in v[0] and "solve_provenance" in v[0], v


def test_an_emptied_marker_is_reported_as_emptied_not_as_dropped(repo):
    """The two are different repairs, so they must not share a message.

    A dropped field is restored by putting it back; an emptied one usually means a writer
    overwrote it with a variable that was unset. Telling them apart in the refusal text is
    what makes the message actionable.
    """
    _, art = repo
    _write(art, corrigendum_note="")
    v = dpl.check()
    assert len(v) == 1 and "EMPTIED" in v[0], v
    _write(art, corrigendum_note=dpl)  # sentinel: delete the key outright
    v2 = dpl.check()
    assert len(v2) == 1 and "EMPTIED" not in v2[0] and "lost" in v2[0], v2


def test_empty_to_empty_stays_silent(repo):
    """A field that never carried a record loses nothing, so it must not become noise.

    This is the boundary the substantive-only filter has to respect: the fix must catch
    substantive -> empty without also catching empty -> absent.
    """
    _, art = repo
    _write(art, corrigendum_note="")
    _git(repo[0], "add", "-A")
    _git(repo[0], "commit", "-q", "-m", "empty note")
    _write(art, corrigendum_note=dpl)  # delete the already-empty key
    assert dpl.check() == [], "an already-empty field carried no record to lose"


def test_an_unrelated_cleared_note_does_not_lift_a_quarantine(repo):
    """`_cleared_deliberately` accepted ANY key pairing "cleared" with "note".

    Confirmed bypass: `cache_cleared_note` -- a GPU-housekeeping remark -- cleared a
    fabrication determination. `cache_cleared` is a REAL key shape in this corpus.

    This is the same defect the sibling `_has_change_note` had already diagnosed and removed
    from itself, so before this fix the two functions disagreed about a rule they both
    implement.
    """
    _, art = repo
    _write(
        art,
        flagged_adversarial=False,
        cache_cleared_note="VRAM cache cleared between runs to avoid OOM",
    )
    v = dpl.check()
    assert any("flagged_adversarial" in x and "LIFTS a quarantine" in x for x in v), v


def test_a_one_word_cleared_note_is_not_a_rationale(repo):
    """The exemption exists so a human states what they re-verified; "ok" is not that.

    The 12-character floor matches `_has_change_note`'s, deliberately -- consistency between
    the two exemptions is worth more than either threshold on its own.
    """
    _, art = repo
    _write(art, flagged_adversarial=False, flagged_adversarial_cleared_note="ok")
    assert dpl.check() != [], "a token string must not clear a determination"


def test_the_documented_cleared_note_convention_still_passes(repo):
    """The tightening must not break the escape hatch the module docstring advertises."""
    _, art = repo
    _write(
        art,
        flagged_adversarial=False,
        flagged_adversarial_cleared_note=(
            "Cleared 2026-07-29: DURATION_TOO_SHORT was a false positive under the "
            "pre-2026-07-03 substrate taxonomy; re-ran adversarial_verify.py, 0 flagged."
        ),
    )
    assert dpl.check() == [], "the documented convention must keep working"


def test_a_strip_staged_but_hidden_in_the_working_tree_is_refused(repo):
    """The index is what a commit LANDS, and it was never examined.

    The default comparison is HEAD vs WORKING TREE, justified by pre-commit stashing unstaged
    changes. That justification was verified against a real installed hook and does hold --
    but it makes correctness depend on an external tool, and only under one driver. Running
    the script directly (the invocation this file's own USAGE section documents for auditing)
    on a tree where the index and the working tree disagree reported OK over a staged strip.
    """
    r, art = repo
    original = art.read_text()
    _write(art, flagged_adversarial=dpl, corrigendum_pending=dpl, corrigendum_note=dpl)
    _git(r, "add", "results/experiment_1_thing.json")  # stage the stripped copy
    art.write_text(original)  # working tree restored: looks innocent
    v = dpl.check()
    assert v, "a staged strip must be refused even when the working tree looks clean"
    assert any("STAGED CONTENT" in x for x in v), v


def test_a_violation_present_on_both_sides_is_reported_once(repo):
    """The union must dedupe, or every ordinary refusal would print twice."""
    _, art = repo
    _write(art, flagged_adversarial=dpl)
    _git(repo[0], "add", "-A")  # same content in the index and the working tree
    v = dpl.check()
    assert len([x for x in v if "LIFTS a quarantine" in x]) == 1, v
    assert not any("STAGED CONTENT" in x for x in v), v


def test_an_absent_index_path_reads_as_MISSING_not_as_a_guard_error(repo):
    """`git show :<path>` has its OWN absent-path spelling, and it is not the `<rev>:` one.

    Captured by running it rather than guessed:

        git show HEAD:nope.json  ->  fatal: path 'nope.json' does not exist in 'HEAD'
        git show :nope.json      ->  fatal: path 'nope.json' does not exist
                                            (neither on disk nor in the index)

    The second does NOT contain the substring "does not exist in", so before the index side
    existed the regex covered only the first spelling.

    Both outcomes happen to be fail-CLOSED (an unmatched stderr raises GuardError, which
    refuses), so this is about the ACCURACY of the refusal rather than its direction: a routine
    absent path should read as "the artifact is gone", not as "the guard could not run". The
    call is defensive today -- a `--cached` diff only ever names paths that are in the index --
    so it is asserted directly here rather than through `check()`, which is the honest way to
    test a branch that the normal call graph does not reach.
    """
    r, _ = repo
    assert dpl._load_index("results/experiment_1_thing.json") is not dpl.MISSING
    assert dpl._load_index("results/definitely_not_staged.json") is dpl.MISSING


def test_a_principle_wrapped_stamp_cannot_be_flipped_to_false_silently(repo):
    """Rule 1's `is True` was an IDENTITY check against the raw value.

    CLAUDE.md's Principle-Annotated Artifact Fields discipline permits ANY field to be written
    as ``{"principle": ..., "value": ...}``. A wrapped stamp is a dict, so `raw is True` is
    False, the OLD side was never recognised as flagged, and flipping it to a bare `false` with
    NO cleared-note scored a clean OK -- the quarantine lifted silently.

    This is origin bug #2 of the QA-Layer Authenticity Discipline reproduced inside the guard
    that cites it: `_unwrap_principle`'s own docstring names that bug and says "This lint must
    not repeat it".

    Not hypothetical shape-wise: the corpus carries 1,699 principle-wrapped top-level fields
    across 676 distinct names, including 44 artifacts that wrap `preconditions_checked`, which
    is itself one of this lint's protected markers.
    """
    _, art = repo
    _write(art, flagged_adversarial={"principle": "the gate keys off this", "value": True})
    _git(repo[0], "add", "-A")
    _git(repo[0], "commit", "-q", "-m", "wrap the stamp")
    _write(art, flagged_adversarial=False)
    v = dpl.check()
    assert any("LIFTS a quarantine" in x for x in v), v


def test_a_principle_wrapped_stamp_still_honours_a_real_cleared_note(repo):
    """Unwrapping must not break the documented escape hatch for wrapped stamps."""
    _, art = repo
    _write(art, flagged_adversarial={"principle": "the gate keys off this", "value": True})
    _git(repo[0], "add", "-A")
    _git(repo[0], "commit", "-q", "-m", "wrap the stamp")
    _write(
        art,
        flagged_adversarial=False,
        flagged_adversarial_cleared_note=(
            "Cleared 2026-07-29: re-ran adversarial_verify.py against this artifact, 0 flagged."
        ),
    )
    assert dpl.check() == [], "a wrapped stamp must still be clearable on purpose"


def test_a_principle_wrapped_stamp_counts_as_something_at_stake_on_deletion(repo):
    """`_protected_content` had the same raw `value is True` check.

    It decides whether deleting an artifact is this lint's business at all, so a wrapped stamp
    failing that test is how a flagged artifact gets deleted without objection.
    """
    r, art = repo
    _write(art, flagged_adversarial={"principle": "p", "value": True})
    for k in ("corrigendum_pending", "corrigendum_note"):
        _write(art, **{k: dpl})  # strip the other protections so ONLY the stamp is at stake
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "wrapped stamp only")
    assert dpl._protected_content(json.loads(art.read_text())) == ["flagged_adversarial"]
    _git(r, "rm", "-q", "results/experiment_1_thing.json")
    v = dpl.check()
    assert v and "DELETED" in v[0], v


def test_cleared_deliberately_never_reports_a_still_flagged_artifact_as_cleared(repo):
    """A wrapped stamp that is STILL True must not read as "deliberately cleared".

    Asserted directly rather than through `check()`, because rule 1 only consults this helper
    once it has decided the new side is not True -- so with rule 1 unwrapping, the wrapped-True
    case never reaches here today. That makes this a guard against a FUTURE caller, and the
    honest way to test an unreachable branch is to call it.

    Without the unwrap, `{"principle":..., "value": True} is True` is False, the early return
    is skipped, and the presence of any cleared-note makes the function answer "yes, cleared"
    about an artifact whose stamp is still set.
    """
    still_flagged = {
        "flagged_adversarial": {"principle": "the gate keys off this", "value": True},
        "flagged_adversarial_cleared_note": "a note left over from an earlier clearing attempt",
    }
    assert dpl._cleared_deliberately(still_flagged) is False

    genuinely_cleared = {
        "flagged_adversarial": False,
        "flagged_adversarial_cleared_note": "Cleared 2026-07-29: re-verified, 0 flagged.",
    }
    assert dpl._cleared_deliberately(genuinely_cleared) is True
