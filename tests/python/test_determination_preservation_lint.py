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
