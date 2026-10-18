#!/usr/bin/env python3
"""REFUSE an LLM-on ARC row whose own instrumentation says the generator was dead.

WHY THIS EXISTS -- the 2026-07-26 concurrency fault, and why nothing stopped it.

`LocalGGUFProposer.generate()` / `.complete_text()`
(`python/carnot/agentic/arc_executable_world_model.py`) do NOT raise when the generator
server is unreachable, HTTP-500s, or dies mid-run. They `return False, "<diagnostic>"`.
Every caller on the live ARC path treats that False as "no induction available this
stall" and continues. So a run whose generator died at action 3 completes all 400 actions,
exits 0, and is written to `results/` as an LLM-on measurement -- while having behaved as
an LLM-OFF agent for 99% of its life. Measured shape (`results/llm_on_contention_rows_
20260726/`): at K=4 concurrency against a `total_slots=4`, `n_ctx=16384` server, all 8
cells came back `generator_healthy_after: False` with 0-2 completions, and all 8 still had
`worker_ok: True` and `worker_returncode: 0`.

The harness ALREADY computes the correct witness -- `llm_on_row_valid`
(`scripts/arc_scored_path_lever_harness.py`) ANDs `responses > 0` with
`generator_healthy_after` and `not server_storm_suspected`, and it was correctly `False` on
all 8. The defect is that NOTHING REFUSES on it. It is a field, not a gate: the cell
wrapper still reports success, the probe still exits 0, and each downstream analyser
re-decides for itself whether to filter. This lint is the missing gate.

WHAT IT CHECKS, per row that claims `llm_enabled: True`:

  DEAD_GENERATOR      `generator_healthy_after` is False -> the generator was not alive at
                      the end of a run that claims to have used it.
  NO_COMPLETIONS      `llm.responses == 0` -> the model produced nothing for this row, so
                      the row is not evidence about the LLM tier whatever else is true.
  CONTENT_ONLY_FAILURE
                      `llm.responses == 0` but `llm.errors == 0` and
                      `llm.content_failures > 0` with the generator alive -> the server
                      ANSWERED every call and nothing it produced was usable. Same
                      severity as NO_COMPLETIONS, different diagnosis: added 2026-07-27
                      because the witness deliberately keeps server failures and content
                      failures apart, and this lint -- its only consumer -- threw the
                      distinction away at the gate, reporting "the model produced nothing"
                      on a row whose own liveness channel said the generator was fine.
  MOSTLY_FAILED_CALLS the majority of `llm.calls` failed, even though some completions
                      landed -> the generator was unavailable for most of the row. Before
                      2026-07-27 any mix of successes and failures was only a WARN, so a
                      mid-game server death that another thread self-healed before teardown
                      (calls=20, responses=2, errors=18, healthy_after=True) exited 0.
  STAMP_FALSE_UNEXPLAINED
                      `llm_on_row_valid` is False while this lint recomputed no FAIL
                      condition -> the harness and the lint disagree about the same row.
                      The lint does not get to silently side with itself: a row the
                      producing harness declared invalid cannot pass the gate on the
                      strength of the checker finding nothing.
  SERVER_STORM        `server_storm_suspected` -> more llama-servers alive after than
                      before, i.e. a self-heal forked a second copy of the model onto the
                      card; the measured wall clock is not the intended configuration's.
  WITNESS_MISSING     the row carries no `generator_healthy_after` AND no
                      `llm_on_row_valid` -> the liveness channel is absent, so the row
                      cannot be audited for this failure class at all. Reported at
                      severity WARN, not FAIL: 130 such rows predate the instrumentation
                      and rewriting history is forbidden (never-prune).
  VALID_STAMP_WRONG   the row stamps `llm_on_row_valid: True` while carrying any of the
                      three FAIL conditions above -> the stamp itself is lying. This is the
                      check that survives the stamp being computed by a future,
                      differently-buggy harness: the lint recomputes from the primitives
                      rather than trusting the derived boolean.

DESIGN NOTE -- why this recomputes instead of reading `llm_on_row_valid`. A guard that
just asserts `llm_on_row_valid is not False` would be a guard that trusts the very
derivation it is supposed to police, and would go blind the moment a harness stops
emitting the field (WITNESS_MISSING) or emits it wrongly (VALID_STAMP_WRONG). The origin
lesson this project keeps re-learning is that a guard which reads a DERIVED value instead
of the underlying object does not fire on its own origin incident. So: primitives only.

WHERE THIS RUNS (wired 2026-07-27; it shipped with NO caller, which reproduced its own
documented defect one level up -- "NOTHING REFUSES on it. It is a field, not a gate"):

  * `.pre-commit-config.yaml:arc-llm-on-liveness-lint`, scoped to `results/*.json`,
    invoked as `--baseline` so enforcement is FORWARD-ONLY. The 55 FAIL findings already
    in the tree are real recordings of the 2026-07-26 fault and its probe arms -- the
    evidence the fix rests on -- and cannot be deleted (never-prune), so they are exempted
    by `ops/arc_llm_on_liveness_baseline.json` and anything NEW refuses the commit.
  * `scripts/research_conductor.py:_log_experiment_completion`, ADVISORY (a warning, not a
    status downgrade), so a freshly-landed deliverable is named in the conductor log at the
    moment it lands rather than at some later commit.

Usage:
    arc_llm_on_liveness_lint.py [PATH ...]        # default: results/
    arc_llm_on_liveness_lint.py --json            # machine-readable
    arc_llm_on_liveness_lint.py --self-test       # fire-on-the-origin-incident proof
    arc_llm_on_liveness_lint.py --baseline        # forward-only: refuse only NEW findings
    arc_llm_on_liveness_lint.py --write-baseline  # OPERATOR: record current FAILs as exempt

Exit: 0 clean, 1 FAIL findings present (or, with --baseline, NEW ones), 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Iterator

FAIL_CODES = (
    "DEAD_GENERATOR",
    "NO_COMPLETIONS",
    "CONTENT_ONLY_FAILURE",
    "MOSTLY_FAILED_CALLS",
    "SERVER_STORM",
    "VALID_STAMP_WRONG",
    "STAMP_FALSE_UNEXPLAINED",
    "WITNESS_SELF_CONTRADICTORY",
)
WARN_CODES = (
    "WITNESS_MISSING",
    "WITNESS_UNAVAILABLE",
    "LLM_TIER_NEVER_ENGAGED",
    "GENERATOR_PARTIAL_FAILURES",
)

# Fraction of a row's generator calls that may fail before the row stops being evidence about the
# LLM tier. Deliberately a MAJORITY threshold, not any-failure: a single transient failure among
# many successes is real degradation but the row still carries real induction, and firing FAIL on
# it would train the operator to ignore this lint. Above this the row spent most of its life
# LLM-off while reporting itself as the LLM-on scored path -- the origin defect exactly.
_MOSTLY_FAILED_FRACTION = 0.5


def _is_row(obj: Any) -> bool:
    """A row is a dict that makes an LLM-on claim, in EITHER of the two shapes this corpus uses.

    1. The instrumented shape: an explicit `llm_enabled` field (the 2026-07 harnesses).
    2. The PRE-INSTRUMENTATION shape: no `llm_enabled` at all, but a ROW-LEVEL assertion that
       the LLM induction tier ran -- either the `induction_attempts_llm_reached` aggregate, or
       an `induction_attempts` list on a per-game row (`game`/`actions` present). Recognising
       ONLY shape 1 would make this lint structurally blind to older rows whose LLM-on claims
       have no liveness witness at all, which is precisely the dead-channel failure mode the
       lint exists to surface (a check that cannot see the un-instrumented case reports a
       clean scan and reads as reassurance).

    PRECISION NOTE -- what is deliberately NOT a row. A first draft also keyed on any
    `model_specs` string that was not the `offline_dsl_induction_no_llm` sentinel. That
    matched the PER-ATTEMPT dicts nested inside `induction_attempts` (each attempt carries its
    own `model_specs`), inflating WITNESS_MISSING from 1 real ARC row to 233, of which 232 were
    sub-dicts that have no liveness witness BY DESIGN and are not measurements at all. An
    individual induction attempt is not a row; only its containing per-game record is. Keying
    on row-level aggregates keeps the check honest about its own scope.
    """
    if not isinstance(obj, dict):
        return False
    if "llm_enabled" in obj:
        return True
    if isinstance(obj.get("induction_attempts_llm_reached"), int):
        return True
    if "induction_attempts" in obj and ("game" in obj or "actions" in obj):
        return True
    return False


def _claims_llm_on(row: dict) -> bool:
    """True if this row asserts the LLM tier ran -- either flag-shaped or evidence-shaped.

    `llm_tier_operational` OVERRIDES `llm_enabled` when present (2026-07-27 review finding
    11). `llm_enabled` reads one env var and is True even for an arm whose installed
    proposer is a stub with no model behind it: all 100 LLM-OFF control rows of the
    2026-07-27 first-win measurement carry `llm_enabled: True`. Keying on it alone made this
    lint report "174 rows, 174 claiming llm_enabled=True" over a corpus in which 100 rows had
    no generator at all -- both a false-claim reading of an honest control arm AND, worse,
    the same misclassification (a stub arm read as an LLM-on arm) that this lint exists to
    refuse. `llm_tier_operational` is the STRUCTURAL fact -- a real, instrumented generator
    was installed -- so an explicit False here means the row is LLM-OFF and makes no claim.
    A dead-but-REAL generator is still operational=True, so it is still refused below."""
    operational = row.get("llm_tier_operational")
    if isinstance(operational, bool):
        if operational:
            return True
        # A False stamp is an ESCAPE HATCH unless it is checked against the primitives. This
        # lint's own design note says a guard that trusts a DERIVED value goes blind the
        # moment that value is wrong; honouring `llm_tier_operational: false` unconditionally
        # would reintroduce exactly that. A row that claims no LLM tier while its own
        # counters record real generator traffic is SELF-CONTRADICTORY, and the primitives
        # win: it is judged as an LLM-on row.
        llm = row.get("llm")
        calls = (llm or {}).get("calls") if isinstance(llm, dict) else None
        responses = (llm or {}).get("responses") if isinstance(llm, dict) else None
        if (isinstance(calls, int) and calls > 0) or (isinstance(responses, int) and responses > 0):
            return True
        return False
    if "llm_enabled" in row:
        return bool(row.get("llm_enabled"))
    return True  # only reached for shape-2 rows, which are LLM-on by construction


def walk_rows(obj: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (json_path, row) for every llm_enabled-bearing dict, at any nesting depth."""
    if isinstance(obj, dict):
        if _is_row(obj):
            yield path or "$", obj
        for key, value in obj.items():
            yield from walk_rows(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from walk_rows(value, f"{path}[{index}]")


def check_row(row: dict) -> list[dict]:
    """Recompute the liveness verdict from PRIMITIVES. Returns a list of findings."""
    if not _claims_llm_on(row):
        return []  # an LLM-off row makes no claim about the generator

    findings: list[dict] = []
    healthy_after = row.get("generator_healthy_after")
    stamp = row.get("llm_on_row_valid")
    llm = row.get("llm")
    responses = (llm or {}).get("responses") if isinstance(llm, dict) else None
    storm = row.get("server_storm_suspected")
    # `calls` / `errors` come from the 2026-07-27 scored-path witness
    # (LocalGGUFProposer.liveness_witness()). They are ABSENT on every pre-existing row,
    # and every branch below is written so that absence leaves the old verdict EXACTLY as
    # it was -- the eight recorded origin cells have no `calls` key and must keep failing.
    calls = (llm or {}).get("calls") if isinstance(llm, dict) else None
    errors = (llm or {}).get("errors") if isinstance(llm, dict) else None
    # `content_failures` is the OTHER half of the server-vs-content split the 2026-07-27 witness
    # deliberately introduced -- and until 2026-07-27 this lint, its ONLY consumer, never read it,
    # which threw the whole distinction away at the gate. See the CONTENT_ONLY_FAILURE branch.
    content_failures = (llm or {}).get("content_failures") if isinstance(llm, dict) else None
    never_engaged = calls == 0  # explicit zero only; None (absent) is NOT "never engaged"
    # THE WITNESS WAS ATTEMPTED AND COULD NOT BE TAKEN. Distinct from both "absent" (an old row
    # that predates the instrumentation -> WITNESS_MISSING) and "measured zero" (a real reading).
    # The 2026-07-27 first-win LLM-OFF control arm installs a `_NoOpProposer` with no
    # `liveness_witness()` method at all and records `calls/responses/errors = -1` plus an
    # explicit `liveness_witness_error`. A NEGATIVE count is a not-measured sentinel, never a
    # measurement: reading it as one made `responses=-1` compare as "not zero" (so no
    # NO_COMPLETIONS) while `llm_on_row_valid=False` had no recomputed explanation, which fired
    # STAMP_FALSE_UNEXPLAINED on all 100 honest control cells. An unauditable row is a WARN.
    sentinel = any(isinstance(v, int) and v < 0 for v in (calls, responses, errors))
    witness_error = row.get("liveness_witness_error")
    if sentinel or witness_error:
        return [
            {
                "code": "WITNESS_UNAVAILABLE",
                "severity": "WARN",
                "detail": "the liveness witness was attempted and could not be taken "
                f"(liveness_witness_error={witness_error!r}, llm={llm!r}) -- negative counts are "
                "a not-measured sentinel, not a measurement, so this row cannot be audited for "
                "generator liveness either way",
            }
        ]

    # WITNESS_MISSING: no liveness channel at all -> unauditable, not provably wrong.
    if healthy_after is None and "llm_on_row_valid" not in row:
        return [
            {
                "code": "WITNESS_MISSING",
                "severity": "WARN",
                "detail": "row claims llm_enabled=True but carries neither "
                "generator_healthy_after nor llm_on_row_valid",
            }
        ]

    if healthy_after is False:
        findings.append(
            {
                "code": "DEAD_GENERATOR",
                "severity": "FAIL",
                "detail": f"generator_healthy_after=False (responses={responses})",
            }
        )
    if responses == 0 and not never_engaged:
        # WHICH KIND of nothing? The witness records server failures and content failures
        # separately precisely because the prescriptions are different, and NO_COMPLETIONS'
        # message ("the model produced nothing") actively misattributes the second kind to the
        # first. Before 2026-07-27 an alive server answering every call with unusable content
        # (calls=1, responses=0, errors=0, content_failures=1, healthy_after=True) was reported
        # as NO_COMPLETIONS -- blaming liveness for a content problem, on a row whose own
        # liveness channel says the generator was fine. Same severity, correct diagnosis.
        content_only = (
            isinstance(content_failures, int)
            and content_failures > 0
            and (errors == 0 or errors is None)
            and healthy_after is not False
        )
        if content_only:
            findings.append(
                {
                    "code": "CONTENT_ONLY_FAILURE",
                    "severity": "FAIL",
                    "detail": f"llm.responses=0 with llm.errors={errors} and "
                    f"llm.content_failures={content_failures} -- the generator was ALIVE and "
                    "answered, but nothing it produced was usable, so this row is not evidence "
                    "about the LLM tier (do NOT read this as a dead server)",
                }
            )
        else:
            findings.append(
                {
                    "code": "NO_COMPLETIONS",
                    "severity": "FAIL",
                    "detail": "llm.responses=0 -- the model produced nothing for this row",
                }
            )
    elif responses == 0 and never_engaged:
        # "produced nothing because it was never asked" is NOT the same defect as "produced
        # nothing though asked". A scored game that never stalled into the induction tier
        # legitimately makes zero generator calls; failing it would be an over-fire that
        # trains the operator to ignore this lint. The row is still not EVIDENCE about the
        # LLM tier, so it is reported -- at WARN.
        findings.append(
            {
                "code": "LLM_TIER_NEVER_ENGAGED",
                "severity": "WARN",
                "detail": "llm.calls=0 -- the induction tier was never reached, so this row "
                "is not evidence about the LLM tier either way",
            }
        )
    if never_engaged and isinstance(errors, int) and errors > 0:
        # A row cannot record server failures it never made calls to produce. This is the
        # anti-gaming branch for the new `calls` field: zeroing `calls` to dodge
        # NO_COMPLETIONS now trips a FAIL of its own instead.
        findings.append(
            {
                "code": "WITNESS_SELF_CONTRADICTORY",
                "severity": "FAIL",
                "detail": f"llm.calls=0 but llm.errors={errors} -- a row with no calls "
                "cannot have recorded server failures",
            }
        )
    if isinstance(errors, int) and errors > 0 and isinstance(responses, int) and responses > 0:
        # DEGRADED-BUT-NOT-DEAD is a spectrum, and reporting all of it at WARN was an escape
        # hatch (found 2026-07-27): a mid-game server death that another thread self-healed --
        # calls=20, responses=2, errors=18, healthy_after=True because the restart landed before
        # teardown -- produced ONLY this WARN, so the gate exited 0 on a row that spent 90% of
        # its life LLM-off. That is the origin defect wearing a healthy stamp. Above a majority
        # failure fraction the row stops being evidence about the LLM tier and this is a FAIL.
        total = errors + responses
        if isinstance(calls, int) and calls > 0:
            total = calls
        frac = errors / float(total) if total else 0.0
        if frac > _MOSTLY_FAILED_FRACTION:
            findings.append(
                {
                    "code": "MOSTLY_FAILED_CALLS",
                    "severity": "FAIL",
                    "detail": f"llm.errors={errors} of {total} calls ({frac:.0%}) failed -- the "
                    "generator was unavailable for the majority of this row, so the row is not "
                    "evidence about the LLM tier even though it recorded some completions",
                }
            )
        else:
            findings.append(
                {
                    "code": "GENERATOR_PARTIAL_FAILURES",
                    "severity": "WARN",
                    "detail": f"llm.errors={errors} alongside llm.responses={responses} -- the "
                    "generator answered some calls and failed others, so this row is partially "
                    "degraded (real completions, but fewer than the run attempted)",
                }
            )
    if storm is True:
        findings.append(
            {
                "code": "SERVER_STORM",
                "severity": "FAIL",
                "detail": f"server_storm_suspected=True "
                f"({row.get('llama_servers_before')}->{row.get('llama_servers_after')} servers)",
            }
        )
    # Only a FAIL condition makes the stamp a LIE. (Before the WARN codes existed every
    # finding here was a FAIL, so filtering on severity leaves pre-existing rows' verdicts
    # bit-identical -- it is the WARN codes added 2026-07-27 that must not be able to
    # manufacture a VALID_STAMP_WRONG on an otherwise-honest row.)
    hard = [f for f in findings if f["severity"] == "FAIL"]
    if stamp is True and hard:
        findings.append(
            {
                "code": "VALID_STAMP_WRONG",
                "severity": "FAIL",
                "detail": "llm_on_row_valid=True while "
                + ",".join(f["code"] for f in hard)
                + " hold",
            }
        )
    if stamp is False and not findings:
        # THE OTHER DIRECTION, and the one that lets a bad row through the GATE rather than just
        # mislabelling it (added 2026-07-27). The harness's own derivation says this row is not
        # a valid LLM-on measurement, and this lint -- recomputing from primitives -- found
        # nothing that FAILS. Exactly one of the two is wrong, and the lint cannot tell which
        # from the row alone, so it must not silently side with itself and exit 0. Measured
        # shape that hits this: a mid-game server death another thread self-healed before
        # teardown, leaving healthy_after=True (so no DEAD_GENERATOR) with real completions on
        # both sides of it (so, before MOSTLY_FAILED_CALLS existed, only a WARN) while
        # `llm_on_row_valid` was correctly False.
        #
        # NOTE the deliberate asymmetry with the DESIGN NOTE above: this does NOT trust the
        # derived stamp as a verdict, it treats a DISAGREEMENT between the stamp and the
        # recomputation as the finding. A lint that can only fire when it already agrees with
        # the derived value is not an independent check.
        # UNAUDITABLE ROWS ARE A WARN, NOT A FAIL (added 2026-08-04).
        #
        # This branch is a DISAGREEMENT detector: the harness stamped the row invalid and the
        # recomputation found nothing failing, so one of them is wrong. That reasoning only holds
        # when there were primitives to recompute FROM. A row with no `llm` block carries no
        # call-level primitives at all, so the lint has not disagreed with the harness -- it has
        # simply been handed nothing to check, and calling that a FAIL blames the row for the
        # lint's own blind spot.
        #
        # The measured shape that hits this: `walk_rows` descends into SUMMARY SUB-OBJECTS such as
        # `raw_row_summary`, which holds {generator_healthy_after, induction_attempts_llm_reached,
        # llm_on_row_valid} and nothing else. Its PARENT row carries the actual reason for the
        # False stamp -- `generator_valid: False`, `terminal_state: "generator_invalid"` -- but
        # check_row only ever sees the child. exp5972's budget=1 smoke cell is an honest row that
        # correctly recorded an invalid generator and was failed for it, blocking every commit in
        # the repo behind a forward-only gate.
        #
        # This module's own DESIGN NOTE already states the principle ("An unauditable row is a
        # WARN"); this path simply did not implement it.
        if not isinstance(llm, dict):
            findings.append(
                {
                    "code": "STAMP_FALSE_UNAUDITABLE",
                    "severity": "WARN",
                    "detail": "llm_on_row_valid=False and this row carries no `llm` block, so "
                    "there are no call-level primitives to recompute from -- the stamp cannot be "
                    "confirmed or contradicted here. Check the parent row for the recorded reason "
                    "(e.g. generator_valid / terminal_state).",
                }
            )
            return findings
        findings.append(
            {
                "code": "STAMP_FALSE_UNEXPLAINED",
                "severity": "FAIL",
                "detail": "llm_on_row_valid=False but no FAIL condition was recomputed from the "
                f"primitives (responses={responses}, errors={errors}, "
                f"content_failures={content_failures}, healthy_after={healthy_after}, "
                f"storm={storm}) -- the harness and this lint disagree about the same row, so "
                "one of them is wrong and the row cannot be counted as evidence until that is "
                "resolved",
            }
        )
    return findings


def scan_paths(paths: list[str]) -> dict:
    """Scan JSON files under `paths` (files or dirs) and return a structured report."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)))
        else:
            files.append(p)

    findings: list[dict] = []
    n_rows = 0
    n_llm_on = 0
    unreadable: list[str] = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            unreadable.append(f)
            continue
        for jpath, row in walk_rows(data):
            n_rows += 1
            if not _claims_llm_on(row):
                continue
            n_llm_on += 1
            for finding in check_row(row):
                findings.append(
                    {
                        **finding,
                        "file": f,
                        "json_path": jpath,
                        "game": row.get("game"),
                        "llm_on_row_valid": row.get("llm_on_row_valid"),
                    }
                )
    return {
        "files_scanned": len(files),
        "files_unreadable": len(unreadable),
        "rows_seen": n_rows,
        "rows_llm_on": n_llm_on,
        "findings": findings,
        "n_fail": sum(1 for x in findings if x["severity"] == "FAIL"),
        "n_warn": sum(1 for x in findings if x["severity"] == "WARN"),
    }


# --------------------------------------------------------------------------------------
# FORWARD-ONLY ENFORCEMENT.
#
# The whole point of this lint is to be a GATE, and a gate nothing calls is not a gate --
# it is the same "it is a field, not a gate" defect the docstring above was written to
# fix, one level up. But `--warn-only` is not enforcement either: it can never refuse
# anything, so it decays into another channel nobody reads.
#
# What blocked wiring it is that the EXISTING corpus fails: 55 FAIL findings across 25
# files, all of them real recordings of the 2026-07-26 concurrency fault and its probes.
# Deleting or rewriting them is forbidden (never-prune) and would destroy the evidence the
# fix rests on. So enforcement is forward-only, the same shape every sibling discipline in
# CLAUDE.md uses: a dated baseline manifest records the findings that already existed, and
# the gate refuses anything NOT in it.
#
# The baseline keys on (file, json_path, code) -- deliberately NOT on the detail string, so
# that a recorded dead cell staying dead is exempt but the SAME row acquiring a NEW failure
# class is not. Regenerate with --write-baseline, which is an operator action: every entry
# added is a row this gate will no longer refuse, so the diff is the audit.
# --------------------------------------------------------------------------------------

DEFAULT_BASELINE = os.path.join("ops", "arc_llm_on_liveness_baseline.json")


def _finding_key(f: dict) -> str:
    return f"{f['file']}::{f['json_path']}::{f['code']}"


def apply_baseline(report: dict, baseline_path: str) -> dict:
    """Split FAIL findings into pre-existing (exempt) and NEW (blocking).

    A missing baseline file is NOT an error and NOT a free pass: every FAIL stays blocking,
    which is the safe direction (a gate whose exemption list vanished must refuse, not admit).
    """
    known: set[str] = set()
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path) as fh:
                data = json.load(fh)
            known = {str(k) for k in (data.get("known_failing") or [])}
        except Exception:
            known = set()
    new_fail = []
    grandfathered = []
    for f in report["findings"]:
        if f["severity"] != "FAIL":
            continue
        (grandfathered if _finding_key(f) in known else new_fail).append(f)
    report["baseline_path"] = baseline_path
    report["baseline_present"] = os.path.exists(baseline_path)
    report["baseline_entries"] = len(known)
    report["n_fail_grandfathered"] = len(grandfathered)
    report["n_fail_new"] = len(new_fail)
    report["new_fail_findings"] = new_fail
    return report


# --------------------------------------------------------------------------------------
# THE FIRE-ON-THE-ORIGIN-INCIDENT PROOF.
#
# A guard is not real until it fires on the recorded failure it was written for. This
# project has shipped a lint that printed OK on a faithful replay of its own origin
# incident. So the self-test does not use synthetic fixtures: it replays the ACTUAL eight
# K=4 cells (verbatim primitives, read off
# results/llm_on_contention_rows_20260726/ladder_b400_seed*.json on 2026-07-26) and the
# ACTUAL matched-config controls that must NOT fire, then mutation-proves each check by
# flipping exactly one primitive.
# --------------------------------------------------------------------------------------

# The 8 dead cells: (game, seed, responses, generator_healthy_after, llm_on_row_valid)
ORIGIN_DEAD_CELLS = [
    ("dc22", 20260724, 1, False, False),
    ("ft09", 20260724, 0, False, False),
    ("sc25", 20260724, 1, False, False),
    ("su15", 20260724, 0, False, False),
    ("dc22", 20260725, 1, False, False),
    ("ft09", 20260725, 0, False, False),
    ("sc25", 20260725, 1, False, False),
    ("su15", 20260725, 2, False, False),
]

# The matched-config control: same K=4, same games, n_ctx=32768 -> all four alive.
ORIGIN_CONTROL_CELLS = [
    ("dc22", 5, True, True),
    ("ft09", 10, True, True),
    ("sc25", 6, True, True),
    ("su15", 6, True, True),
]


def _row(responses: int, healthy_after: bool, stamp: bool, **kw) -> dict:
    row = {
        "llm_enabled": True,
        "ran": True,
        "actions": 396,
        "levels": 0,
        "llm": {"responses": responses, "errors": 0},
        "generator_healthy_before": True,
        "generator_healthy_after": healthy_after,
        "server_storm_suspected": False,
        "llm_on_row_valid": stamp,
    }
    row.update(kw)
    return row


def self_test() -> int:
    failures: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # 1. FIRES ON THE ORIGIN INCIDENT: all 8 recorded K=4 cells must produce a FAIL.
    for game, seed, responses, healthy, stamp in ORIGIN_DEAD_CELLS:
        got = check_row(_row(responses, healthy, stamp, game=game, seed=seed))
        codes = {f["code"] for f in got}
        expect(
            any(f["severity"] == "FAIL" for f in got),
            f"ORIGIN CELL NOT CAUGHT: K=4 {game} seed={seed} -> {codes or 'NO FINDINGS'}",
        )
        expect(
            "DEAD_GENERATOR" in codes,
            f"ORIGIN CELL {game}/{seed}: expected DEAD_GENERATOR, got {codes}",
        )
        if responses == 0:
            expect(
                "NO_COMPLETIONS" in codes,
                f"ORIGIN CELL {game}/{seed}: responses=0 but no NO_COMPLETIONS, got {codes}",
            )

    # 2. DOES NOT OVER-FIRE: the matched-config n_ctx=32768 controls must be clean.
    for game, responses, healthy, stamp in ORIGIN_CONTROL_CELLS:
        got = check_row(_row(responses, healthy, stamp, game=game))
        expect(
            not got,
            f"CONTROL CELL FALSE POSITIVE: K=4/n_ctx=32768 {game} -> {[f['code'] for f in got]}",
        )

    # 3. MUTATION PROOFS -- flip exactly one primitive, the verdict must flip with it.
    base = _row(6, True, True)
    expect(not check_row(base), "mutation base is not clean")

    m = dict(base, generator_healthy_after=False)
    expect(
        {f["code"] for f in check_row(m)} == {"DEAD_GENERATOR", "VALID_STAMP_WRONG"},
        "mutation: healthy_after False did not raise DEAD_GENERATOR+VALID_STAMP_WRONG",
    )

    m = dict(base, llm={"responses": 0, "errors": 0})
    expect(
        {f["code"] for f in check_row(m)} == {"NO_COMPLETIONS", "VALID_STAMP_WRONG"},
        "mutation: responses=0 did not raise NO_COMPLETIONS+VALID_STAMP_WRONG",
    )

    m = dict(base, server_storm_suspected=True)
    expect(
        {f["code"] for f in check_row(m)} == {"SERVER_STORM", "VALID_STAMP_WRONG"},
        "mutation: storm did not raise SERVER_STORM+VALID_STAMP_WRONG",
    )

    # 4. THE STAMP IS NOT TRUSTED. A row whose harness stamped it valid while dead must
    #    still FAIL -- this is the check that survives a future harness's own bug.
    m = dict(base, generator_healthy_after=False, llm_on_row_valid=True)
    expect(
        "VALID_STAMP_WRONG" in {f["code"] for f in check_row(m)},
        "mutation: a lying llm_on_row_valid=True stamp was believed",
    )
    #    ...and the same row with the stamp REMOVED entirely must still FAIL on the
    #    primitive, i.e. the lint does not depend on the derived field existing.
    m = dict(base, generator_healthy_after=False)
    m.pop("llm_on_row_valid")
    expect(
        {f["code"] for f in check_row(m)} == {"DEAD_GENERATOR"},
        "mutation: dropping llm_on_row_valid blinded the primitive check",
    )

    # 5. AN LLM-OFF ROW MAKES NO CLAIM: must never fire, even with a dead generator.
    m = dict(base, llm_enabled=False, generator_healthy_after=False)
    expect(not check_row(m), "an llm_enabled=False row must never be flagged")

    # 6. WITNESS_MISSING is WARN, not FAIL (130 pre-instrumentation rows; never-prune).
    got = check_row({"llm_enabled": True, "game": "old", "actions": 400})
    expect(
        [f["code"] for f in got] == ["WITNESS_MISSING"] and got[0]["severity"] == "WARN",
        f"uninstrumented row: expected one WARN WITNESS_MISSING, got {got}",
    )

    # 7. THE WALKER MUST REACH A NESTED ROW. The recorded cells live at cells[i].row --
    #    a walker that only looked at the top level would report a clean scan on the
    #    origin artifact. (This is the exact shape of the shipped lint that read the
    #    wrong location and printed OK on its own incident.)
    nested = {"probe": "x", "cells": [{"worker_ok": True, "row": _row(0, False, False)}]}
    found = list(walk_rows(nested))
    expect(len(found) == 1, f"walker missed the nested row: {found}")
    expect(
        any(f["severity"] == "FAIL" for f in check_row(found[0][1])),
        "walker found the nested row but the check did not fire on it",
    )

    # 8. THE NEW `calls` FIELD MUST NOT WEAKEN THE ORIGIN VERDICT. Re-run every origin cell
    #    with an EXPLICIT calls count equal to its response count (the shape the scored
    #    witness now emits): a dead generator that answered 0-2 calls must still FAIL.
    for game, seed, responses, healthy, stamp in ORIGIN_DEAD_CELLS:
        row = _row(responses, healthy, stamp, game=game, seed=seed)
        row["llm"] = {"calls": max(1, responses), "responses": responses, "errors": 1}
        codes = {f["code"] for f in check_row(row)}
        expect(
            "DEAD_GENERATOR" in codes,
            f"ORIGIN CELL {game}/{seed} in the NEW witness shape lost DEAD_GENERATOR: {codes}",
        )

    # 9. calls == 0 -> WARN, not FAIL. "never asked" is not "asked and got nothing".
    got = check_row(
        {
            "llm_enabled": True,
            "game": "never_stalled",
            "actions": 12,
            "llm": {"calls": 0, "responses": 0, "errors": 0},
            "generator_healthy_after": None,
            "llm_on_row_valid": False,
        }
    )
    expect(
        [f["code"] for f in got] == ["LLM_TIER_NEVER_ENGAGED"] and got[0]["severity"] == "WARN",
        f"calls=0 row: expected one WARN LLM_TIER_NEVER_ENGAGED, got {got}",
    )

    # 10. ...and the DODGE is closed: zeroing calls while recording errors is a FAIL.
    got = check_row(
        {
            "llm_enabled": True,
            "game": "liar",
            "actions": 400,
            "llm": {"calls": 0, "responses": 0, "errors": 7},
            "generator_healthy_after": True,
            "llm_on_row_valid": True,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        "WITNESS_SELF_CONTRADICTORY" in codes and "VALID_STAMP_WRONG" in codes,
        f"calls=0/errors=7 row: expected WITNESS_SELF_CONTRADICTORY+VALID_STAMP_WRONG, got {codes}",
    )

    # 11. ASKED AND GOT NOTHING is still a FAIL (this is the eval-concurrency shape).
    got = check_row(
        {
            "llm_enabled": True,
            "game": "asked",
            "actions": 400,
            "llm": {"calls": 9, "responses": 0, "errors": 9},
            "generator_healthy_after": False,
            "llm_on_row_valid": False,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        {"DEAD_GENERATOR", "NO_COMPLETIONS"} <= codes,
        f"calls=9/responses=0 row: expected DEAD_GENERATOR+NO_COMPLETIONS, got {codes}",
    )

    # 12. A WARN ALONE MUST NOT MANUFACTURE VALID_STAMP_WRONG on an honest row.
    got = check_row(
        {
            "llm_enabled": True,
            "game": "partial",
            "actions": 400,
            "llm": {"calls": 10, "responses": 8, "errors": 2},
            "generator_healthy_after": True,
            "llm_on_row_valid": True,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        codes == {"GENERATOR_PARTIAL_FAILURES"},
        f"partial-failure row: expected only the WARN, got {codes}",
    )

    # 13. AN ALIVE SERVER ANSWERING UNUSABLY is a content problem, not a liveness one. Before
    # 2026-07-27 this row reported NO_COMPLETIONS -- "the model produced nothing" -- blaming
    # liveness on a row whose own liveness channel says the generator was fine, and throwing
    # away the server/content split the witness deliberately records.
    got = check_row(
        {
            "llm_enabled": True,
            "game": "terse",
            "actions": 400,
            "llm": {"calls": 1, "responses": 0, "errors": 0, "content_failures": 1},
            "generator_healthy_after": True,
            "llm_on_row_valid": False,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        "CONTENT_ONLY_FAILURE" in codes and "NO_COMPLETIONS" not in codes,
        f"content-only row: expected CONTENT_ONLY_FAILURE and NOT NO_COMPLETIONS, got {codes}",
    )
    expect(
        all(f["severity"] == "FAIL" for f in got if f["code"] == "CONTENT_ONLY_FAILURE"),
        "CONTENT_ONLY_FAILURE must be a FAIL: the row is still not evidence about the LLM tier",
    )

    # 14. A MID-GAME DEATH THAT ANOTHER THREAD SELF-HEALED. healthy_after=True (the restart
    # landed before teardown) and responses>0, so before 2026-07-27 this produced ONLY a WARN
    # and the gate exited 0 -- on a row that spent 90% of its life LLM-off, which is the origin
    # defect wearing a healthy stamp.
    got = check_row(
        {
            "llm_enabled": True,
            "game": "healed",
            "actions": 400,
            "llm": {"calls": 20, "responses": 2, "errors": 18},
            "generator_healthy_after": True,
            "llm_on_row_valid": False,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        "MOSTLY_FAILED_CALLS" in codes,
        f"90%-failed row: expected MOSTLY_FAILED_CALLS (FAIL), got {codes}",
    )
    expect(
        any(f["severity"] == "FAIL" for f in got),
        "a row whose generator was unavailable for the majority of the run must not pass",
    )

    # 15. THE HARNESS SAYS INVALID AND THE LINT FINDS NOTHING -> the two disagree, and the lint
    # does not get to silently side with itself and exit 0.
    got = check_row(
        {
            "llm_enabled": True,
            "game": "disputed",
            "actions": 400,
            "llm": {"calls": 5, "responses": 5, "errors": 0, "content_failures": 0},
            "generator_healthy_after": True,
            "llm_on_row_valid": False,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        codes == {"STAMP_FALSE_UNEXPLAINED"},
        f"disputed row: expected exactly STAMP_FALSE_UNEXPLAINED, got {codes}",
    )

    # 16. ...but a False stamp the lint ALREADY EXPLAINS is not a disagreement. A game that
    # never stalled into the induction tier legitimately stamps False; firing here would be an
    # over-fire on every honest never-engaged row (it fired on exactly that row when this check
    # was first written, which is how the scope was found).
    got = check_row(
        {
            "llm_enabled": True,
            "game": "never_stalled_stamped",
            "actions": 12,
            "llm": {"calls": 0, "responses": 0, "errors": 0},
            "generator_healthy_after": True,
            "llm_on_row_valid": False,
        }
    )
    codes = {f["code"] for f in got}
    expect(
        codes == {"LLM_TIER_NEVER_ENGAGED"},
        f"explained-False-stamp row: expected only the WARN, got {codes}",
    )

    for msg in failures:
        print(f"SELF-TEST FAIL: {msg}")
    if failures:
        print(f"\n{len(failures)} self-test failure(s) -- THE GUARD IS NOT TRUSTWORTHY")
        return 1
    print(
        f"SELF-TEST OK: fired on all {len(ORIGIN_DEAD_CELLS)} recorded origin cells (in BOTH "
        f"the original and the 2026-07-27 scored-witness shape), clean on all "
        f"{len(ORIGIN_CONTROL_CELLS)} matched-config controls, 16 mutation proofs passed."
    )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", default=None, help="files or dirs (default: results/)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--self-test", action="store_true", help="fire-on-origin-incident proof")
    ap.add_argument(
        "--warn-only", action="store_true", help="report but always exit 0 (audit mode)"
    )
    ap.add_argument(
        "--baseline",
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
        help="forward-only mode: exempt FAILs recorded in this manifest, refuse only NEW ones "
        f"(default path when given bare: {DEFAULT_BASELINE})",
    )
    ap.add_argument(
        "--write-baseline",
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
        help="OPERATOR ACTION: record every current FAIL as pre-existing and exit 0",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.write_baseline:
        report = scan_paths(args.paths or ["results"])
        fails = [f for f in report["findings"] if f["severity"] == "FAIL"]
        payload = {
            "generated_utc": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(timespec="seconds"),
            "why": "Forward-only enforcement baseline. Every key here is an LLM-on row this "
            "gate will NOT refuse, because it was already in the tree when the gate was "
            "wired. These are real recordings of the 2026-07-26 generator concurrency fault "
            "and its probe arms -- evidence the fix rests on, preserved per never-prune. "
            "Adding an entry is an operator decision: it exempts a row from the gate.",
            "n_known_failing": len(fails),
            "known_failing": sorted(_finding_key(f) for f in fails),
        }
        os.makedirs(os.path.dirname(args.write_baseline) or ".", exist_ok=True)
        with open(args.write_baseline, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.write_baseline}: {len(fails)} pre-existing FAIL findings exempted")
        return 0

    report = scan_paths(args.paths or ["results"])
    if args.baseline:
        report = apply_baseline(report, args.baseline)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"scanned {report['files_scanned']} files "
            f"({report['files_unreadable']} unreadable), "
            f"{report['rows_seen']} rows, {report['rows_llm_on']} claiming llm_enabled=True"
        )
        shown = report.get("new_fail_findings") if args.baseline else None
        for f in shown if shown is not None else report["findings"]:
            if f["severity"] == "FAIL":
                print(
                    f"  FAIL {f['code']:18} {f.get('game') or '?':6} "
                    f"{f['file']}{f['json_path']}: {f['detail']}"
                )
        n_warn_files = len({f["file"] for f in report["findings"] if f["severity"] == "WARN"})
        print(
            f"\n{report['n_fail']} FAIL, {report['n_warn']} WARN "
            f"(WARN = uninstrumented rows across {n_warn_files} files)"
        )
        if args.baseline:
            print(
                f"forward-only vs {report['baseline_path']} "
                f"(present={report['baseline_present']}, {report['baseline_entries']} exempt): "
                f"{report['n_fail_grandfathered']} pre-existing, "
                f"{report['n_fail_new']} NEW -- only NEW findings refuse"
            )
    blocking = report["n_fail_new"] if args.baseline else report["n_fail"]
    return 0 if (args.warn_only or blocking == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
