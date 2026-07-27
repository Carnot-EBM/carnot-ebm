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

Usage:
    arc_llm_on_liveness_lint.py [PATH ...]        # default: results/
    arc_llm_on_liveness_lint.py --json            # machine-readable
    arc_llm_on_liveness_lint.py --self-test       # fire-on-the-origin-incident proof

Exit: 0 clean, 1 FAIL findings present, 2 usage/IO error.
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
    "SERVER_STORM",
    "VALID_STAMP_WRONG",
    "WITNESS_SELF_CONTRADICTORY",
)
WARN_CODES = ("WITNESS_MISSING", "LLM_TIER_NEVER_ENGAGED", "GENERATOR_PARTIAL_FAILURES")


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
    """True if this row asserts the LLM tier ran -- either flag-shaped or evidence-shaped."""
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
    never_engaged = calls == 0  # explicit zero only; None (absent) is NOT "never engaged"

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

    for msg in failures:
        print(f"SELF-TEST FAIL: {msg}")
    if failures:
        print(f"\n{len(failures)} self-test failure(s) -- THE GUARD IS NOT TRUSTWORTHY")
        return 1
    print(
        f"SELF-TEST OK: fired on all {len(ORIGIN_DEAD_CELLS)} recorded origin cells (in BOTH "
        f"the original and the 2026-07-27 scored-witness shape), clean on all "
        f"{len(ORIGIN_CONTROL_CELLS)} matched-config controls, 12 mutation proofs passed."
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
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    report = scan_paths(args.paths or ["results"])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"scanned {report['files_scanned']} files "
            f"({report['files_unreadable']} unreadable), "
            f"{report['rows_seen']} rows, {report['rows_llm_on']} claiming llm_enabled=True"
        )
        for f in report["findings"]:
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
    return 0 if (args.warn_only or report["n_fail"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
