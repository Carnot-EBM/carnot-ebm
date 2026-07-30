#!/usr/bin/env python3
"""CLI for the treatment-activation pre-flight (REQ-HARNESS-6050).

Point it at a directory of per-cell JSON records and it diffs the two arms' ACTION TRACES,
classifies every matched pair, and PASSES or REFUSES the full grid. Exit code 0 = PASS,
1 = REFUSE, 2 = usage/no comparable cells.

    # the retrospective validation: the 2026-07-29 retention grid, which it must refuse
    scripts/treatment_activation_preflight.py \
        --cells results/arc_engine_retention_20260729/cells \
        --control ret0 --treatment ret1

    # with an A/A noise floor, which is what makes a PASS interpretable at all
    scripts/treatment_activation_preflight.py --cells <dir> \
        --control pre --treatment post --replicate postb

The cell format is the one `run_bounded_progress` already writes: a JSON object with a
`result.action_trace` list, a top-level `status`, and `result.timed_out`. Filenames are
`<arm>__<cell>__*.json`. Nothing is ever written -- this reads evidence and prints a verdict.

WHY A CLI AND NOT JUST A LIBRARY. The pre-flight has to be runnable in the minutes between a
probe finishing and a grid being launched, by whoever is at the keyboard, without writing a
bespoke analysis script each time -- because writing one is exactly the step that got skipped
on 2026-07-29, three times in one day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

from carnot.analysis.treatment_activation_preflight import (  # noqa: E402
    PASS,
    classify_trace_pair,
    format_report,
    preflight_verdict,
)


class AmbiguousCellRecord(Exception):
    """More than one record matched one arm+cell, so which one to read is not determined.

    Raised rather than resolved (2026-07-30 review). The previous code took ``sorted(matches)[0]``
    and discarded the rest SILENTLY: a second seed or a re-run replicate (``..__s2.json``) would
    have been neither read, nor reported, nor counted as MISSING -- it would simply have vanished,
    and the verdict would have been computed from an arbitrary one of several available runs.
    That is the same silent-data-selection error class this whole module exists to catch, so it
    exits 2 (usage) instead of quietly picking. The caller resolves it with ``--suffix``.
    """


def _load_arm_cell(cells_dir: str, arm: str, cell: str, *, suffix: str | None = None) -> tuple:
    """Return ``(trace, complete, filename)`` for one arm of one cell.

    ``complete`` means the run ended on its OWN terms. It is False when the cell's status is
    anything but `ok`, or when the run reports `timed_out` -- the two ways a run gets cut short
    by the harness rather than by its own decision. That distinction is what keeps a wall-clock
    cap from being read as a treatment effect (and vice versa), so it is read from the cell
    record rather than guessed from trace lengths.

    ``suffix`` (e.g. ``s1``) disambiguates when several records exist for the same arm+cell; with
    no suffix and several matches this raises rather than choosing one. Absent cell -> a
    ``(None, False, None)`` triple, which classifies as MISSING downstream.
    """
    matches = sorted(
        f for f in os.listdir(cells_dir) if f.startswith(f"{arm}__{cell}__") and f.endswith(".json")
    )
    if suffix:
        matches = [f for f in matches if f == f"{arm}__{cell}__{suffix}.json"]
    if not matches:
        return None, False, None
    if len(matches) > 1:
        raise AmbiguousCellRecord(
            f"{len(matches)} records match arm {arm!r} cell {cell!r} in {cells_dir}: "
            f"{matches}. Pass --suffix to say which run to read; refusing to pick one "
            f"silently, because discarding a replicate without reporting it is the same "
            f"missing-observation error this pre-flight exists to detect."
        )
    with open(os.path.join(cells_dir, matches[0])) as fh:
        d = json.load(fh)
    res = d.get("result") or {}
    return (
        res.get("action_trace"),
        (d.get("status") == "ok" and not res.get("timed_out")),
        matches[0],
    )


def _pairs(
    cells_dir: str, arm_a: str, arm_b: str, cells: list[str], *, suffix: str | None = None
) -> tuple[dict, dict]:
    """Classify every cell, and return the per-cell provenance alongside the classifications.

    The provenance (which FILE each arm was read from) is returned and printed because a verdict
    computed from an unnamed subset of the available records is not checkable by anyone else.
    """
    out: dict = {}
    provenance: dict = {}
    for cell in cells:
        ta, ca, fa = _load_arm_cell(cells_dir, arm_a, cell, suffix=suffix)
        tb, cb, fb = _load_arm_cell(cells_dir, arm_b, cell, suffix=suffix)
        out[cell] = classify_trace_pair(ta, tb, a_complete=ca, b_complete=cb)
        provenance[cell] = {arm_a: fa, arm_b: fb}
    return out, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cells", required=True, help="directory of <arm>__<cell>__*.json records")
    ap.add_argument("--control", required=True, help="control arm name (the A of the A/B)")
    ap.add_argument("--treatment", required=True, help="treatment arm name (the B)")
    ap.add_argument(
        "--replicate",
        default=None,
        help="A/A replicate arm (byte-identical to --treatment). Without it a PASS "
        "is uninterpretable, because the harness is not deterministic.",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--suffix",
        default=None,
        help="run suffix (e.g. s1) to select when several records exist per arm+cell; "
        "without it, multiple matches are a usage error rather than an arbitrary pick",
    )
    ap.add_argument(
        "--planned-n-cells",
        type=int,
        default=None,
        help="size of the grid you intend to run (defaults to the probed count)",
    )
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.cells):
        print(f"no such cells directory: {args.cells}", file=sys.stderr)
        return 2

    # Cells are the UNION over both arms, not just the control's.
    #
    # A first version enumerated the control arm alone, which is wrong in a way that matters here:
    # a cell the TREATMENT produced and the control did not would have been invisible -- not even
    # counted as MISSING -- so a drop asymmetry favouring the control would have been silently
    # hidden. The held-out 31b-vs-9b grid is exactly that shape: 3 of its 8 cells are missing on
    # one arm only, and that grid flagged the asymmetry itself. A census that can only see one
    # arm's failures is the same missing-vs-zero error this tool exists to prevent, one level up.
    def _cells_of(arm: str) -> set:
        return {
            f.split("__")[1]
            for f in os.listdir(args.cells)
            if f.startswith(f"{arm}__") and "__" in f
        }

    control_cells, treatment_cells = _cells_of(args.control), _cells_of(args.treatment)
    cells = sorted(control_cells | treatment_cells)
    # An arm with ZERO cells is a USAGE error (exit 2), not a refusal (exit 1) -- almost always a
    # typo'd arm name. Reporting it as "12 MISSING cells, underpowered" would send the caller
    # looking for a measurement problem that does not exist. Checked per arm, so a typo in EITHER
    # name is caught even though the census is the union.
    for name, found in ((args.control, control_cells), (args.treatment, treatment_cells)):
        if not found:
            print(f"no cells found for arm {name!r} in {args.cells}", file=sys.stderr)
            return 2

    try:
        ab, ab_provenance = _pairs(
            args.cells, args.control, args.treatment, cells, suffix=args.suffix
        )
        aa, aa_provenance = (
            _pairs(args.cells, args.treatment, args.replicate, cells, suffix=args.suffix)
            if args.replicate
            else (None, {})
        )
    except AmbiguousCellRecord as exc:
        print(f"ambiguous cell record: {exc}", file=sys.stderr)
        return 2

    verdict = preflight_verdict(
        ab, alpha=args.alpha, planned_n_cells=args.planned_n_cells, noise_pairs=aa
    )
    # Which file each number came from, recorded on the verdict itself, so the result is
    # checkable by someone who was not at the keyboard.
    verdict["record_provenance_ab"] = ab_provenance
    verdict["record_provenance_aa"] = aa_provenance or None
    if args.json:
        print(json.dumps(verdict, indent=2, sort_keys=True))
    else:
        print(
            format_report(
                verdict,
                title=f"TREATMENT-ACTIVATION PRE-FLIGHT  "
                f"{args.control} vs {args.treatment}"
                + (f"  (A/A: {args.replicate})" if args.replicate else ""),
            )
        )
        for cell in cells:
            files = ab_provenance.get(cell, {})
            aa_files = (aa_provenance or {}).get(cell, {})
            print(
                f"  read {cell:8s} A/B "
                + " ".join(f"{k}={v}" for k, v in files.items())
                + (
                    ""
                    if not aa_files
                    else "   A/A " + " ".join(f"{k}={v}" for k, v in aa_files.items())
                )
            )
    return 0 if verdict["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
