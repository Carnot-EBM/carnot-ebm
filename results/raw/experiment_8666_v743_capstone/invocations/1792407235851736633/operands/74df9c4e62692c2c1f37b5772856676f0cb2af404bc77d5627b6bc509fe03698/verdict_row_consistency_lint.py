#!/usr/bin/env python3
"""Does an artifact's VERDICT survive contact with its own per-row data?

WHY THIS EXISTS. The project already checks an artifact against ITSELF -- acceptance-gate
self-reports (`summarize_artifact.py`), fabrication patterns (`adversarial_verify.py`),
determination preservation, freshness. None of them ask whether the headline is SUPPORTED BY
THE ROWS. In one session (2026-08-11/12) three artifacts had headlines their own rows
contradicted:

  * exp6252 reported `gate_met: True`. Its mandated random ablation was byte-identical to the
    baseline on 20 of 25 rows, because `UniformGoalEnergy` hashed `repr()` and numpy
    abbreviates arrays over 1000 elements -- so the control literally WAS the baseline and the
    gate's second clause could never fail. -> DEGENERATE_CONTROL.
  * exp6251 reported `gate_met: True` on 1 win, 1 loss, and 2 games pinned at the 0.0 floor
    and 1.0 ceiling where no delta was possible. A pooled mean hid the split.
    -> NO_HEADROOM_MAJORITY, WINS_NOT_EXCEEDING_LOSSES.
  * exp6254 would have reported a clean null while EVERY row's metric was None, from a
    store-path bug that made every arm return no candidate. -> ALL_ROWS_NULL.

Each was caught by a human reading rows instead of the verdict string. That is not a process.
The conductor plans from `honest_verdict`, so a wrong verdict propagates into the retro, the
planner and the exclusion decisions -- the error compounds rather than staying local.

WHAT THIS IS NOT. It is not a fabrication detector; `adversarial_verify.py` owns that. It does
not judge whether a result is INTERESTING. It asks one question: do the rows support the
headline?

FAIL DIRECTION, STATED EXPLICITLY (QA-Layer Authenticity Discipline). Advisory by default:
exit 0 with warnings, because these are heuristics over free-form artifacts and a hard block on
a fuzzy match is how a guard starts punishing people for other people's data (see the
2026-08-13 artifact-freshness fix). ONE class exits non-zero by default -- ALL_ROWS_NULL --
because "every row is empty" is unambiguous and never a real result. `--strict` escalates every
class to exit 1. An artifact with no recognised row container is SKIPPED, not passed: absence of
rows is absence of evidence, and it is reported as `skipped`, never as clean.

Usage:
    python3 scripts/verdict_row_consistency_lint.py results/experiment_1234.json
    python3 scripts/verdict_row_consistency_lint.py --recent 20
    python3 scripts/verdict_row_consistency_lint.py --strict results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Containers that actually hold per-unit rows in this corpus, measured over recent artifacts
# rather than guessed. `MODEL_SPECS`, `events`, `gates_evaluated` and friends are lists of dicts
# too, but they are not experiment units, so matching on "list of dicts" alone would produce
# nonsense findings.
ROW_CONTAINERS = (
    "per_game_results",
    "per_predicate",
    "rows",
    "per_cell_results",
    "cells",
    "per_unit_rows",
)

# A verdict claiming a POSITIVE outcome. Only these are checked for wins-vs-losses: a verdict
# that already says "not met" or "null" is not over-claiming, and flagging it would train people
# to ignore this tool.
_POSITIVE_TOKENS = ("gate_met", "_met_", "improved", "beats", "positive", "wins", "success")
_NEGATIVE_TOKENS = (
    "not_met",
    "gate_not_met",
    "no_reliable",
    "null",
    "unusable",
    "retired",
    "blocked",
    "no_improvement",
    "negative",
)


def _verdict(d: dict) -> str:
    v = d.get("honest_verdict")
    if isinstance(v, dict):  # principle-annotated fields are legal project-wide
        v = v.get("value")
    return str(v or "")


def _claims_positive(d: dict) -> bool:
    v = _verdict(d).lower()
    if any(t in v for t in _NEGATIVE_TOKENS):
        return False
    return bool(d.get("gate_met")) or any(t in v for t in _POSITIVE_TOKENS)


def _rows(d: dict) -> tuple[str | None, list[dict]]:
    for key in ROW_CONTAINERS:
        v = d.get(key)
        if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
            return key, v
    return None, []


def _flatten_row(row: dict, prefix: str = "", depth: int = 0) -> dict:
    """Flatten one level of nesting so per-arm values are visible as `arm.field`.

    NOT COSMETIC -- this is the fix for the lint's own founding bug. The first version scanned
    only top-level row keys, and exp6252 (the artifact this tool exists to catch) stores its
    arms nested: `row["arms"]["uniform_random"]["nodes_expanded"]`. So the DEGENERATE_CONTROL
    check found nothing on the exact case it was written for -- a pattern list narrower than
    its concept, which is the failure class this project keeps hitting. Depth is capped at 2
    because artifacts nest arms one or two deep and an unbounded walk would start comparing
    unrelated bookkeeping.
    """
    out: dict = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            # Booleans are real per-row outcomes (`fires_on_real_win`, `plan_found`), not
            # bookkeeping. Dropping them is why PAIRED_METRIC_COLLAPSE could not see exp6260's
            # fires=True / specificity=0.0 pair. Coerced so numeric checks can compare them.
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, dict) and depth < 2:
            out.update(_flatten_row(v, prefix=f"{key}.", depth=depth + 1))
    return out


def _numeric_fields(rows: list[dict]) -> list[str]:
    """Fields numeric in a majority of rows -- the candidate metrics."""
    counts: dict[str, int] = {}
    for r in rows:
        for k in _flatten_row(r):
            counts[k] = counts.get(k, 0) + 1
    return [k for k, n in counts.items() if n >= max(2, len(rows) // 2)]


def _metric_like(name: str) -> bool:
    """Is this field a MEASURED OUTCOME rather than bookkeeping?

    Exclusions are SUFFIX-anchored, not bare substring. The first version excluded any name
    containing `_n` or `_s`, which threw away `best_of_n_held` -- the headline metric of the
    very artifact this check was meant to catch. That is the same unanchored-substring bug the
    project's own TAUTOLOGY check hit when "meta" matched inside "meta_tensor", and it is worth
    naming twice because it is this easy to reintroduce.
    """
    n = name.lower()
    leaf = n.rsplit(".", 1)[-1]
    if leaf.endswith(("_s", "_n", "_count", "_seed", "_ms")):
        return False
    if any(t in leaf for t in ("wall", "duration", "seconds", "nodes", "n_samples", "llm_calls")):
        return False
    return any(
        t in n
        for t in (
            "held",
            "fidelity",
            "accuracy",
            "score",
            "auroc",
            "precision",
            "recall",
            "delta",
            "rate",
            "mean",
            # Added 2026-08-13: the two-sided vocabulary this project uses for goal predicates.
            # Their absence is why PAIRED_METRIC_COLLAPSE could not see exp6260's
            # fires_on_real_win / specificity pair -- the check ran and matched nothing.
            "specificity",
            "sensitivity",
            "fires",
            "plan_found",
            "hollow",
        )
    )


def check_all_rows_null(d: dict, rows: list[dict]) -> list[str]:
    """Every row's metric fields are None/absent while a verdict is claimed.

    exp6254 wrote `held=None` on every arm from a store-path bug and would have reported a
    plausible zero-comparable null. An empty measurement is not a result.
    """
    if not _verdict(d):
        return []
    metric_keys = {k for r in rows for k in r if _metric_like(k)}
    if not metric_keys:
        return []
    any_value = any(
        isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
        for r in rows
        for k in metric_keys
    )
    if any_value:
        return []
    return [
        f"ALL_ROWS_NULL: {len(rows)} row(s) and not one numeric value among "
        f"{sorted(metric_keys)[:4]} -- the verdict summarises nothing measured"
    ]


def check_degenerate_control(d: dict, rows: list[dict]) -> list[str]:
    """A control/ablation arm identical to the baseline on most rows.

    exp6252's `uniform_random` ablation matched `flat_none` exactly on 20 of 25 rows because a
    hashing bug made it a constant, i.e. literally the baseline. A control that cannot differ
    cannot fail, so a gate requiring "beats the control" could never fail either.
    """
    out = []
    fields = _numeric_fields(rows)
    control_like = [
        f
        for f in fields
        if any(t in f.lower() for t in ("random", "uniform", "control", "ablation", "shuffled"))
    ]
    base_like = [
        f
        for f in fields
        if any(t in f.lower() for t in ("flat", "baseline", "none", "n1", "control_"))
    ]
    for c in control_like:
        for b in base_like:
            if c == b:
                continue
            pairs = [
                (r.get(c), r.get(b))
                for r in rows
                if isinstance(r.get(c), (int, float)) and isinstance(r.get(b), (int, float))
            ]
            if len(pairs) < 3:
                continue
            same = sum(1 for x, y in pairs if x == y)
            if same / len(pairs) >= 0.8:
                out.append(
                    f"DEGENERATE_CONTROL: `{c}` equals `{b}` on {same}/{len(pairs)} rows -- a "
                    "control that cannot differ from the baseline cannot fail, so any gate "
                    "requiring it to be beaten is vacuous"
                )
    return out


def check_no_headroom(d: dict, rows: list[dict]) -> list[str]:
    """Most rows pinned at a floor or ceiling, where no difference was possible.

    exp6251 had 2 of 4 games at 0.0 and 1.0 respectively; only two could discriminate and they
    split one win, one loss. The pooled mean hid it.
    """
    out = []
    for f in _numeric_fields(rows):
        if not _metric_like(f):
            continue
        vals = [
            r[f]
            for r in rows
            if isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool)
        ]
        if len(vals) < 3:
            continue
        # A BINARY-valued field is 0/1 by nature, so "pinned at 0/1" says nothing about it.
        # Coercing booleans into the numeric view -- needed by PAIRED_METRIC_COLLAPSE -- made
        # this check fire on every flag field: noise manufactured by a fix for another check.
        if set(vals) <= {0, 1}:
            continue
        pinned = sum(1 for v in vals if v in (0.0, 0, 1.0, 1))
        if pinned / len(vals) >= 0.5:
            out.append(
                f"NO_HEADROOM_MAJORITY: `{f}` is pinned at 0.0/1.0 on {pinned}/{len(vals)} rows "
                "-- those rows could not move in either direction, so a pooled mean over them "
                "reports the pinning, not the lever"
            )
    return out


def check_wins_vs_losses(d: dict, rows: list[dict]) -> list[str]:
    """A positive verdict while per-row losses meet or exceed wins.

    Reads any `*delta*` field: positive is a win, negative a loss. exp6251 claimed gate-met on
    1 win and 1 loss.
    """
    if not _claims_positive(d):
        return []
    out = []
    for f in _numeric_fields(rows):
        if "delta" not in f.lower():
            continue
        vals = [
            r[f]
            for r in rows
            if isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool)
        ]
        wins = sum(1 for v in vals if v > 0)
        losses = sum(1 for v in vals if v < 0)
        if wins + losses >= 2 and losses >= wins:
            out.append(
                f"WINS_NOT_EXCEEDING_LOSSES: verdict claims a positive outcome but `{f}` shows "
                f"{wins} win(s) against {losses} loss(es) across rows"
            )
    return out


def check_coverage_shortfall(d: dict, rows: list[dict]) -> list[str]:
    """A positive verdict while fewer units were comparable than the roster declares."""
    if not _claims_positive(d):
        return []
    roster = d.get("roster")
    n_comp = d.get("n_games_comparable", d.get("n_comparable"))
    if isinstance(roster, list) and isinstance(n_comp, int) and 0 < n_comp < len(roster):
        return [
            f"COVERAGE_SHORTFALL: verdict claims a positive outcome on {n_comp} of "
            f"{len(roster)} roster units -- state which units were not measured and why"
        ]
    return []


def check_paired_metric_collapse(d: dict, rows: list[dict]) -> list[str]:
    """A claimed gain on one axis while the SAME row collapses on its paired axis.

    ADDED 2026-08-13 after this lint failed to catch exp6260, which is the honest test of any
    guard. That artifact reported `gate_met: True` because one goal predicate finally fired on
    a real win -- while that same predicate's specificity was 0.0, meaning it fired on
    everything, including the opening board. It had traded constant-False for constant-True.
    The gain was real on the axis measured and worthless on the axis that makes it meaningful.

    THE GENERAL SHAPE, which is not ARC-specific: a two-sided measurement where an arm wins on
    side A and bottoms out on side B. Any pair of fields sharing a row prefix where one is at
    an extreme while its sibling improves is worth a human look. Reported when the verdict
    claims a positive outcome, since an honest negative is not over-claiming.
    """
    if not _claims_positive(d):
        return []
    out = []
    fields = [f for f in _numeric_fields(rows) if _metric_like(f)]
    # Pair fields that share a prefix, e.g. arm_b.fires / arm_b.specificity.
    for f in fields:
        prefix = f.rsplit(".", 1)[0] if "." in f else None
        if not prefix:
            continue
        sibs = [g for g in fields if g != f and g.startswith(f"{prefix}.")]
        for g in sibs:
            bad = [
                r
                for r in rows
                if isinstance(r.get(f), (int, float))
                and isinstance(r.get(g), (int, float))
                and r[f] > 0
                and r[g] == 0.0
            ]
            if bad:
                out.append(
                    f"PAIRED_METRIC_COLLAPSE: on {len(bad)} row(s) `{f}` is positive while its "
                    f"sibling `{g}` is 0.0 -- a gain on one side of a two-sided measurement "
                    "with the other side bottomed out is usually the opposite degeneracy, not a win"
                )
    return out


CHECKS = (
    check_all_rows_null,
    check_degenerate_control,
    check_no_headroom,
    check_wins_vs_losses,
    check_coverage_shortfall,
    check_paired_metric_collapse,
)
# Unambiguous enough to block on its own: every row empty is never a real result.
HARD_CLASSES = ("ALL_ROWS_NULL",)


def check_artifact(path: Path) -> tuple[str, list[str]]:
    """Return (status, findings). status is `ok`, `findings`, `skipped` or `unreadable`."""
    try:
        d = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return ("unreadable", [f"{type(exc).__name__}: {exc}"])
    if not isinstance(d, dict):
        return ("unreadable", ["top level is not an object"])
    key, rows = _rows(d)
    if not rows:
        # Absence of rows is absence of evidence, NOT a pass.
        return ("skipped", ["no recognised row container"])
    rows = [{**r, **_flatten_row(r)} for r in rows]  # flat view beside the originals
    findings: list[str] = []
    for fn in CHECKS:
        try:
            findings.extend(fn(d, rows))
        except Exception as exc:  # noqa: BLE001
            findings.append(f"CHECK_ERROR in {fn.__name__}: {exc!r}"[:160])
    return ("findings" if findings else "ok", findings)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument(
        "--recent", type=int, default=0, help="check the N most recently modified results/*.json"
    )
    ap.add_argument(
        "--strict", action="store_true", help="exit 1 on ANY finding, not just the hard classes"
    )
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.paths]
    if args.recent:
        allj = sorted((REPO / "results").glob("experiment_*.json"), key=lambda p: p.stat().st_mtime)
        paths = allj[-args.recent :]
    if not paths:
        print("verdict-row-consistency: nothing to check")
        return 0

    hard = soft = 0
    n_ok = n_skipped = n_unreadable = 0
    for p in paths:
        status, findings = check_artifact(p)
        if status == "ok":
            n_ok += 1
            continue
        if status == "skipped":
            n_skipped += 1
            continue
        if status == "unreadable":
            n_unreadable += 1
            continue
        print(f"\n  {p.name}")
        for f in findings:
            is_hard = any(f.startswith(c) for c in HARD_CLASSES)
            hard += is_hard
            soft += not is_hard
            print(f"    [{'BLOCK' if is_hard else 'warn '}] {f}")

    # COVERAGE IS ALWAYS PRINTED, never only on failure. A first version printed a bare "OK"
    # while silently skipping 57 of 60 artifacts, which is precisely the guard-is-green-while-
    # blind state this project keeps finding in other guards. A reader must be able to tell
    # "checked and clean" from "could not check".
    checked = n_ok + hard + soft
    print(
        f"\nverdict-row-consistency: checked {checked}, skipped {n_skipped} (no per-unit rows), "
        f"unreadable {n_unreadable}."
    )
    if n_skipped > checked and n_skipped:
        print(
            "  NOTE: most artifacts carry no per-unit row container, so their headline claims "
            "are not falsifiable from the artifact alone. That is a reporting-shape gap in those "
            "experiments, not a clean bill of health from this tool."
        )
    if hard or soft:
        print(
            f"  {hard} blocking, {soft} advisory. These say the ROWS do not support the "
            "headline -- read the rows before citing the verdict, and correct the artifact "
            "rather than the reader."
        )
    elif checked:
        print("  No row/verdict contradictions among the artifacts that could be checked.")
    return 1 if (hard or (args.strict and soft)) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
