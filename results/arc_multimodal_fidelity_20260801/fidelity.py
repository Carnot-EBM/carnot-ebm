"""STEP 3+4: the fidelity measurement and its mandatory text negative control.

THE QUESTION. Given a 64x64 ARC colour-index grid, can gemma-4-31B-it's vision tower
read back the exact integer at a named (row, col)? ARC grades induced engines by
EXACT match, and the engine's Python indexes specific cells against specific integer
values, so "approximately the right colour" is worth nothing here.

PRE-REGISTERED DESIGN DECISIONS (fixed before any result was seen):

1. SAMPLE SIZE AND PRECISION. n = 128 probe cells per scheme, stratified so each of
   the 16 colour indices is the ground truth for exactly 8 probes. Worst-case 95%
   binomial CI half-width is 1.96*sqrt(0.25/128) = 8.7 percentage points. Chance for a
   16-way choice is 6.25%. So this n cannot resolve a 3pp difference, but it CAN
   decisively separate "at chance" from "usable for exact-match induction (>=90%)",
   which is the only distinction the gate needs to make.

2. PAIRED PROBES. The SAME 128 (row, col) cells, in the same order, are used for every
   scheme including the text control. Cell difficulty is therefore held constant and
   the schemes differ only in how the grid was presented.

3. THE GRID IS DELIBERATELY THE EASY CASE. It is an 8x8 arrangement of solid 8x8-cell
   blocks -- 64 blocks, each of the 16 colours used exactly 4 times. Real ARC frames
   are finer-grained than this. Combined with a maximally-separated nameable palette
   (see common.PALETTE), this makes the measurement a CHARITABLE UPPER BOUND: if the
   tower cannot read indices off THIS, it cannot read them off a real frame either.
   The escalation rule was fixed in advance: only if the charitable case PASSES does a
   harder, per-cell-dense grid need to be run. A failure here needs no escalation.

4. THE NEGATIVE CONTROL IS NOT OPTIONAL. The identical 128 questions are also asked
   over the shipped run-length TEXT encoding (`_rle_grid`'s exact format). If the text
   arm also scores near zero, then the failure is in the QUESTION FORMAT -- "read cell
   (r,c)" is just a hard thing to ask -- and NO conclusion about vision would be
   licensed. Both arms are always reported.

5. NO TUNING TO PASS. Each scheme is reported at whatever it scores. Rendering
   parameters are fixed up front (1, 8, 16 pixels per cell) and are not adjusted in
   response to results.

POST-HOC AMENDMENT, 2026-08-01, MARKED AS SUCH (it is NOT pre-registered).
An adversarial review after the run found that `parse()`'s positional fallback was
ingesting row and column numbers as colour predictions. The fix is in `parse()` below,
with the evidence. It is FORWARD-ONLY: the completed run cannot be re-scored, because
that run persisted only the parsed predictions and not the model's replies. Every
exact_match figure published from that run is therefore a LOWER BOUND, and the artifact
says so. Raw replies are now persisted (`transcript`) precisely so this can never again
be uncorrectable. Note also that `SUPPLEMENT_BAR = 0.25` in build_artifact.py was
introduced AFTER the numbers were seen and is not part of the pre-registration above;
the only bars fixed in advance are the 0.90 replace/escalation rule (decision 3) and
the 6.25% chance rate (decision 1).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time

import numpy as np

sys.path.insert(
    0,
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/mmgate",
)
from common import (  # noqa: E402
    LEGEND,
    N_COLORS,
    PALETTE,
    chat,
    image_part,
    render_png,
    rle_grid,
    text_part,
)

SD = (
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/mmgate"
)
SEED = 20260801
GRID_H = GRID_W = 64
BLOCK = 8
N_PROBES = 128
BATCH = 8
PX_SCHEMES = [1, 8, 16]


def build_grid(rng: np.random.Generator) -> np.ndarray:
    """8x8 blocks of 8x8 cells; each of the 16 colours used exactly 4 times."""
    order = np.repeat(np.arange(N_COLORS), 4)
    rng.shuffle(order)
    blocks = order.reshape(GRID_H // BLOCK, GRID_W // BLOCK)
    return np.repeat(np.repeat(blocks, BLOCK, axis=0), BLOCK, axis=1).astype(np.uint8)


def build_probes(grid: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """128 (row, col, true_value), exactly 8 per colour index."""
    probes = []
    for v in range(N_COLORS):
        rs, cs = np.where(grid == v)
        pick = rng.choice(len(rs), size=N_PROBES // N_COLORS, replace=False)
        for i in pick:
            probes.append((int(rs[i]), int(cs[i]), int(v)))
    rng.shuffle(probes)
    return probes


INSTR = f"""This is a {GRID_H}x{GRID_W} grid of coloured cells.
Rows are numbered 0 to {GRID_H - 1} from TOP to BOTTOM.
Columns are numbered 0 to {GRID_W - 1} from LEFT to RIGHT.

Each cell is one of these 16 colours. This legend maps each colour to its integer index:
{LEGEND}

Report the INTEGER INDEX of the cell at each requested (row, column).
Answer with one line per request, in the exact format `row,col=index`, and nothing else.
Requests:
"""


def ask(parts_prefix, batch, max_tokens=9000):
    """One batched request.

    MAX_TOKENS IS A CORRECTNESS PARAMETER, NOT A PERFORMANCE ONE. gemma-4 always emits
    a chain-of-thought before its answer, and on the HARD schemes (a 64x64-pixel image
    where each ARC cell is a single pixel) that CoT runs long -- calibration measured
    4000 tokens exhausted mid-CoT, returning content="" with finish_reason="length".
    Scoring that as "the vision tower got the colour wrong" would be a fabricated
    negative: the model never reached the answer at all. So the budget is set well
    above the observed worst case, and any residual truncation is counted and reported
    SEPARATELY rather than silently folded into the error rate.

    RAISED 7000 -> 9000 AFTER A MEASURED PATHOLOGY, and the reason matters because it
    is easy to mistake for tuning. It is not: a larger budget cannot make a wrong
    colour right, it only decides whether the model reaches an answer at all, and it
    is applied identically to every scheme including the text control. At 7000 the RLE
    text arm truncated on most batches, so nearly every batch paid a full 7000-token
    dead attempt AND THEN an 11000-token retry -- about 530s of decode at the measured
    34 tok/s, against roughly 120s when the first attempt succeeds. That double-pay,
    not the model and not the GPU, is what made the run infeasible; the decode rate
    itself never degraded. Sizing the first attempt above the observed CoT length
    removes the wasted attempt and, as a side effect, REDUCES the number of probes
    that end up unparseable -- i.e. it makes the measurement more complete, not more
    flattering.
    """
    q = INSTR + "\n".join(f"row {r}, column {c}" for r, c, _ in batch)
    raw, finish, ntok = chat(parts_prefix + [text_part(q)], max_tokens=max_tokens)
    return raw, finish, ntok


# A whole LINE that is nothing but an answer. Anchored at both ends on purpose --
# see the CORRECTED note in parse(). Accepts `3,7=5`, `= 5`, `- 5`, `5`.
_ANSWER_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\d+\s*,\s*\d+\s*)?=?\s*(\d{1,2})\s*[.,;]?\s*$")


def parse(raw: str, batch) -> tuple[list[int | None], str]:
    """Pull `r,c=v` answers out of the reply.

    CORRECTED 2026-08-01 AFTER AN ADVERSARIAL REVIEW, and the original defect is worth
    stating plainly because it silently biased every number this harness has ever
    produced.

    THE BUG. The previous fallback was
    `re.findall(r"(?<![\\d,])(\\d{1,2})(?![\\d])", raw)` filtered to 0..15, taking the
    first len(batch) hits POSITIONALLY. When gemma-4 answers in prose instead of the
    requested `row,col=index` form -- "The cell at row 3, column 7 is yellow, which is
    index 5." -- that regex scoops the ROW and the COLUMN as if they were colour
    predictions. Run on exactly that reply the old parser returned [3, 7, 5, 9, ...]
    against a truth of [5, 9, ...]: the row and column were scored as colours and every
    real answer was shifted down the list. 54 of this run's 128 probe cells have a row
    or column index below 16, so the contamination was available on 42% of probes.

    IT WAS NOT HYPOTHETICAL. Measured on the run's own recorded predictions, a wrong
    prediction equals the TRUTH OF AN EARLIER PROBE IN THE SAME BATCH far more often
    than it equals the truth of a later one -- the exact signature of a shifted
    alignment. Fixed-shift totals, k=+1/+2/+3 (earlier) against k=-1/-2/-3 (later):
    text_rle 27/23/16 against 5/5/4; image_px16 18/15/8 against 6/4/3. Those shift-hits
    are also concentrated almost entirely in the batches that scored 0-2 out of 8
    (4.64/batch for text_rle) and essentially absent from the batches that scored 6+
    (0.00/batch) -- so the striking 8/8-versus-0/8 bimodality this experiment reported
    is substantially THIS PARSER losing alignment, not the model mislocating a whole
    batch's rows and columns.

    DIRECTION OF THE BIAS. A misaligned prediction is scored WRONG, so every
    exact_match this harness has emitted is a LOWER BOUND. That is the safe direction
    for the gate's headline conclusion (the tower is below the exact-match bar a
    fortiori), but it is NOT safe for comparing two arms, because the bias is bigger in
    the text control than in px16.

    THE FIX. Coordinate echoes only, or a fallback restricted to lines that are
    NOTHING BUT an answer -- a prose sentence containing an integer no longer
    qualifies. The fallback additionally demands EXACTLY len(batch) answer lines, so a
    reply that dropped or doubled one answer is recorded as unparseable rather than
    silently shifted. An unanswered probe is scored as a non-answer, which the strict /
    parsed-only pair of metrics already reports honestly.

    Returns (predictions, parse_mode) so the mode is auditable per batch.
    """
    found = {}
    for m in re.finditer(r"(\d+)\s*,\s*(\d+)\s*=\s*(\d+)", raw):
        found[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    out: list[int | None] = []
    if found:
        for r, c, _ in batch:
            v = found.get((r, c))
            out.append(v if (v is not None and 0 <= v < N_COLORS) else None)
        if sum(x is not None for x in out) >= len(batch) // 2:
            return out, "coord_echo"
    # RESTRICTED fallback: the model answered in order, one answer per line, without
    # echoing coordinates. Whole-line match only, and the count must be exact.
    lines = [m.group(1) for ln in raw.splitlines() if (m := _ANSWER_LINE.match(ln))]
    cand = [int(x) for x in lines]
    if len(cand) == len(batch) and all(0 <= n < N_COLORS for n in cand):
        return cand, "answer_lines"
    return (out if out else [None] * len(batch)), "unparseable"


def run_scheme(name: str, parts_prefix_fn, probes, grid) -> dict:
    t0 = time.time()
    preds: list[int | None] = []
    truncated = 0
    # RAW REPLIES ARE PERSISTED (added 2026-08-01 after an adversarial review).
    # The first run of this harness saved only the parsed `predictions`, so when the
    # parser was later found to be scooping row/column numbers as colours there was no
    # way to re-score the run -- correcting it would have cost a fresh ~4 GPU-hours.
    # Scoring must stay auditable and re-derivable from the model's actual words, so
    # every reply is kept verbatim alongside the parse mode that consumed it.
    transcript: list[dict] = []
    for i in range(0, len(probes), BATCH):
        batch = probes[i : i + BATCH]
        raw, finish, ntok = ask(parts_prefix_fn(), batch)
        if finish == "length":
            # ONE retry with a much larger budget before conceding. A truncated CoT is
            # a harness limit, not a measurement, so we spend the tokens rather than
            # record a negative we cannot attribute to the vision path.
            # 11000, not more: the largest prompt in this run is the px=16 image at
            # ~4200 tokens, and the server is launched with `--parallel 1 -c 16384` so
            # one slot owns the whole 16384-token pool. 4200 + 11000 leaves headroom;
            # asking for 13000 would exceed the pool and turn a retry into a 500.
            raw2, finish2, ntok2 = ask(parts_prefix_fn(), batch, max_tokens=11000)
            if finish2 != "length" or len(raw2) > len(raw):
                raw, finish, ntok = raw2, finish2, ntok2
            if finish == "length":
                truncated += 1
        got, mode = parse(raw, batch)
        preds.extend(got)
        ok = sum(1 for (_, _, t), p in zip(batch, got, strict=True) if p == t)
        transcript.append(
            {
                "batch_index": i // BATCH,
                "probes": [[r, c, t] for r, c, t in batch],
                "raw": raw,
                "finish_reason": finish,
                "completion_tokens": ntok,
                "parse_mode": mode,
                "parsed": got,
                "n_correct": ok,
            }
        )
        print(
            f"    [{name}] batch {i // BATCH + 1:2d}/{len(probes) // BATCH}: "
            f"{ok}/{len(batch)} ({finish}, {ntok}tok, {mode})",
            flush=True,
        )
    truth = [t for _, _, t in probes]
    n_ok = sum(1 for t, p in zip(truth, preds, strict=True) if p == t)
    n_parsed = sum(1 for p in preds if p is not None)
    conf: dict[str, int] = {}
    for t, p in zip(truth, preds, strict=True):
        if p is not None and p != t:
            conf[f"{t}->{p}"] = conf.get(f"{t}->{p}", 0) + 1
    return {
        "scheme": name,
        "cells_probed": len(probes),
        "n_correct": n_ok,
        # HEADLINE metric: strict. Every probe is in the denominator; an unanswered
        # probe counts against the scheme, because an induction pipeline that cannot
        # get an answer out of the model is no better off than one that gets a wrong
        # answer.
        "exact_match": round(n_ok / len(probes), 4),
        # DIAGNOSTIC companion: accuracy over probes that produced a parseable answer
        # at all. Reported alongside so a reader can tell "the tower misread the
        # colour" apart from "the model never finished thinking". These two being far
        # apart is itself the finding for that scheme.
        "exact_match_over_parsed": round(n_ok / n_parsed, 4) if n_parsed else None,
        "n_parsed": n_parsed,
        "n_unparseable": len(probes) - n_parsed,
        "n_batches_truncated_after_retry": truncated,
        "confusions": dict(sorted(conf.items(), key=lambda kv: -kv[1])),
        "predictions": preds,
        "parse_modes": [t["parse_mode"] for t in transcript],
        "transcript": transcript,
        "duration_s": round(time.time() - t0, 2),
    }


def main():
    rng = np.random.default_rng(SEED)
    grid = build_grid(rng)
    probes = build_probes(grid, rng)
    rle = rle_grid(grid)

    meta = {
        "grid_shape": [GRID_H, GRID_W],
        "block_cells": BLOCK,
        "palette": [{"index": i, "name": n, "rgb": list(c)} for i, (n, c) in enumerate(PALETTE)],
        "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
        "probes_sha256": hashlib.sha256(json.dumps(probes).encode()).hexdigest(),
        "rle_chars": len(rle),
        "colours_present": sorted(int(v) for v in np.unique(grid)),
    }
    print(json.dumps(meta, indent=1)[:900])

    results = []
    # NEGATIVE CONTROL FIRST, deliberately: it is the arm that can invalidate the whole
    # experiment, so it should not be something we get to look at only after the
    # vision numbers are in.
    print("  --- TEXT CONTROL (shipped RLE encoding) ---", flush=True)
    txt_prefix = (
        "Here is the grid, run-length encoded. Each line is `r<row>:<value>x<count>,...` "
        "covering that row's columns left to right with no gaps."
        f"\n{rle}\n\n"
    )
    results.append(run_scheme("text_rle", lambda: [text_part(txt_prefix)], probes, grid))

    for px in PX_SCHEMES:
        png = render_png(grid, px)
        print(
            f"  --- IMAGE px_per_cell={px} ({GRID_H * px}x{GRID_W * px}px, {len(png)}B) ---",
            flush=True,
        )
        r = run_scheme(f"image_px{px}", lambda p=png: [image_part(p)], probes, grid)
        r["px_per_cell"] = px
        r["image_pixels"] = [GRID_H * px, GRID_W * px]
        r["image_bytes"] = len(png)
        results.append(r)
        with open(f"{SD}/rendered_px{px}.png", "wb") as f:
            f.write(png)

    out = {"meta": meta, "results": results}
    with open(f"{SD}/fidelity_raw.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"  {r['scheme']:12s} exact={r['exact_match']:.3f} "
            f"({r['n_correct']}/{r['cells_probed']}) unparsed={r['n_unparseable']} "
            f"top_conf={list(r['confusions'].items())[:4]}"
        )

    # PRE-REGISTERED ESCALATION (design decision 3). The blocks grid is the charitable
    # case. If any image scheme clears 0.90 on it, the charitable result is NOT
    # sufficient to license "vision can replace the exact text" -- a real ARC frame is
    # far finer-grained -- so the harder dense grid must be run before any such claim.
    if any(r["exact_match"] >= 0.90 for r in results if r["scheme"].startswith("image_")):
        print(
            "\n  charitable case PASSED -> escalation to the dense grid is "
            "REQUIRED (see escalate.py)"
        )


if __name__ == "__main__":
    main()
