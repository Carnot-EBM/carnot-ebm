#!/usr/bin/env python3
"""Run the SHIPPED defect detector over every frozen induced-engine candidate on disk.

WHAT THIS ANSWERS. 22 of 40 frozen stall-path candidates were unusable (2026-07-31/08-01). That
is the binding constraint on the whole induce -> verify -> plan -> execute pipeline: no metric,
representation or search change matters when the engine is not valid Python, or is inert. This
pass classifies EVERY failure across every frozen candidate, so the classes can be counted and
ranked instead of recalled.

THE CORPORA, and why they are stratified rather than pooled.

  * PRIMARY (gemma-4-31B-it-qat, the CURRENT live generator):
      - 48 best-of-N candidates, `results/arc_induce_bestofn_20260731/` -- SINGLE-SHOT. Each
        candidate is one completion with no retry, so the raw per-completion failure rate is
        visible.
      - 124 A/B cells, `results/arc_object_perception_ab_change_fidelity_20260801/` -- each cell
        is the output of a 3-TRY induce loop, so its failures are the ones that survived retry.
        116 produced an engine file; 8 exhausted all three tries.
    These two measure DIFFERENT things and must not be averaged. Single-shot gives the failure
    rate of one generation; the A/B corpus gives the residual after the shipped retry, and the
    gap between them IS the measured value of retrying.

  * SECONDARY (Qwen3.5-9B-MTP, RETIRED as the generator on 2026-07-28):
      - 240 A/B cells, `results/arc_object_perception_ab_20260728/`. Kept strictly separate and
        never pooled with the primary: the generator differs, so any pooled rate would confound
        the taxonomy with the model switch. Its value is the CONTRAST -- whether a failure class
        is a property of this model or of induced-code generation generally -- plus the fact
        that it is the only corpus that recorded `stop_type` per cell.

WHY THE SHIPPED DETECTOR AND NOTHING ELSE. Every classification here is
`arc_engine_static_validation.validate_engine_code`'s own output. Writing a second classifier
would produce a taxonomy of the second classifier. Where the shipped detector is deliberately
coarse -- it folds IndentationError into `syntax_error`, and reports inertness through a
separate non-defect entry point -- that coarseness is recorded alongside, not patched out.

BOUNDS. Every unit of work that executes generated code, and every window rebuild, runs in a
killable child. See `classify_worker.py` and `window_worker.py` for why each one needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results"
SCRATCH = Path(
    os.environ.get(
        "ARC_GENTAX_SCRATCH",
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/gentax",
    )
)

BESTOFN = RESULTS / "arc_induce_bestofn_20260731"
ABCF = RESULTS / "arc_object_perception_ab_change_fidelity_20260801"
AB0728 = RESULTS / "arc_object_perception_ab_20260728"

# A window rebuild steps a real environment; the 2026-08-01 rescue pass needed 240s for tr87.
WINDOW_TIMEOUT_S = 300.0
# A classification runs the detector (whose own dry run is bounded at 30s by default) plus
# `engine_changes_anything`, which has no bound of its own. 120s leaves room for both to be
# slow-but-finite while still killing a genuine hang.
CLASSIFY_TIMEOUT_S = 120.0


def _inputs_checksum() -> str:
    """Content address of everything this pass reads plus the detector it reads them with.

    A taxonomy is only about the corpus it was derived from WITH the detector it was derived by,
    so both go into the hash. It changes if either changes -- which is the point.
    """
    import hashlib

    parts = []
    for p in [
        BESTOFN / "bestofn_scored.json",
        BESTOFN / "harness" / "bon" / "gpu1" / "bon.json",
        BESTOFN / "split.json",
        ABCF / "rows.json",
        ABCF / "meta.json",
        AB0728 / "rows.json",
        REPO / "python" / "carnot" / "agentic" / "arc_engine_static_validation.py",
        HERE / "classify_worker.py",
        HERE / "window_worker.py",
    ]:
        if p.exists():
            parts.append(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}")
    return hashlib.md5("|".join(sorted(parts)).encode()).hexdigest()  # noqa: S324


def run_worker(worker: str, job: dict, tag: str, timeout: float) -> dict:
    """One killable child. A timeout is UNDETERMINED, never a zero and never a clean result."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    jp = SCRATCH / f"job_{tag}.json"
    jp.write_text(json.dumps(job, default=str))
    try:
        pr = subprocess.run(  # noqa: S603
            [sys.executable, str(HERE / worker), str(jp)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "worker_timeout", "timeout_s": timeout}
    lines = (pr.stdout or "").strip().splitlines()
    if not lines:
        return {"status": "worker_no_output", "stderr": (pr.stderr or "")[-400:]}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"status": "worker_bad_output", "stdout": lines[-1][:300]}


# ---------------------------------------------------------------------------
# Candidate enumeration. Each corpus knows its own layout; nothing is inferred.
# ---------------------------------------------------------------------------


def rebuild_bestofn_shown() -> dict[str, dict]:
    """Reconstruct the EXACT `shown` rows the frozen best-of-N run validated against.

    THIS EXISTS BECAUSE THE OBVIOUS SHORTCUT IS WRONG, and it was caught by disagreeing with the
    frozen record rather than by review. The first version of this pass fed every corpus the
    window from `build_progress_window(game)` + `_split_prefix_heldout`. That is exactly right for
    the A/B corpora (verified per game against `meta.json`'s recorded `n_shown`, 20 of 20) and
    exactly WRONG for best-of-N, which validated against `split.py`'s `_shown` derived from a
    CAPTURED prefix (`capture/<game>/transitions1.pkl`). The two differ: ft09 has 17 prefix rows
    and 4 shown, while the rebuilt window yields 3 transitions. `engine_changes_anything` is
    monotone in the transition set -- more transitions can only turn False into True -- so the
    error had exactly one direction, and all 10 disagreements were True (frozen) -> False (mine),
    which is what made the diagnosis certain rather than plausible.

    THE HISTORICAL `k` MATTERS AND IS NOT THE CURRENT ONE. `_induce_transitions_k()` returned 8
    when this corpus was generated (2026-07-31) and returns None -- show ALL transitions -- since
    2026-08-01. Calling the live function here would silently reconstruct a DIFFERENT prompt's
    shown set. So k is pinned to the historical 8 and the reconstruction is PROVEN with split.py's
    own two checks (every shown row's rendered line appears in the frozen prompt text, and the
    prompt's ACTION-line count equals the shown count) plus agreement with the `n_shown` recorded
    in `split.json`. A game that fails any check is dropped, not guessed at.
    """
    sys.path.insert(0, str(REPO / "python"))
    import pickle

    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3

    cap = BESTOFN / "harness" / "capture"
    split = json.loads((BESTOFN / "split.json").read_text())
    recorded = {r["game"]: r for r in split["rows"] if r.get("call_index") == 1}
    out_dir = SCRATCH / "bon_shown"
    out_dir.mkdir(parents=True, exist_ok=True)

    def line(t) -> str:
        click = f" data={t.data}" if t.data else ""
        return (
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (FULL, run-length) = {e3._rle_delta_compact(t.grid, t.next_grid)}"  # noqa: SLF001
        )

    k_historical = 8
    status: dict[str, dict] = {}
    for g in sorted(recorded):
        tp = cap / g / "transitions1.pkl"
        pp = cap / g / "prompt1_combined.txt"
        if not tp.exists() or not pp.exists():
            status[g] = {"status": "capture_missing"}
            continue
        with open(tp, "rb") as fh:
            prefix = list(pickle.load(fh))
        prompt = pp.read_text()
        changed = [
            t for t in prefix if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
        ]
        noop = [t for t in prefix if np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))]
        shown = changed[: max(0, k_historical - 2)] + noop[:2]
        checks = {
            "every_shown_line_in_prompt": all(line(t) in prompt for t in shown),
            "n_action_lines_matches_shown": prompt.count("--- ACTION") == len(shown),
            "n_shown_matches_split_json": len(shown) == recorded[g].get("n_shown"),
        }
        if not all(checks.values()):
            status[g] = {"status": "reconstruction_unproven", "checks": checks}
            continue
        p = out_dir / f"{g}_shown.pkl"
        with open(p, "wb") as fh:
            pickle.dump({"shown": shown, "held": [], "cell": 1}, fh)
        status[g] = {
            "status": "ok",
            "path": str(p),
            "n_shown": len(shown),
            "n_prefix": len(prefix),
            "checks": checks,
        }
    return status


def bestofn_units() -> list[dict]:
    """48 single-shot best-of-N candidates.

    THE `.txt` ON DISK IS THE RAW COMPLETION, NOT THE CODE. `bestofn.py` writes the server's
    `content` verbatim and separately records `code_sha256_16` of
    `e3._extract_python(text) or text.strip()` -- the fenced block, not the prose around it. The
    first version of this function read the `.txt` directly and 47 of 48 shas disagreed, which is
    the only reason the mismatch was caught: this pass RE-DERIVES the extraction and CHECKS the
    sha per candidate, so a future change to `_extract_python` cannot silently re-point the
    taxonomy at different bytes than the frozen record was scored on. Rows whose sha does not
    reproduce are marked and excluded from rate denominators rather than quietly classified.

    Pairing is by `completion_file` recorded in `bon.json`, never by list order.
    """
    sys.path.insert(0, str(REPO / "python"))
    from carnot.agentic import arc_executable_world_model as e3

    bon_dir = BESTOFN / "harness" / "bon" / "gpu1"
    bon = json.loads((bon_dir / "bon.json").read_text())
    scored = json.loads((BESTOFN / "bestofn_scored.json").read_text())
    by_key = {(c["game"], c["candidate"]): c for c in scored["candidates"]}

    extracted_dir = SCRATCH / "bon_extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    shown_status = rebuild_bestofn_shown()

    units = []
    for r in bon["rows"]:
        game, k = r["game"], r["candidate"]
        src = bon_dir / r["completion_file"]
        code = None
        sha_ok = None
        p = None
        if src.exists():
            text = src.read_text()
            code = e3._extract_python(text) or text.strip()  # noqa: SLF001
            import hashlib

            sha_ok = hashlib.sha256(code.encode()).hexdigest()[:16] == r["code_sha256_16"]
            p = extracted_dir / f"{game}_k{k}.py"
            p.write_text(code)
        c = by_key.get((game, k), {})
        units.append(
            {
                "corpus": "bestofn_31B_single_shot",
                "cell": f"{game}_k{k}",
                "game": game,
                "code_path": str(p) if p else None,
                "code_present": p is not None,
                "code_sha_reproduces": sha_ok,
                # The frozen `_shown` rows, NOT a rebuilt window -- see `rebuild_bestofn_shown`.
                "window_pkl_override": (shown_status.get(game) or {}).get("path"),
                "shown_reconstruction": shown_status.get(game),
                "completion_chars": len(src.read_text()) if src.exists() else None,
                "code_chars": r.get("code_chars"),
                "stop_type": r.get("stop_type"),
                "predicted_n": r.get("predicted_n"),
                "budget": r.get("n_predict_requested") or 4096,
                "retry_tries": 1,
                "sampler": r.get("sampler"),
                "temperature": r.get("temperature"),
                "frozen_defect_kinds": sorted(r.get("defect_kinds") or []),
                "frozen_engine_changes_anything": r.get("engine_changes_anything"),
                "frozen_usable": r.get("usable"),
                "frozen_generate_would_accept": r.get("generate_would_accept"),
                "frozen_criteria": c.get("criteria"),
                "frozen_has_engine": c.get("has_engine"),
                "frozen_plan_found": c.get("plan_found"),
                "frozen_shipped_gate_passes": c.get("shipped_gate_passes"),
                "frozen_score_status": c.get("score_status"),
            }
        )
    return units


def abcf_units() -> list[dict]:
    """124 A/B cells from the 3-try induce loop; 116 have an engine file, 8 do not."""
    rows = json.loads((ABCF / "rows.json").read_text())
    units = []
    for r in rows:
        cell = f"{r['game']}__r{r['replicate']}__{r['arm']}"
        p = ABCF / "engines" / cell / r["game"] / "world_model.py"
        units.append(
            {
                "corpus": "abcf_31B_after_3_tries",
                "cell": cell,
                "game": r["game"],
                "arm": r["arm"],
                "code_path": str(p) if p.exists() else None,
                "code_present": p.exists(),
                "stop_type": None,  # not recorded by run_ab.py
                "budget": 4096,
                "retry_tries": 3,
                "induce_ok": bool(r["induce_ok"]),
                "induce_msg": r.get("induce_msg"),
                "frozen_change_fidelity": (r.get("heldout") or {}).get("change_fidelity"),
            }
        )
    return units


def ab0728_units() -> list[dict]:
    """240 cells on the RETIRED Qwen3.5-9B generator. Contrast stratum, never pooled."""
    rows_p = AB0728 / "rows.json"
    if not rows_p.exists():
        return []
    rows = json.loads(rows_p.read_text())
    units = []
    for r in rows:
        cell = r.get("cell_id") or f"{r['game']}__r{r['replicate']}__{r['arm']}"
        cands = sorted((AB0728 / "e3").glob(f"{cell}/{r['game']}/world_model.py"))
        if not cands:
            cands = sorted((AB0728 / "cells").glob(f"{cell}/**/world_model.py"))
        p = cands[0] if cands else None
        units.append(
            {
                "corpus": "ab0728_qwen9B_retired_generator",
                "cell": cell,
                "game": r["game"],
                "arm": r.get("arm"),
                "code_path": str(p) if p else None,
                "code_present": p is not None,
                "stop_type": r.get("stop_type"),
                "budget": 4096,
                "retry_tries": 3,
                "induce_ok": bool(r.get("induce_ok")),
                "induce_msg": r.get("induce_msg"),
                "generated_tokens": r.get("generated_tokens"),
            }
        )
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-secondary", action="store_true", help="also run the Qwen9B contrast")
    ap.add_argument("--only", default=None, help="restrict to one corpus (substring match)")
    ap.add_argument("--out", default=str(HERE / "classification.json"))
    args = ap.parse_args()

    t0 = time.time()
    units = bestofn_units() + abcf_units()
    if args.with_secondary:
        units += ab0728_units()
    if args.only:
        units = [u for u in units if args.only in u["corpus"]]

    games = sorted({u["game"] for u in units if u.get("code_present")})
    print(f"{len(units)} units across {len(games)} games; rebuilding windows")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    windows: dict[str, str] = {}
    window_status: dict[str, dict] = {}
    for g in games:
        wp = SCRATCH / f"{g}_window.pkl"
        if wp.exists():
            windows[g] = str(wp)
            window_status[g] = {"status": "ok", "cached": True}
            print(f"  {g}: cached")
            continue
        r = run_worker(
            "window_worker.py", {"game": g, "window_pkl": str(wp)}, f"win_{g}", WINDOW_TIMEOUT_S
        )
        window_status[g] = r
        if r.get("status") == "ok" and wp.exists():
            windows[g] = str(wp)
            print(f"  {g}: shown={r['n_shown']} held={r['n_heldout']}")
        else:
            print(f"  {g}: NOT REBUILT ({r.get('status')})")

    print(f"\nclassifying {len(units)} units")
    out_rows = []
    for i, u in enumerate(units):
        if not u.get("code_present"):
            # NO CODE AT ALL is itself a class, and the most severe one: the generator was asked
            # and, after however many tries this corpus allows, produced nothing loadable. The
            # detector cannot be run on a file that does not exist, so the row is classified from
            # the induce message the harness recorded rather than being dropped.
            out_rows.append({**u, "status": "no_code_file", "defect_kinds": [], "parses": None})
            continue
        r = run_worker(
            "classify_worker.py",
            {
                "cell": u["cell"],
                "game": u["game"],
                "corpus": u["corpus"],
                "code_path": u["code_path"],
                "window_pkl": u.get("window_pkl_override") or windows.get(u["game"]),
                "stop_type": u.get("stop_type"),
                "budget": u.get("budget"),
            },
            f"cls_{u['corpus'][:6]}_{u['cell']}",
            CLASSIFY_TIMEOUT_S,
        )
        out_rows.append({**u, **r})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(units)} ({time.time() - t0:.0f}s)")

    payload = {
        "what_this_is": __doc__,
        "duration_s": round(time.time() - t0, 1),
        "n_units": len(out_rows),
        # METHODOLOGY, on the DETAIL file and not only on the scored artifact. This file carries
        # GGUF strings inside the induce messages it quotes from the frozen corpora, so a reader
        # (or `adversarial_verify.py`) meeting it alone would reasonably ask what compute produced
        # it. The answer is none: no model is loaded and no token is generated here.
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_note": (
            "The shipped mechanical detector run against pre-existing frozen candidates. GGUF "
            "names appearing in quoted `induce_msg` fields identify the generator that produced "
            "the CORPUS; none was invoked by this pass."
        ),
        "model_specs": {
            "invoked": False,
            "generator_of_the_primary_corpora": "unsloth/gemma-4-31B-it (qat UD-Q4_K_XL / Q4_K_M)",
            "generator_of_the_secondary_contrast_corpus": (
                "unsloth/Qwen3.5-9B-MTP-GGUF -- RETIRED 2026-07-28, never pooled with the primary"
            ),
        },
        "random_seed": 20260801,
        "random_seed_note": (
            "The classification is deterministic -- frozen code re-executed against frozen "
            "transitions. The seed is recorded because the downstream analysis (bootstrap, "
            "permutation test) consumes it."
        ),
        "reproducibility_checksum": _inputs_checksum(),
        "windows": window_status,
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
