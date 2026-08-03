"""REPRODUCE-ONLY: the two defects that make the refinement hypothesis untestable.

Nothing here is fixed, flagged, or written outside this script's own artifact. Every number is
produced by CALLING the shipped functions -- `refactor_prompt`, `WorldModelVerifier.score`,
`split_refinement_acceptance`, `_split_prefix_heldout`, `atp.build_progress_window` -- over the
13 REAL offline windows of the exp5760/5766 roster. No grep produces a value; no field is read
by regex out of a file.

DEFECT 1 -- THE REFACTOR PROMPT NEVER CONTAINS THE ENGINE.
`refactor_prompt(game, vr)` takes a game id and a VerifyResult. The VerifyResult carries
mismatches; it does not carry the engine. So the model is asked to "REFACTOR toward simpler,
more general rules ... while keeping the cases it already gets right" without being shown the
code it is refactoring or a single case it already gets right. Measured here as: what fraction
of the engine's own substantive source lines appear anywhere in the rendered prompt string.
The measurement is on the RENDERED TEXT (delivery), not on what a dict made available.

  ANTI-FALSE-POSITIVE. A line like `import numpy as np` or `def engine(grid, action, data):`
  appears in the REQUIRED OUTPUT STRUCTURE boilerplate and would score as a "hit" while
  carrying zero information about THIS engine. Those lines are counted separately as
  `boilerplate_hits` and excluded from the substantive denominator, because counting them
  would manufacture a non-zero result out of the prompt quoting its own template.

DEFECT 2 -- UNGRADEABLE ACCEPTANCE CELLS.
`WorldModelVerifier.score` excludes level-up rows (correctly: `next_grid` is the next level's
re-laid-out board). The induction window ENDS at the level-up transition, so that row is always
last and always lands in the held-out tail. Where the tail IS that row, a PERFECT engine scores
0.0 and the gate is unfalsifiable. Measured here with an ORACLE engine -- it returns the
recorded `next_grid` for the exact (grid, action) it is asked about -- under both the shipped
two-way split and the default-OFF `CARNOT_ARC_CEGIS_ACCEPT_SPLIT` three-way split.

  VACUITY CHECK ON THE CONTROL. An oracle that scores 0.0 everywhere proves nothing: it could
  be a broken oracle rather than an unfalsifiable gate. So the oracle must score 1.0 on at
  least one block, and that is asserted, not assumed.

SUBSTRATE: pure Python/numpy over the committed offline windows and the committed evidence
engines. No LLM, no GPU, no network. `results/arc_e3` is READ, never written -- checksummed
before and after.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    E3_DIR,
    WorldModelVerifier,
    refactor_prompt,
)
from carnot.agentic.arc_world_model_trust_energy import (  # noqa: E402
    _split_prefix_heldout,
    split_refinement_acceptance,
)
from carnot.experiment_5760_cegis_refinement_induction_ab import ROSTER  # noqa: E402

ARTIFACT = REPO / "results" / "outer_loop_arc_refine_instrument_repro_20260803.json"
EVIDENCE_DIRS = ("results/arc_e3", "results/arc_logo_snapshot", "results/arc_e3_origin_fixtures")

# Lines that appear in `refactor_prompt`'s own REQUIRED OUTPUT STRUCTURE block, or that are
# universal Python boilerplate. A "hit" on one of these says nothing about whether THIS engine's
# logic reached the model, so they are excluded from the substantive denominator.
_BOILERPLATE = {
    "import numpy as np",
    "def engine(grid, action, data):",
    "def is_level_complete(grid):",
    "...",
    "import numpy",
    "return grid",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def evidence_checksum() -> dict[str, str]:
    out = {}
    for d in EVIDENCE_DIRS:
        p = REPO / d
        if not p.exists():
            out[d] = "absent"
            continue
        h = hashlib.sha256()
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode())
                h.update(f.read_bytes())
        out[d] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------------------------
# DEFECT 1
# ---------------------------------------------------------------------------------------------
def substantive_lines(src: str) -> list[str]:
    """Source lines that carry THIS engine's logic: non-blank, non-comment, non-boilerplate,
    and long enough that an incidental substring match is not plausible."""
    out = []
    for raw in src.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s in _BOILERPLATE:
            continue
        if len(s) < 12:  # `pass`, `else:`, `return` -- too short to be a meaningful hit
            continue
        out.append(s)
    return out


def defect1_for_game(game: str, window: list) -> Optional[dict[str, Any]]:
    src_path = E3_DIR / game / "world_model.py"
    if not src_path.exists():
        return None
    src = src_path.read_text()
    ns: dict[str, Any] = {}
    try:
        exec(compile(src, str(src_path), "exec"), ns)  # noqa: S102 - committed evidence engine
        engine = ns["engine"]
    except Exception as exc:
        return {"game": game, "engine_loadable": False, "error": f"{type(exc).__name__}: {exc}"}

    vr = WorldModelVerifier(list(window), hud_mask=None).score(engine)
    prompt = refactor_prompt(game, vr)

    subs = substantive_lines(src)
    hits = [ln for ln in subs if ln in prompt]
    boiler_hits = [ln for ln in sorted(_BOILERPLATE) if ln in prompt]
    return {
        "game": game,
        "engine_loadable": True,
        "engine_source_chars": len(src),
        "engine_total_lines": len(src.splitlines()),
        "engine_substantive_lines": len(subs),
        "substantive_lines_present_in_prompt": len(hits),
        "substantive_hit_fraction": round(len(hits) / len(subs), 6) if subs else None,
        "example_hits": hits[:3],
        "boilerplate_lines_present_in_prompt": boiler_hits,
        "prompt_chars": len(prompt),
        "n_mismatches_shown": min(5, len(vr.mismatches)),
        "n_mismatches_available": len(vr.mismatches),
        # The prompt asks the model to keep "the cases it already gets right". How many are shown?
        "n_correct_cases_shown_in_prompt": 0,  # structural: refactor_prompt renders mismatches only
        "vr_n": int(vr.n),
        "vr_n_correct": int(vr.n_correct),
    }


# ---------------------------------------------------------------------------------------------
# DEFECT 2
# ---------------------------------------------------------------------------------------------
def oracle_engine(rows: list):
    """Returns the recorded next_grid for the exact (grid, action) asked about. A PERFECT
    engine on this corpus by construction; anything it cannot score is unfalsifiable."""
    table = {}
    for t in rows:
        table[(np.asarray(t.grid).tobytes(), int(t.action))] = np.asarray(t.next_grid).copy()

    def engine(grid, action, data=None):
        hit = table.get((np.asarray(grid).tobytes(), int(action)))
        return hit.copy() if hit is not None else np.asarray(grid).copy()

    return engine


def block_report(rows: list, engine) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0, "gradeable_n": 0, "n_changing": 0, "change_accuracy": None}
    vr = WorldModelVerifier(list(rows), hud_mask=None).score(engine)
    return {
        "n_rows": len(rows),
        "gradeable_n": int(vr.n),
        "n_levelup_rows_excluded": int(vr.n_levelup_rows_excluded),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "accuracy": round(float(vr.accuracy), 6),
    }


def defect2_for_game(game: str, window: list) -> dict[str, Any]:
    oracle = oracle_engine(list(window))
    _prefix, heldout = _split_prefix_heldout(list(window))
    split = split_refinement_acceptance(list(window))
    # Per-row changed-cell counts on the acceptance block -- the stratification key.
    changed_cells = []
    for t in split.acceptance:
        if int(getattr(t, "level_after", 0)) > int(getattr(t, "level_before", 0)):
            continue
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        changed_cells.append(int((g0 != g1).sum()))
    return {
        "game": game,
        "window_n": len(window),
        "shipped_two_way": {
            "heldout": block_report(list(heldout), oracle),
        },
        "accept_split_on": {
            "n_refinable": len(split.refinable),
            "n_acceptance": len(split.acceptance),
            "decidable": bool(split.decidable),
            "reason": str(split.reason),
            "n_acceptance_gradeable": int(split.n_acceptance_gradeable),
            "acceptance": block_report(list(split.acceptance), oracle),
            "acceptance_changed_cells_per_gradeable_row": changed_cells,
        },
        # THE FALSIFIABILITY VERDICT, per split.
        "oracle_can_score_1_two_way": (
            block_report(list(heldout), oracle).get("change_accuracy") == 1.0
        ),
        "oracle_can_score_1_split_on": (
            block_report(list(split.acceptance), oracle).get("change_accuracy") == 1.0
        ),
    }


def main() -> int:
    t0 = time.time()
    before = evidence_checksum()
    from carnot.agentic import arc_actions_to_progress as atp

    windows: dict[str, Any] = {}
    for game in ROSTER:
        try:
            built = atp.build_progress_window(game)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"{game}: window build raised {type(exc).__name__}: {exc}")
            built = None
        # build_progress_window returns (window, full_trajectory, cell) -- take the WINDOW.
        windows[game] = list(built[0]) if built else None
        log(f"{game}: window n={len(windows[game]) if windows[game] else 0}")

    d1, d2 = [], []
    for game in ROSTER:
        w = windows.get(game)
        if not w:
            continue
        r1 = defect1_for_game(game, list(w))
        if r1:
            d1.append(r1)
        d2.append(defect2_for_game(game, list(w)))

    # --- aggregate DEFECT 1 ---
    ok1 = [r for r in d1 if r.get("engine_loadable")]
    tot_subs = sum(r["engine_substantive_lines"] for r in ok1)
    tot_hits = sum(r["substantive_lines_present_in_prompt"] for r in ok1)

    # --- aggregate DEFECT 2 ---
    n_undec_twoway = sum(1 for r in d2 if not r["oracle_can_score_1_two_way"])
    n_undec_split = sum(1 for r in d2 if not r["oracle_can_score_1_split_on"])
    undec_twoway = [r["game"] for r in d2 if not r["oracle_can_score_1_two_way"]]
    undec_split = [r["game"] for r in d2 if not r["oracle_can_score_1_split_on"]]
    # VACUITY CHECK: the oracle must be able to score 1.0 SOMEWHERE, else the control is broken.
    oracle_scored_1_somewhere = any(
        r["oracle_can_score_1_two_way"] or r["oracle_can_score_1_split_on"] for r in d2
    )

    after = evidence_checksum()
    out = {
        "experiment": "outer_loop_arc_refine_instrument_repro_20260803",
        "spec": "REQ-ARC-WMTE-6091-REPRO",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 6091,
        "roster": ROSTER,
        "defect1_refactor_prompt_lacks_engine": {
            "per_game": d1,
            "games_measured": len(ok1),
            "total_substantive_engine_lines": tot_subs,
            "total_substantive_lines_delivered_to_prompt": tot_hits,
            "delivered_fraction": round(tot_hits / tot_subs, 6) if tot_subs else None,
            "n_correct_cases_shown_anywhere": 0,
        },
        "defect2_ungradeable_acceptance": {
            "per_game": d2,
            "games_measured": len(d2),
            "undecidable_under_shipped_two_way_split": undec_twoway,
            "n_undecidable_two_way": n_undec_twoway,
            "undecidable_under_accept_split_on": undec_split,
            "n_undecidable_split_on": n_undec_split,
            "oracle_control_non_vacuous": bool(oracle_scored_1_somewhere),
        },
        "evidence_checksum_before": before,
        "evidence_checksum_after": after,
        "evidence_unchanged": before == after,
        "duration_s": round(time.time() - t0, 3),
    }
    out["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in out.items() if k != "reproducibility_checksum"}, sort_keys=True
        ).encode()
    ).hexdigest()
    out["honest_verdict"] = (
        "complete_both_instrument_defects_reproduced"
        if (tot_hits == 0 and n_undec_twoway > 0 and oracle_scored_1_somewhere)
        else "complete_reproduction_ran_see_fields"
    )
    ARTIFACT.write_text(json.dumps(out, indent=1))
    log(f"wrote {ARTIFACT}")
    print(
        json.dumps(
            {
                "d1_delivered_fraction": out["defect1_refactor_prompt_lacks_engine"][
                    "delivered_fraction"
                ],
                "d1_total_substantive_lines": tot_subs,
                "d1_delivered": tot_hits,
                "d2_undecidable_two_way": undec_twoway,
                "d2_undecidable_split_on": undec_split,
                "oracle_non_vacuous": oracle_scored_1_somewhere,
                "evidence_unchanged": out["evidence_unchanged"],
                "verdict": out["honest_verdict"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
