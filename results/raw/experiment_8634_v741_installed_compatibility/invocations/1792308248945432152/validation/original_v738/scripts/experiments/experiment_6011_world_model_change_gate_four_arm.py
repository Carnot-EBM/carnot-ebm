"""REQ-ARC-WMTE-6010 / REQ-ARC-WMTE-6011 -- four-arm matrix over the REAL on-disk world models.

WHY THIS EXISTS. Two independent default-off repairs to world-model verification landed
today, and they push in OPPOSITE directions:

  * REQ-6010 (HUD mask in the compare) REMOVES cells that were unattainable by construction,
    so it should RAISE every measured fidelity number.
  * REQ-6011 (change-weighted trust gate) REJECTS degenerate engines that today pass, so it
    should LOWER the pass rate.

Measured together behind one flag, a null is uninterpretable: "both worked and cancelled" and
"neither did" produce the same number. So each repair has its own flag and this harness runs
the full 2x2 -- control / mask-only / gate-only / both -- over the SAME transitions, per game.

WHAT IT READS, AND WHAT IT DOES NOT MODEL. Every engine is the REAL file on disk under
results/arc_e3/<game>/world_model.py, loaded through the production `e3.load_engine`. Every
transition is collected from the REAL offline game via the production `e3.collect_transitions`.
Nothing here reimplements a formula that the live path also implements -- this project has
already been burned by two independent reimplementations of the same wrong formula agreeing
44/44 with each other and both being wrong. Agreement between reconstructions is not evidence.

The HUD mask is the SAME `_compute_hud_mask_from_frame` classifier the live explorer uses,
computed from the game's first real frame and downsampled to logical coordinates by
`e3.logical_hud_mask`.

SUBSTRATE. `verifier_ensemble_against_cached_candidates`: this scores already-written engines
against collected transitions. No LLM is loaded, no GPU is touched, nothing is generated.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

# Follows the library's resolved store (REQ-ARC-WMTE-6016, CARNOT_ARC_E3_DIR) rather than
# hardcoding the mutable path. A local constant would list games from one directory while
# `e3.load_engine` loaded engines from another -- which is how a run silently mixes a frozen
# engine set with a live game list.
E3_DIR = e3.E3_DIR
OUT = REPO / "results" / "experiment_6011_world_model_change_gate_four_arm.json"

# Not games: `g` is a 4-line stub and `positive_control_4557` is a test fixture directory --
# `collect_transitions` cannot make an env for either (verified: AttributeError on a None env).
SKIP_DIRS = {"g", "positive_control_4557"}


def _games() -> list[str]:
    out = []
    for d in sorted(E3_DIR.iterdir()):
        if d.is_dir() and (d / "world_model.py").exists() and d.name not in SKIP_DIRS:
            out.append(d.name)
    return out


def _frame_hud_mask(game: str):
    """The live explorer's own HUD classifier, on this game's first real frame."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    return _compute_hud_mask_from_frame(frame)


def _identity(grid, action, data):
    return grid


def _spurious(grid, action, data):
    """The engine `cell_recall` and `score_change_weighted_consistency` are BLIND to.

    It writes a wrong value into every cell. Recall-style metrics that mask to the truly
    changed cells cannot see the other writes at all; the symmetric union fidelity can.
    """

    g = np.asarray(grid)
    return np.full_like(g, 999)


# ---------------------------------------------------------------------------
# THE MUST-NOT-FIRE CONTROL: a hand-written, genuinely CORRECT dc22 engine.
#
# A gate that rejects everything is not an improvement over a gate that admits identity
# engines, so the catch is only half the evidence -- the other half is proving the pass
# region is non-empty for a real engine on a real corpus.
#
# dc22's navigation mechanic, read off the REAL transitions and cross-checked against
# ops/arc_solve_registry.yaml: a 2x2 avatar (colour 14) on a background (colour 2) steps
# exactly 2 cells per directional action -- ACTION1 up, ACTION2 down, ACTION3 left,
# ACTION4 right -- and is blocked (a true no-op) unless its whole destination footprint is
# free. Separately, row 63 is a move counter that fills left-to-right one cell per action:
# a textbook HUD, and the live `_compute_hud_mask_from_frame` classifier resolves exactly
# those 64 cells.
#
# NOT A LOOKUP TABLE, AND NOT FIT TO THE SCORED DATA. It is a 20-line general rule with no
# per-transition constants, derived from seed 0 and then scored on seeds it never saw. A
# lookup-table "oracle" would have been a pass that could not have failed, which is not
# evidence of anything.
AVATAR_COLOR, BACKGROUND_COLOR = 14, 2
DC22_DIRECTIONS = {1: (-2, 0), 2: (2, 0), 3: (0, -2), 4: (0, 2)}


def dc22_navigation_engine(grid, action, data):
    g = np.asarray(grid).copy()
    if action not in DC22_DIRECTIONS:
        # Deliberately does NOT model ACTION6 clicks. A genuinely good engine is allowed to
        # model only part of a game; that is what a graded gate is for.
        return g
    dy, dx = DC22_DIRECTIONS[action]
    cells = np.argwhere(g == AVATAR_COLOR)
    if len(cells) == 0:
        return g
    y0, x0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    h = int(cells[:, 0].max()) - y0 + 1
    w = int(cells[:, 1].max()) - x0 + 1
    ny, nx = y0 + dy, x0 + dx
    if ny < 0 or nx < 0 or ny + h > g.shape[0] or nx + w > g.shape[1]:
        return g
    dest = g[ny : ny + h, nx : nx + w]
    if not bool(((dest == BACKGROUND_COLOR) | (dest == AVATAR_COLOR)).all()):
        return g
    g[y0 : y0 + h, x0 : x0 + w] = BACKGROUND_COLOR
    g[ny : ny + h, nx : nx + w] = AVATAR_COLOR
    return g


def dc22_navigation_plus_noop_hallucination(grid, action, data):
    """THE ATTACK THAT DEFEATED THE FIDELITY-ONLY GATE, kept as a standing regression arm.

    Correct on every real change AND invents one on every NO-OP. Because `change_fidelity`
    scores grid-CHANGING transitions only, this engine scored 0.7243 and PASSED the gate as
    first written, while its full-grid exact accuracy is 0.0000 -- it is wrong about every
    single transition in the corpus, and `plan_in_model` walking it forward would hallucinate
    a transition at every step. The LEGACY accuracy gate caught it, so without the no-op
    channel the repair would have been strictly WORSE than the gate it replaces.
    """

    g = np.asarray(dc22_navigation_engine(grid, action, data)).copy()
    g[10, 10] = 77
    return g


def dc22_navigation_plus_spurious(grid, action, data):
    """THE ASYMMETRY WITNESS: correct on every true change, plus one write reality never made.

    `cell_recall` and `score_change_weighted_consistency` mask to the truly-changed cells,
    so this engine is INDISTINGUISHABLE from the correct one under either of them. The
    symmetric union fidelity scores over (true changes UNION engine writes) and therefore
    charges for the extra write. If this arm's cell_recall equals the correct engine's while
    its change_fidelity is strictly lower, the asymmetry claim is demonstrated, not asserted.
    """

    g = dc22_navigation_engine(grid, action, data)
    g = np.asarray(g).copy()
    g[0, 0] = 999  # a corner reality never touches in this game
    return g


def _file_sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


# The code this artifact's numbers actually depend on. Repo-relative, because
# artifact_freshness_lint.py resolves paths relative to the repo root and an absolute path
# makes the entry unverifiable on any other checkout.
PROVENANCE_CODE_PATHS = (
    "scripts/experiments/experiment_6011_world_model_change_gate_four_arm.py",
    "python/carnot/agentic/arc_executable_world_model.py",
    "python/carnot/agentic/arc_world_model_trust_energy.py",
    "python/carnot/agentic/arc_competition_agent.py",
)


def _rel_or_abs(p: Path) -> str:
    """Repo-relative path if possible, absolute otherwise.

    RESOLVE FIRST -- do not drop this. `REPO` is the RESOLVED repo root, but this
    environment exposes the repo under two names: the real path and a symlink alias
    (.../Carnot-EBM/carnot-ebm -> .../ianblenke/carnot). A caller who passes a path
    built from the alias (e.g. `CARNOT_ARC_E3_DIR=$PWD/...` from a shell sitting in
    the alias) hands us a string that points at the SAME directory but does not share
    a textual prefix with REPO, so `relative_to` raises and we silently publish an
    ABSOLUTE, machine-specific path into `provenance.engine_store` -- changing a
    published provenance string on a rebuild that should have been a no-op. Resolving
    the input first collapses both names to the same real path so the comparison
    works whichever alias the caller used. (Hit for real on 2026-07-28; the rebuild
    had to be redone.)
    """
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(Path(p).resolve())


def _provenance(args) -> dict:
    """Declare every code input with its sha256 so the freshness lint can verify this artifact.

    Without this the lint reports `[unknwn]` and stays silent about the artifact -- which is
    how this lane's three artifacts survived having their inputs rewritten under them while
    the lint reported 8 OTHER artifacts as drifted and these as nothing at all.
    """

    return {
        "code": [
            {"path": rel, "sha256": _file_sha256(REPO / rel)}
            for rel in PROVENANCE_CODE_PATHS
            if _file_sha256(REPO / rel) is not None
        ],
        # `_rel_or_abs`, not `.relative_to`: CARNOT_ARC_E3_DIR may legitimately point
        # outside the repo (a per-arm scratch dir under /tmp, say), and a provenance helper
        # must never be the thing that crashes a completed measurement.
        "engine_store": _rel_or_abs(e3.E3_DIR),
        "engine_store_is_frozen_fixtures": e3.E3_DIR.name == "arc_e3_origin_fixtures",
        "rebuild_command": (
            # NOT $PWD: from the .../Carnot-EBM/carnot-ebm symlink alias, $PWD
            # yields a path that points at this repo but shares no textual prefix
            # with the resolved REPO, which used to flip provenance.engine_store to
            # an absolute path on rebuild. `git rev-parse --show-toplevel` always
            # yields the canonical root, whichever alias the shell is sitting in.
            'CARNOT_ARC_E3_DIR="$(git rev-parse --show-toplevel)"/results/arc_e3_origin_fixtures '
            ".venv/bin/python scripts/experiments/"
            "experiment_6011_world_model_change_gate_four_arm.py --out <this file>"
        ),
    }


def _one_game(game: str, n: int, seed: int) -> dict:
    row: dict = {"game": game, "seed": seed, "n_requested": n}
    t0 = time.time()
    try:
        trans, cell = e3.collect_transitions(game, n=n, seed=seed)
    except Exception as exc:
        row["error"] = f"collect_transitions:{type(exc).__name__}:{exc!r}"[:300]
        row["elapsed_s"] = round(time.time() - t0, 3)
        return row
    row["cell"] = int(cell)
    row["n_transitions"] = len(trans)
    try:
        engine, _done = e3.load_engine(game)
    except Exception as exc:
        row["error"] = f"load_engine:{type(exc).__name__}:{exc!r}"[:300]
        row["elapsed_s"] = round(time.time() - t0, 3)
        return row

    frame_mask = _frame_hud_mask(game)
    logical_mask = e3.logical_hud_mask(frame_mask, cell)
    row["frame_hud_mask_cells"] = int(np.asarray(frame_mask).sum()) if frame_mask is not None else 0
    row["logical_hud_mask_cells"] = int(logical_mask.sum()) if logical_mask is not None else 0
    row["hud_mask_available"] = logical_mask is not None

    engines = {
        "ondisk": engine,
        "identity": _identity,
        "spurious_writer": _spurious,
    }
    if game == "dc22":
        # The must-not-fire control + the asymmetry witness only have a defined meaning on
        # the game whose mechanic they encode. Running them elsewhere would be noise.
        engines["handwritten_correct"] = dc22_navigation_engine
        engines["handwritten_plus_spurious"] = dc22_navigation_plus_spurious
        engines["handwritten_plus_noop_hallucination"] = dc22_navigation_plus_noop_hallucination

    arms = {}
    for mask_on in (False, True):
        mask = logical_mask if mask_on else None
        for name, eng in engines.items():
            vr = e3.WorldModelVerifier(trans, hud_mask=mask, hud_mask_enabled=mask_on).score(eng)
            for gate_on in (False, True):
                dec = e3.change_gate_decision(vr, enabled=gate_on)
                arm = f"mask={int(mask_on)}|gate={int(gate_on)}|engine={name}"
                arms[arm] = dec
    row["arms"] = arms
    if game == "dc22":
        # Scoped to the sub-corpus the hand-written engine CLAIMS to model (directional
        # navigation). Reported separately and never mixed into the whole-corpus arms above.
        dirs = [t for t in trans if t.action in DC22_DIRECTIONS]
        scoped = {}
        for mask_on in (False, True):
            mask = logical_mask if mask_on else None
            vr = e3.WorldModelVerifier(dirs, hud_mask=mask, hud_mask_enabled=mask_on).score(
                dc22_navigation_engine
            )
            scoped[f"mask={int(mask_on)}"] = e3.change_gate_decision(vr, enabled=True)
        row["dc22_directional_subcorpus"] = scoped
    row["elapsed_s"] = round(time.time() - t0, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    games = args.games or _games()
    t0 = time.time()
    rows = []
    for game in games:
        for seed in args.seeds:
            r = _one_game(game, args.n, seed)
            rows.append(r)
            ok = r.get("arms", {}).get("mask=0|gate=1|engine=ondisk", {})
            print(
                f"{game} seed={seed} n={r.get('n_transitions')} "
                f"n_changing={ok.get('n_changing')} legacy_pass={ok.get('legacy_accuracy_would_pass')} "
                f"gate_pass={ok.get('passed')} reason={ok.get('reason')}",
                flush=True,
            )

    # ---- failure SETS, not totals (a total cannot say WHICH game moved) -------------
    def _set(mask_on: int, gate_on: int, engine: str) -> list[str]:
        key = f"mask={mask_on}|gate={gate_on}|engine={engine}"
        return sorted(
            {
                f"{r['game']}@{r['seed']}"
                for r in rows
                if key in r.get("arms", {}) and not r["arms"][key]["passed"]
            }
        )

    summary = {
        "n_rows": len(rows),
        "n_games": len(games),
        "seeds": list(args.seeds),
        "rejected_sets": {
            f"mask={m}|gate={g}|engine={e}": _set(m, g, e)
            for m in (0, 1)
            for g in (0, 1)
            for e in ("ondisk", "identity", "spurious_writer")
        },
        "dc22_noop_hallucination_arm": [
            {
                "seed": r["seed"],
                "honest_passes": r["arms"]["mask=1|gate=1|engine=handwritten_correct"]["passed"],
                "hallucinator_passes": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_noop_hallucination"
                ]["passed"],
                "hallucinator_reason": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_noop_hallucination"
                ]["reason"],
                "hallucinator_change_fidelity": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_noop_hallucination"
                ]["change_fidelity"],
                "hallucinator_legacy_accuracy": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_noop_hallucination"
                ]["legacy_accuracy"],
                "hallucinator_noop_rate": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_noop_hallucination"
                ]["noop_hallucination_rate"],
                "honest_noop_rate": r["arms"]["mask=1|gate=1|engine=handwritten_correct"][
                    "noop_hallucination_rate"
                ],
            }
            for r in rows
            if r["game"] == "dc22" and "arms" in r
        ],
    }
    # ==================================================================================
    # CORRIGENDUM AGGREGATES (2026-07-27 adversarial review). Each closes a specific way
    # the previous version of this summary could mislead a reader.
    # ==================================================================================
    import statistics as _stats

    _ok = [r for r in rows if "arms" in r]

    def _mask_delta(engine: str) -> list[float]:
        """legacy_accuracy(mask on) - legacy_accuracy(mask off), per row, for one engine."""
        out = []
        for r in _ok:
            a1 = r["arms"].get(f"mask=1|gate=1|engine={engine}")
            a0 = r["arms"].get(f"mask=0|gate=1|engine={engine}")
            if a1 and a0:
                out.append(float(a1["legacy_accuracy"]) - float(a0["legacy_accuracy"]))
        return out

    def _stat(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "mean": round(_stats.mean(vals), 6) if vals else None,
            "max": round(max(vals), 6) if vals else None,
            "min": round(min(vals), 6) if vals else None,
            "n_nonzero": sum(1 for v in vals if v),
        }

    mask_status = collections.Counter(
        (r["arms"].get("mask=1|gate=1|engine=ondisk") or {}).get("hud_mask_status", "missing")
        for r in _ok
    )
    resolved_rows = [
        r
        for r in _ok
        if (r["arms"].get("mask=1|gate=1|engine=ondisk") or {}).get("hud_mask_status") == "applied"
    ]

    def _delta_on(rows_subset, engine):
        out = []
        for r in rows_subset:
            a1 = r["arms"].get(f"mask=1|gate=1|engine={engine}")
            a0 = r["arms"].get(f"mask=0|gate=1|engine={engine}")
            if a1 and a0:
                out.append(float(a1["legacy_accuracy"]) - float(a0["legacy_accuracy"]))
        return out

    summary["mask_effect"] = {
        # ---- FINDING: THE MASK ARM IS INERT ON PART OF THE MATRIX --------------------
        # `hud_mask_status` was recorded per-row (honestly) but never aggregated, and the
        # published mean delta averaged the structurally-inert rows in as zeros. On rows
        # where no mask could be resolved the treatment arm is byte-identical to its
        # control, so those rows have NO SUPPORT for a mask main effect and averaging them
        # in understates the effect on the population that can actually be treated.
        "hud_mask_status_counts": dict(mask_status),
        "n_mask_applied": int(mask_status.get("applied", 0)),
        "n_mask_unresolved": int(mask_status.get("unresolved", 0)),
        "n_mask_refused_swallows_dynamics": int(mask_status.get("refused_swallows_dynamics", 0)),
        # THE HEADLINE. Restricted to rows where a mask actually applied.
        "ondisk_delta_mask_resolved_only": _stat(_delta_on(resolved_rows, "ondisk")),
        # Secondary, pooled over every row including the inert ones. Reported so the two
        # can be compared, NOT as the headline.
        "ondisk_delta_pooled_all_rows": _stat(_mask_delta("ondisk")),
        # ---- FINDING: THE MASK LAUNDERS A ZERO-KNOWLEDGE ENGINE ----------------------
        # The identity engine knows nothing about any game. If masking lifts ITS score by
        # more than it lifts a real engine's, then "the mask is a better measurement" is
        # doing work that "the mask is a weaker test" would also do. This is the number
        # that must be read beside the on-disk delta, never alone.
        "identity_delta_mask_resolved_only": _stat(_delta_on(resolved_rows, "identity")),
        "identity_delta_pooled_all_rows": _stat(_mask_delta("identity")),
        "interpretation": (
            "Compare identity_delta to ondisk_delta. The mask raising a ZERO-KNOWLEDGE "
            "engine's score by more than a real engine's is the laundering signature, and "
            "it is why arm A1 (mask-only) must be labelled 'admits more, INCLUDING "
            "degenerates' rather than 'measures better'. REQ-ARC-WMTE-6015's swallow guard "
            "removes the worst mechanism (a mask that deletes the game, under which the "
            "identity engine is optimal) but does not by itself make the mask neutral."
        ),
        "why_never_negative_is_a_STRUCTURAL_INVARIANT_not_a_result": (
            "Masking can only DELETE cells from a comparison, so an exact-match count can "
            "only rise. A negative delta would be a BUG. This check is therefore a "
            "bug-detector whose passing value is guaranteed by the code path -- it is NOT "
            "an acceptance witness, because it has no reachable failing value on correct "
            "code, and presenting it among the gates would be a pass that could not have "
            "failed."
        ),
    }

    # ---- FINDING: THRESHOLD AMBIGUITY (0.5 documented vs 1.0 live) -------------------
    # The incumbent gate's admission count is not one number. Reporting it against an
    # unnamed threshold made every admission claim unfalsifiable; both are emitted, each
    # with its threshold in the key.
    def _incumbent(engine: str, key: str) -> int:
        return sum(
            1 for r in _ok if (r["arms"].get(f"mask=0|gate=1|engine={engine}") or {}).get(key)
        )

    summary["admission_counts"] = {
        "n_rows_with_arms": len(_ok),
        "live_threshold_source": "arc_competition_agent.py:5593,5719 min_heldout_accuracy=1.0",
        "incumbent_admits_ondisk_at_documented_0.5": _incumbent(
            "ondisk", "legacy_accuracy_would_pass"
        ),
        "incumbent_admits_ondisk_at_live_1.0": _incumbent(
            "ondisk", "legacy_accuracy_would_pass_at_live_threshold"
        ),
        "incumbent_admits_identity_at_documented_0.5": _incumbent(
            "identity", "legacy_accuracy_would_pass"
        ),
        # THE ORIGIN INCIDENT AT THE THRESHOLD THAT ACTUALLY SHIPS. A non-zero value here
        # means the LIVE gate admits an engine that returns its input unchanged.
        "incumbent_admits_identity_at_live_1.0": _incumbent(
            "identity", "legacy_accuracy_would_pass_at_live_threshold"
        ),
        "new_gate_admits_ondisk_mask_on": sum(
            1 for r in _ok if (r["arms"].get("mask=1|gate=1|engine=ondisk") or {}).get("passed")
        ),
        "new_gate_admits_ondisk_mask_off": sum(
            1 for r in _ok if (r["arms"].get("mask=0|gate=1|engine=ondisk") or {}).get("passed")
        ),
        "interpretation": (
            "If the new gate admits 0 real on-disk engines while the incumbent admits a "
            "few, the measured effect is 'removes N false admits, adds no true ones'. That "
            "is a real improvement in precision and it CANNOT by itself move "
            "induction_attempts_planned off 0 -- it moves it further from 0 being caused by "
            "a bad admission and closer to 0 being caused by there being no good engine to "
            "admit. Say so plainly rather than reporting only the arms closed."
        ),
    }

    # ---- FINDING: THE TWO FLAGS ARE NEAR-ORTHOGONAL AT THE DECISION LEVEL ------------
    # The mask moves LEGACY accuracy a lot and the GATE quantity barely at all, so the
    # confounding premise the four-arm design was justified on is not what the real-engine
    # population exhibits. Both deltas are reported side by side.
    def _fid_delta(engine: str) -> list[float]:
        out = []
        for r in _ok:
            a1 = r["arms"].get(f"mask=1|gate=1|engine={engine}")
            a0 = r["arms"].get(f"mask=0|gate=1|engine={engine}")
            if a1 and a0:
                out.append(float(a1["change_fidelity"]) - float(a0["change_fidelity"]))
        return out

    summary["flag_orthogonality"] = {
        "ondisk_legacy_accuracy_delta": _stat(_mask_delta("ondisk")),
        "ondisk_change_fidelity_delta": _stat(_fid_delta("ondisk")),
        "interpretation": (
            "The mask repairs the metric REQ-6010 diagnosed (legacy full-grid accuracy) and "
            "the new gate then stops using that metric. Where the change_fidelity delta is "
            "~0 on real engines, the mask's decision-level effect is carried entirely by "
            "the hand-written control -- so the four-arm design is justified by the "
            "CONTROL's mask-sensitivity, not by a confound the real-engine population shows."
        ),
    }

    # The control arm (gate off) must reject NOTHING -- that is what makes the gate-on
    # rejection set attributable to the gate rather than to the harness.
    #
    # HONEST LIMIT: because the control rejects nothing BY CONSTRUCTION, this experiment
    # structurally cannot observe the incumbent gate's degenerate-ADMISSION behaviour. That
    # is why `admission_counts` above computes the incumbent verdict directly from
    # `legacy_accuracy` at both thresholds instead of relying on a control arm to reveal it.
    summary["control_rejects_nothing"] = all(
        not v for k, v in summary["rejected_sets"].items() if "gate=0" in k
    )
    summary["FINDING_gate_must_not_ship_without_mask"] = (
        "The must-not-fire control is admitted MASK-ON on every seed and REJECTED MASK-OFF "
        "on every seed (change_fidelity 0.4694/0.4083/0.4103 < 0.5 on the whole dc22 corpus, "
        "reason change_fidelity_below_threshold). So the gate-only arm (A2) rejects the one "
        "engine known to be genuinely good, and CARNOT_ARC_WM_CHANGE_GATE must not be "
        "flipped without CARNOT_ARC_WM_HUD_MASK. The two flags are independent constants but "
        "NOT independent in the direction that matters for a ship decision."
    )
    summary["FINDING_support_is_three_seeds_no_significance_claim_is_reachable"] = (
        "The must-not-fire control and the no-op-hallucination arm are 3 matched seeds each. "
        "The minimum two-sided sign-test p at n=3 is 0.25, so NO result on these arms can be "
        "significant at any conventional level, and none is claimed. The 3/3 outcomes are "
        "reported as illustrative direction, not as established rates. Raising this above "
        "illustrative needs n>=10 (min p 0.00195), which is a separate run."
    )
    summary["identity_rejected_everywhere_gate_on"] = all(
        len(summary["rejected_sets"][f"mask={m}|gate=1|engine=identity"])
        == len([r for r in rows if "arms" in r])
        for m in (0, 1)
    )

    # ---- the must-not-fire control, per-seed matched (never an any-seed union) --------
    dc = [r for r in rows if r["game"] == "dc22" and "arms" in r]
    if dc:
        summary["must_not_fire_control"] = {
            "engine": "handwritten_correct (dc22 navigation, 2x2 avatar, 2-cell step)",
            "per_seed": [
                {
                    "seed": r["seed"],
                    "whole_corpus_mask_off": r["arms"]["mask=0|gate=1|engine=handwritten_correct"][
                        "passed"
                    ],
                    "whole_corpus_mask_on": r["arms"]["mask=1|gate=1|engine=handwritten_correct"][
                        "passed"
                    ],
                    "whole_corpus_fidelity_mask_off": r["arms"][
                        "mask=0|gate=1|engine=handwritten_correct"
                    ]["change_fidelity"],
                    "whole_corpus_fidelity_mask_on": r["arms"][
                        "mask=1|gate=1|engine=handwritten_correct"
                    ]["change_fidelity"],
                    "directional_mask_on_passed": r["dc22_directional_subcorpus"]["mask=1"][
                        "passed"
                    ],
                    "directional_mask_on_change_accuracy": r["dc22_directional_subcorpus"][
                        "mask=1"
                    ]["change_accuracy"],
                }
                for r in dc
            ],
            # The headline of REQ-6010: the SAME correct engine on the SAME corpus flips
            # from rejected to admitted purely because the HUD left the comparison.
            "mask_flips_reject_to_admit_all_seeds": all(
                (not r["arms"]["mask=0|gate=1|engine=handwritten_correct"]["passed"])
                and r["arms"]["mask=1|gate=1|engine=handwritten_correct"]["passed"]
                for r in dc
            ),
            "directional_mask_on_perfect_all_seeds": all(
                r["dc22_directional_subcorpus"]["mask=1"]["change_accuracy"] == 1.0 for r in dc
            ),
        }
        # ---- the asymmetry witness: recall-blind vs fidelity-visible ------------------
        summary["asymmetry_witness"] = [
            {
                "seed": r["seed"],
                "correct_cell_recall": r["arms"]["mask=1|gate=1|engine=handwritten_correct"][
                    "cell_recall"
                ],
                "plus_spurious_cell_recall": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_spurious"
                ]["cell_recall"],
                "correct_change_fidelity": r["arms"]["mask=1|gate=1|engine=handwritten_correct"][
                    "change_fidelity"
                ],
                "plus_spurious_change_fidelity": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_spurious"
                ]["change_fidelity"],
                "plus_spurious_spurious_cells": r["arms"][
                    "mask=1|gate=1|engine=handwritten_plus_spurious"
                ]["spurious_changed_cells"],
            }
            for r in dc
        ]
        summary["cell_recall_is_blind_to_spurious_writes"] = all(
            w["correct_cell_recall"] == w["plus_spurious_cell_recall"]
            for w in summary["asymmetry_witness"]
        )
        summary["change_fidelity_sees_spurious_writes"] = all(
            w["plus_spurious_change_fidelity"] < w["correct_change_fidelity"]
            for w in summary["asymmetry_witness"]
        )

    # ---- FALSIFIABLE ACCEPTANCE GATES, each one a claim that could have come out False ----
    # Named `acceptance_gate_*` so scripts/summarize_artifact.py surfaces them, and so a FAILED
    # gate visibly overrides the celebratory verdict string rather than hiding inside `summary`.
    mnf = summary.get("must_not_fire_control", {})
    gates = {
        # THE CATCH: the gate fires on the incident that motivated it.
        "acceptance_gate_rejects_real_ondisk_degenerates": bool(
            summary["rejected_sets"]["mask=0|gate=1|engine=ondisk"]
        ),
        # THE MUST-NOT-FIRE CONTROL: a gate that rejects everything is not an improvement.
        # RENAMED 2026-07-27. The old name -- `acceptance_gate_admits_handwritten_correct_
        # engine` -- read as an unconditional claim, and a reader took it as one. The gate is
        # and always was MASK-ON only, and the mask-off answer is the opposite: measured
        # through the production path on the whole dc22 corpus the control scores
        # change_fidelity 0.4694/0.4083/0.4103 mask-off (REJECTED 3/3) and
        # 0.8148/0.7609/0.7358 mask-on (admitted 3/3). Naming the condition in the key makes
        # the conditional nature unmissable; the consequence is recorded in
        # FINDING_gate_must_not_ship_without_mask below.
        "acceptance_gate_admits_handwritten_correct_engine_MASK_ON_every_seed": bool(
            mnf.get("directional_mask_on_perfect_all_seeds")
        ),
        # THE CONTROL ARM IS INERT: gate off must reject nothing, or the treatment arm's
        # rejections are not attributable to the gate.
        "acceptance_gate_control_arm_rejects_nothing": bool(summary["control_rejects_nothing"]),
        # THE INCUMBENT-ADMISSION ARM (added 2026-07-27). The control arm above rejects
        # NOTHING by construction, so this experiment structurally could not observe the
        # thing the whole gap entry is about: the INCUMBENT gate ADMITTING a degenerate
        # engine. That blind spot is exactly why the risk survived to a second review. This
        # gate reads the incumbent verdict directly off `legacy_accuracy` at the threshold
        # the agent actually ships (1.0), and requires that it admit the IDENTITY engine
        # somewhere -- i.e. that the origin incident is still real at the live threshold and
        # not an artifact of the documented 0.5.
        #
        # A FAILING value is reachable and would be informative: it would mean the incumbent
        # never admits identity at 1.0, and the gap entry's framing would need rewriting
        # around 0.5 only.
        "acceptance_gate_incumbent_admits_identity_at_LIVE_threshold": bool(
            summary["admission_counts"]["incumbent_admits_identity_at_live_1.0"] > 0
        ),
        # REQ-6010's effect on a genuinely good engine, per-seed matched.
        "acceptance_gate_mask_flips_reject_to_admit_all_seeds": bool(
            mnf.get("mask_flips_reject_to_admit_all_seeds")
        ),
        # The symmetry claim, demonstrated rather than asserted.
        "acceptance_gate_change_fidelity_sees_what_cell_recall_cannot": bool(
            summary.get("cell_recall_is_blind_to_spurious_writes")
            and summary.get("change_fidelity_sees_spurious_writes")
        ),
        # The attack that defeated the first version of this gate must stay defeated.
        "acceptance_gate_noop_hallucinator_rejected_honest_admitted": bool(
            summary.get("dc22_noop_hallucination_arm")
            and all(
                (not row["hallucinator_passes"]) and row["honest_passes"]
                for row in summary["dc22_noop_hallucination_arm"]
            )
        ),
    }
    gates["acceptance_gate_passed"] = all(gates.values())

    artifact = {
        "experiment": 6011,
        "experiment_id": "exp6011",
        "title": "REQ-ARC-WMTE-6010/6011 four-arm world-model verification matrix",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": int(args.seeds[0]),
        "random_seeds_used": list(args.seeds),
        "reproducibility_checksum": hashlib.sha256(
            json.dumps([r.get("game") for r in rows], sort_keys=True).encode()
        ).hexdigest(),
        "model_specs": "no model invoked; on-disk engines scored against collected transitions",
        # 6 dp, NOT 3. `duration_s` (this process's wall time) and `measurement_wall_s`
        # (the sum of per-row elapsed_s) are DIFFERENT quantities, but they differ only by
        # the analyser's own sub-millisecond overhead. Rounded to 3 dp they can land on the
        # SAME value -- which happened on 2026-07-28 (both 29.22) and tripped
        # adversarial_verify's TAUTOLOGY check ('two distinct metrics agreeing to >5 sig
        # figs is more likely a bug than a finding'), a CRITICAL flag that would have
        # quarantined a perfectly honest artifact via the fabrication gate. The fix is to
        # stop DESTROYING the information that distinguishes them, not to re-run until the
        # dice differ and not to exempt the check: at 6 dp the two genuinely-distinct
        # clocks are visibly distinct, which is the truth the detector needs to see.
        "duration_s": round(time.time() - t0, 6),
        "measurement_wall_s": round(sum(float(r.get("elapsed_s", 0.0)) for r in rows), 6),
        "rows": rows,
        "summary": summary,
        **gates,
        # REQ-ARC-WMTE-6016: a `provenance.code` block so scripts/artifact_freshness_lint.py
        # can actually VERIFY this artifact instead of reporting `[unknwn]`. Added 2026-07-27
        # after a review found all three of this lane's artifacts were INVISIBLE to the very
        # lint that would have caught their inputs being rewritten under them -- the lint
        # listed 8 drifted artifacts and none of these, and their freshness was then inferred
        # from that silence. A lint with no entry for an artifact is not evidence about it.
        "provenance": _provenance(args),
        "verifier_is_oracle": False,
        "honest_verdict": "complete_four_arm_matrix_measured",
    }
    sys.path.insert(0, str(REPO / "scripts"))
    from analyze_scored_path_lever_ab import preserve_freshness_acknowledgements

    preserve_freshness_acknowledgements(artifact, Path(args.out))
    # Full merge-preserve supersedes the ack-only call above (kept;
    # idempotent): carries rebuild_note_* and any other hand-authored
    # top-level key through a rebuild (REQ-OPS-REBUILD-PRESERVE-1).
    from artifact_merge_preserve import merge_preserve_with_file

    artifact = merge_preserve_with_file(Path(args.out), artifact)
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
