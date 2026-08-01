#!/usr/bin/env python3
"""Score ONE engine under every mask arm, through the SHIPPED `WorldModelVerifier`.

WHY THERE ARE SEVEN ARMS AND NOT TWO. "Mask on" is not one setting. `WorldModelVerifier`
resolves three independent things before a single grid is compared, and each can turn a
requested mask into no masking at all:

  1. `hud_mask_enabled` -- the module flag, or the explicit per-arm override. Passing
     `hud_mask=<array>` WITHOUT it is the documented silent no-op
     (`arc_world_model_trust_energy.hud_mask_silently_dropped`: supplied=True, enabled=False).
     Every masked arm here passes `hud_mask_enabled=True` explicitly, so the module default
     stays untouched and the arm cannot be silently dropped.
  2. Is there a mask to give? 14 of the 20 roster games ship an EMPTY default mask. Empty is a
     REFUSAL in the capture artifact's own words, and it resolves to `unresolved` -- not to
     masking.
  3. THE SWALLOW GUARD (REQ-ARC-WMTE-6015), which can refuse a real, supplied, enabled mask
     after measuring it. And WHICH CORPUS it measures on changes the verdict: the constructor's
     own docstring says a caller holding the whole corpus SHOULD pre-compute the check on it,
     because "a tail that happens to contain no genuine state change has ALL of its changed
     cells inside the HUD -- an honest mask then looks exactly like a swallowing one". The A/B
     tail is 3-4 transitions. Judging a mask on 3 transitions is exactly the false positive
     that paragraph describes, so both provenances are measured rather than one being chosen:

       *_swallow_full   verdict computed on the WHOLE window (shown + held-out). This is what
                        the docstring instructs a caller in this position to do, so it is the
                        arm the headline numbers come from.
       *_swallow_slice  verdict computed by the verifier itself on the graded tail alone, the
                        bare `WorldModelVerifier(held, hud_mask=m, hud_mask_enabled=True)`
                        construction. Reported as a sensitivity, because it is the literal
                        reading of "use the shipped verifier" and a reader should be able to
                        see whether the choice of corpus moved anything.

  *_forced_guard_bypassed  A DIAGNOSTIC, NEVER A HEADLINE. The guard is handed a record that
                        reads clean so the mask is applied unconditionally. This exists for one
                        reason: on tn36's best-of-N window the ONLY cells that move are the
                        progress bar's, so masking that row deletes every changing transition
                        and the guard refuses -- correctly, by its own rules. Without this arm
                        the answer to "do the six perfect scores survive masking" would be "the
                        guard declined to find out", which is true and useless. The forced arm
                        says what the number WOULD be, and is labelled a counterfactual
                        everywhere it appears. The synthetic record keeps every MEASURED field
                        of the real one under `real_*` keys, so nothing is hidden by overriding
                        it -- a reviewer can read the true verdict off the forced arm's own
                        record.

NOTHING HERE MUTATES A SHIPPED DEFAULT. `SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED` is not
touched and `CARNOT_ARC_WM_HUD_MASK` is not set; every arm passes its setting explicitly. The
unmasked arm is `hud_mask_enabled=False`, which is what the A/B ran, and it is re-derived here
rather than copied so it can be checked against the A/B's recorded value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Fields lifted off VerifyResult for every arm. `hud_mask_status` and `hud_mask_cells` are the
# PROOF-OF-ACTIVATION pair: `applied` with a non-zero cell count is the only combination that
# means grids were actually collapsed. Any other status is a masked-named arm that masked
# nothing, and the artifact reports it as such rather than as a second column of numbers.
_FIELDS = (
    "n",
    "n_correct",
    "accuracy",
    "cell_recall",
    "n_changing",
    "n_changes_correct",
    "change_accuracy",
    "change_fidelity",
    "correct_changed_cells",
    "spurious_changed_cells",
    "invented_changed_cells",
    "invented_change_rate",
    "n_noop",
    "n_noop_hallucinated",
    "noop_hallucination_rate",
    "noop_channel_measurable",
    "n_levelup_rows_excluded",
)


def _vr_dict(vr: Any) -> dict:
    out: dict = {}
    for f in _FIELDS:
        v = getattr(vr, f)
        out[f] = (
            bool(v)
            if isinstance(v, (bool, np.bool_))
            else (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 6))
        )
    out["hud_mask_status"] = str(vr.hud_mask_status)
    out["hud_mask_cells"] = int(vr.hud_mask_cells)
    out["hud_mask_swallow_source"] = str(vr.hud_mask_swallow_source)
    sw = dict(vr.hud_mask_swallow)
    # On a forced arm the three fields the guard READS are overridden, so the MEASURED
    # quantities live under `real_*`. Falling back to them here means the forced arm still
    # reports the true verdict in the same columns as every other arm -- otherwise the one arm
    # a reviewer most needs to audit would be the one showing nulls.
    out["swallow_reason"] = sw.get("real_reason", sw.get("reason"))
    out["swallow_swallows"] = bool(sw.get("real_swallows", sw.get("swallows")))
    out["swallow_changed_cell_overlap"] = sw.get(
        "real_changed_cell_overlap", sw.get("changed_cell_overlap")
    )
    out["swallow_raw_changing"] = sw.get(
        "real_raw_changing_transitions", sw.get("raw_changing_transitions")
    )
    out["swallow_masked_changing"] = sw.get(
        "real_masked_changing_transitions", sw.get("masked_changing_transitions")
    )
    out["swallow_n_transitions"] = sw.get("real_n_transitions", sw.get("n_transitions"))
    out["guard_bypassed_on_this_arm"] = bool(sw.get("SYNTHETIC_GUARD_BYPASS"))
    return out


def _forced_record(real: dict) -> dict:
    """A swallow record that reads clean to the guard, carrying the real one inside it.

    `hud_mask_swallow_clean` requires exactly `checked and not swallows and reason == "ok"`.
    Those three are overridden and NOTHING else is: every measured quantity of the true verdict
    survives under a `real_` prefix and lands in the artifact via `VerifyResult.hud_mask_swallow`,
    so the arm carries its own disclosure. Overriding a guard silently would be the failure this
    whole re-score is auditing; overriding it in public, with the true verdict attached, is a
    stated counterfactual.
    """
    out = {f"real_{k}": v for k, v in real.items()}
    out.update(
        {
            "checked": True,
            "swallows": False,
            "reason": "ok",
            "SYNTHETIC_GUARD_BYPASS": True,
            "why": (
                "REQ-ARC-WMTE-6015 deliberately bypassed to measure the counterfactual score. "
                "Never a headline. The true measured verdict is under the real_* keys."
            ),
        }
    )
    return out


def score_all_arms(
    engine: Any,
    graded: list,
    full_corpus: list,
    masks: dict,
) -> dict:
    """`graded` is what gets scored (the held-out tail); `full_corpus` only feeds the guard."""

    from carnot.agentic import arc_executable_world_model as e3

    arms: dict = {}

    def run(name: str, mask: np.ndarray | None, enabled: bool, swallow: dict | None) -> None:
        try:
            # `hud_mask_swallow=None` is NOT the same as omitting it: omitting makes the
            # verifier compute the verdict on its own slice, which is the `*_slice` arms'
            # whole point. Passing None explicitly would too (the constructor's default IS
            # None), so the branch is kept literal rather than clever.
            kw: dict = {"hud_mask": mask, "hud_mask_enabled": enabled}
            if swallow is not None:
                kw["hud_mask_swallow"] = swallow
            arms[name] = _vr_dict(e3.WorldModelVerifier(list(graded), **kw).score(engine))
        except Exception as exc:  # noqa: BLE001
            arms[name] = {"arm_error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    run("unmasked", None, False, None)

    swallow_meta: dict = {}
    for key, arm in (("default", "default"), ("conditional", "conditional")):
        m = masks.get(key)
        if m is None:
            # NO MASK IS STILL AN ARM, AND IT IS SCORED. The first version of this function
            # short-circuited here and emitted a stub with no `change_fidelity`, which made the
            # downstream analysis DROP those games -- so the masked primary silently ran on the
            # 6 masked games instead of all 20, and the masked ranking on 5 candidates instead
            # of 31. A missing number reads as an absent unit, and an absent unit changes the
            # test. Running the verifier with `hud_mask=None, hud_mask_enabled=True` is also the
            # honest construction: it is exactly what the live path does when the flag is on and
            # the detector resolved nothing, and it records `hud_mask_status == "unresolved"` --
            # the explicit "asked for masking, had none to give" state, which is the fact here.
            for suffix in ("swallow_full", "swallow_slice", "forced_guard_bypassed"):
                run(f"{arm}_{suffix}", None, True, None)
                arms[f"{arm}_{suffix}"]["no_mask_available"] = True
            swallow_meta[key] = {"no_mask_available": True}
            continue
        full_rec = e3.hud_mask_swallow_check(list(full_corpus), m)
        swallow_meta[key] = dict(full_rec)
        run(f"{arm}_swallow_full", m, True, full_rec)
        run(f"{arm}_swallow_slice", m, True, None)
        run(f"{arm}_forced_guard_bypassed", m, True, _forced_record(full_rec))

    return {"arms": arms, "full_corpus_swallow_check": swallow_meta}
