"""The four arms of the convention-perturbation transfer battery.

WHY NEW ARMS AND NOT THE EXISTING A / B2 / G3
---------------------------------------------
Both of the existing harness's reference arms became INVALID as controls the moment the two
levers were flipped on, because `_fd_gate`'s precedence is explicit-kwarg > env > default and
neither arm pins the full flag set:

  arm A  pins only {tier_exhaustion: False, frontier_gradient: False}.  Since 2026-07-25 it
         therefore INHERITS edge_bar_hud_mask=True + both HUD safety stages=True, so "the
         baseline" now silently ships the HUD treatment.
  arm B2 pins only {tier_exhaustion, tier_uniform_random, frontier_gradient}.  It likewise
         inherits the whole HUD trio, so the arm that was the CONTROL for the HUD A/B has
         become the HUD TREATMENT.

Re-running either against today's module would be measuring a treatment against itself.  So
every arm below pins ALL SEVEN gated flags explicitly.  An arm's behaviour is then a pure
function of its own kwargs and cannot drift when a SUBMITTED_* default changes again.

THE 2x2 FACTORIAL
-----------------
    CTRL   frontier OFF, HUD OFF   -- the pre-2026-07-25 agent, reconstructed exactly
    FRONT  frontier ON,  HUD OFF   -- replicates the configuration the frontier flip was
                                      measured in (the frontier A/B ran before the HUD flip)
    HUDO   frontier OFF, HUD ON    -- the HUD lever alone
    SHIP   frontier ON,  HUD ON    -- today's shipped live configuration

Main effects and the interaction are all identifiable:
    frontier effect given HUD off = FRONT - CTRL
    frontier effect given HUD on  = SHIP  - HUDO
    HUD effect given frontier on  = SHIP  - FRONT   (the shipped HUD claim's own contrast)
    HUD effect given frontier off = HUDO  - CTRL
    interaction                   = (SHIP - FRONT) - (HUDO - CTRL)

Attribution matters here because this project has already once credited a result to the
wrong mechanism (exp5950: the sampler was not the r11l fix).  A single "today's agent vs
yesterday's agent" contrast could not have told the two levers apart.
"""

from __future__ import annotations

# The seven flags every arm pins.  Named once so an arm cannot silently omit one.
GATED_FLAGS = (
    "tier_exhaustion",
    "tier_uniform_random",
    "tier_click_vocab_only",
    "frontier_gradient",
    "edge_bar_hud_mask",
    "hud_mask_collapse_guard",
    "hud_mask_stage2_confirm",
)

_FRONTIER_ON = {
    "tier_exhaustion": True,
    "tier_uniform_random": True,
    "tier_click_vocab_only": True,
    "frontier_gradient": False,  # measured WORSE than B2; stayed off at the flip
}
_FRONTIER_OFF = {
    "tier_exhaustion": False,
    "tier_uniform_random": False,
    "tier_click_vocab_only": False,
    "frontier_gradient": False,
}
_HUD_ON = {
    "edge_bar_hud_mask": True,
    "hud_mask_collapse_guard": True,
    "hud_mask_stage2_confirm": True,
}
_HUD_OFF = {
    "edge_bar_hud_mask": False,
    "hud_mask_collapse_guard": False,
    "hud_mask_stage2_confirm": False,
}

CPTB_ARMS = {
    "CTRL": {
        "label": "pre_flip_agent_all_six_levers_off_explicitly_pinned",
        "kwargs": {**_FRONTIER_OFF, **_HUD_OFF},
        "deterministic": True,
        "frontier": False,
        "hud": False,
    },
    "FRONT": {
        "label": "frontier_trio_on_hud_off",
        "kwargs": {**_FRONTIER_ON, **_HUD_OFF},
        "deterministic": False,
        "frontier": True,
        "hud": False,
    },
    "HUDO": {
        "label": "hud_trio_on_frontier_off",
        "kwargs": {**_FRONTIER_OFF, **_HUD_ON},
        "deterministic": True,
        "frontier": False,
        "hud": True,
    },
    "SHIP": {
        "label": "shipped_live_configuration_both_levers_on",
        "kwargs": {**_FRONTIER_ON, **_HUD_ON},
        "deterministic": False,
        "frontier": True,
        "hud": True,
    },
}

for _k, _v in CPTB_ARMS.items():
    missing = [f for f in GATED_FLAGS if f not in _v["kwargs"]]
    if missing:  # pragma: no cover - a construction error, not a runtime one
        raise AssertionError(f"arm {_k} does not pin {missing}: it would inherit a default")
