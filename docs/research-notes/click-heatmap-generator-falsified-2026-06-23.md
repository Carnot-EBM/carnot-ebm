# Click-heatmap-as-generator — falsified before build (outer-loop, 2026-06-23)

Operator asked to tackle the click-heatmap-generator lever (the other PURSUE_MED from the
`.427`-improvement analysis). Per the discipline the macro investigation taught — **falsify first,
build second** — I ran the note's pre-flight falsifier. **It came back DEAD. No generator built, no
CNN trained: ~30 minutes, decisive.**

## The lever (premise)
Our `SmallFrameChangeCNN` has a per-pixel `click_head` (64×64 sigmoid heatmap), but
`rich_action_candidates` only enumerates clicks at OBJECT CENTROIDS (`max_click=48`). A heatmap
GENERATOR (top-k of the per-pixel field) would add OFF-centroid click candidates the enumerator omits
— **but only helps if winning clicks actually fall off centroids.**

## Falsifier 1 — solver trajectories (circular, reported for completeness)
For every coord-bearing winning click in the 6 solved click-games (cd82/cn04/dc22/ft09/sp80/su15, 90
clicks), classified against the centroid enumerator: **OFF = 0/90 (0%)**. But this is CIRCULAR — those
trajectories came from a centroid-only solver, so they are centroid-covered by construction. Not proof.

## Falsifier 2 — ARC Public Demo HUMAN replays (NON-circular, decisive)
`data/arc_public_demo_human_replay_corpus` — free-clicking humans, 25 games, CC BY 4.0. Over every
human ACTION6 click that CHANGED the frame (`frame_delta > 0`), classified (x,y) against the centroid
enumerator. **4097 effective clicks:**

| class | count | % |
|---|---|---|
| exact centroid | 653 | 16% |
| near (≤2px of a centroid) | 2619 | 64% |
| on-object (non-centroid pixel of a detected object) | 789 | 19% |
| **OFF (empty/seam cell, no centroid covers)** | **36** | **0.9%** |

**Even free-clicking humans, who can click anywhere, land 99.1% of their effective clicks on or near an
object centroid; only 0.9% are truly off-object.** VERDICT: `DEAD_human_effective_clicks_centroid_covered`.
The near-2px (64%) clicks are functionally the centroid (same object); the on-object (19%) clicks select
a detected object the enumerator already proposes a centroid for. The heatmap generator's pixel-precision
targets the 0.9% off-object slice — negligible.

## The deeper insight (consistent with the macro finding)
ARC click games are **OBJECT-level interactions** (click an object to select/toggle it), so the
discriminating signal is WHICH object, not WHERE-precisely on the canvas. The centroid enumerator
already enumerates every object. So the click-game wall (su15/ft09/etc. stuck at L1) is **NOT a
candidate-coverage problem** — all the right clicks are already available — it is a
**sequencing/GUIDANCE problem** (which object, in which order). This is the SAME generation-guidance
wall the macro investigation found: both the depth lever (macros) and the coverage lever (click-heatmap)
are dead because the binding constraint is neither depth nor coverage — it is **generation guidance**,
which the `.427` action-effect predictor and `.428` goal-energy / expansion-prior attack.

## Disposition
Falsifiers ran on the prototype branch `outer-loop/click-heatmap` (`proto_click_centroid_falsifier.py`,
`proto_click_human_falsifier.py`), now **purged**. **Nothing built, nothing retained** — the findings are
preserved in this note. The click-heatmap-
generator candidate is RETIRED (premise falsified on non-circular human data). Two PURSUE-ranked levers
from the `.427`-improvement workflow are now both empirically dead (macro depth + click coverage),
sharply concentrating `.429+` on the GUIDANCE class.
