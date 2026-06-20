# ARC action-efficiency score-lever fix (2026-06-20)

Operator: "let's address the action-efficiency score-lever problems." After `.417` A1 (frame-change
prune) and A2 (imitation prior) both logged honest negatives with the SAME signature
(`solve_rate_guard_failed`, 4/8 -> 3/8), the outer-loop diagnosed the root cause and shipped a
two-layer fix. Diagnosis via an 8-agent design workflow (4 forensic readers -> 3 design proposals ->
synthesis), all facts verified against the actual code.

## Diagnosis (two compounding causes)

**P1 — Measurement/formulation (the visible failure).** The gate `_verdict`
(`scripts/kaggle/arc_local_submission_gate.py`) compared raw `solved_count`. But the live explorer
reaches FIRST LEVELUP only at ~7724-7792 actions on an ~8000-action budget (97% of cap) on the games
it solves (baseline CORE: lp85=7792, m0r0=7789, sp80=7724, vc33=7731, median 7760). A lever that
merely REORDERS that chaotic near-budget trajectory flips one knife-edge solve 4<->3 => automatic
FAIL regardless of merit. Proof: A1's no-op recall was 0.053 (it pruned almost NOTHING) yet m0r0 went
solved@7789 -> unsolved@7803 and lp85 went 7792 -> 7905; both levers' `positive_control` PASSED. So
the METRIC was broken, not the levers — the 4->3 was order-perturbation noise, not a causal removal.

**P2 — Architecture (the real ceiling).** The levers are <=0.3%-action micro-optimizations
(A1: 7760->7766; A2: 7760->7733) on a fundamentally UNDIRECTED ~7800-action search (offline BFS solves
the same games in ~21 -> a 370x gap). Candidate-set pruning/biasing acts on per-frame fan-out width,
which cannot close a 370x gap. The 370x is dominated by the RESET-replay-from-root NAVIGATION tax plus
undirected frontier order — neither of which any candidate filter touches.

## The fix

### Layer 1 — outer-loop shipped NOW (gate + roadmap prompts; zero race with the conductor's explorer edits)

**Gate `_verdict` redesign (commit eefd3b15a) — CORE set-containment.** CORE := the games the baseline
solves {lp85,m0r0,sp80,vc33}. A lever must preserve EVERY core solve (set, not count) and cut median
actions on the FIXED core denominator (+inf for any core game not solved); new fringe solves are a
reported BONUS, NEVER netted against a core loss. This is the ONLY relaxation that still FAILs a config
trading core solves for fringe (A2 swapped 3 core for 2 fringe). Legacy count fallback keeps it working
against an old baseline JSON. `measure()` now emits `solved_games` + `actions_by_game`.

Verified: four+ fixtures (`tests/python/test_arc_submission_gate_verdict.py`) — A1->FAIL(lost m0r0),
A2->FAIL(lost core)+bonus-reported, positive->PASS(IMPROVED), neutral->PASS(non-inferior), bonus,
legacy fallback. On the REAL A1 artifact: control(baseline-vs-itself)->PASS non-inferior (the old gate
false-failed this), A1 with_prune->FAIL lost m0r0 (still caught). Noise-robust WITHOUT going soft.

**Roadmap A3/A4/A6/B2 prompt steering (commit e0af579a1)** — the solve-rate-preserving lever contract:
- A3 (adaptive budget): DEFER not DELETE — move the top candidate to the FRONT of `node['untested']`
  and KEEP the tail; never return `[rows[0]]`. Commit only on re-visited frames. Report
  `committed_frame_fraction_on_winning_path` (if ~0 the lever is architecturally inert).
- A4 (lazy best-first): keep `depth` PRIMARY sort key; lazy top-K is SCORING not FILTERING; DROP w=5
  (the explorer's own comment documents a value-override regressed the baseline) -> sweep {0,0.5,1,2};
  a weight wins only if `core_solves_preserved AND median_actions_on_core < w=0` AND wall-time in budget.
- A6 (integration): wire only CORE-containment-passing levers (value_weight=0 is an acceptable null);
  diagnose the real 97%-of-cost axis (RESET-replay nav tax) by MEASURING `forward_walk_hit_rate` (the
  `.416 nav fix half-engaged: `_serve` sets `awaiting` but `_ingest` still gates on `h != origin`).
- B2: notes the gate is already shipped; canonicalizes + CI-guards the fixtures + measures headroom
  budget B* (no blind cap raise).

### Layer 2 — conductor implements (explorer code; outer-loop does NOT touch)

- A3: `apply_adaptive_budget` returns the full REORDERED list (committed-to-front), not `[rows[0]]`.
- A4: best_first over the lazy value head, depth-primary `_frontier` blend, w=0 control short-circuit.
- A6: densify forward-edge `adj` so a known edge shortcuts RESET-replay (keep replay as the floor).

## Best bet + risks

**Best bet to actually move 7800:** A4 (the only queued lever that changes the GLOBAL trajectory, now
affordable via the `.416 232x lazy eval) to SHIP a weight; A6's measured replay-tax elimination is the
higher-ceiling bet to COLLAPSE the number. A3 acts on fan-out width (the axis A1 proved moves <=0.3%) —
likely an honest null. Realistic A4 outcome: honest null at most weights (v3 head is weak, LOO 0.674),
genuine upside only if a small w=0.5 routes deep wins now that cost is removed.

**Risks flagged:** (1) gate relaxation must not hide a real regression — CORE-containment + bonus-never-
netted is the only safe relaxation; the four fixtures are mandatory. (2) Do NOT bump the eval budget
blind (masks real slowdowns); B2 must MEASURE B*. (3) Multi-seed is theater on the deterministic E3
path — do not add it. (4) best-first wall-time must stay in the live per-step budget.

Cross-refs: `scripts/kaggle/arc_local_submission_gate.py`, `tests/python/test_arc_submission_gate_verdict.py`,
`research-roadmap.yaml` (A3/A4/A6/B2), `docs/research-notes/arc-417-shaping-action-efficiency.md`,
`docs/research-notes/loopwm-2606.18208-ingestion-2026-06-20.md` (A3 = the ACT/PonderNet adaptive budget).
