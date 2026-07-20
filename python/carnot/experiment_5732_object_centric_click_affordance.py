"""Experiment 5732: is there an OBJECT-CENTRIC, non-base-rate click-affordance
representation that a within-frame contrastive objective can learn, where the
falsified frame-only / global-conv representations (exp5727, exp5730) could not?

WHY THIS EXISTS (read before the code)
--------------------------------------
`ops/verifier_gaps.md :: GAP-ARCH-FRAME-CHANGE-PREDICTOR` is
`attempted_and_falsified`: two structurally different action-effect
representations both collapsed to the SAME degenerate structure -- a per-action-
TYPE base rate the live `PersistentAEM` memory already owns for free.

  * exp5727 (hand-crafted linear `cross_game_features_v3`): LOO AUROC 0.844, but
    an action-id-ONLY control scored 0.883 -> `frame_adds_over_action_id = -0.039`.
    The frame representation SUBTRACTS.
  * exp5730 (learned global-pooled conv `SmallFrameChangeCNN`): held-out AUROC
    0.539 (5-seed mean) vs an action-id-only control of 0.549 ->
    `frame_adds_over_action_id = -0.010`. Again subtracts. The one apparent
    positive (click discrimination within action-6, 0.918 AUROC on seed 4547)
    did NOT survive a 5-seed re-run (mean 0.570) and lost to its own
    untrained/random-init structural control (mean 0.580). Seed luck.

The revised missing discriminator (from the gap entry): a representation that
captures WHICH specific action, at WHICH specific state/location, produces a
change -- a genuine action x frame INTERACTION term NOT reducible to the
per-action-type marginal `PersistentAEM` already has.

THIS EXPERIMENT (design: docs/research-notes/arc-action-effect-representation-
redesign-2026-07-19.md; operator-approved). Replace the raw-frame representation
with an OBJECT-CENTRIC one -- featurize each action-6 click candidate by its
TARGET OBJECT (translation-invariant `object_hash` context: color, area,
`is_rect`, a normalized shape key; containment depth + adjacency degree from
`blob_topology`; a KxK local-neighborhood patch around the object centroid) --
and train it with a WITHIN-FRAME CONTRASTIVE RANKING objective that ranks the
object that actually changed the frame above the inert objects present in the
SAME frame under the SAME action-type. Restricting to action-6 clicks (the only
action-type with a real negative class -- directional keys are ~93-98% change, a
survivorship artifact) and ranking WITHIN a frame means the per-action-type
marginal and any global scalar CANCEL by construction (a constant-per-candidate
scorer gets within-frame grouped AUROC = 0.5 exactly; verified below), so the
model can only win by using object structure -- closing the escape hatch that
made exp5727/exp5730 mirages.

WITHIN-FRAME CONTRAST FROM THE CACHED CORPUS (no env stepping)
-------------------------------------------------------------
A within-frame group is all action-6 clicks that share the same `(env,
state_key)` frame. Because a NO-OP click leaves the frame unchanged, the human's
subsequent click is made from the SAME frame (same `state_key`); a click that
DOES change the frame moves to a new `state_key`. So a `(env, state_key)` group
that contains both a `changed` and a `no-op` click is exactly a set of DISTINCT
target objects tried from one frame, some inert and (at least) one that changed
it -- the within-frame changing-vs-inert contrast, with the no-op negatives the
design's Sec.6 mitigation asks for already present in the corpus (they are the
human's own no-op clicks, the offline analogue of what `ObjectHistorySalience
Prior` collects live). No env stepping / after-frame is needed.

TWO EVALUATIONS, BOTH REQUIRED, BOTH vs BOTH CONTROLS
----------------------------------------------------
1. OFFLINE CROSS-GAME LOO (exp5727 methodology): leave-one-GAME-out; does the
   object-effect model transfer to a held-out game? (asks whether a UNIVERSAL
   object->effect prior exists; the benchmark's design may make this null -- a
   cross-game null WITH an online positive is a PARTIAL WIN, not a failure.)
2. ONLINE PREFIX-CAUSAL (exp5730-style but CAUSAL): walk each game's trajectory;
   score each click using only transitions observed BEFORE it; does an object-
   identity effect memory (keyed on `object_hash`) rank the next click better
   than `PersistentAEM`'s action-type+click-bucket memory WITHIN the game? This
   is the live-relevant question (per the hidden-game framing).

TWO CONTROLS (both required):
  C1 -- action-type + click-bucket base rate (the exp5730 `action_id_only` /
        `PersistentAEM`-equivalent control): the "free" signal the agent has.
        Within a frame, action-type is constant, so C1's within-frame signal is
        entirely its per-16px-click-bucket change rate.
  C2 -- object-property-only base rate: the SAME architecture + within-frame
        ranking objective as the object model, but on JUST (color, area,
        `is_rect`). Guards the subtler trap that object salience alone explains
        the changing-vs-inert split, re-deriving `ColorBlobSaliencePrior` for
        nothing. (Isolates the FEATURE contribution: object model - C2 = the
        lift from patch/shape/containment/adjacency over bare object properties.)

5 SEEDS + UNTRAINED STRUCTURAL CONTROL, matching exp5730's rigor: every learned
metric is reported min/max/mean/std over 5 seeds, and the object model must beat
its own untrained/random-init baseline (a real learned signal must beat random
weights, not just chance). The gate holds on the WORST seed, not the mean --
the exact discipline that caught exp5730's seed-luck false positive.

Substrate: CPU-only. Featurizes cached replay clicks (connected-component
segmentation + a small torch MLP), scores against cached candidates; no
GGUF/LLM/GPU. `inference_substrate: verifier_ensemble_against_cached_candidates`
(1s floor).

Spec refs: REQ-ARC-FCP-5732, SCENARIO-ARC-FCP-5732-WITHIN-FRAME-CANCELS-MARGINAL,
SCENARIO-ARC-FCP-5732-TWO-CONTROL-GATE, SCENARIO-ARC-FCP-5732-RETIRE-OR-PARTIAL.
Prior work extended: exp5727 (frame_adds_over_action_id -0.039),
exp5730 (-0.010, seed-luck null).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch import nn

from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_color_blob_salience import (
    ColorBlobSaliencePrior,
    blob_at_click,
    blob_topology,
    connected_color_blobs,
    object_hash,
)

# Reuse exp5730's corpus building blocks verbatim (imported, not re-implemented)
# so the corpus/split/AUROC math is methodologically identical to the base-rate
# audit this experiment extends.
from carnot.experiment_4547_frame_change_predictor import (
    _is_trainable,
    binary_auroc,
    load_cached_examples,
    split_train_heldout_by_game,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5732_object_centric_click_affordance.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

# exp5730's seed set, verbatim, so the seed-robustness comparison is directly comparable.
SEEDS = (4547, 7, 99, 123, 2024)
RANDOM_SEED = SEEDS[0]

NUM_COLORS = 16
COLOR_VOCAB = 17  # colors 0..15 + status_bar_color 16, clipped into this vocab.
PATCH_RADIUS = 2  # KxK = 5x5 local color-index patch around the object centroid.
SHAPE_BUCKETS = 16  # hashing-trick buckets for the (color-independent) normalized shape key.
CLICK_BUCKET_SIZE = 16  # PersistentAEM's click-bucket size (control C1), verbatim.
MLP_HIDDEN = 16
MLP_STEPS = 400  # bounded gradient steps per training (keeps CPU runtime bounded).
MLP_BATCH_PAIRS = 256
MLP_LR = 0.02
SMOOTHING = 1.0  # Laplace smoothing for the count-based memories/base rates.

# Gate thresholds (exp5730's discipline).
ADD_MARGIN = 0.05  # object model must add > this over max(C1, C2) AUROC.
CHANCE = 0.5
# Pre-registered minimum within-frame (changed x inert) pairs for an AUROC-delta
# claim. CLAUDE.md requires N>=30 for a percentage-point delta; a grouped-AUROC
# comparison of two models needs more resolution, so require >=100 held-out
# within-frame pairs. Below this the coverage-collapse mode (design Sec.6) is
# reported honestly rather than forced into a claim. The measured corpus has
# 32,518 within-frame pairs across 1,894 contrast groups in 16 games (probe
# 2026-07-19), well above this floor, so coverage collapse is NOT triggered.
MIN_WITHIN_FRAME_PAIRS = 100
# A held-out game enters the UNWEIGHTED per-game LOO mean (exp5727's methodology,
# which is robust to one game dominating -- lp85 alone holds 27,491 of the 32,518
# pairs) only if it has >= this many held-out within-frame pairs; below it a
# per-game AUROC is too noisy to average. CLAUDE.md N>=30 for a delta claim.
MIN_GAME_PAIRS = 30
MIN_GAMES_FOR_CLAIM = 5  # the LOO mean needs at least this many qualifying games.
# Guard: skip the O(n_blobs x cells) containment-tree computation on a pathological
# frame with more blobs than this (degree/depth default to 0); real designed ARC
# frames are well under it. Keeps the offline featurizer's wall-time bounded.
MAX_BLOBS_FOR_TOPOLOGY = 400

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; complete: object features add over BOTH base rates (learned "
        "interaction signal) / partial win (cross-game null + online positive) / retire "
        "(neither eval clears the gate on the worst seed), or success: on a clean win."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- segments cached replay frames + "
        "trains/scores a small CPU MLP over object features; no GGUF/LLM/GPU (1s floor)."
    ),
    "solve_provenance": (
        "representation/verifier-building INFRASTRUCTURE work, not a game solve; it does "
        "NOT fit the live_agent_self_discovery / development_proxy / outer_loop_re solve "
        "taxonomy (no game level is claimed), so it is declared explicitly as such."
    ),
    "object_features_add_over_baselines": (
        "THE HEADLINE -- (object-model within-frame grouped AUROC) minus max(C1, C2 AUROC), "
        "reported for BOTH the offline cross-game LOO and the online prefix-causal eval; "
        "positive means object structure adds a real interaction signal over the base rates "
        "exp5727/exp5730 found were all that the frame-only representations had."
    ),
    "n_within_frame_pairs": (
        "held-out within-frame (changed x inert) pair count -- the effective sample size for "
        "the grouped-AUROC claim; below the pre-registered minimum the result is a coverage "
        "collapse (design Sec.6), reported honestly, not a claim."
    ),
    "c1_auroc": (
        "control C1 -- action-type + click-bucket base rate within-frame grouped AUROC; the "
        "free per-16px-bucket signal the live agent already has (PersistentAEM). Seed-free."
    ),
    "c2_auroc": (
        "control C2 -- object-property-only (color, area, is_rect) model, SAME within-frame "
        "ranking objective as the object model; isolates the lift of patch/shape/topology "
        "over bare object salience so a ColorBlobSaliencePrior re-derivation cannot pass."
    ),
    "object_model_auroc_over_seeds": (
        "the object model's within-frame grouped AUROC min/max/mean/std over 5 seeds; the "
        "gate is applied on the WORST seed (kills the exp5730 seed-luck mirage)."
    ),
    "untrained_structural_control_over_seeds": (
        "same object-model architecture, random init, no training -- a real learned signal "
        "must beat random weights on the worst seed, not just chance (exp5730 discipline)."
    ),
    "gate_passed": (
        "bool per eval (offline / online): object adds > 0.05 over max(C1,C2) on the WORST "
        "seed AND beats the untrained control AND positive control passes AND pairs >= min."
    ),
    "marginal_cancellation_check": (
        "the structural claim, verified: a constant-per-candidate scorer gets within-frame "
        "grouped AUROC == 0.5 exactly (the marginal cannot rank within a frame), so the "
        "objective cannot be minimized by a per-action-type base rate -- unlike exp5727/5730."
    ),
    "positive_control_passed": (
        "in-sample within-frame grouped AUROC (train==test) > 0.5 -- the harness can learn "
        "the objective when structure exists; a null is informative only if this passed."
    ),
    "recommendation": (
        "full win (name the live-path-eligible scorer) / partial win (route the live path "
        "from click-buckets to object identity) / retire the object-centric offline lineage "
        "to the exclusion manifest (operator-only) and close the gap with the online-only bound."
    ),
    "model_specs": (
        "the real compute substrate actually run -- a small torch MLP over object features "
        "(no LLM); required so a third party can re-run the exact model under audit."
    ),
    "random_seed": "primary determinism seed (exp5730's 4547); full seed set in `seeds`.",
    "random_seeds_used": "the 5 seeds; every learned metric is reported across all of them.",
    "reproducibility_checksum": "content hash of inputs+metrics; catches silent corpus/model drift.",
    "preconditions_checked": "records resources verified (torch, corpus cached) before the run.",
    "duration_s": "wall-clock of corpus load + featurize + multi-seed LOO + online eval; no LLM/GPU, 1s floor.",
    "verifier_is_oracle": (
        "false -- the label is raw pixel-change ground truth, oracle-DISTINCT from the LEARNED "
        "MLP scoring the object features; no moat/gate claim is made."
    ),
    "prior_work_extended": (
        "exp5727 + exp5730 -- the two base-rate-mirage nulls this representation+objective change "
        "is built to break; cited by id + verdict + the exact number that motivated the redesign."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "solve_provenance",
    "object_features_add_over_baselines",
    "n_within_frame_pairs",
    "c1_auroc",
    "c2_auroc",
    "object_model_auroc_over_seeds",
    "untrained_structural_control_over_seeds",
    "gate_passed",
    "recommendation",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "verifier_is_oracle",
    "prior_work_extended",
    "field_principles",
    "requirements",
    "scenarios",
    "preconditions_checked",
    "positive_control_passed",
    "marginal_cancellation_check",
    "missing_verifier_gaps",
)


# --------------------------------------------------------------------------- #
# Object-centric featurizer (composes existing repo primitives; no new perception infra)
# --------------------------------------------------------------------------- #

_PRIOR = ColorBlobSaliencePrior()


def _shape_only_bucket(blob: Any) -> int:
    """Hash the color-INDEPENDENT normalized shape into SHAPE_BUCKETS.

    `object_hash` folds color + normalized shape together; for a shape feature we
    want the shape alone so a square button generalizes across colors. Normalize
    cells so the bbox top-left is the origin, sort, hash, bucket.
    """
    cells = blob.cells
    min_y = min(y for y, _ in cells)
    min_x = min(x for _, x in cells)
    normalized = tuple(sorted((y - min_y, x - min_x) for y, x in cells))
    digest = hashlib.sha1(repr(normalized).encode("utf-8")).hexdigest()
    return int(digest, 16) % SHAPE_BUCKETS


def _containment_depth(children: dict[int, list[int]], n: int) -> dict[int, int]:
    """Depth of each blob in the containment tree (0 = not enclosed by anything)."""
    parent: dict[int, int] = {}
    for p, kids in children.items():
        for k in kids:
            parent[k] = p
    depth: dict[int, int] = {}
    for b in range(n):
        d = 0
        cur = b
        seen = {cur}
        while cur in parent:
            cur = parent[cur]
            if cur in seen:  # pragma: no cover - defensive against a degenerate cycle
                break
            seen.add(cur)
            d += 1
        depth[b] = d
    return depth


def _target_blob_index(blobs: list[Any], x: int, y: int) -> int | None:
    """Index of the blob a click at (x, y) lands in, else the nearest-centroid blob
    (blob_at_click's logic, returning the INDEX so adjacency/containment can be read)."""
    for i, blob in enumerate(blobs):
        if blob.contains_xy(x, y):
            return i
    if not blobs:
        return None
    return min(range(len(blobs)), key=lambda i: math.dist((float(y), float(x)), blobs[i].centroid))


def decompose_frame(grid: np.ndarray) -> dict[str, Any] | None:
    """One connected-component decomposition of an OFFLINE contrast frame (used for the
    ~1,894 contrast-group frames only, so the O(n_blobs x cells) containment tree is
    affordable). Returns blobs + per-blob object_hash + adjacency degree + containment
    depth + the background color + a color-count map -- everything the offline per-click
    featurizer needs, computed once per distinct frame.

    Frames with more than MAX_BLOBS_FOR_TOPOLOGY components skip the containment/adjacency
    computation (degree/depth default to 0) so a pathological frame cannot blow up runtime;
    the object_hash / color / shape / patch features are still computed for those frames.
    """
    try:
        blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)
    except Exception:
        return None
    n = len(blobs)
    if n == 0:
        return None
    degree: dict[int, int] = defaultdict(int)
    depth: dict[int, int] = {}
    object_hashes: dict[int, str] = {}
    if n <= MAX_BLOBS_FOR_TOPOLOGY:
        try:
            topo = blob_topology(grid)
            blobs = topo["blobs"]
            object_hashes = topo["object_hashes"]
            for i, j in topo["adjacency_list"]:
                degree[i] += 1
                degree[j] += 1
            depth = _containment_depth(topo["children"], len(blobs))
        except Exception:
            object_hashes = {i: object_hash(b) for i, b in enumerate(blobs)}
    else:  # pragma: no cover - pathological-frame guard, not hit on designed ARC frames
        object_hashes = {i: object_hash(b) for i, b in enumerate(blobs)}
    flat = grid.reshape(-1)
    color_counts = Counter(int(v) for v in flat.tolist())
    bg = int(max(color_counts.items(), key=lambda kv: kv[1])[0]) if color_counts else 0
    return {
        "blobs": blobs,
        "object_hashes": object_hashes,
        "degree": degree,
        "depth": depth,
        "color_counts": color_counts,
        "bg": bg,
        "grid": grid,
    }


def precompute_click_object_hashes(examples: list[Any]) -> dict[tuple[str, int, int], str]:
    """Fast per-click object_hash map for the ONLINE eval, WITHOUT the expensive
    containment tree. Groups all action-6 clicks by state_key, decomposes each distinct
    frame ONCE with the scipy-vectorized `connected_color_blobs`, records the clicked
    object's hash, and discards the blobs (so 42k+ distinct frames never sit in memory).
    The hash is identical to the offline topology path's (same segmentation + object_hash).
    """
    coords_by_frame: dict[str, set[tuple[int, int]]] = defaultdict(set)
    frame_by_key: dict[str, Any] = {}
    for ex in examples:
        if int(ex.action_id) != 6 or ex.x is None or ex.y is None or not _is_trainable(ex):
            continue
        sk = str(ex.state_key)
        coords_by_frame[sk].add((int(ex.x), int(ex.y)))
        if sk not in frame_by_key:
            frame_by_key[sk] = ex.frame
    out: dict[tuple[str, int, int], str] = {}
    for sk, coords in coords_by_frame.items():
        try:
            grid = np.asarray(grid_of(frame_by_key[sk]), dtype=np.int16)
            blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)
        except Exception:
            continue
        for x, y in coords:
            blob = blob_at_click(blobs, x, y)
            out[(sk, x, y)] = object_hash(blob) if blob is not None else ""
    return out


def object_features(decomp: dict[str, Any], x: int, y: int) -> dict[str, Any] | None:
    """Object-centric feature dict for a click at (x, y). Full vector + the C2 subset
    (color, area, is_rect) sliced from it, plus the object_hash (online memory key)."""
    blobs = decomp["blobs"]
    ti = _target_blob_index(blobs, int(x), int(y))
    if ti is None:
        return None
    blob = blobs[ti]
    grid = decomp["grid"]
    h_grid, w_grid = grid.shape
    color = int(blob.color)
    area = int(blob.pixel_count)
    bh, bw = int(blob.height), int(blob.width)
    is_rect = 1.0 if area == bh * bw else 0.0

    # C2 block: color one-hot(COLOR_VOCAB) + [area_norm, is_rect]  -- "logistic on (color, area, is_rect)".
    color_oh = [0.0] * COLOR_VOCAB
    color_oh[min(max(color, 0), COLOR_VOCAB - 1)] = 1.0
    area_norm = min(float(area), 256.0) / 256.0
    c2 = list(color_oh) + [area_norm, is_rect]

    # Richer object features the object model adds over C2.
    aspect = max(bh, bw) / max(1.0, float(min(bh, bw)))
    rich = [
        min(float(bh), 32.0) / 32.0,
        min(float(bw), 32.0) / 32.0,
        min(aspect, 8.0) / 8.0,
        float(blob.area_fraction),
        1.0 / (1.0 + float(decomp["color_counts"].get(color, 0))),
        1.0 if _PRIOR.is_status_bar_like(blob) else 0.0,
        1.0 if _PRIOR.is_button_like_blob(blob) else 0.0,
        min(float(decomp["depth"].get(ti, 0)), 6.0) / 6.0,
        min(float(decomp["degree"].get(ti, 0)), 12.0) / 12.0,
    ]
    shape_oh = [0.0] * SHAPE_BUCKETS
    shape_oh[_shape_only_bucket(blob)] = 1.0

    # KxK color-index patch around the object centroid (color-index-based, kept small
    # per design Sec.6 so raw-pixel dependence does not dominate). NOTE (deviation,
    # documented in the artifact): the design cites object_centric_slots as the patch
    # extractor, but that function returns density-summary SLOTS, not a raw patch, so
    # the faithful realization of "KxK local-neighborhood patch around the object
    # centroid" is a direct grid extraction here; the object_centric_slots locality
    # concept is realized as the `local_density` feature below.
    cy = int(round(blob.centroid[0]))
    cx = int(round(blob.centroid[1]))
    r = PATCH_RADIUS
    patch: list[float] = []
    non_bg = 0
    real_cells = 0
    bg = decomp["bg"]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            py, px = cy + dy, cx + dx
            if 0 <= py < h_grid and 0 <= px < w_grid:
                v = int(grid[py, px])
                patch.append(min(max(v, 0), COLOR_VOCAB - 1) / float(COLOR_VOCAB - 1))
                real_cells += 1
                if v != bg:
                    non_bg += 1
            else:
                patch.append(1.0)  # out-of-bounds sentinel (distinct from real colors 0..0.94)
    local_density = float(non_bg / real_cells) if real_cells else 0.0

    full = c2 + rich + shape_oh + patch + [local_density]
    return {
        "full": full,
        "c2": c2,
        "object_hash": decomp["object_hashes"].get(ti, ""),
    }


# --------------------------------------------------------------------------- #
# Model + within-frame contrastive ranking objective + grouped-AUROC metric
# --------------------------------------------------------------------------- #


class RankMLP(nn.Module):
    """Small MLP: object features -> a scalar affordance score."""

    def __init__(self, in_dim: int, hidden: int = MLP_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _grouped_within_frame_auroc(
    groups: list[dict[str, Any]], scores_by_group: list[list[float]]
) -> tuple[float | None, int]:
    """Weighted mean of per-frame AUROCs; weight = n_changed*n_inert pairs in the group.

    Within one group every candidate shares action-type (click), so a constant-per-
    candidate scorer scores 0.5 in every group -- the marginal cancels by construction.
    """
    acc = 0.0
    tot_pairs = 0
    for grp, scores in zip(groups, scores_by_group):
        labels = grp["labels"]
        npos = int(sum(labels))
        nneg = int(len(labels) - npos)
        if npos == 0 or nneg == 0:
            continue
        auc = binary_auroc(labels, scores)
        w = npos * nneg
        acc += auc * w
        tot_pairs += w
    if tot_pairs == 0:
        return None, 0
    return float(acc / tot_pairs), int(tot_pairs)


def _build_pairs(groups: list[dict[str, Any]], feat_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """All within-frame (changed, inert) feature-vector pairs from the training groups."""
    pos_rows: list[list[float]] = []
    neg_rows: list[list[float]] = []
    for grp in groups:
        feats = grp[feat_key]
        labels = grp["labels"]
        pos_idx = [i for i, y in enumerate(labels) if y == 1]
        neg_idx = [i for i, y in enumerate(labels) if y == 0]
        if not pos_idx or not neg_idx:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                pos_rows.append(feats[pi])
                neg_rows.append(feats[ni])
    if not pos_rows:
        return torch.empty(0), torch.empty(0)
    return (
        torch.tensor(pos_rows, dtype=torch.float32),
        torch.tensor(neg_rows, dtype=torch.float32),
    )


def _train_rank_mlp(
    train_groups: list[dict[str, Any]], feat_key: str, in_dim: int, seed: int
) -> RankMLP:
    """Train a RankMLP with within-frame pairwise logistic (BPR) ranking loss.

    A model whose output depends only on action-type is stuck: within a group all
    candidates share action-type, so it cannot separate the pair and the loss is
    log(2) with no within-group gradient. The only way to reduce the loss is to
    use the object features -- the structural fix for the exp5727/5730 base-rate hole.
    """
    torch.manual_seed(int(seed))
    model = RankMLP(in_dim)
    pos, neg = _build_pairs(train_groups, feat_key)
    if pos.numel() == 0:
        return model  # untrainable (no pairs); returns random-init (handled upstream)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    n_pairs = pos.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    model.train()
    for _step in range(MLP_STEPS):
        idx = torch.randint(0, n_pairs, (min(MLP_BATCH_PAIRS, n_pairs),), generator=gen)
        sp = model(pos[idx])
        sn = model(neg[idx])
        loss = -torch.log(torch.sigmoid(sp - sn) + 1e-9).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _score_groups(model: RankMLP, groups: list[dict[str, Any]], feat_key: str) -> list[list[float]]:
    model.eval()
    out: list[list[float]] = []
    with torch.no_grad():
        for grp in groups:
            feats = grp[feat_key]
            if not feats:
                out.append([])
                continue
            t = torch.tensor(feats, dtype=torch.float32)
            out.append([float(v) for v in model(t).tolist()])
    return out


# --------------------------------------------------------------------------- #
# Control C1: PersistentAEM-style action-type + click-bucket base rate
# --------------------------------------------------------------------------- #


def _click_bucket(x: int, y: int) -> tuple[int, int]:
    return int(x) // CLICK_BUCKET_SIZE, int(y) // CLICK_BUCKET_SIZE


def _fit_click_bucket_rate(train_groups: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    counts: dict[tuple[int, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )  # bucket -> [changed, total]
    for grp in train_groups:
        for (bx, by), label in zip(grp["buckets"], grp["labels"]):
            counts[(bx, by)][1] += 1
            counts[(bx, by)][0] += int(label)
    rate: dict[tuple[int, int], float] = {}
    for bucket, (chg, tot) in counts.items():
        rate[bucket] = (chg + SMOOTHING) / (tot + 2.0 * SMOOTHING)
    return rate


def _c1_scores(
    groups: list[dict[str, Any]], rate: dict[tuple[int, int], float]
) -> list[list[float]]:
    out: list[list[float]] = []
    for grp in groups:
        out.append([rate.get(b, 0.5) for b in grp["buckets"]])
    return out


# --------------------------------------------------------------------------- #
# Corpus -> within-frame contrast groups
# --------------------------------------------------------------------------- #


def build_contrast_groups(
    examples: list[Any], decomp_cache: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group trainable action-6 clicks by (env, state_key); keep groups with both a
    changed and a no-op click, featurizing each click's target object."""
    by_frame: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for ex in examples:
        if int(ex.action_id) != 6 or ex.x is None or ex.y is None or not _is_trainable(ex):
            continue
        by_frame[(str(ex.env), str(ex.state_key))].append(ex)

    groups: list[dict[str, Any]] = []
    diag = {
        "click_frames": len(by_frame),
        "contrast_groups": 0,
        "featurized_clicks": 0,
        "dropped_no_target": 0,
        "total_pairs": 0,
    }
    for (env, sk), rows in by_frame.items():
        n_changed = sum(1 for e in rows if e.changed)
        n_noop = len(rows) - n_changed
        if n_changed == 0 or n_noop == 0:
            continue
        decomp = decomp_cache.get(sk)
        if decomp is None:
            grid = grid_of(rows[0].frame)
            decomp = decompose_frame(np.asarray(grid, dtype=np.int16))
            decomp_cache[sk] = decomp if decomp is not None else {}
        if not decomp:
            continue
        full_feats: list[list[float]] = []
        c2_feats: list[list[float]] = []
        labels: list[int] = []
        buckets: list[tuple[int, int]] = []
        hashes: list[str] = []
        for e in rows:
            gx, gy = int(e.x), int(e.y)
            feat = object_features(decomp, gx, gy)
            if feat is None:
                diag["dropped_no_target"] += 1
                continue
            full_feats.append(feat["full"])
            c2_feats.append(feat["c2"])
            labels.append(1 if e.changed else 0)
            buckets.append(_click_bucket(gx, gy))
            hashes.append(feat["object_hash"])
        if sum(labels) == 0 or sum(labels) == len(labels) or len(labels) < 2:
            continue
        groups.append(
            {
                "env": env,
                "state_key": sk,
                "full": full_feats,
                "c2": c2_feats,
                "labels": labels,
                "buckets": buckets,
                "hashes": hashes,
            }
        )
        diag["contrast_groups"] += 1
        diag["featurized_clicks"] += len(labels)
        npos = sum(labels)
        diag["total_pairs"] += npos * (len(labels) - npos)
    return groups, diag


# --------------------------------------------------------------------------- #
# Online prefix-causal eval (object-hash memory vs click-bucket memory)
# --------------------------------------------------------------------------- #


def online_prefix_causal(
    examples: list[Any], click_hash: dict[tuple[str, int, int], str]
) -> dict[str, Any]:
    """Per game, walk the trajectory in (guid, step_index) order; score each action-6
    click causally from the two count memories updated only from EARLIER clicks; compare
    object-hash-memory AUROC to click-bucket-memory AUROC within each game.

    Aggregated UNWEIGHTED across qualifying games (>= MIN_GAME_PAIRS pairs) so no single
    game (lp85 holds ~85% of the clicks) dominates -- the exp5727-style per-game mean; a
    pair-weighted variant is reported alongside for transparency.
    """
    by_game: dict[str, list[Any]] = defaultdict(list)
    for ex in examples:
        if int(ex.action_id) != 6 or ex.x is None or ex.y is None or not _is_trainable(ex):
            continue
        by_game[str(ex.env)].append(ex)

    per_game: list[dict[str, Any]] = []
    total_scored = 0
    total_hash_covered = 0
    for game, rows in by_game.items():
        rows = sorted(rows, key=lambda e: (str(e.guid), int(e.step_index)))
        hash_mem: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [changed, total]
        bucket_mem: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        labels: list[int] = []
        hs: list[float] = []
        bs: list[float] = []
        hash_covered = 0
        for e in rows:
            gx, gy = int(e.x), int(e.y)
            oh = click_hash.get((str(e.state_key), gx, gy), "")
            bucket = _click_bucket(gx, gy)
            hm = hash_mem[oh]
            bm = bucket_mem[bucket]
            h_score = (hm[0] + SMOOTHING) / (hm[1] + 2.0 * SMOOTHING)
            b_score = (bm[0] + SMOOTHING) / (bm[1] + 2.0 * SMOOTHING)
            if hm[1] > 0:
                hash_covered += 1
            labels.append(1 if e.changed else 0)
            hs.append(h_score)
            bs.append(b_score)
            chg = int(e.changed)  # update memories AFTER scoring (causal / no lookahead)
            hm[1] += 1
            hm[0] += chg
            bm[1] += 1
            bm[0] += chg
        npos = int(sum(labels))
        nneg = int(len(labels) - npos)
        pairs = npos * nneg
        if pairs == 0:
            continue
        h_auc = binary_auroc(labels, hs)
        b_auc = binary_auroc(labels, bs)
        total_scored += len(labels)
        total_hash_covered += hash_covered
        per_game.append(
            {
                "game": game,
                "n": len(labels),
                "n_changed": npos,
                "pairs": int(pairs),
                "qualifies": bool(pairs >= MIN_GAME_PAIRS),
                "hash_memory_auroc": round(h_auc, 4),
                "click_bucket_memory_auroc": round(b_auc, 4),
                "delta_hash_minus_bucket": round(h_auc - b_auc, 4),
                "hash_covered_fraction": round(hash_covered / len(labels), 4) if labels else 0.0,
            }
        )

    qualifying = [r for r in per_game if r["qualifies"]]
    total_pairs = sum(r["pairs"] for r in qualifying)

    def _mean(vals: list[float]) -> float | None:
        return float(mean(vals)) if vals else None

    def _wmean(rows: list[dict[str, Any]], key: str) -> float | None:
        tot = sum(r["pairs"] for r in rows)
        if tot == 0:
            return None
        return float(sum(r[key] * r["pairs"] for r in rows) / tot)

    hash_mean = _mean([r["hash_memory_auroc"] for r in qualifying])
    bucket_mean = _mean([r["click_bucket_memory_auroc"] for r in qualifying])
    add = None if (hash_mean is None or bucket_mean is None) else float(hash_mean - bucket_mean)
    return {
        "eval": "online_prefix_causal",
        "aggregation": "unweighted_mean_over_qualifying_games",
        "object_hash_memory_auroc": None if hash_mean is None else round(hash_mean, 6),
        "click_bucket_memory_auroc": None if bucket_mean is None else round(bucket_mean, 6),
        "object_features_add_over_baselines": None if add is None else round(add, 6),
        "pair_weighted_object_hash_memory_auroc": _round_or_none(
            _wmean(qualifying, "hash_memory_auroc")
        ),
        "pair_weighted_click_bucket_memory_auroc": _round_or_none(
            _wmean(qualifying, "click_bucket_memory_auroc")
        ),
        "n_within_frame_pairs": int(total_pairs),
        "n_scored": int(total_scored),
        "n_games": len(per_game),
        "n_qualifying_games": len(qualifying),
        "hash_covered_fraction": round(total_hash_covered / total_scored, 4)
        if total_scored
        else 0.0,
        "seed_invariant": True,
        "note": (
            "Both memories are count-based (Laplace-smoothed) and deterministic given the "
            "trajectory order, so the online eval is seed-invariant (worst==mean==best). Only "
            "the KEY differs: object_hash identity vs 16px click bucket. The DELTA is the "
            "apples-to-apples 'does object identity rank better than the click-bucket base rate'. "
            "hash_covered_fraction is how often the object_hash memory was non-empty at scoring "
            "time -- low coverage means object identity rarely recurs before the causal test point."
        ),
        "per_game": sorted(per_game, key=lambda r: -r["delta_hash_minus_bucket"]),
    }


def _round_or_none(v: float | None, nd: int = 6) -> float | None:
    return None if v is None else round(v, nd)


# --------------------------------------------------------------------------- #
# Offline cross-game LOO eval
# --------------------------------------------------------------------------- #


def offline_cross_game_loo(groups: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Leave-one-GAME-out: for each held-out game with >= MIN_GAME_PAIRS held-out
    within-frame pairs, train on the OTHER games' within-frame groups and evaluate the
    held-out game's within-frame grouped AUROC, per seed, for the object model, C2, C1
    (seed-free), and the untrained object model. Aggregated as the UNWEIGHTED mean over
    qualifying games (exp5727 methodology) so lp85 (~85% of pairs) does not dominate;
    the pair-weighted mean is reported alongside for transparency."""
    games = sorted({g["env"] for g in groups})
    # NOTE: groups[i]["full"] is a LIST OF feature rows, so the feature DIM is len(row[0]).
    in_dim = len(groups[0]["full"][0]) if groups and groups[0]["full"] else 0
    c2_dim = len(groups[0]["c2"][0]) if groups and groups[0]["c2"] else 0

    def _test_pairs(test_groups: list[dict[str, Any]]) -> int:
        return sum(
            int(sum(g["labels"])) * int(len(g["labels"]) - sum(g["labels"])) for g in test_groups
        )

    per_game_rows: list[dict[str, Any]] = []
    total_heldout_pairs = 0

    for game in games:
        train_groups = [g for g in groups if g["env"] != game]
        test_groups = [g for g in groups if g["env"] == game]
        test_pairs = _test_pairs(test_groups)
        qualifies = bool(test_pairs >= MIN_GAME_PAIRS and train_groups)
        if test_pairs == 0 or not train_groups:
            continue
        total_heldout_pairs += test_pairs

        rate = _fit_click_bucket_rate(
            train_groups
        )  # C1: click-bucket base rate from TRAIN (seed-free)
        c1_auc, _ = _grouped_within_frame_auroc(test_groups, _c1_scores(test_groups, rate))

        row: dict[str, Any] = {
            "game": game,
            "test_pairs": int(test_pairs),
            "qualifies": qualifies,
            "c1_auroc": _round_or_none(c1_auc, 4),
            "object_auroc_per_seed": {},
            "c2_auroc_per_seed": {},
            "untrained_auroc_per_seed": {},
        }
        if qualifies:  # only spend training compute on games that enter the LOO mean
            for seed in seeds:
                obj_model = _train_rank_mlp(train_groups, "full", in_dim, seed)
                obj_auc, _ = _grouped_within_frame_auroc(
                    test_groups, _score_groups(obj_model, test_groups, "full")
                )
                c2_model = _train_rank_mlp(train_groups, "c2", c2_dim, seed)
                c2_auc, _ = _grouped_within_frame_auroc(
                    test_groups, _score_groups(c2_model, test_groups, "c2")
                )
                torch.manual_seed(int(seed))
                untrained = RankMLP(in_dim)
                unt_auc, _ = _grouped_within_frame_auroc(
                    test_groups, _score_groups(untrained, test_groups, "full")
                )
                row["object_auroc_per_seed"][str(seed)] = _round_or_none(obj_auc, 6)
                row["c2_auroc_per_seed"][str(seed)] = _round_or_none(c2_auc, 6)
                row["untrained_auroc_per_seed"][str(seed)] = _round_or_none(unt_auc, 6)
        per_game_rows.append(row)

    qual = [r for r in per_game_rows if r["qualifies"]]

    def _seed_vals(rows: list[dict[str, Any]], field: str, seed: int) -> list[float]:
        return [v for r in rows if (v := r[field].get(str(seed))) is not None]

    def _summ_unweighted(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
        per_seed = {}
        for s in seeds:
            vals = _seed_vals(rows, field, s)
            if vals:
                per_seed[str(s)] = round(float(mean(vals)), 6)  # unweighted mean over games
        if not per_seed:
            return {
                "mean": None,
                "min": None,
                "max": None,
                "std": None,
                "n_seeds": 0,
                "per_seed": {},
            }
        vv = list(per_seed.values())
        return {
            "mean": round(float(mean(vv)), 6),
            "min": round(float(min(vv)), 6),
            "max": round(float(max(vv)), 6),
            "std": round(float(pstdev(vv)) if len(vv) > 1 else 0.0, 6),
            "n_seeds": len(vv),
            "per_seed": per_seed,
        }

    object_summary = _summ_unweighted(qual, "object_auroc_per_seed")
    c2_summary = _summ_unweighted(qual, "c2_auroc_per_seed")
    untrained_summary = _summ_unweighted(qual, "untrained_auroc_per_seed")

    c1_vals = [r["c1_auroc"] for r in qual if r["c1_auroc"] is not None]
    c1_loo = float(mean(c1_vals)) if c1_vals else None
    # pair-weighted C1 for transparency (dominated by lp85)
    c1_wtot = sum(r["test_pairs"] for r in qual if r["c1_auroc"] is not None)
    c1_weighted = (
        float(
            sum(r["c1_auroc"] * r["test_pairs"] for r in qual if r["c1_auroc"] is not None)
            / c1_wtot
        )
        if c1_wtot
        else None
    )

    # object_features_add_over_baselines per seed = object[seed] - max(C1, C2[seed]); worst seed governs.
    add_per_seed: dict[str, float] = {}
    for s in seeds:
        o = object_summary["per_seed"].get(str(s))
        c2v = c2_summary["per_seed"].get(str(s))
        if o is None:
            continue
        base = max([b for b in (c1_loo, c2v) if b is not None], default=CHANCE)
        add_per_seed[str(s)] = round(float(o - base), 6)
    add_vals = list(add_per_seed.values())
    add_summary = (
        {
            "mean": round(float(mean(add_vals)), 6),
            "min": round(float(min(add_vals)), 6),
            "max": round(float(max(add_vals)), 6),
            "std": round(float(pstdev(add_vals)) if len(add_vals) > 1 else 0.0, 6),
            "per_seed": add_per_seed,
        }
        if add_vals
        else {"mean": None, "min": None, "max": None, "std": None, "per_seed": {}}
    )

    beats_untrained = bool(
        object_summary["min"] is not None
        and untrained_summary["max"] is not None
        and object_summary["min"] > untrained_summary["max"]
    )

    return {
        "eval": "offline_cross_game_loo",
        "aggregation": "unweighted_mean_over_qualifying_games",
        "n_games": len(per_game_rows),
        "n_qualifying_games": len(qual),
        "min_game_pairs_to_qualify": MIN_GAME_PAIRS,
        "n_within_frame_pairs": int(total_heldout_pairs),
        "n_qualifying_within_frame_pairs": int(sum(r["test_pairs"] for r in qual)),
        "c1_auroc": _round_or_none(c1_loo),
        "c1_auroc_pair_weighted": _round_or_none(c1_weighted),
        "c2_auroc_over_seeds": c2_summary,
        "object_model_auroc_over_seeds": object_summary,
        "untrained_structural_control_over_seeds": untrained_summary,
        "object_features_add_over_baselines_over_seeds": add_summary,
        "object_beats_untrained_worst_seed": beats_untrained,
        "per_game": per_game_rows,
    }


# --------------------------------------------------------------------------- #
# Positive control + marginal-cancellation sanity check
# --------------------------------------------------------------------------- #


def _marginal_cancellation_check(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """A constant-per-candidate scorer must get within-frame grouped AUROC == 0.5 exactly.
    This is the structural proof the objective cannot be won by a per-action-type marginal."""
    const_scores = [[1.0] * len(g["labels"]) for g in groups]
    auc, pairs = _grouped_within_frame_auroc(groups, const_scores)
    return {
        "constant_scorer_within_frame_grouped_auroc": None if auc is None else round(auc, 6),
        "n_pairs": int(pairs),
        "passes": bool(auc is not None and abs(auc - 0.5) < 1e-9),
        "note": (
            "binary_auroc returns 0.5 for tied scores; a constant score ties every candidate "
            "in every frame, so the within-frame grouped AUROC is exactly 0.5. A per-action-type "
            "base rate is constant within action-6, hence structurally cannot rank within a frame "
            "-- the escape hatch exp5727/exp5730's pointwise objective left open is closed."
        ),
    }


def _positive_control(groups: list[dict[str, Any]], seed: int, in_dim: int) -> dict[str, Any]:
    """In-sample (train==test) within-frame grouped AUROC must exceed 0.5: the harness can
    learn the within-frame objective when object structure separates the classes."""
    model = _train_rank_mlp(groups, "full", in_dim, seed)
    auc, pairs = _grouped_within_frame_auroc(groups, _score_groups(model, groups, "full"))
    return {
        "in_sample_within_frame_grouped_auroc": None if auc is None else round(auc, 6),
        "n_pairs": int(pairs),
        "passes": bool(auc is not None and auc > CHANCE),
    }


# --------------------------------------------------------------------------- #
# Verdict / gate / driver
# --------------------------------------------------------------------------- #


def _checksum(payload: dict[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    digest = hashlib.sha256(
        json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def run(
    *, root: Path | str = REPO_ROOT, seeds: tuple[int, ...] = SEEDS, write: bool = True
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5732: featurize, control, seed-robustly probe, offline+online."""
    t0 = time.time()
    root_path = Path(root)
    preconditions: dict[str, Any] = {"torch_import": True, "corpus_cached": False}

    examples = load_cached_examples(root_path, limit=None)
    preconditions["corpus_cached"] = bool(examples)
    train_examples, heldout_examples = split_train_heldout_by_game(examples)
    all_examples = list(examples)

    decomp_cache: dict[str, dict[str, Any]] = {}
    groups, group_diag = build_contrast_groups(all_examples, decomp_cache)

    positive_control = (
        _positive_control(groups, seeds[0], len(groups[0]["full"][0]))
        if groups and groups[0]["full"]
        else {"passes": False}
    )
    marginal_check = _marginal_cancellation_check(groups) if groups else {"passes": False}

    offline = (
        offline_cross_game_loo(groups, seeds)
        if groups
        else {"eval": "offline_cross_game_loo", "n_within_frame_pairs": 0, "n_qualifying_games": 0}
    )
    # Fast, topology-free object_hash map for the online eval (avoids decomposing 42k+
    # distinct frames with the O(n_blobs x cells) containment tree).
    click_hash = precompute_click_object_hashes(all_examples)
    online = online_prefix_causal(all_examples, click_hash)

    # ---- gate ----
    pc_ok = bool(positive_control.get("passes"))
    mc_ok = bool(marginal_check.get("passes"))

    off_add = offline.get("object_features_add_over_baselines_over_seeds", {})
    off_add_worst = off_add.get("min") if isinstance(off_add, dict) else None
    off_pairs = int(offline.get("n_within_frame_pairs", 0))
    off_qual_games = int(offline.get("n_qualifying_games", 0))
    offline_gate = bool(
        pc_ok
        and mc_ok
        and off_add_worst is not None
        and off_add_worst > ADD_MARGIN
        and offline.get("object_beats_untrained_worst_seed") is True
        and off_pairs >= MIN_WITHIN_FRAME_PAIRS
        and off_qual_games >= MIN_GAMES_FOR_CLAIM
    )

    on_add = online.get("object_features_add_over_baselines")
    on_pairs = int(online.get("n_within_frame_pairs", 0))
    on_qual_games = int(online.get("n_qualifying_games", 0))
    online_gate = bool(
        pc_ok
        and mc_ok
        and on_add is not None
        and on_add > ADD_MARGIN
        and on_pairs >= MIN_WITHIN_FRAME_PAIRS
        and on_qual_games >= MIN_GAMES_FOR_CLAIM
    )

    # ---- verdict / recommendation (decision matrix, design Sec.5) ----
    off_add_str = f"{off_add_worst:+.3f}" if off_add_worst is not None else "NA"
    on_add_str = f"{on_add:+.3f}" if on_add is not None else "NA"
    if not pc_ok:
        verdict = "complete: object_centric_positive_control_failed_harness_uninformative"
        recommendation = (
            "The in-sample within-frame objective did not exceed chance -- the harness could not "
            "learn even train==test; do NOT draw conclusions, investigate featurizer/corpus first."
        )
    elif offline_gate and online_gate:
        verdict = (
            "success: object_features_add_over_both_base_rates_offline_"
            + off_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
            + "_and_online_"
            + on_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
        )
        recommendation = (
            "Object-centric features carry a real, non-base-rate action x frame interaction signal "
            "on BOTH the cross-game LOO and the online prefix-causal eval, beating C1 (click-bucket) "
            "and C2 (object-property-only) and the untrained control on the worst of 5 seeds. Ship "
            "the object-feature scorer to the live path (rich_action_candidates.frame_change_scorer "
            "slot / LiveActionEffectScorer), replacing the falsified SmallFrameChangeCNN term."
        )
    elif online_gate and not offline_gate:
        verdict = (
            "complete: object_centric_partial_win_cross_game_null_offline_"
            + off_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
            + "_but_online_within_game_positive_"
            + on_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
        )
        recommendation = (
            "PARTIAL WIN (design Sec.5): no UNIVERSAL cross-game object->effect prior (LOO null, "
            "as the benchmark's shared-nothing design makes plausible), but object IDENTITY beats "
            "the click-bucket base rate WITHIN a game online. Redirect the live online memory from "
            "click-buckets to object_hash identity (generalize ObjectHistorySaliencePrior as the "
            "PersistentAEM-equivalent key). Do NOT ship a cross-game-trained offline scorer."
        )
    elif offline_gate and not online_gate:
        verdict = (
            "complete: object_centric_cross_game_positive_offline_"
            + off_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
            + "_online_within_game_null"
        )
        recommendation = (
            "A transferable cross-game object->effect prior exists offline, but the online object-hash "
            "memory did not beat the click-bucket memory within a game (likely low object_hash recurrence "
            "before the causal test point). Ship the offline object scorer; keep the click-bucket online "
            "memory. Investigate why object identity does not recur enough for an online track record."
        )
    else:
        verdict = (
            "complete: object_centric_representation_no_add_over_base_rates_offline_"
            + off_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
            + "_online_"
            + on_add_str.replace("+", "plus").replace("-", "minus").replace(".", "p")
            + "_retire_lineage_recommended"
        )
        recommendation = (
            "RETIRE (Failed-Experiment Rerun Discipline, pre-registered): object features beat NEITHER "
            "the cross-game LOO gate NOR the online prefix-causal gate by >0.05 on the worst seed. "
            "RECOMMEND (operator-only, do NOT self-apply) adding the object-centric OFFLINE action-effect "
            "predictor lineage to ops/exclusion_manifest.yaml citing this artifact, and closing "
            "GAP-ARCH-FRAME-CHANGE-PREDICTOR with the honest bound: no offline-learnable, non-base-rate "
            "action-effect representation exists on this corpus; action-effect is online-within-game only "
            "via the existing ObjectHistorySaliencePrior memory. This extends exp5727/exp5730 as a 5th, "
            "structurally-different (object-centric + within-frame contrastive) negative on the same gap."
        )

    coverage_collapsed = bool(
        off_pairs < MIN_WITHIN_FRAME_PAIRS or on_pairs < MIN_WITHIN_FRAME_PAIRS
    )

    artifact: dict[str, Any] = {
        "experiment": "experiment_5732_object_centric_click_affordance",
        "schema": "carnot.arc_object_centric_click_affordance_5732.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": "not_a_game_solve_representation_verifier_infrastructure",
        "verifier_is_oracle": False,
        "target": "does an OBJECT-CENTRIC within-frame contrastive representation add a non-base-rate "
        "click action-effect signal over the action-type+click-bucket (C1) and object-property (C2) base rates",
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(seeds),
        "object_features_add_over_baselines": {
            "offline_cross_game_loo": {
                "worst_seed": off_add_worst,
                "mean": off_add.get("mean") if isinstance(off_add, dict) else None,
                "per_seed": off_add.get("per_seed") if isinstance(off_add, dict) else {},
            },
            "online_prefix_causal": on_add,
        },
        "n_within_frame_pairs": {
            "offline_cross_game_loo": off_pairs,
            "online_prefix_causal": on_pairs,
            "minimum_required": MIN_WITHIN_FRAME_PAIRS,
            "coverage_collapsed": coverage_collapsed,
        },
        "c1_auroc": {
            "offline_cross_game_loo": offline.get("c1_auroc"),
            "online_prefix_causal": online.get("click_bucket_memory_auroc"),
        },
        "c2_auroc": {
            "offline_cross_game_loo": offline.get("c2_auroc_over_seeds"),
            "online_prefix_causal_note": "C2 (learned object-property MLP) has no online analogue; "
            "the online eval's second baseline is C1 (click-bucket memory).",
        },
        "object_model_auroc_over_seeds": {
            "offline_cross_game_loo": offline.get("object_model_auroc_over_seeds"),
            "online_prefix_causal_object_hash_memory_auroc": online.get("object_hash_memory_auroc"),
        },
        "untrained_structural_control_over_seeds": offline.get(
            "untrained_structural_control_over_seeds"
        ),
        "gate_passed": {
            "offline_cross_game_loo": offline_gate,
            "online_prefix_causal": online_gate,
            "both": bool(offline_gate and online_gate),
            "neither": bool(not offline_gate and not online_gate),
            "add_margin_threshold": ADD_MARGIN,
        },
        "offline_cross_game_loo": offline,
        "online_prefix_causal": online,
        "positive_control_passed": pc_ok,
        "positive_control": positive_control,
        "marginal_cancellation_check": marginal_check,
        "false_negative_risk_checked": bool(pc_ok and mc_ok),
        "recommendation": recommendation,
        "corpus_summary": {
            "corpus_examples_loaded": int(len(examples)),
            "train_examples": int(len(train_examples)),
            "heldout_examples": int(len(heldout_examples)),
            "game_count": int(len({ex.env for ex in examples if ex.env})),
            "within_frame_group_diagnostics": group_diag,
            "distinct_frames_decomposed": int(len([d for d in decomp_cache.values() if d])),
        },
        "model_specs": [
            {
                "name": "RankMLP",
                "framework": "torch",
                "device": "cpu",
                "architecture": f"Linear(in->{MLP_HIDDEN}) -> ReLU -> Linear({MLP_HIDDEN}->1), "
                "within-frame pairwise logistic (BPR) ranking loss",
                "object_feature_dim": int(len(groups[0]["full"][0]))
                if groups and groups[0]["full"]
                else 0,
                "c2_feature_dim": int(len(groups[0]["c2"][0])) if groups and groups[0]["c2"] else 0,
                "features": "object_hash context (color one-hot, area, is_rect), height/width/aspect/"
                "area_fraction/color_rarity, is_status_bar_like, is_button_like, containment_depth, "
                "adjacency_degree, normalized-shape hash buckets, 5x5 color-index centroid patch, local_density",
                "train_steps": MLP_STEPS,
                "no_llm": True,
                "note": "no GGUF/LLM invoked; object features are computed by classical connected-"
                "component segmentation (arc_color_blob_salience) over cached replay frames.",
            }
        ],
        "prior_work_extended": {
            "exp5727_perception_action_effect_adequacy": {
                "verdict": "complete: action_effect_above_chance_but_driven_by_action_base_rate_not_"
                "frame_representation_honest_null_on_perception",
                "number": "frame_adds_over_action_id = -0.039 (hand-crafted linear features SUBTRACT)",
                "role": "the base-rate control this extends; motivated the object-centric + within-frame redesign.",
            },
            "exp5730_cnn_baserate_audit": {
                "verdict": "complete: cnn_held_out_auroc_is_action_id_base_rate_and_seed_luck_mirage_"
                "frame_adds_-0.010_no_robust_within_action_signal_matching_exp5727_null",
                "number": "frame_adds_over_action_id = -0.010 (global-pooled conv SUBTRACTS); the click "
                "discrimination win was 5-seed seed luck (mean 0.570, untrained 0.580).",
                "role": "the harness skeleton reused here (corpus/split/5-seed/untrained control); the "
                "seed-luck null this experiment's worst-seed gate is designed to not repeat.",
            },
        },
        "prior_failures": [
            {
                "experiment_id": "exp5727",
                "verdict": "frame_adds_over_action_id_-0.039_base_rate_mirage",
                "diagnosed_root_cause": "pointwise changed/no-op objective (minimizable by the per-action-"
                "type marginal) over a global/frame-only representation with no object localization, on a "
                "survivorship-biased corpus (no negative class outside action-6).",
                "addressed_by": "object-centric per-candidate representation + WITHIN-FRAME contrastive "
                "ranking that structurally cancels the action-type marginal (constant scorer -> AUROC 0.5) "
                "+ action-6-only restriction (the only channel with a real negative class) + a second "
                "control (object-property-only) + an online prefix-causal eval the prior runs lacked. This "
                "is a genuine representation+objective change, NOT a retrain/re-tune of the falsified scorer.",
                "retire_if_same_verdict": True,
            },
            {
                "experiment_id": "exp5730",
                "verdict": "frame_adds_over_action_id_-0.010_and_within_action_click_discrimination_seed_luck_null",
                "diagnosed_root_cause": "same base-rate collapse for a learned global-pooled conv; the one "
                "within-action click signal was a lone-seed artifact that lost to its untrained control.",
                "addressed_by": "object-localized features (a within-frame difference between two objects is "
                "representable, which a global pool cannot resolve) + the worst-of-5-seeds gate + the "
                "untrained-control requirement retained verbatim so a seed-luck value cannot pass.",
                "retire_if_same_verdict": True,
            },
        ],
        "preconditions_checked": preconditions,
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "requirements": ["REQ-ARC-FCP-5732"],
        "scenarios": [
            "SCENARIO-ARC-FCP-5732-WITHIN-FRAME-CANCELS-MARGINAL",
            "SCENARIO-ARC-FCP-5732-TWO-CONTROL-GATE",
            "SCENARIO-ARC-FCP-5732-RETIRE-OR-PARTIAL",
        ],
        "duration_s": round(time.time() - t0, 3),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)

    errors = [f for f in REQUIRED_ARTIFACT_FIELDS if f not in artifact]
    if errors:
        raise ValueError(f"missing required artifact fields: {errors}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")

    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])
    off = artifact["object_features_add_over_baselines"]["offline_cross_game_loo"]
    on = artifact["object_features_add_over_baselines"]["online_prefix_causal"]
    print(
        f"OFFLINE object_features_add_over_baselines: worst_seed={off['worst_seed']} mean={off['mean']}"
    )
    print(f"ONLINE  object_features_add_over_baselines: {on}")
    print(f"c1_auroc={artifact['c1_auroc']}")
    print(f"n_within_frame_pairs={artifact['n_within_frame_pairs']}")
    print(
        f"marginal_cancellation_check={artifact['marginal_cancellation_check'].get('passes')} "
        f"(const AUROC={artifact['marginal_cancellation_check'].get('constant_scorer_within_frame_grouped_auroc')})"
    )
    print(f"positive_control_passed={artifact['positive_control_passed']}")
    print(f"gate_passed={artifact['gate_passed']}")
    print(f"duration_s={artifact['duration_s']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
