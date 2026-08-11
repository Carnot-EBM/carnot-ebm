"""REQ-ARC-WMTE-6248: Pinductor-style REx refinement over induced ARC world models.

WHAT THIS IS. A refinement harness that keeps a POPULATION of candidate engines and
searches over them, instead of refining one lineage greedily. The mechanisms come from
Pinductor (arXiv:2605.13740, "Learning POMDP World Models from Observations with
Language-Model Priors"): UCB1 parent selection over the candidate tree, and
query-by-committee (QBC) vote entropy to pick WHICH failing transitions the refactor
prompt shows. Reimplemented from the paper's description -- no reference code copied
(inspiration tier per the project's untrusted-code discipline).

WHY ONE CODE PATH FOR BOTH ARMS. The A/B this serves (experiment 6248) compares the
current production shape (linear: always refine the latest candidate, mismatches in
corpus order) against the REx shape (UCB1 parent + QBC-ordered mismatches). Running
both through `run_rex` with different flags makes the LLM-call budget equal by
construction, so budget cannot confound the comparison.

WHAT IS DELIBERATELY NOT HERE. No particle filter or belief machinery (ARC frames are
fully observed); no near-best softmax final selection (deterministic argmax keeps a
small paired A/B interpretable); one proposal per round (the tree provides the
population across rounds). See the plan note for the full deviation list:
docs/research-notes/pinductor-rex-refinement-plan-2026-08-09.md

REQ-ARC-WMTE-6250 ADDS `run_rex_ensemble`: 6248's gate found REx does NOT beat linear as a
blanket replacement (2 of 6 games), but the two arms fail on DIFFERENT games, and a
retrospective check found the VALID score (known online, before HELD is ever seen) predicts
the HELD-optimal arm on 6 of 6 games. `run_rex_ensemble` runs both arms and keeps whichever
one's VALID score is higher -- see its own docstring for the full contract.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = [
    "RexNode",
    "ucb1_pick",
    "committee_entropy",
    "qbc_order_mismatches",
    "load_engine_from_source",
    "run_rex",
    "run_rex_ensemble",
]


@dataclass
class RexNode:
    """One candidate engine in the refinement tree."""

    idx: int
    source: str
    # VALID-slice score used for selection (NOT the held-out metric the A/B reports).
    valid_fidelity: float
    # Full mismatch list from the VALID scoring pass, in corpus order. The refactor
    # prompt bounds to the first 5, so ORDER is what QBC controls.
    mismatches: list[dict] = field(default_factory=list)
    parent: Optional[int] = None
    # Times this node was picked as a refinement parent (UCB1 visit count).
    n_visit: int = 0
    # Kept for the artifact: exact-match stats alongside the graded score.
    valid_accuracy: float = 0.0
    valid_n: int = 0
    valid_n_correct: int = 0


def ucb1_pick(nodes: Sequence[RexNode], *, c: float = 1.0) -> int:
    """Pick the index of the node with the largest UCB1 score.

    score(n) = quality(n) + c * sqrt(ln(N_total + 1) / (n_visit + 1))

    The `(n_visit + 1)` denominator is deliberate: a fresh node gets the LARGEST
    exploration bonus but never an infinite one. Pinductor's own release notes record
    that an `n_visit == 0 -> +inf` shortcut degenerated their search (every round
    picked the newest node and the quality term was never consulted).
    """
    if not nodes:
        raise ValueError("ucb1_pick needs at least one node")
    total = sum(n.n_visit for n in nodes) + 1
    best_idx, best_score = nodes[0].idx, -math.inf
    for n in nodes:
        score = float(n.valid_fidelity) + c * math.sqrt(math.log(total + 1) / (n.n_visit + 1))
        if score > best_score:
            best_idx, best_score = n.idx, score
    return best_idx


def committee_entropy(prediction_hashes: Sequence[Optional[int]]) -> float:
    """Shannon entropy of the committee's prediction distribution for one transition.

    Each committee member contributes a hash of its predicted next grid (None = the
    member crashed on this transition; crashes are excluded rather than counted as a
    distinct "prediction", because two members crashing with different tracebacks are
    not disagreeing about dynamics). 0.0 = all agree; log(k) = all k members differ.
    """
    hashes = [h for h in prediction_hashes if h is not None]
    if not hashes:
        return 0.0
    counts = Counter(hashes)
    n = len(hashes)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def qbc_order_mismatches(
    mismatches: Sequence[dict],
    entropy_by_transition_index: dict[int, float],
) -> list[dict]:
    """Return mismatches sorted by committee disagreement, highest first.

    Each mismatch dict carries `i`, the VALID-slice transition index (the
    `WorldModelVerifier.score` contract). A mismatch whose index has no recorded
    entropy sorts as 0.0 (no measured disagreement). The sort is stable, so within an
    entropy tie the original corpus order is preserved.
    """
    return sorted(
        mismatches,
        key=lambda m: -float(entropy_by_transition_index.get(m.get("i", -1), 0.0)),
    )


def load_engine_from_source(source: str, tag: str = "candidate"):
    """Import an engine from source text without touching any engine store.

    Writes the text to a throwaway temp file and imports it, mirroring what
    `arc_executable_world_model._load_engine_from` does for the on-disk store. Used to
    run the QBC committee: every candidate must predict, but only the current parent
    lives in the store file.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"arc_rex_{tag}_", delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"arc_rex_{tag}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine")
    finally:
        path.unlink(missing_ok=True)


def _grid_hash(grid: np.ndarray) -> int:
    return hash(np.asarray(grid).tobytes())


def _committee_entropies(sources: Sequence[str], transitions: Sequence[Any]) -> dict[int, float]:
    """Per-transition QBC entropy over every candidate proposed so far.

    A candidate whose source fails to import contributes nothing anywhere (it cannot
    predict); a candidate that imports but crashes on one transition is excluded from
    that transition only, per `committee_entropy`'s crash rule.
    """
    engines = []
    for k, src in enumerate(sources):
        try:
            engines.append(load_engine_from_source(src, tag=f"c{k}"))
        except Exception:
            continue
    out: dict[int, float] = {}
    for i, t in enumerate(transitions):
        hashes: list[Optional[int]] = []
        for eng in engines:
            try:
                pred = eng(np.asarray(t.grid).copy(), t.action, t.data)
                hashes.append(_grid_hash(np.asarray(pred)))
            except Exception:
                hashes.append(None)
        out[i] = committee_entropy(hashes)
    return out


def run_rex(
    game: str,
    proposer: Any,
    *,
    train: Sequence[Any],
    valid: Sequence[Any],
    cell: int,
    budget: int,
    score_candidate: Callable[[str], dict],
    read_store_source: Callable[[], Optional[str]],
    write_store_source: Callable[[str], None],
    make_verify_result: Callable[[RexNode, list[dict]], Any],
    use_ucb1: bool,
    use_qbc: bool,
    ucb1_c: float = 1.0,
) -> dict:
    """Run one refinement cell: 1 induce + (budget-1) refinements. Returns a summary.

    The LLM-facing side stays behind three injected callables so the loop itself is
    unit-testable without a GPU:
      - `score_candidate(source)` -> dict with `valid_fidelity`, `mismatches`,
        `valid_accuracy`, `valid_n`, `valid_n_correct` (the VALID-slice verifier pass).
      - `read_store_source()` / `write_store_source(text)` -> the ISOLATED engine
        store file the proposer reads and writes (never the shared results/arc_e3).
      - `make_verify_result(node, ordered_mismatches)` -> the VerifyResult-shaped
        object `proposer.refactor` expects, carrying the (possibly QBC-reordered)
        mismatches.

    `use_ucb1=False, use_qbc=False` is the `linear` baseline arm; both True is the
    `rex` treatment arm. Same code path, same budget -- parity by construction.
    """
    nodes: list[RexNode] = []
    llm_calls = 0
    events: list[dict] = []

    ok, detail = proposer.induce(game, list(train), cell)
    llm_calls += 1
    events.append({"round": 0, "kind": "induce", "ok": bool(ok), "detail": str(detail)[:200]})
    if ok:
        src = read_store_source()
        if src:
            s = score_candidate(src)
            nodes.append(
                RexNode(
                    idx=0,
                    source=src,
                    valid_fidelity=float(s.get("valid_fidelity", 0.0)),
                    mismatches=list(s.get("mismatches", [])),
                    parent=None,
                    valid_accuracy=float(s.get("valid_accuracy", 0.0)),
                    valid_n=int(s.get("valid_n", 0)),
                    valid_n_correct=int(s.get("valid_n_correct", 0)),
                )
            )
    if not nodes:
        return {
            "game": game,
            "nodes": [],
            "llm_calls": llm_calls,
            "events": events,
            "final_idx": None,
            "final_source": None,
            "final_valid_fidelity": None,
        }

    for rnd in range(1, budget):
        parent_idx = ucb1_pick(nodes, c=ucb1_c) if use_ucb1 else nodes[-1].idx
        parent = nodes[parent_idx]
        parent.n_visit += 1

        ordered = list(parent.mismatches)
        if use_qbc and ordered:
            entropies = _committee_entropies([n.source for n in nodes], valid)
            ordered = qbc_order_mismatches(ordered, entropies)

        # The refactor prompt reads the CURRENT store file as "the engine being
        # refined" -- write the chosen parent there first so prompt and lineage agree.
        write_store_source(parent.source)
        vr = make_verify_result(parent, ordered)
        try:
            ok, detail = proposer.refactor(game, vr)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, repr(exc)[:200]
        llm_calls += 1
        events.append(
            {
                "round": rnd,
                "kind": "refactor",
                "parent": parent_idx,
                "ok": bool(ok),
                "detail": str(detail)[:200],
            }
        )
        if not ok:
            continue
        src = read_store_source()
        if not src or src == parent.source:
            events.append({"round": rnd, "kind": "no_new_source", "parent": parent_idx})
            continue
        s = score_candidate(src)
        nodes.append(
            RexNode(
                idx=len(nodes),
                source=src,
                valid_fidelity=float(s.get("valid_fidelity", 0.0)),
                mismatches=list(s.get("mismatches", [])),
                parent=parent_idx,
                valid_accuracy=float(s.get("valid_accuracy", 0.0)),
                valid_n=int(s.get("valid_n", 0)),
                valid_n_correct=int(s.get("valid_n_correct", 0)),
            )
        )

    # Deterministic final pick: best VALID fidelity, earliest on ties (declared
    # deviation D2 from the paper's near-best softmax).
    final = max(nodes, key=lambda n: (n.valid_fidelity, -n.idx))
    return {
        "game": game,
        "nodes": [
            {
                "idx": n.idx,
                "parent": n.parent,
                "valid_fidelity": round(n.valid_fidelity, 4),
                "valid_accuracy": round(n.valid_accuracy, 4),
                "n_visit": n.n_visit,
                "n_mismatches": len(n.mismatches),
            }
            for n in nodes
        ],
        # Full sources, index-aligned with `nodes`. For callers that need to score
        # EVERY candidate on a held slice (the any-candidate trust marker) -- not
        # meant to be written into an artifact verbatim.
        "node_sources": [n.source for n in nodes],
        "llm_calls": llm_calls,
        "events": events,
        "final_idx": final.idx,
        "final_source": final.source,
        "final_valid_fidelity": round(final.valid_fidelity, 4),
    }


def run_rex_ensemble(
    game: str,
    proposer: Any,
    *,
    train: Sequence[Any],
    valid: Sequence[Any],
    cell: int,
    budget: int,
    score_candidate: Callable[[str], dict],
    read_store_source: Callable[[], Optional[str]],
    write_store_source: Callable[[str], None],
    make_verify_result: Callable[[RexNode, list[dict]], Any],
    ucb1_c: float = 1.0,
) -> dict:
    """Run BOTH arms (linear, rex) SEQUENTIALLY and keep whichever final candidate
    scores higher on VALID -- the only signal available online, before HELD is ever
    seen. REQ-ARC-WMTE-6248 rejected REx as a BLANKET replacement (2 of 6 games
    improved), but a retrospective check on that same run's own data found VALID-score
    selection would have matched the HELD-optimal arm on 6 of 6 games (REQ-ARC-WMTE-6250)
    -- the two arms fail on different games, and picking per-game recovers each arm's
    wins without its losses.

    ONE store, used by BOTH arms, is correct and sufficient: `run_rex` always starts
    with `proposer.induce(...)`, which resets whatever the store held -- there is
    nothing left from one arm for the other to accidentally inherit. This mirrors
    exactly how experiment 6248's own harness already ran both arms back to back
    against the same `_store_file(game)` path; this function just fuses the two calls
    the harness made separately into one, so a caller gets the ensemble decision
    without re-deriving the sequencing.

    Costs ~2x the LLM-call budget of a single arm (both arms run to completion
    regardless of which one is kept) -- affordable for a rare event like induction, not
    for anything that fires every action.
    """
    linear = run_rex(
        game,
        proposer,
        train=train,
        valid=valid,
        cell=cell,
        budget=budget,
        score_candidate=score_candidate,
        read_store_source=read_store_source,
        write_store_source=write_store_source,
        make_verify_result=make_verify_result,
        use_ucb1=False,
        use_qbc=False,
    )
    rex = run_rex(
        game,
        proposer,
        train=train,
        valid=valid,
        cell=cell,
        budget=budget,
        score_candidate=score_candidate,
        read_store_source=read_store_source,
        write_store_source=write_store_source,
        make_verify_result=make_verify_result,
        use_ucb1=True,
        use_qbc=True,
        ucb1_c=ucb1_c,
    )
    arms = {"linear": linear, "rex": rex}
    available = {name: r for name, r in arms.items() if r.get("final_source")}
    if not available:
        chosen_arm = None
    elif len(available) == 1:
        chosen_arm = next(iter(available))
    else:
        chosen_arm = max(available, key=lambda name: available[name]["final_valid_fidelity"])
    chosen = arms[chosen_arm] if chosen_arm else None
    return {
        "game": game,
        "linear": linear,
        "rex": rex,
        "chosen_arm": chosen_arm,
        "chosen_final_source": chosen.get("final_source") if chosen else None,
        "chosen_final_valid_fidelity": chosen.get("final_valid_fidelity") if chosen else None,
        "total_llm_calls": linear.get("llm_calls", 0) + rex.get("llm_calls", 0),
    }
