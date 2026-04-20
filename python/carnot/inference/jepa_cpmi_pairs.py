"""jepa_cpmi_pairs — CPMI contrastive pair builder for JEPA training (arXiv 2604.10660).

**Researcher summary (RETRO-063):**
    The JEPA predictor AUC was stuck at 0.4444 across three consecutive retrains (v8, v9, v10)
    despite switching from BCE to PURE min-form loss.  Root cause: ALL three variants still
    operated on SCALAR loss values computed on STEP-LEVEL labels.  Step-level labels are noisy —
    intermediate steps in a correct chain can look temporarily wrong — so the model hedges to
    P=0.5 because the step-level signal is incoherent.

    The CPMI fix (arXiv 2604.10660) sidesteps step-level noise entirely by constructing
    EXPLICIT contrastive pairs: one (correct_chain, incorrect_chain) pair per question, where
    "correct" and "incorrect" are determined by the final ANSWER verdict, not by intermediate
    step labels.  The margin loss then forces:

        E(incorrect_chain) > E(correct_chain) + margin

    for each pair.  There is no BCE, no PURE, no step-level label involved — just a pairwise
    ordering constraint between two whole chains for the same question.

**Why this works where PURE failed:**
    PURE improved OVER BCE by aggregating step scores with min(), but still trained on
    individual chain labels (correct=1, incorrect=0) rather than on RELATIVE pairs.
    The model can still hedge by making ALL chain scores near 0.5 — the per-chain gradient
    is non-zero but the contrastive gradient (which requires seeing both chains together)
    is zero.

    Contrastive margin loss sees BOTH chains from the same question in a single gradient step
    and explicitly penalises the model if the ordering is wrong.  This is the same mechanism
    that produced AUC=1.0 in the NUP Probe v4 experiments.

**CPMI = Contrastive Pair Margin Inference (internal name for arXiv 2604.10660 approach).**

**Hard negative selection:**
    The "hardest" incorrect chain is the one with the MOST cot_steps.  More steps means
    more elaborate wrong reasoning — more likely to fool a naive classifier — so the model
    has to work harder to correctly rank it below the correct chain.  This is the standard
    hard-negative mining heuristic from metric learning (e.g. FaceNet, 2015).

Spec: REQ-LEARN-065, REQ-LEARN-066,
      SCENARIO-LEARN-101, SCENARIO-LEARN-102, SCENARIO-LEARN-103
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# JEPACPMIPair dataclass
# ---------------------------------------------------------------------------


@dataclass
class JEPACPMIPair:
    """One contrastive pair: a (correct chain, incorrect chain) from the same question.

    **Detailed explanation for engineers:**
        Each ``JEPACPMIPair`` encapsulates everything needed to compute one term of the
        CPMI margin loss for a single question.  The pair is pre-embedded — both
        ``correct_embeddings`` and ``incorrect_embeddings`` are lists of JAX arrays,
        one array per CoT step — so the training loop never needs to call the embed
        function again.

        ``hard_negative_step_idx`` records WHICH step index in the incorrect chain was
        identified as the most misleading one (currently: the last step, i.e. len-1,
        since the hard negative selection picks the longest chain).  This is preserved
        for debugging and for potential future hard-step-mining losses.

        ``pair_quality`` is a scalar in [0, 1] that reflects how "informative" this pair
        is.  Currently computed as:
            pair_quality = len(correct_cot_steps) / max(len(incorrect_cot_steps), 1)
        A value close to 1.0 means the correct and incorrect chains have similar lengths,
        making the contrast more challenging for the model.

    Attributes:
        question_id:           Identifier for the source question (from FOVER corpus).
        correct_embeddings:    One jnp.ndarray per step in the CORRECT chain.
        incorrect_embeddings:  One jnp.ndarray per step in the INCORRECT chain.
        hard_negative_step_idx: Index of the hardest step in incorrect chain (or None).
        pair_quality:          Pair informativeness score in [0, 1].

    Spec: REQ-LEARN-065
    """

    question_id: str
    correct_embeddings: list  # list[jnp.ndarray]
    incorrect_embeddings: list  # list[jnp.ndarray]
    hard_negative_step_idx: Optional[int]
    pair_quality: float


# ---------------------------------------------------------------------------
# CPMIContrastiveLoss
# ---------------------------------------------------------------------------


class CPMIContrastiveLoss:
    """Contrastive hinge margin loss for JEPA CPMI pairs (arXiv 2604.10660).

    **Detailed explanation for engineers:**
        This loss operates on JEPACPMIPair objects.  For each pair it:
        1. Computes E_correct  = chain_energy(model, correct_embeddings)
        2. Computes E_incorrect = chain_energy(model, incorrect_embeddings)
        3. Adds L_i = max(0, margin - (E_incorrect - E_correct))

        The final loss = mean(L_i) over all pairs.  When E_incorrect >= E_correct + margin,
        L_i = 0 (constraint satisfied, no gradient needed).  When the gap is too small,
        L_i > 0 (push the incorrect chain's energy up and the correct chain's down).

        **Chain energy aggregation modes:**
        - 'mean': Average step energy.  Robust to single noisy steps.
        - 'max':  Maximum step energy.  Highlights the "most suspicious" step.
        - 'min':  Minimum step energy.  The weakest-link aggregation from PURE.
                  (PURE showed that min alone is insufficient — we provide all three for
                   ablations in upcoming experiments.)

        **Why no BCE or PURE here?**
        Both BCE and PURE operate on per-chain labels (correct=1 / incorrect=0).  The
        contrastive margin loss operates on PAIRS and directly optimises the relative
        ordering, which is the quantity AUC measures.  Training to maximise AUC directly
        (or its lower bound, the margin ranking loss) is the correct objective for JEPA.

    Args:
        margin:           Minimum required energy gap (E_incorrect - E_correct >= margin).
                          Default 1.0.
        chain_energy_mode: How to aggregate per-step scores.  One of 'mean', 'max', 'min'.
                          Default 'mean'.

    Spec: REQ-LEARN-066
    """

    def __init__(self, margin: float = 1.0, chain_energy_mode: str = "mean") -> None:
        """Create a CPMIContrastiveLoss.

        Args:
            margin:           Minimum required gap between incorrect and correct energy.
            chain_energy_mode: Step aggregation mode ('mean', 'max', or 'min').

        Spec: REQ-LEARN-066
        """
        if chain_energy_mode not in ("mean", "max", "min"):
            raise ValueError(
                f"chain_energy_mode must be 'mean', 'max', or 'min', got {chain_energy_mode!r}"
            )
        self.margin = margin
        self.chain_energy_mode = chain_energy_mode

    def chain_energy(
        self,
        model: Callable,
        embeddings: list,
    ) -> float:
        """Compute a single scalar energy for an entire chain of step embeddings.

        **Detailed explanation for engineers:**
            Each embedding in ``embeddings`` is passed through ``model`` (a callable that
            returns a scalar or 1-D array for each step).  The scalar per-step scores are
            collected and then aggregated by self.chain_energy_mode:
            - 'mean': mean of all scores (default, robust).
            - 'max':  max of all scores (highlights worst-looking step).
            - 'min':  min of all scores (weakest-link / PURE-style).

            If ``embeddings`` is empty, returns 0.0 (neutral energy, no gradient).

        Args:
            model:      Callable mapping jnp.ndarray -> scalar.
            embeddings: List of jnp.ndarray, one per CoT step.

        Returns:
            Scalar float representing the chain's aggregate energy.

        Spec: REQ-LEARN-066
        """
        if not embeddings:
            return 0.0

        scores = [float(model(emb)) for emb in embeddings]

        if self.chain_energy_mode == "mean":
            return sum(scores) / len(scores)
        elif self.chain_energy_mode == "max":
            return max(scores)
        else:  # 'min'
            return min(scores)

    def compute_loss(
        self,
        model: Callable,
        pairs: list[JEPACPMIPair],
    ) -> float:
        """Compute mean contrastive hinge margin loss over all CPMI pairs.

        **Detailed explanation for engineers:**
            For each pair p in pairs:
                E_correct   = chain_energy(model, p.correct_embeddings)
                E_incorrect = chain_energy(model, p.incorrect_embeddings)
                L_p = max(0, margin - (E_incorrect - E_correct))

            Returns mean(L_p).  Returns 0.0 for an empty list via zero_if_empty().

        Args:
            model: Callable mapping jnp.ndarray -> scalar energy.
            pairs: List of JEPACPMIPair objects.

        Returns:
            Mean pair loss as a Python float.

        Spec: REQ-LEARN-066, SCENARIO-LEARN-101, SCENARIO-LEARN-102
        """
        if not pairs:
            return self.zero_if_empty(pairs)

        total = 0.0
        for pair in pairs:
            e_correct = self.chain_energy(model, pair.correct_embeddings)
            e_incorrect = self.chain_energy(model, pair.incorrect_embeddings)
            gap = e_incorrect - e_correct
            total += max(0.0, self.margin - gap)
        return total / len(pairs)

    def zero_if_empty(self, pairs: list) -> float:
        """Return 0.0 when pair list is empty — prevents NaN gradients at training start.

        **Detailed explanation for engineers:**
            At the very start of training (or when the corpus yields no valid pairs),
            the loss must be a well-defined float, not NaN or ZeroDivisionError.
            This method guards that edge case.

        Args:
            pairs: List of JEPACPMIPair (may be empty).

        Returns:
            0.0 always (empty → no gradient).

        Spec: REQ-LEARN-066, SCENARIO-LEARN-103
        """
        return 0.0


# ---------------------------------------------------------------------------
# JEPACPMIPairBuilder
# ---------------------------------------------------------------------------


class JEPACPMIPairBuilder:
    """Build CPMI contrastive pairs from a FOVER corpus for JEPA training.

    **Detailed explanation for engineers:**
        This builder takes a list of FOVERCorpusEntry objects (from fover_corpus.py) and
        produces a list of JEPACPMIPair objects.  The pairing algorithm:

        1. Group entries by ``question_id`` (using the ``question`` field as the ID,
           since FOVER entries from the Exp 442 pipeline use question_id as the question).
        2. For each group: split into correct_entries (is_correct=True) and
           incorrect_entries (is_correct=False).
        3. If the group has at least 1 correct AND at least 1 incorrect entry:
           - Pick the BEST correct entry: the one with the MOST cot_steps (most detailed).
           - Pick the HARDEST incorrect entry: the one with the MOST cot_steps (most
             elaborate wrong reasoning — hardest for the model to correctly rank lower).
           - Embed each step using embed_fn.
           - Compute pair_quality = len(correct.cot_steps) / max(len(incorrect.cot_steps), 1).
           - Yield a JEPACPMIPair.
        4. Groups with only correct or only incorrect entries are skipped — they cannot
           produce a contrastive pair.

        **Fallback: synthetic pairs**
        If the real corpus yields fewer than min_pairs pairs, build_synthetic_pairs() is
        called to pad the batch.  Synthetic pairs inject a single arithmetic error into
        an otherwise correct chain: correct chain sums correctly, incorrect chain has
        one off-by-one.  These are not realistic reasoning chains but provide a non-zero
        contrastive gradient signal to warm up the model before real data is available.

    Args:
        embed_fn:  Callable mapping a text string -> jnp.ndarray (step embedding).
        min_pairs: Minimum number of real pairs before synthetic fallback is used.
                   Default 5.

    Spec: REQ-LEARN-065
    """

    def __init__(self, embed_fn: Callable[[str], jnp.ndarray], min_pairs: int = 5) -> None:
        """Create a JEPACPMIPairBuilder.

        Args:
            embed_fn:  Callable mapping text str -> jnp.ndarray.
            min_pairs: Minimum real pairs required; fewer triggers synthetic fallback.
        """
        self.embed_fn = embed_fn
        self.min_pairs = min_pairs

    def build_pairs(self, fover_entries: list) -> list[JEPACPMIPair]:
        """Build contrastive pairs from FOVER corpus entries.

        **Detailed explanation for engineers:**
            Groups entries by question_id, then for each question that has both correct
            and incorrect responses, picks the best correct entry and the hardest incorrect
            entry (most cot_steps).  Embeds each step and yields a JEPACPMIPair.

            Entries with no cot_steps default to a single empty-string step so that the
            embedding function always has at least one call to make.

        Args:
            fover_entries: List of FOVERCorpusEntry objects (or dicts with same fields).

        Returns:
            List of JEPACPMIPair objects, one per valid question group.

        Spec: REQ-LEARN-065, SCENARIO-LEARN-101
        """
        # Group by question_id (the 'question' field in FOVERCorpusEntry).
        groups: dict[str, dict] = {}
        for entry in fover_entries:
            qid = entry.question if hasattr(entry, "question") else entry.get("question", "")
            if qid not in groups:
                groups[qid] = {"correct": [], "incorrect": []}
            if entry.is_correct if hasattr(entry, "is_correct") else entry.get("is_correct", False):
                groups[qid]["correct"].append(entry)
            else:
                groups[qid]["incorrect"].append(entry)

        pairs: list[JEPACPMIPair] = []
        for qid, group in groups.items():
            correct_entries = group["correct"]
            incorrect_entries = group["incorrect"]
            if not correct_entries or not incorrect_entries:
                # Cannot form a contrastive pair without both sides.
                continue

            # Best correct = most steps (most detailed chain).
            best_correct = max(
                correct_entries,
                key=lambda e: len(e.cot_steps if hasattr(e, "cot_steps") else e.get("cot_steps", [])),
            )
            # Hardest incorrect = most steps (most elaborate wrong reasoning).
            hardest_incorrect = max(
                incorrect_entries,
                key=lambda e: len(e.cot_steps if hasattr(e, "cot_steps") else e.get("cot_steps", [])),
            )

            correct_steps = (
                best_correct.cot_steps if hasattr(best_correct, "cot_steps")
                else best_correct.get("cot_steps", [])
            )
            incorrect_steps = (
                hardest_incorrect.cot_steps if hasattr(hardest_incorrect, "cot_steps")
                else hardest_incorrect.get("cot_steps", [])
            )

            # Extract step texts, falling back to empty string for steps without text.
            def _step_text(step: dict | str) -> str:
                if isinstance(step, dict):
                    return step.get("step_text", "") or step.get("text", "")
                return str(step)

            correct_texts = [_step_text(s) for s in correct_steps] or [""]
            incorrect_texts = [_step_text(s) for s in incorrect_steps] or [""]

            correct_embeddings = [self.embed_fn(t) for t in correct_texts]
            incorrect_embeddings = [self.embed_fn(t) for t in incorrect_texts]

            pair_quality = len(correct_steps) / max(len(incorrect_steps), 1)
            hard_negative_step_idx = len(incorrect_steps) - 1 if incorrect_steps else None

            pairs.append(
                JEPACPMIPair(
                    question_id=qid,
                    correct_embeddings=correct_embeddings,
                    incorrect_embeddings=incorrect_embeddings,
                    hard_negative_step_idx=hard_negative_step_idx,
                    pair_quality=pair_quality,
                )
            )

        return pairs

    def build_synthetic_pairs(self, n_pairs: int) -> list[JEPACPMIPair]:
        """Generate synthetic contrastive pairs as a fallback for small real corpora.

        **Detailed explanation for engineers:**
            Each synthetic pair consists of:
            - CORRECT chain: two steps with accurate arithmetic.
              Step 1: "The value is X."
              Step 2: "Therefore the answer is X + X = {2*X}."  (correct)
            - INCORRECT chain: two steps with one off-by-one error.
              Step 1: "The value is X."
              Step 2: "Therefore the answer is X + X = {2*X + 1}."  (wrong by 1)

            The pair_quality = 1.0 for all synthetic pairs (equal step counts).
            The question_id is "synthetic_{i}" for i in range(n_pairs).

            These pairs exist solely to give the model a non-zero contrastive gradient
            signal when the real corpus is too small.  They are explicitly labelled
            n_synthetic_pairs in experiment artifacts so they are not confused with real data.

        Args:
            n_pairs: Number of synthetic pairs to generate.

        Returns:
            List of n_pairs JEPACPMIPair objects.

        Spec: REQ-LEARN-065, SCENARIO-LEARN-103
        """
        pairs: list[JEPACPMIPair] = []
        for i in range(n_pairs):
            x = i + 1
            correct_steps = [
                f"The value is {x}.",
                f"Therefore the answer is {x} + {x} = {2 * x}.",
            ]
            incorrect_steps = [
                f"The value is {x}.",
                f"Therefore the answer is {x} + {x} = {2 * x + 1}.",
            ]
            correct_embeddings = [self.embed_fn(t) for t in correct_steps]
            incorrect_embeddings = [self.embed_fn(t) for t in incorrect_steps]
            pairs.append(
                JEPACPMIPair(
                    question_id=f"synthetic_{i}",
                    correct_embeddings=correct_embeddings,
                    incorrect_embeddings=incorrect_embeddings,
                    hard_negative_step_idx=1,
                    pair_quality=1.0,
                )
            )
        return pairs
