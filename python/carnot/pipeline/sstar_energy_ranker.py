"""S* energy-based candidate pre-ranker (arXiv 2502.14382).

**Researcher summary:**
    S* (arXiv 2502.14382) generates N candidate solutions and uses distinguishing
    execution tests (tests that differentiate candidates) as an oracle for
    tournament selection. Execution tests are expensive: running code + collecting
    output for every candidate on every test. This module implements a pre-filter
    that uses Carnot's static energy (bag-of-tokens embedding) to rank candidates
    BEFORE running execution tests. If the energy correctly identifies the best
    candidate, we skip 75% of test executions (for N=4 candidates).

**How energy ranking works:**
    1. Each candidate code string is tokenized via Python's stdlib ``tokenize``.
    2. Token frequencies are accumulated into a fixed-size vector (vocab_size=256).
       This is the "bag-of-tokens" embedding from REQ-CODE-002.
    3. The L1 norm of the embedding (total token count) is used as a proxy energy.
       The intuition: correct implementations tend to be simpler and shorter than
       over-complicated buggy candidates. This is an Occam's razor prior.
    4. Candidates are sorted ascending by energy (lowest = predicted-best).
    5. select_top_k() returns the top-k lowest-energy candidates.

**Important limitation:**
    This is a STATIC energy — it does not execute the code. It cannot distinguish
    candidates that are syntactically similar but semantically different (e.g.,
    "return a + b" vs "return a - b" — same token count, different operators at
    different hash buckets). The experiment (Exp 787) measures how often this
    static signal correlates with actual correctness. If the correlation is low,
    the verdict is "energy_prefilter_random" — which is an honest, informative
    scientific result motivating future trained-model approaches (Exp 785).

Spec: REQ-RANK-001, REQ-RANK-002, SCENARIO-RANK-001, SCENARIO-RANK-002
"""

from __future__ import annotations

from dataclasses import dataclass, field

from carnot.verify.python_types import code_to_embedding


@dataclass
class SStarConfig:
    """Configuration for the S* energy pre-ranker.

    **Detailed explanation for engineers:**
        n_candidates controls how many candidates we expect per problem — this
        matches the N in the S* paper (typical: 4 or 8). energy_top_k is how
        many candidates to retain after energy pre-filtering; default 1 means
        we select the single best candidate and skip all execution tests.
        vocab_size is the embedding dimension for code_to_embedding.

    Attributes:
        n_candidates: Expected number of candidates per problem (S* paper N=4).
        energy_top_k: How many top candidates to return from select_top_k().
        vocab_size: Embedding dimension for bag-of-tokens representation.
    """

    n_candidates: int = 4
    energy_top_k: int = 1
    vocab_size: int = 256


@dataclass
class CandidateWithEnergy:
    """A code candidate paired with its computed static energy score.

    **Detailed explanation for engineers:**
        Holds the original candidate string plus the scalar energy. Lower energy
        means the ranker predicts this candidate is more likely to be correct.
        The original_index tracks which position the candidate occupied in the
        input list so callers can recover the pre-ranking order if needed.

    Attributes:
        code: The candidate Python code string.
        energy: Scalar energy score (lower = more likely correct).
        original_index: Position of this candidate in the input candidates list.
    """

    code: str
    energy: float
    original_index: int


class SStarEnergyRanker:
    """Rank code candidates by Carnot static energy before running execution tests.

    **Detailed explanation for engineers:**
        This class implements the energy pre-filtering stage of the S* pipeline.
        It is intentionally stateless beyond its config — no model weights, no
        training required. The energy computation uses code_to_embedding() from
        python_types.py (REQ-CODE-002), which is deterministic and fast (<1ms
        per candidate on CPU).

        Usage pattern:
            ranker = SStarEnergyRanker(config=SStarConfig(n_candidates=4))
            ranked = ranker.rank_by_energy(candidates, problem_context)
            # ranked[0] is the predicted-best candidate (lowest energy)
            top1 = ranker.select_top_k(candidates, problem_context, k=1)

    Spec: REQ-RANK-001, REQ-RANK-002
    """

    def __init__(self, config: SStarConfig | None = None) -> None:
        """Initialize the ranker with configuration.

        Args:
            config: SStarConfig instance. If None, uses default SStarConfig().
        """
        self.config = config if config is not None else SStarConfig()

    def compute_energy(self, code: str) -> float:
        """Compute static structural energy for a single code candidate.

        **Detailed explanation for engineers:**
            Embeds the code as a bag-of-tokens vector via code_to_embedding(),
            then returns the L1 norm (sum of all token counts). The L1 norm
            equals the total token count. Lower norm = fewer tokens = simpler
            code = lower energy.

            Why L1 norm as energy? It encodes an Occam's razor prior: among
            candidates that could be correct, simpler ones (fewer tokens) are
            preferred. This is a weak signal — it helps distinguish long
            over-complicated candidates from concise correct ones, but cannot
            distinguish candidates of equal token length.

        Args:
            code: Python source code string for one candidate.

        Returns:
            Non-negative float energy score (lower = predicted more correct).

        Spec: REQ-RANK-001
        """
        embedding = code_to_embedding(code, vocab_size=self.config.vocab_size)
        # L1 norm = total token count — simple complexity proxy.
        return float(embedding.sum())

    def rank_by_energy(
        self,
        candidates: list[str],
        problem_context: str = "",  # noqa: ARG002 — reserved for future semantic energy
    ) -> list[str]:
        """Sort candidates by energy ascending (lowest energy = predicted best).

        **Detailed explanation for engineers:**
            Computes energy for each candidate, sorts ascending, returns the
            reordered list. The problem_context argument is reserved for future
            semantic energy computation (e.g., embedding the problem description
            and comparing against candidate embeddings for relevance scoring).
            It is unused in this static implementation.

            Ties in energy are broken by original order (stable sort) so the
            output is deterministic.

        Args:
            candidates: List of Python code strings (N candidates per problem).
            problem_context: Problem description string (unused; reserved for
                future semantic energy extensions).

        Returns:
            Reordered list of candidates, lowest energy first. Length equals
            len(candidates). No candidates are dropped.

        Spec: REQ-RANK-001, SCENARIO-RANK-001
        """
        scored: list[CandidateWithEnergy] = [
            CandidateWithEnergy(code=c, energy=self.compute_energy(c), original_index=i)
            for i, c in enumerate(candidates)
        ]
        # Stable sort: ties broken by original index (stable=True is Python's default).
        scored.sort(key=lambda x: (x.energy, x.original_index))
        return [item.code for item in scored]

    def select_top_k(
        self,
        candidates: list[str],
        problem_context: str = "",
        k: int | None = None,
    ) -> list[str]:
        """Return the top-k lowest-energy candidates.

        **Detailed explanation for engineers:**
            Calls rank_by_energy() then slices the first k results. k defaults
            to self.config.energy_top_k (typically 1 for S* pre-filtering).
            This is the list of candidates we would submit to the execution
            oracle, skipping all other candidates.

        Args:
            candidates: List of Python code strings (N candidates per problem).
            problem_context: Problem description (passed through to rank_by_energy).
            k: Number of top candidates to return. Defaults to config.energy_top_k.

        Returns:
            List of up to k candidate strings, sorted lowest energy first.

        Spec: REQ-RANK-001
        """
        effective_k = k if k is not None else self.config.energy_top_k
        ranked = self.rank_by_energy(candidates, problem_context)
        return ranked[:effective_k]
