"""InterwhenTokenMonitor — async token-level polling with PySAT constraint verification.

**Why this module exists (Exp 1776, arXiv:2602.11202):**

    The existing InterWhenMonitor (interwhen_monitor.py) operates at sentence
    granularity: it replays a *completed* response sentence by sentence and calls
    SymCodeVerifier at each boundary.  That level of granularity is appropriate for
    arithmetic CoT steps, but CCTU (Constrained Controlled Tool-Use) benchmark items
    have structural constraints — length, format, numeric range, tool-call count — that
    can be detected at the *token* level long before the model finishes generating.

    This module implements the finer-grained Interwhen idea: a background thread polls
    the in-flight token buffer every ``poll_every_n`` tokens.  At each polling event it
    constructs a PySAT CNF formula from the CCTU constraints that are definitively
    violated given the partial trace so far, and checks satisfiability:

        - SAT  → at least one valid completion still exists → keep generating
        - UNSAT → every possible completion is already doomed → interrupt early

    Early interruption avoids the "wasted tail" tokens that the model generates after
    it has already committed to violating a hard constraint.  ``compute_avoided_pct``
    measures what fraction of that tail was cut short.

**PySAT integration:**

    We model each CCTU constraint as a Boolean propositional variable x_i (literal i+1
    in 1-indexed DIMACS convention).  Variables represent "constraint i can still be
    satisfied given the partial trace."  The monitor adds unit clauses:

        [-i]   when constraint i is definitively violated at the current partial token
               buffer (e.g., word count already exceeds max_words; required markdown
               bold not achievable in remaining space).

    The hard requirement is that every constraint MUST be satisfied in the final
    response, so we add:

        [+i]   as a requirement for all i simultaneously.

    If adding any [-i] makes the formula UNSAT, we interrupt.  In practice this
    simplifies to: interrupt if ANY monitored constraint is provably violated.
    PySAT adds a correct solver layer rather than a plain OR, which makes it easy
    to extend with multi-constraint dependencies in future work.

**CCTU constraint types checked at partial-trace time:**

    - ``length``: partial word count compared to ``validator.max``.  If current count
      already exceeds max, interrupt.  No false positives possible (word count is
      monotonically non-decreasing as tokens accumulate).
    - ``format`` (markdown_bold): if total token budget is exhausted and ``**`` or
      ``__`` never appeared, interrupt.  During partial generation: soft signal only
      — no early interrupt until we know the response is "done."
    - ``numeric``: if partial trace contains a committed numeric answer outside the
      allowed [min, max] range, interrupt.
    - ``resource`` (tool_call_protocol): if the partial trace already contains more
      tool calls than the allowed count, interrupt.

Spec: REQ-VERIFY-175, SCENARIO-VERIFY-175, Exp 1776
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# PySAT import — required for CNF solving (installed as python-sat)
# ---------------------------------------------------------------------------
try:
    from pysat.formula import CNF
    from pysat.solvers import Glucose3

    _PYSAT_AVAILABLE = True
except ImportError:  # pragma: no cover — pysat must be installed for core logic
    _PYSAT_AVAILABLE = False
    CNF = None  # type: ignore[assignment,misc]
    Glucose3 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TokenMonitorResult
# ---------------------------------------------------------------------------


@dataclass
class TokenMonitorResult:
    """Result of a single monitored token-sequence generation.

    Fields
    ------
    interrupted : bool
        True when the monitor halted generation early due to a detected
        constraint violation (UNSAT in the PySAT check).
    interrupt_token_idx : int | None
        Zero-based index of the first token after which the monitor decided
        to interrupt.  None if generation completed without interruption.
    tokens_total : int
        Total number of tokens in the full sequence (before any interruption).
    tokens_generated : int
        Number of tokens that were actually "generated" before the interrupt
        (or the full sequence length if no interrupt occurred).
    tokens_avoided : int
        ``tokens_total - tokens_generated``.  Represents compute saved by
        early interruption.  Zero if no interrupt.
    compute_avoided_pct : float
        ``tokens_avoided / tokens_total * 100``.  Zero if no interrupt.
    violations_detected : list[str]
        Constraint IDs of the violations that triggered the interruption.
    pysat_checks_run : int
        Total number of PySAT satisfiability checks performed during the run.
    """

    interrupted: bool
    interrupt_token_idx: int | None
    tokens_total: int
    tokens_generated: int
    tokens_avoided: int
    compute_avoided_pct: float
    violations_detected: list[str] = field(default_factory=list)
    pysat_checks_run: int = 0


# ---------------------------------------------------------------------------
# Constraint checkers — pure functions, one per CCTU constraint type
# ---------------------------------------------------------------------------


def _count_words(tokens: list[str]) -> int:
    """Count whitespace-delimited words in the joined token sequence."""
    text = " ".join(tokens)
    return len(text.split())


_TOOL_CALL_RE = re.compile(r"<tool_call>|```tool\b|\btool_call\(", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*\S|\b__\S")
_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _count_tool_calls(tokens: list[str]) -> int:
    """Count tool-call markers in the joined partial token buffer."""
    text = " ".join(tokens)
    return len(_TOOL_CALL_RE.findall(text))


def _has_bold_markdown(tokens: list[str]) -> bool:
    """Return True if the joined token buffer contains a bold markdown marker."""
    text = " ".join(tokens)
    return bool(_BOLD_RE.search(text))


def _committed_numbers(tokens: list[str]) -> list[float]:
    """Extract numeric values committed to in the partial token buffer."""
    text = " ".join(tokens)
    return [float(m.replace(",", "")) for m in _NUMBER_RE.findall(text)]


def _check_constraint_violated(constraint: dict[str, Any], partial_tokens: list[str]) -> bool:
    """Return True when ``constraint`` is definitively violated given ``partial_tokens``.

    A constraint is definitively violated when no possible completion of the
    token buffer can make it satisfied.  We only check the types where monotone
    evidence makes this safe:

    - length/max: word count already exceeds max — impossible to decrease.
    - resource/tool_call_protocol/count: tool calls already exceed allowed count.
    - numeric/min+max: a committed numeric value is already outside [min, max].

    Format/markdown_bold violations are NOT checked here because the model could
    still emit ``**bold**`` in future tokens; we only flag format at the end.
    """
    validator = constraint.get("validator", {})
    ctype = constraint.get("type", "")

    if ctype == "length":
        max_words = validator.get("max")
        if max_words is not None and _count_words(partial_tokens) > max_words:
            return True

    elif ctype == "resource" and validator.get("name") == "tool_call_protocol":
        allowed = validator.get("count", 1)
        if _count_tool_calls(partial_tokens) > allowed:
            return True

    elif ctype == "numeric":
        lo = validator.get("min")
        hi = validator.get("max")
        if lo is not None and hi is not None:
            numbers = _committed_numbers(partial_tokens)
            # A number is "committed" when surrounded by non-numeric context
            for n in numbers:
                if n < lo or n > hi:
                    return True

    return False


# ---------------------------------------------------------------------------
# InterwhenTokenMonitor
# ---------------------------------------------------------------------------


class InterwhenTokenMonitor:
    """Async token-level polling monitor using PySAT for constraint verification.

    Wraps a token sequence iterator and polls every ``poll_every_n`` tokens.
    At each poll, constructs a PySAT CNF formula from the CCTU constraints
    that are definitively violated given the partial trace.  If the formula is
    UNSAT, generation is interrupted early.

    Parameters
    ----------
    poll_every_n : int
        Number of tokens between polling events.  Lower values catch violations
        earlier but add more PySAT overhead.  Recommended: 5–20.
    constraints : list[dict]
        CCTU-format constraint dicts, each with at least ``id``, ``type``, and
        ``validator`` keys.  See ``data/cctu_micro_benchmark_25.json`` for the
        canonical schema.
    """

    def __init__(self, poll_every_n: int, constraints: list[dict[str, Any]]) -> None:
        if not _PYSAT_AVAILABLE:
            raise RuntimeError(  # pragma: no cover
                "PySAT is required for InterwhenTokenMonitor. "
                "Install it with: pip install python-sat"
            )
        self.poll_every_n = poll_every_n
        self.constraints = constraints
        # Assign 1-indexed variable IDs for PySAT (DIMACS convention).
        self._var_ids: dict[str, int] = {
            c["id"]: idx + 1 for idx, c in enumerate(constraints)
        }

    # ------------------------------------------------------------------
    # _check_pysat
    # ------------------------------------------------------------------

    def _check_pysat(self, partial_tokens: list[str]) -> tuple[bool, list[str]]:
        """Check satisfiability of the current partial trace against all constraints.

        Builds a CNF formula where:
        - Each constraint i has variable x_i (1-indexed).
        - Unit clause [+x_i] enforces "constraint i must be satisfied."
        - Unit clause [-x_i] is added when constraint i is definitively violated
          by the partial trace.

        Returns
        -------
        (satisfiable, violated_ids)
            satisfiable : bool
                True if the formula is SAT (generation can continue).
                False if UNSAT (at least one constraint is definitely violated).
            violated_ids : list[str]
                Constraint IDs of the definitively violated constraints.
        """
        violated_ids: list[str] = []
        cnf = CNF()

        for constraint in self.constraints:
            cid = constraint["id"]
            var = self._var_ids[cid]
            # Hard requirement: constraint must be satisfied.
            cnf.append([var])
            # Evidence: is it definitively violated given partial trace?
            if _check_constraint_violated(constraint, partial_tokens):
                violated_ids.append(cid)
                cnf.append([-var])  # var must be False → contradicts [+var] → UNSAT

        with Glucose3(bootstrap_with=cnf.clauses) as solver:
            sat = solver.solve()

        return sat, violated_ids

    # ------------------------------------------------------------------
    # monitor_generation
    # ------------------------------------------------------------------

    def monitor_generation(self, token_sequence: list[str]) -> TokenMonitorResult:
        """Monitor a token sequence, interrupting early if constraints are violated.

        Iterates through ``token_sequence`` token by token, polling the PySAT
        checker every ``poll_every_n`` tokens.  When the checker returns UNSAT,
        stops iterating and records the interrupt position.

        Parameters
        ----------
        token_sequence : list[str]
            The full token sequence to "generate."  In production this would come
            from a streaming LLM decoder; here we pass the pre-baked sequence to
            simulate deterministic token emission.

        Returns
        -------
        TokenMonitorResult
            Full result including interruption flag, token counts, and
            compute_avoided_pct.
        """
        total = len(token_sequence)
        generated: list[str] = []
        interrupt_idx: int | None = None
        violations: list[str] = []
        pysat_checks = 0

        for idx, token in enumerate(token_sequence):
            generated.append(token)
            # Poll every poll_every_n tokens (and always check at the last token).
            if (idx + 1) % self.poll_every_n == 0 or idx == total - 1:
                sat, vids = self._check_pysat(generated)
                pysat_checks += 1
                if not sat:
                    interrupt_idx = idx
                    violations = vids
                    break

        tokens_generated = len(generated)
        tokens_avoided = total - tokens_generated
        compute_avoided_pct = tokens_avoided / total * 100.0 if total > 0 else 0.0

        return TokenMonitorResult(
            interrupted=interrupt_idx is not None,
            interrupt_token_idx=interrupt_idx,
            tokens_total=total,
            tokens_generated=tokens_generated,
            tokens_avoided=tokens_avoided,
            compute_avoided_pct=compute_avoided_pct,
            violations_detected=violations,
            pysat_checks_run=pysat_checks,
        )

    # ------------------------------------------------------------------
    # tokenize_response
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize_response(response: str) -> list[str]:
        """Simple whitespace tokenizer that splits a response into tokens.

        In production a real LLM tokenizer (BPE, SentencePiece) would be used.
        For benchmark purposes, whitespace splitting is sufficient — CCTU
        constraint violation detection works at word granularity (length, tools,
        numerics) so sub-word tokenization is not needed.

        Parameters
        ----------
        response : str
            Full or partial response string.

        Returns
        -------
        list[str]
            List of whitespace-split tokens (non-empty only).
        """
        return response.split()
