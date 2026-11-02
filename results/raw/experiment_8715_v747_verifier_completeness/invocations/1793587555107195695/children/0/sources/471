"""Φ (phi) measurement module — Zenil α_t exogenous-grounding metric for Carnot.

**Why this module exists (researcher summary):**
    Exp 1031 found that Carnot's energy filter performed WORSE than a plain
    temperature-percentile baseline: accuracy 0.196 vs 1.0.  Zenil (2026,
    arXiv 2601.05280) Theorem 4 tells us WHY this matters: for a
    self-distillation loop to converge, the exogenous verifier must contribute
    α_t > 0 at every distillation step.  If Carnot's energy function merely
    agrees with temperature at all decision boundaries, α_t = 0 and the loop
    degenerates — the student learns nothing new from the verifier.

**What α_t measures:**
    α_t is the fraction of examples where Carnot's energy verdict CHANGED the
    selection versus what temperature-only selection would have chosen.  It is
    the "information delta" that Carnot contributes.  A value of 0.0 means
    Carnot is redundant (never changes anything); 1.0 means Carnot always
    disagrees with temperature (maximally independent, but possibly wrong).
    The sweet spot is α_t ∈ (0.1, 0.5) — enough disagreement to add signal,
    not so much that the verifier is just noise.

**AND-composition (Phase-3 k=5):**
    Per the Deep Think Round-9 4-step recipe: composing k independent verifiers
    with AND logic shrinks the joint null space exponentially.  A candidate
    "passes" the AND-composed verifier only when ALL k individual verifiers
    agree it is correct.  This raises the bar, increases false-negative rate,
    but dramatically reduces false positives — which is the right trade-off for
    a self-distillation teacher that must only teach on verified examples.

    bypass_rate = fraction of examples where the AND verdict disagrees with
    at least one individual verifier.  High bypass_rate = the verifiers are
    orthogonal (good: different null spaces); low bypass_rate = the verifiers
    share a null space (bad: AND adds no new coverage).

Spec: REQ-PHI-001 (alpha_t measurement), REQ-PHI-002 (AND-composition bypass rate),
      REQ-PHI-003 (convergence gate — alpha_t > 0 required for FR-11 loop).
"""

from __future__ import annotations

from typing import NamedTuple


class VerdictRecord(NamedTuple):
    """Single verifier verdict for one candidate.

    Fields:
        example_id: Unique identifier for the (question, candidate) pair.
        verdict: Binary string — "correct" or "incorrect".
        score: Numeric score produced by the verifier (energy, temperature,
               confidence, etc.).  Used only for diagnostics; the verdict
               field is what drives α_t computation.
    """

    example_id: str
    verdict: str
    score: float


class AlphaTResult(NamedTuple):
    """Result of a single α_t measurement.

    Fields:
        alpha_t: Fraction of examples where Carnot's verdict differs from
                 temperature-only baseline.  This IS the Zenil α_t metric.
        delta_example_ids: IDs of examples where the two verifiers disagreed.
        n_total: Total number of examples evaluated.
        n_disagreements: Raw count of disagreements (= alpha_t * n_total).
    """

    alpha_t: float
    delta_example_ids: list[str]
    n_total: int
    n_disagreements: int


class AndCompositionResult(NamedTuple):
    """Result of AND-composing k verifiers.

    Fields:
        and_verdicts: List of VerdictRecords where verdict = AND of all k.
        bypass_rate: Fraction of examples where AND disagrees with at least
                     one individual verifier (i.e. AND added new information).
        n_passed: Number of examples that passed all k verifiers.
        n_bypassed: Number of examples where AND changed the outcome.
    """

    and_verdicts: list[VerdictRecord]
    bypass_rate: float
    n_passed: int
    n_bypassed: int


def measure_alpha_t(
    energy_verdicts: list[VerdictRecord],
    temperature_verdicts: list[VerdictRecord],
) -> AlphaTResult:
    """Measure Carnot's exogenous grounding contribution as α_t.

    α_t is the fraction of examples where Carnot's energy-based verdict
    CHANGED the selection relative to the temperature-percentile baseline.
    This directly operationalises Zenil Theorem 4's convergence condition:
    the self-distillation loop requires α_t > 0 at every step.

    If α_t = 0, Carnot is provably redundant — every selection it would make
    is identical to what temperature alone would make.  In that case the loop
    adds no new signal and cannot converge to a better model.

    Args:
        energy_verdicts: Carnot energy verifier verdicts — list of
            VerdictRecord(example_id, verdict, score).  verdict ∈
            {"correct", "incorrect"}.  Length must equal temperature_verdicts.
        temperature_verdicts: Temperature-only baseline verdicts — same
            structure.  example_ids must match energy_verdicts in order.

    Returns:
        AlphaTResult with alpha_t, delta_example_ids, n_total, n_disagreements.

    Raises:
        ValueError: If the two lists have different lengths.
        ValueError: If example_ids do not match at corresponding positions.
    """
    if len(energy_verdicts) != len(temperature_verdicts):
        raise ValueError(
            f"energy_verdicts and temperature_verdicts must have the same length; "
            f"got {len(energy_verdicts)} vs {len(temperature_verdicts)}"
        )

    delta_ids: list[str] = []
    for ev, tv in zip(energy_verdicts, temperature_verdicts):
        if ev.example_id != tv.example_id:
            raise ValueError(
                f"example_id mismatch at corresponding positions: "
                f"{ev.example_id!r} vs {tv.example_id!r}"
            )
        if ev.verdict != tv.verdict:
            delta_ids.append(ev.example_id)

    n_total = len(energy_verdicts)
    n_disagreements = len(delta_ids)
    alpha_t = n_disagreements / n_total if n_total > 0 else 0.0

    return AlphaTResult(
        alpha_t=alpha_t,
        delta_example_ids=delta_ids,
        n_total=n_total,
        n_disagreements=n_disagreements,
    )


def and_compose_verifiers(
    all_verifier_verdicts: list[list[VerdictRecord]],
) -> AndCompositionResult:
    """AND-compose k verifier verdict lists into a single combined verdict.

    A candidate passes the AND-composed verifier only when ALL k individual
    verifiers mark it as "correct".  This is the Phase-3 k=5 recipe from
    Deep Think Round-9: AND-composition shrinks the joint null space
    exponentially in k, making specification gaming much harder.

    bypass_rate captures how often the AND composition produces a different
    outcome than any individual verifier would produce alone.  High bypass_rate
    means the verifiers are informationally orthogonal — good, because it
    means AND is adding new filtering power beyond any single verifier.

    Args:
        all_verifier_verdicts: A list of k lists, each of length n.  Entry
            all_verifier_verdicts[i][j] is verifier i's verdict for example j.
            All lists must have the same length and matching example_ids.

    Returns:
        AndCompositionResult with and_verdicts, bypass_rate, n_passed, n_bypassed.

    Raises:
        ValueError: If fewer than 1 verifier is provided.
        ValueError: If verifier lists have different lengths.
        ValueError: If example_ids do not match across verifiers.
    """
    if len(all_verifier_verdicts) < 1:
        raise ValueError("Need at least 1 verifier list for AND-composition.")

    n = len(all_verifier_verdicts[0])
    for i, vlist in enumerate(all_verifier_verdicts[1:], start=1):
        if len(vlist) != n:
            raise ValueError(f"Verifier list {i} has length {len(vlist)}, expected {n}.")

    # Validate that example_ids match across all verifier lists
    ref_ids = [r.example_id for r in all_verifier_verdicts[0]]
    for i, vlist in enumerate(all_verifier_verdicts[1:], start=1):
        for j, (ref_id, rec) in enumerate(zip(ref_ids, vlist)):
            if ref_id != rec.example_id:
                raise ValueError(
                    f"Verifier {i}, position {j}: example_id {rec.example_id!r} "
                    f"does not match reference {ref_id!r}."
                )

    and_verdicts: list[VerdictRecord] = []
    n_passed = 0
    n_bypassed = 0

    for j in range(n):
        individual_verdicts = [vlist[j].verdict for vlist in all_verifier_verdicts]
        # AND: all must be "correct"
        and_verdict = "correct" if all(v == "correct" for v in individual_verdicts) else "incorrect"

        # bypass = AND disagrees with at least one individual verifier
        if any(v != and_verdict for v in individual_verdicts):
            n_bypassed += 1

        if and_verdict == "correct":
            n_passed += 1

        # Use the first verifier's score as representative (AND has no single score)
        representative_score = all_verifier_verdicts[0][j].score
        and_verdicts.append(
            VerdictRecord(
                example_id=ref_ids[j],
                verdict=and_verdict,
                score=representative_score,
            )
        )

    bypass_rate = n_bypassed / n if n > 0 else 0.0

    return AndCompositionResult(
        and_verdicts=and_verdicts,
        bypass_rate=bypass_rate,
        n_passed=n_passed,
        n_bypassed=n_bypassed,
    )
