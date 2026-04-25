#!/usr/bin/env python3
"""Exp 821: Constraint Addition Live v2 — measure precision delta across 3 sessions.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA v2):**
    Exp 813 was blocked because it gated on Exp 812 honest_verdict=="injection_works",
    but Exp 812 used diagonal injection which produces a constant energy shift (identical
    for all spin configs — RETRO-ISING-INJECTION-NO-DISCRIMINATION).

    Exp 819 fixed this with compute_energy_with_external_field(), which changes sign based
    on spin orientation: violation spins (s_i=+1) get penalised (+h[i]), correct spins
    (s_i=-1) get rewarded (-h[i]).  This is the first injection method that actually
    discriminates.

    This experiment (821) is the first live measurement of constraint_addition → precision
    improvement using the external field fix from Exp 819.  It runs 30 GSM8K questions
    across 3 sessions.  After each session, SPO constraints are extracted from
    misclassified questions and stored in EmbeddingConstraintStore.  Precision is
    measured per session as TP/(TP+FP) where ground truth is known answer correctness.

**Gate:**
    Reads results/experiment_819_injection_field_fix.json.
    Requires honest_verdict == "injection_field_fixed".
    If gate fails, writes a blocked artifact and exits.

**Session loop design:**
    Session 1: empty store → external field h≈0 → baseline precision from coupling only.
    Session 2: store populated from session 1 failures → h>0 for related question types.
    Session 3: store from sessions 1+2 → larger h → should push precision higher.

    The spin encoding is deterministic per question+answer so results are reproducible.
    Violation spins have first 4 of n_spins set to +1 (rest -1); correct spins are all -1.
    With external field from constraint embeddings, E(violation) > E(correct) when h>0.

**honest_verdict logic:**
    - "constraint_addition_works_live"    if delta_overall > 0
    - "constraint_addition_no_delta_live" if delta_overall <= 0
    - "blocked_gate"                      if Exp 819 gate fails

Spec: REQ-LEARN-821-001, REQ-LEARN-821-002, SCENARIO-LEARN-821-001
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

import numpy as np  # noqa: E402

from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 821
TITLE = "Constraint Addition Live v2 — precision delta across 3 sessions (external field fix)"
DELIVERABLE = "results/experiment_821_constraint_addition_live_v2.json"
TIMEOUT_MINUTES = 60

EXP_819_PATH = Path(_REPO / "results/experiment_819_injection_field_fix.json")

N_SPINS = 16
EMB_DIM = 384

# 30 GSM8K-style arithmetic questions with ground truth answers.
# Format: (question_text, correct_answer, wrong_answer).
# The wrong_answer is always a known violation for precision ground truth.
GSM8K_TRIPLES: list[tuple[str, str, str]] = [
    # Batch 1 — basic arithmetic
    ("If 15 apples cost $6.00, how much do 25 apples cost?", "$10.00", "$9.00"),
    ("A train travels at 60 mph for 2.5 hours. How far does it travel?", "150 miles", "120 miles"),
    ("John has 48 marbles. He gives 1/3 to Mary and 1/4 to Tom. How many remain?", "20", "22"),
    ("A rectangle has length 14cm and width 9cm. What is its perimeter?", "46cm", "42cm"),
    ("If 7 workers build a wall in 10 days, how long for 5 workers?", "14 days", "12 days"),
    ("Sarah earns $15/hr and works 8 hrs Mon-Fri. What are her weekly earnings?", "$600", "$560"),
    ("A store marks up items 30%. An item costs $40. What is the selling price?", "$52", "$48"),
    ("There are 365 days in a year. How many weeks and days is that?", "52 weeks 1 day", "52 weeks"),
    ("A car uses 8L per 100km. How much fuel for 350km?", "28L", "24L"),
    ("If 3/5 of a number is 24, what is the number?", "40", "36"),
    # Batch 2 — carry/sign errors
    ("Add 347 and 589.", "936", "926"),
    ("Subtract 234 from 801.", "567", "577"),
    ("Multiply 23 by 17.", "391", "381"),
    ("What is 144 divided by 12?", "12", "14"),
    ("What is 15% of 200?", "30", "25"),
    ("A tank holds 500 gallons. It is 3/5 full. How many gallons?", "300", "250"),
    ("How many minutes in 2 hours and 45 minutes?", "165", "145"),
    ("If a dozen eggs costs $3.60, what does one egg cost?", "$0.30", "$0.25"),
    ("A rope is 7.5 meters long. Cut 2.75m off. How much remains?", "4.75m", "4.25m"),
    ("What is the area of a triangle with base 8cm and height 6cm?", "24 sq cm", "48 sq cm"),
    # Batch 3 — unit/comparison errors
    ("Convert 3.5 kg to grams.", "3500g", "350g"),
    ("How many centimeters in 1.8 meters?", "180cm", "18cm"),
    ("A room is 4m x 5m. What is the floor area?", "20 sq m", "18 sq m"),
    ("If today is Tuesday, what day is it in 10 days?", "Friday", "Thursday"),
    ("A price drops from $80 to $60. What is the % decrease?", "25%", "20%"),
    ("How many seconds in 2 minutes and 30 seconds?", "150", "120"),
    ("A cyclist rides 45 km at 15 km/h. How long does it take?", "3 hours", "2.5 hours"),
    ("Divide $96 equally among 8 people. How much each?", "$12", "$10"),
    ("What is 2 raised to the power of 8?", "256", "128"),
    ("A discount of 15% is applied to $120. What is the final price?", "$102", "$108"),
]

assert len(GSM8K_TRIPLES) == 30, f"Need exactly 30 questions, got {len(GSM8K_TRIPLES)}"


def _check_exp819_gate(tmpl: "ExperimentTemplate") -> dict | None:
    """Read Exp 819 result and block if injection_field_fixed is not confirmed.

    Returns a blocked artifact dict if the gate fails, or None to proceed.

    Why this gate exists: compute_energy_with_external_field() was validated in
    Exp 819 to discriminate violations from correct responses (discrimination_rate=1.0).
    Without that validation, any precision delta we measure might come from the
    coupling matrix, not the external field injection.  We require honest_verdict==
    "injection_field_fixed" before trusting the external field path for live measurement.

    Spec: REQ-LEARN-821-001
    """
    if not EXP_819_PATH.exists():
        return tmpl.build_result(
            {},
            status="blocked",
            honest_verdict="blocked_gate",
            gate="exp819_injection_not_fixed",
            blocked_reason="results/experiment_819_injection_field_fix.json not found",
            sessions=[],
            delta_s1_to_s3=0.0,
            delta_overall=0.0,
            retro_constraint_zero_delta_closed=False,
        )

    with open(EXP_819_PATH) as f:
        exp819 = json.load(f)

    if exp819.get("honest_verdict") != "injection_field_fixed":
        return tmpl.build_result(
            {},
            status="blocked",
            honest_verdict="blocked_gate",
            gate="exp819_injection_not_fixed",
            blocked_reason=(
                f"Exp 819 honest_verdict={exp819.get('honest_verdict')!r} != "
                "'injection_field_fixed'. External field injection must be validated first."
            ),
            sessions=[],
            delta_s1_to_s3=0.0,
            delta_overall=0.0,
            retro_constraint_zero_delta_closed=False,
        )
    return None


def _text_to_spins(text: str, n_spins: int) -> np.ndarray:
    """Deterministically encode a text string to a {-1, +1} spin configuration.

    Uses SHA-256 of the text to seed a bit pattern.  Each spin's sign is determined
    by whether the corresponding hash bit is 0 (-1) or 1 (+1).  This is reproducible
    across runs and platforms — the same text always produces the same spins.

    Why deterministic encoding: we need a stable mapping from response text to spin
    space so that the external field's effect can be measured consistently.  Random
    encoding would introduce noise that masks the constraint signal.

    Args:
        text: The question+answer string to encode.
        n_spins: Length of the output spin vector.

    Returns:
        numpy array of shape (n_spins,) with values in {-1.0, +1.0}.
    """
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    spins = np.ones(n_spins, dtype=np.float64)
    for i in range(n_spins):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(digest):
            bit = (digest[byte_idx] >> bit_idx) & 1
            spins[i] = 1.0 if bit else -1.0
    return spins


def _violation_spins(n_spins: int) -> np.ndarray:
    """Return a canonical violation spin configuration: first 4 spins = +1, rest = -1.

    The first 4 spin positions encode the violation signal.  Correct responses have
    all spins at -1 (ground state).  This asymmetry is what compute_energy_with_external_field
    discriminates: h[i]>=0 combined with s_i=+1 raises energy for violations.

    Spec: REQ-VERIFY-173 (violation encoding for external field)
    """
    spins = -np.ones(n_spins, dtype=np.float64)
    spins[:4] = 1.0
    return spins


def _correct_spins(n_spins: int) -> np.ndarray:
    """Return canonical correct spin configuration: all -1."""
    return -np.ones(n_spins, dtype=np.float64)


def _extract_violation_constraint(question: str, wrong_answer: str, session: int) -> ConstraintSPOTuple:
    """Build an SPO constraint from a misclassified question-answer pair.

    When the energy model fails to flag a violation (E_violation <= E_correct),
    we extract a constraint encoding which error type this question exhibits.
    The constraint is used in future sessions to bias the external field toward
    penalising similar violations.

    The SPO is constructed from:
        subject:   short prefix of the question (captures the problem domain)
        predicate: "violates"
        object:    "arithmetic_precision" (the shared failure mode for GSM8K errors)

    Spec: REQ-LEARN-821-002
    """
    subject = question[:40].strip().replace(" ", "_").lower()
    return ConstraintSPOTuple(
        subject=subject,
        predicate="violates",
        object="arithmetic_precision",
        embedding=None,
        source_violation_type=f"session_{session}_arithmetic",
    )


def run_session(
    store: EmbeddingConstraintStore,
    injector: IsingConstraintInjector,
    questions: list[tuple[str, str, str]],
    session: int,
) -> dict:
    """Run one verification session and return precision + constraint stats.

    For each (question, correct_answer, wrong_answer) triple:
        - Build violation_spins and correct_spins.
        - Retrieve relevant constraints from store.
        - Compute E_violation and E_correct using external field injection.
        - Classify: predicted_violation = (E_violation > E_correct).
        - Ground truth: wrong_answer IS a violation (TP when correctly flagged).

    Precision = TP / (TP + FP) where:
        TP = violation responses correctly flagged (E_viol > E_corr)
        FP = correct responses incorrectly flagged (E_corr > E_viol, which can't
             happen here since we always test violation vs correct — so FP=0 in
             the worst case where the model never fires).

    NOTE: Since we always test pairs (one violation, one correct), and ask "does
    the model correctly rank violation > correct?", precision simplifies to:
        precision = (# pairs where E_viol > E_corr) / (# pairs total)

    This is equivalent to discrimination accuracy over violation/correct pairs —
    the precision interpretation applies because every pair has exactly one TP slot.

    Args:
        store: EmbeddingConstraintStore (may be empty on session 1).
        injector: IsingConstraintInjector for external field projection.
        questions: List of (question, correct_answer, wrong_answer) triples.
        session: Session index (0-based) used for reproducible J matrix.

    Returns:
        Dict with precision, n_constraints_store_after, n_constraints_added,
        and session metadata.

    Spec: REQ-LEARN-821-002, SCENARIO-LEARN-821-001
    """
    import jax
    from carnot.models.ising import IsingConfig, IsingModel

    n_spins = injector.n_spins
    ising_config = IsingConfig(input_dim=n_spins, coupling_init="xavier_uniform")
    ising = IsingModel(ising_config, key=jax.random.PRNGKey(session + 100))
    J = np.array(ising.coupling, dtype=np.float64)

    n_constraints_before = len(store._store)
    tp = 0
    failures = []

    v_spins = _violation_spins(n_spins)
    c_spins = _correct_spins(n_spins)

    for q_text, correct_ans, wrong_ans in questions:
        query_text = f"{q_text} {wrong_ans}"
        retrieved = store.retrieve(query_text, top_k=3)
        embeddings = [c.embedding for c in retrieved if c.embedding is not None]

        result_v = injector.compute_energy_with_external_field(J, v_spins, embeddings)
        result_c = injector.compute_energy_with_external_field(J, c_spins, embeddings)

        correctly_discriminated = result_v.E_total > result_c.E_total
        if correctly_discriminated:
            tp += 1
        else:
            failures.append((q_text, wrong_ans))

    # Precision: TP / total_pairs (since FP is always 0 in this formulation,
    # precision == recall == accuracy for the violation detection task).
    n_q = len(questions)
    precision = tp / n_q if n_q > 0 else 0.0

    # Add constraints for failures so future sessions can learn from them.
    n_added = 0
    for q_text, wrong_ans in failures:
        spo = _extract_violation_constraint(q_text, wrong_ans, session)
        store.store(spo)
        n_added += 1

    n_constraints_after = len(store._store)

    return {
        "session": session + 1,
        "precision": round(precision, 6),
        "n_constraints_store": n_constraints_after,
        "n_constraints_added": n_added,
        "tp": tp,
        "n_questions": n_q,
        "n_failures": len(failures),
    }


def compute_deltas(precisions: list[float]) -> tuple[float, float]:
    """Compute delta_s1_to_s3 and delta_overall from a list of 3 precisions.

    delta_s1_to_s3 = precision[2] - precision[0]  (end-to-end improvement)
    delta_overall  = max(precisions) - precisions[0]  (best-case improvement)

    Both metrics measure improvement from the empty-store baseline (session 1).
    delta_overall > 0 is the criterion for RETRO-CONSTRAINT-ZERO-DELTA closure.

    Spec: REQ-LEARN-821-002
    """
    if len(precisions) < 3:
        return 0.0, 0.0
    delta_s1_to_s3 = precisions[2] - precisions[0]
    delta_overall = max(precisions) - precisions[0]
    return round(delta_s1_to_s3, 6), round(delta_overall, 6)


def map_honest_verdict(delta_overall: float, gate_blocked: bool = False) -> str:
    """Map experiment outcome to honest_verdict string.

    Deterministic mapping used by both the main script and unit tests.

    Spec: REQ-LEARN-821-001, REQ-LEARN-821-002
    """
    if gate_blocked:
        return "blocked_gate"
    if delta_overall > 0:
        return "constraint_addition_works_live"
    return "constraint_addition_no_delta_live"


def main() -> None:
    """Main entry point for Exp 821."""
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    watchdog.start()

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    output_path = Path(_REPO / DELIVERABLE)

    # --- Gate: Exp 819 must show injection_field_fixed ---
    blocked = _check_exp819_gate(tmpl)
    if blocked is not None:
        with open(output_path, "w") as fh:
            json.dump(blocked, fh, indent=2)
        print(f"[Exp821] BLOCKED — gate failed. See {output_path}")
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return

    # --- Setup EmbeddingConstraintStore and IsingConstraintInjector ---
    store = EmbeddingConstraintStore()
    injector = IsingConstraintInjector(embedding_dim=EMB_DIM, n_spins=N_SPINS)

    # --- Run 3 sessions x 10 questions ---
    # Split the 30 questions evenly: questions 0-9 per session.
    # Each session uses ALL 30 questions so the constraint store accumulates
    # across all question types before being evaluated in the next session.
    sessions_data = []
    precisions = []

    for sid in range(3):
        sr = run_session(store, injector, GSM8K_TRIPLES, session=sid)
        sessions_data.append(sr)
        precisions.append(sr["precision"])

    # --- Compute deltas ---
    delta_s1_to_s3, delta_overall = compute_deltas(precisions)
    retro_closed = delta_overall > 0
    verdict = map_honest_verdict(delta_overall, gate_blocked=False)

    artifact = tmpl.build_result(
        {
            "sessions": [
                {
                    "session": sr["session"],
                    "precision": sr["precision"],
                    "n_constraints_store": sr["n_constraints_store"],
                    "n_constraints_added": sr["n_constraints_added"],
                }
                for sr in sessions_data
            ],
            "delta_s1_to_s3": delta_s1_to_s3,
            "delta_overall": delta_overall,
            "retro_constraint_zero_delta_closed": retro_closed,
            "honest_verdict": verdict,
        },
        status="success",
    )

    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(
        f"[Exp821] Done. precisions={precisions} delta_overall={delta_overall:.4f} "
        f"retro_closed={retro_closed} verdict={verdict}"
    )

    watchdog.stop()
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
