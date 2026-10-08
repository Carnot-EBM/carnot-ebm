# Lessons from the OpenAI math repository (read 2026-10-08)

Source: https://github.com/openai/math (Apache-2.0, last push 2026-10-06, Lean).
Status of this note: a record of what was read, what was judged useful, and what was
rejected. Nothing here is built. The operator asked for the research to be written down
so a later session does not re-derive it.

## What the repository is

- 722 manuscripts in 372 result families. An unreleased internal OpenAI model produced
  them.
- Stated procedure: about 3 hours of ChatGPT Pro compute per result. About 4,000
  problems were posed. A significance filter reduced them to the 372 families.
- Stated exceptions: two results used a different procedure. One writeup was edited by a
  human for readability.
- Verification is uneven. Some results have Lean proofs and some do not. The README says
  the unformalized results "could have issues".
- Lean side: a `formalization.yaml` catalogue links each paper to its formalization. A
  `ComparatorChallenges/` directory holds one `.lean` statement and one `.json` config
  per result. The Comparator tool (leanprover/comparator) checks a proof against the
  fixed statement. It uses `landrun` as a sandbox and `lean4export` to export the proof.
- Corrections are published as new versions. Old versions stay available.

## What I did not check

- I read the README, the manuscript map head, the catalogue head and the Comparator
  README. I did not read any paper or any Lean file.
- I did not judge any mathematical claim. The headline claims are unreviewed by me and I
  found no independent review. Treat them as claims.
- The `landrun` description below comes from memory and is not verified.

## What transfers to Carnot

1. **Selection denominator.** The repository states about 4,000 posed and 372 kept. That
   is the number a reader needs to judge how much of the output is signal. Carnot has
   thousands of experiment artifacts and one headline (the FoVer verifier result, AUROC
   0.9131). No artifact states how many verifier or configuration variants were tried
   before that headline was chosen. This is the main gap the repository exposes.
2. **Fixed statement, separate proof.** Comparator checks a proof against a statement the
   model cannot change. Carnot already does this in part with sealed predictions and
   frozen gates. No new work needed.
3. **Per-claim verification status in a machine-readable file.** Carnot has this in
   pieces: `docs/research-notes/paper_v6_anchored_claim_matrix.md` (dated 2026-05-07),
   the G1 to G4 gate in `scripts/publication_gate.py` (G4 checks one headline artifact),
   and the claim-refutation audit in `ops/experiment_claim_audit_report.md`.
4. **Versioned corrections with old versions kept.** Same as Carnot's never-prune rule.
5. **Retire saturated evaluations.** Carnot did this when it retired `double_well` and
   `rosenbrock`.

## What does not transfer

- **Lean and Comparator.** A Lean proof checks a formal statement, not the paper's
  claim. Whether the statement matches the claim is a separate risk. It is the same risk
  as a guard whose pattern list is narrower than its concept. Under the Circularity
  rule, a Lean-checked result is `execution_grounded`: the checker is the oracle. It is
  real, but it is not the open case Carnot's verifier targets, where no cheap oracle
  exists. Carnot's verifiers target structural errors, not proofs. The cost of adopting
  Lean is high and the benefit is low.

## Recommendation

Do these, in this order. Do not add a new rule. The 2026-08-21 plan
(`docs/research-notes/cumulative-coherence-rule-to-check-2026-08-21.md`) says to
consolidate, and the existing audit and gate machinery already covers most of this.

1. **Count the headline's selection denominator.** One research task, read-only. Count
   the verifier and configuration variants that were tried on FoVer before 0.9131 was
   fixed as the headline. Write the count and the method into one results artifact.
   Falsifiable outcome: either the headline survives a stated multiple-comparisons
   correction, or the paper narrows. This is the only item that could change a claim.
2. **Add one bug class to the existing claim-refutation audit.** Ask the reviewer,
   "how many candidates were tried to get this headline, and is that number stated?"
   This is a prompt edit in the existing audit. It is not a new rule and not a new lint.
3. **Refresh the anchored claim matrix when the paper is next touched.** It is five
   months old. Do not build a new catalogue. If it is regenerated, give it a status
   column (measured, reproduced, flagged, retracted, narrowed) and a denominator column.
   The paper is operator-curated, so the operator decides when.

Do not do these:

- Do not adopt Lean or Comparator.
- Do not start a `landrun` sandbox project now. Idea only. From memory, `landrun`
  restricts files and network through the Linux Landlock kernel feature and does not
  intercept syscalls the way gVisor does. If true, it may avoid the nvproxy driver-version
  block recorded in `CLAUDE.md` Security Requirements. Verify first: install it, run one
  CPU JAX hypothesis inside it, and check that GPU access still works. Revisit only if a
  task needs GPU isolation.

## Risks and honest limits of this recommendation

- Item 1 may find nothing wrong, or it may find the denominator is large. Either result
  is useful. A large denominator does not prove the headline false. It raises the bar.
- Item 1 depends on what counts as a "variant tried". That definition must be written
  before counting, or the count can be tuned to the answer.
- The comparison to OpenAI's 4,000 to 372 is loose. Their filter is a significance filter
  on research problems. Carnot's selection is of one headline from many measurements.
