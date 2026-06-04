# Product Headline Status Doc Proposal - 2026-06-04

This is a proposal for the operator. It does not edit the operator-curated technical report.

## Current Status

- FoVer methods headline 0.9131 remains the sole defensible headline.
- The product code-repair headline stays demoted.
- Exp 3798 reran the code-repair candidate and reproduced delta=0.0pp, so the historical +18pp code-repair product headline did not survive.
- Exp 2090 CRANE is not usable as product-headline support because the live artifact re-check is CRITICAL despite Exp 3799's stale G4 stamp.
- A clean operator GPU HumanEval rerun with full provenance is the only path to restore a product headline.

## Proposed Technical-Report Change

Retire or correct the demoted HumanEval code-repair prose in `docs/technical-report.md` that still presents old product-headline numbers as if they were defensible live headline results. Replace that prose with the FoVer methods headline, or explicitly state that the product headline is awaiting a clean operator rerun.

## Evidence

- Result artifact: `results/experiment_3812_product_headline_status_consolidation.json`
- Verdict: `complete: product_headline_status_recorded_code_repair_false_crane_false_sole_defensible_fover_0.9131_stays_demoted_doc_proposal_emitted_operator_curated_doc_unedited`
- Operator-curated documents unedited: true
